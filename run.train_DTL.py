import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from data.RPF_counter_v3 import *
from model.translation_model import TranslationModel
from model_pretrain_two_tasks import PretrainingTrainer
from model_finetune import FineTuneTrainer
from data.dataset import TranslationDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 设置本进程可见 GPU
rank = int(os.environ['LOCAL_RANK'])        # torchrun 会设
world_size = int(os.environ['WORLD_SIZE'])  # 一般等于 nproc_per_node
torch.cuda.set_device(rank)
dist.init_process_group('nccl', rank=rank, world_size=world_size)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# load file
lib_path = "/home/user/data3/rbase/translation_pred/models/lib"
tx_meta_file = lib_path + '/transcript_meta.pkl'
tx_cds_file = lib_path + '/transcript_cds.pkl'
tx_seq_file = lib_path + '/tx_seq.v48.pkl'
dataset_dir = '/home/user/data3/rbase/translation_pred/data/dataset'

# split transcripts
chrom_train = ["chr" + str(i) for i in range(1,17)] + ["X"]
chrom_valid = ["chr" + str(i) for i in range(17,21)]
chrom_test = ["chr" + str(i) for i in range(21,23)] + ["Y"]
tissue_types = ["HEK293 cell", "HEK293T cell", "HeLa cell", "brain"]

# load dataset
max_len = 2000
dataset_name = 'SRR15513148_49_50_51_52'
train_dataset_path = os.path.join(dataset_dir, dataset_name + ".train.h5")
val_dataset_path = os.path.join(dataset_dir, dataset_name + ".valid.h5")
train_dataset = TranslationDataset.from_h5(train_dataset_path, lazy=True)
val_dataset = TranslationDataset.from_h5(val_dataset_path, lazy=True)

# Create model
translation_model = TranslationModel(
    d_seq = 4, 
    d_count = 10, 
    d_model = 512,
    pmt_len = 3,
    num_tissues = len(tissue_types),
    d_ff = 2048, 
    heads = 16,
    number_of_layers = 12, 
    max_seq_len = max_len, 
    PE = "RoPE"
    ).cuda(rank)

translation_model = DDP(
    translation_model, 
    device_ids=[rank], 
    output_device=rank,
    find_unused_parameters=True # freeze parameters
    )

# Define the pre-training trainer
beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-9
epoch_num = 50
trainer = PretrainingTrainer(
    model = translation_model,
    model_name = "ribomodel_16h_512.max_2000.DTL",
    dataset = train_dataset,
    val_dataset = val_dataset,
    batch_size = 10,
    checkpoint_dir = '/home/user/data3/rbase/translation_pred/models/checkpoint',
    log_dir = '/home/user/data3/rbase/translation_pred/models/log/pretrain',
    world_size = world_size,
    rank = rank,
    resume = True,
    save_every=1,
    epoch_num = epoch_num,
    learning_rate = 0.001,
    warmup_perc = 0.2,
    accumulation_steps = 5,
    beta = (beta_1, beta_2),
    epsilon = epsilon
)
trainer.pretrain()

# Define the fine-tune trainer
fine_trainer = FineTuneTrainer(
    model = translation_model,
    model_name = "ribomodel_16h_512.max_2000.DTL",
    dataset = train_dataset,
    val_dataset = val_dataset,
    batch_size = 32,
    checkpoint_dir = '/home/user/data3/rbase/translation_pred/models/checkpoint',
    log_dir = '/home/user/data3/rbase/translation_pred/models/log/fine_tune',
    world_size = world_size,
    rank = rank,
    resume = True,
    epoch_num = 100,
    full_model_epoch_perc = 0.5,
    learning_rate = 0.0001,
    warmup_perc = 0.2,
    accumulation_steps = 5,
    beta = (beta_1, beta_2),
    epsilon = epsilon
)
fine_trainer.fine_tune()

# DDP end
dist.destroy_process_group()