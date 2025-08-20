import os, json
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from data.RPF_counter_v3 import *
from model.translation_model_v2 import TranslationModel
from model_pretrain_three_tasks import PretrainingTrainer
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
dataset_config = '/home/user/data/rbase/ribosome_profiling/Cell_line/read_count/file_dict.json'
dataset_dir = '/home/user/data3/rbase/translation_pred/data/dataset'
# load dataset config file
len_dataset = 4
with open(dataset_config, "r") as f:
    datasets = json.load(f)

# split transcripts
chrom_train = ["chr" + str(i) for i in range(1,18)] + ["X"]
chrom_valid = ["chr" + str(i) for i in range(18,21)]
chrom_test = ["chr" + str(i) for i in range(21,23)] + ["Y"]
tissue_types = datasets["tissue_type"][:len_dataset]

# load dataset
max_len = 2000
dataset_name = "datasets-" + str(len_dataset)
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
epoch_num = 100
trainer = PretrainingTrainer(
    model = translation_model,
    model_name = "ribomodel_16h_512.max_2000.TTL",
    dataset = train_dataset,
    val_dataset = val_dataset,
    batch_size = 12,
    checkpoint_dir = '/home/user/data3/rbase/translation_pred/models/checkpoint',
    log_dir = '/home/user/data3/rbase/translation_pred/models/log/pretrain',
    world_size = world_size,
    rank = rank,
    resume = True,
    save_every = 1,
    epoch_num = epoch_num,
    mask_learn_task_warmup_perc = {"seq": 0, "count": 0.3, "tissue": 0.3},
    mask_perc = {"seq": 0.15, "count": 0.3, "tissue": 0.15},
    learning_rate = 0.001,
    lr_warmup_perc = 0.3,
    accumulation_steps = 5,
    beta = (beta_1, beta_2),
    epsilon = epsilon
)
trainer.pretrain()

# DDP end
dist.destroy_process_group()