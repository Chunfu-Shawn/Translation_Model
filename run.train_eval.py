import os
import torch
import pickle
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
# from livelossplot import PlotLosses
from utils import HiddenPrints
from data.RPF_counter import *
from model.translation_model import TranslationModel
from model_pretrain_two_tasks import PretrainingTrainer
from model_finetune import FineTuneTrainer
from data.dataset_generator import DatasetGenerator

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
RPF_count_file = '/home/user/data3/yaoc/translation_model/model/SRR15513148_49_50_51_52.read_count.pkl'
lib_path = "/home/user/data3/rbase/translation_pred/models/lib"
tx_meta_file = lib_path + '/transcript_meta.pkl'
tx_cds_file = lib_path + '/transcript_cds.pkl'
tx_seq_file = lib_path + '/tx_seq.v48.pkl'
tx_coding_emb_file = lib_path + '/transcript_coding_embedding.pkl'

with open(tx_meta_file, 'rb') as f:
    tx_meta = pickle.load(f)
with open(tx_cds_file, 'rb') as f:
    tx_cds = pickle.load(f)
with open(RPF_count_file, 'rb') as f:
    RPF_count = pickle.load(f)
with open(tx_seq_file, 'rb') as f:
    tx_seq = pickle.load(f)
with open(tx_coding_emb_file, 'rb') as f:
    coding_emb = pickle.load(f)

# split transcripts
chrom_train = ["chr" + str(i) for i in range(1,17)] + ["X"]
chrom_valid = ["chr" + str(i) for i in range(17,21)]
chrom_test = ["chr" + str(i) for i in range(21,23)] + ["Y"]
tissue_types = ["HEK293 cell", "HEK293T cell", "HeLa cell", "brain"]

# create data generator
max_len = 2000
train_dataset_grt = DatasetGenerator(tx_seq, tx_meta, tx_cds, chrom_train, 
                                     all_tissue_types=tissue_types, min_length=200, max_length=max_len,
                                     mask_perc=[0.1, 0.4], motif_file_path=lib_path +"/RBP_motif_annotation.v1.tsv")
val_dataset_grt = DatasetGenerator(tx_seq, tx_meta, tx_cds, chrom_valid, 
                                   all_tissue_types=tissue_types, min_length=200, max_length=max_len,
                                   mask_perc=[0.1, 0.4], motif_file_path=lib_path +"/RBP_motif_annotation.v1.tsv")
# create dataset
train_dataset = train_dataset_grt.generate_dataset(RPF_count, coding_emb, "brain")
val_dataset = val_dataset_grt.generate_dataset(RPF_count, coding_emb, "brain")

# Create model
translation_model = TranslationModel(
    d_seq = 4, 
    d_count = 10, 
    d_model = 256,
    pmt_len = 3,
    num_tissues = len(tissue_types),
    d_ff = 2048, 
    heads = 8,
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
    model_name = "ribomodel_8h_256.max_2000.less_seq_pred.TTL",
    dataset = train_dataset,
    val_dataset = val_dataset,
    batch_size = 20,
    checkpoint_dir = '/home/user/data3/rbase/translation_pred/models/checkpoint',
    log_dir = '/home/user/data3/rbase/translation_pred/models/log/pretrain',
    world_size = world_size,
    rank = rank,
    resume = True,
    save_every=3,
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
    model_name = "ribomodel_8h_256.max_2000.less_seq_pred.MTL",
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
    accumulation_steps = 2,
    beta = (beta_1, beta_2),
    epsilon = epsilon
)
fine_trainer.fine_tune()

# DDP end
dist.destroy_process_group()