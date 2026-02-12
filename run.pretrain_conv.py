import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from data.RPF_counter_v3 import *
from model.translation_base_model_conv import TranslationBaseModel
from model.mask_heads import PsiteDensityHead
from train.model_pretrain import PretrainingTrainer
from data.dataset import TranslationDataset
from utils import print_param_counts

rank = int(os.environ['LOCAL_RANK'])        # torchrun 会设
world_size = int(os.environ['WORLD_SIZE'])  # 一般等于 nproc_per_node
torch.cuda.set_device(rank)
dist.init_process_group('nccl', rank=rank, world_size=world_size)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# load dataset
dataset_dir = '/home/user/data3/yaoc/translation_model/data/input_dataset/'
dataset_name = "datasets_4_cell_types_6k.no_rl.depth0.1_cov0.1"
train_dataset_path = os.path.join(dataset_dir, dataset_name + ".train.h5")
val_dataset_path = os.path.join(dataset_dir, dataset_name + ".valid.h5")
train_dataset = TranslationDataset.from_h5(train_dataset_path, lazy=True)
val_dataset = TranslationDataset.from_h5(val_dataset_path, lazy=True)

# create model
base_model = TranslationBaseModel.from_config("config/base_model_conv_384d_8h_10l_4c.yaml").cuda(rank)
base_model.model_name = base_model.model_name
# create heads
base_model.add_head(
    "count",
    PsiteDensityHead.create_from_model(
        base_model,
        d_pred_h = 256
        ),
    overwrite = True
)
print(base_model.model_name)
print(base_model.list_heads())
print_param_counts(base_model)

# wrap with DDP
base_model = DDP(
    base_model,
    device_ids=[rank],
    output_device=rank
    # find_unused_parameters=True # freeze parameters
)

# Define the pre-training trainer
epoch_num = 20
trainer = PretrainingTrainer(
    model = base_model,
    dataset = train_dataset,
    val_dataset = val_dataset,
    dataset_name = "6k_depth0.1_cov0.1",
    batch_size = 20,
    checkpoint_dir = '/home/user/data3/rbase/translation_model/models/checkpoint',
    log_dir = '/home/user/data3/rbase/translation_model/models/log',
    world_size = world_size,
    rank = rank,
    resume = True,
    save_every = 1,
    epoch_num = epoch_num,
    mask_value = 0,
    head_warmup_perc = {"count": 0},
    mask_perc = {"count": (0.3, 1.5), "cell": 0.05},
    learning_rate = 0.001,
    lr_warmup_perc = 0.3,
    accumulation_steps = 1,
    beta = (0.9, 0.98),
    epsilon = 1e-9,
    weight_decay = 0.01
)
trainer.pretrain()

# DDP end
dist.destroy_process_group()