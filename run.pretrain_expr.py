import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from data.RPF_counter_v3 import *
from model.translation_base_model_expr import TranslationBaseModel
from model.mask_heads import PsiteDensityHead
from train.model_pretrain_expr import PretrainingTrainer
from utils import print_param_counts

rank = int(os.environ['LOCAL_RANK'])        # torchrun 会设
world_size = int(os.environ['WORLD_SIZE'])  # 一般等于 nproc_per_node
torch.cuda.set_device(rank)
dist.init_process_group('nccl', rank=rank, world_size=world_size)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# load dataset
dataset_dir = '/home/user/data3/rbase/translation_model/data/dataset/'
## human
human_dataset_name = "human_7c_8k_depth0.1_cov0.1_rpm1"
human_train_dataset_path = os.path.join(dataset_dir, human_dataset_name + ".train.h5")
human_val_dataset_path = os.path.join(dataset_dir, human_dataset_name + ".valid.h5")
## macaque
# macaque_dataset_name = "macaque_4c_15k_depth0.1_cov0.1_rpm1"
# macaque_train_dataset_path = os.path.join(dataset_dir, macaque_dataset_name + ".train.h5")
# macaque_val_dataset_path = os.path.join(dataset_dir, macaque_dataset_name + ".valid.h5")

# create model
base_model = TranslationBaseModel.from_config(
    "/home/user/data3/rbase/translation_model/models/src/config/base_model_expr_384d_8h_10l_64env_16ad.yaml"
    ).cuda(rank)
# create heads
base_model.add_head(
    "count",
    PsiteDensityHead.create_from_model(
        base_model,
        d_pred_h = 384
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
    output_device=rank,
    # find_unused_parameters=True # freeze parameters
)

# trainer
epoch_num = 20
trainer = PretrainingTrainer(
    model = base_model,
    dataset_paths = [human_train_dataset_path,],
    val_dataset_paths = [human_val_dataset_path],
    dataset_name = "human_7c_8k_depth0.1_cov0.1_rpm1",
    batch_size = 20,
    checkpoint_dir = '/home/user/data3/rbase/translation_model/models/checkpoint/pretrain',
    log_dir = '/home/user/data3/rbase/translation_model/models/log/pretrain',
    world_size = world_size,
    rank = rank,
    resume = True,
    save_every = 1,
    epoch_num = epoch_num,
    mask_value = 0,
    mask_perc = {"count": (1.5, 1.5), "species": 0.2, "cell": 0.2},
    expr_noise_std = 0.1,
    learning_rate = 0.001,
    lr_warmup_perc = 0.3,
    accumulation_steps = 1,
    balance_classes = False,
    beta = (0.9, 0.9),
    epsilon = 1e-9,
    weight_decay = 0.01,
)
trainer.pretrain()

# DDP end
dist.destroy_process_group()