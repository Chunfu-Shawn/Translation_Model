import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from data.RPF_counter_v3 import *
from model.translation_base_model import TranslationBaseModel
from model.mask_heads import TranslationProfileHead
from train.model_pretrain import PretrainingTrainer
from utils import print_param_counts

rank = int(os.environ['LOCAL_RANK'])        # torchrun 会设
world_size = int(os.environ['WORLD_SIZE'])  # 一般等于 nproc_per_node
torch.cuda.set_device(rank)
dist.init_process_group('nccl', rank=rank, world_size=world_size)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# load dataset
dataset_dir = '/public-supool/home/annie/translation_model/dataset/'
## human
human_t_dataset_name = "human_tissue_22c_6k_depth0.1_cov0.1_rpm1"
human_t_train_dataset_path = os.path.join(dataset_dir, human_t_dataset_name + ".train.h5")
human_t_val_dataset_path = os.path.join(dataset_dir, human_t_dataset_name + ".valid.h5")
human_cl_dataset_name = "human_cell_line_18c_6k_depth0.1_cov0.1_rpm1"
human_cl_train_dataset_path = os.path.join(dataset_dir, human_cl_dataset_name + ".train.h5")
human_cl_val_dataset_path = os.path.join(dataset_dir, human_cl_dataset_name + ".valid.h5")
## macaque
macaque_dataset_name = "macaque_4c_6k_depth0.1_cov0.1_rpm1"
macaque_train_dataset_path = os.path.join(dataset_dir, macaque_dataset_name + ".train.h5")
macaque_val_dataset_path = os.path.join(dataset_dir, macaque_dataset_name + ".valid.h5")
## mouse
mouse_dataset_name = "mouse_3c_6k_depth0.1_cov0.1_rpm1"
mouse_train_dataset_path = os.path.join(dataset_dir, mouse_dataset_name + ".train.h5")
mouse_val_dataset_path = os.path.join(dataset_dir, mouse_dataset_name + ".valid.h5")

# create model
base_model = TranslationBaseModel.from_config(
    "/public-supool/home/annie/translation_model/src/config/base_model_expr_384d_16h_12l_128env_32ad.yaml"
    ).cuda(rank)
# create heads
base_model.add_head(
    "count",
    TranslationProfileHead.create_from_model(
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
)

# trainer
epoch_num = 60
trainer = PretrainingTrainer(
    model = base_model,
    dataset_paths = [human_t_train_dataset_path, human_cl_train_dataset_path, macaque_train_dataset_path, mouse_train_dataset_path],
    val_dataset_paths = [human_t_val_dataset_path, human_cl_val_dataset_path, macaque_val_dataset_path, mouse_val_dataset_path],
    dataset_name = "hs_22c_18c_rm_4c_mm_3c_6k_depth0.1_cov0.1_rpm1",
    batch_size = 50,
    checkpoint_dir = '/public-supool/home/annie/translation_model/checkpoint/pretrain',
    log_dir = '/public-supool/home/annie/translation_model/log/pretrain',
    world_size = world_size,
    rank = rank,
    resume = True,
    save_every = 1,
    epoch_num = epoch_num,
    mask_value = 0,
    mask_perc = {"count": (0.4, 1.5), "species": 0.15, "cell": 0.15},
    alpha_limit = (4.0, 4.0),
    expr_noise_std = 0.1,
    learning_rate = 0.001,
    lr_warmup_perc = 0.3,
    accumulation_steps = 2,
    balance_classes = False,
    beta = (0.9, 0.98),
    epsilon = 1e-9,
    weight_decay = 0.01,
)
trainer.pretrain()

# DDP end
dist.destroy_process_group()