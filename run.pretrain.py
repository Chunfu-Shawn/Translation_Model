import os, json
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from data.RPF_counter_v3 import *
from model.translation_base_model import TranslationBaseModel
from model.output_heads import DensityPredictorHead, CellClassificationHead
from model_pretrain import PretrainingTrainer
from data.dataset import TranslationDataset

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
rank = int(os.environ['LOCAL_RANK'])        # torchrun 会设
world_size = int(os.environ['WORLD_SIZE'])  # 一般等于 nproc_per_node
torch.cuda.set_device(rank)
dist.init_process_group('nccl', rank=rank, world_size=world_size)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# load dataset
dataset_dir = '/home/user/data3/rbase/translation_pred/data/dataset'
dataset_name = "datasets_8_cell_types_4k"
train_dataset_path = os.path.join(dataset_dir, dataset_name + ".train.h5")
val_dataset_path = os.path.join(dataset_dir, dataset_name + ".valid.h5")
train_dataset = TranslationDataset.from_h5(train_dataset_path, lazy=True)
val_dataset = TranslationDataset.from_h5(val_dataset_path, lazy=True)

# create model
# "config/base_model_512d_16h_12l_3c_1drop.yaml"
# "config/base_model_384d_8h_10l_3c_1drop.yaml"
# "config/base_model_256d_8h_8l_3c_1drop.yaml"
base_model = TranslationBaseModel.from_config("config/base_model_512d_16h_12l_8c_1drop.yaml").cuda(rank)

# create heads
base_model.add_head(
    "count",
    DensityPredictorHead.create_from_model(
        base_model,
        d_pred_h = 64
        ),
    overwrite = True
)
base_model.add_head(
    "cell",
    CellClassificationHead.create_from_model(
        base_model,
        d_pred_h = 64
        ),
    overwrite = True
)
print(base_model.list_heads())

# wrap with DDP
base_model = DDP(
    base_model,
    device_ids=[rank],
    output_device=rank,
    find_unused_parameters=True # freeze parameters
    )

# Define the pre-training trainer
epoch_num = 50
trainer = PretrainingTrainer(
    model = base_model,
    dataset = train_dataset,
    val_dataset = val_dataset,
    batch_size = 5,
    checkpoint_dir = '/home/user/data3/rbase/translation_pred/models/checkpoint',
    log_dir = '/home/user/data3/rbase/translation_pred/models/log',
    world_size = world_size,
    rank = rank,
    resume = True,
    save_every = 1,
    epoch_num = epoch_num,
    mask_value = 0,
    mask_learn_task_warmup_perc = {"count": 0, "cell": 0.3},
    mask_perc = {"count": 0.3, "cell": 0.3},
    learning_rate = 0.0005,
    lr_warmup_perc = 0.2,
    accumulation_steps = 6,
    beta = (0.9, 0.98),
    epsilon = 1e-9,
    weight_decay = 0.01
)
trainer.pretrain()

# DDP end
dist.destroy_process_group()