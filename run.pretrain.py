import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from data.RPF_counter_v3 import *
from model.translation_base_model_adaLNZero import TranslationBaseModel
from model.mask_heads import PsiteDensityHead
from train.model_pretrain import PretrainingTrainer
from data.translation_dataset import TranslationDataset
from utils import print_param_counts

rank = int(os.environ['LOCAL_RANK'])        # torchrun 会设
world_size = int(os.environ['WORLD_SIZE'])  # 一般等于 nproc_per_node
torch.cuda.set_device(rank)
dist.init_process_group('nccl', rank=rank, world_size=world_size)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# load dataset
dataset_dir = '/home/user/data3/rbase/translation_model/data/dataset/'
dataset_name = "7c_4k_depth0.1_cov0.1_rpm1"
train_dataset_path = os.path.join(dataset_dir, dataset_name + ".train.h5")
val_dataset_path = os.path.join(dataset_dir, dataset_name + ".valid.h5")
train_dataset = TranslationDataset.from_h5(train_dataset_path, lazy=True)
val_dataset = TranslationDataset.from_h5(val_dataset_path, lazy=True)

########### Stage 1 for pre-training ##########
# create model
base_model = TranslationBaseModel.from_config("config/base_model_adaLNZero_384d_8h_10l_7c_8ad.yaml").cuda(rank)
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
    dataset = train_dataset,
    val_dataset = val_dataset,
    dataset_name = dataset_name,
    batch_size = 12,
    checkpoint_dir = '/home/user/data3/rbase/translation_model/models/checkpoint/pretrain',
    log_dir = '/home/user/data3/rbase/translation_model/models/log/pretrain',
    world_size = world_size,
    rank = rank,
    resume = True,
    save_every = 1,
    epoch_num = epoch_num,
    mask_value = 0,
    mask_perc = {"count": (0.3, 1.5), "cell": 0.2},
    learning_rate = 0.001,
    lr_warmup_perc = 0.3,
    accumulation_steps = 4,
    balance_classes = False,
    beta = (0.9, 0.98),
    epsilon = 1e-9,
    weight_decay = 0.01
)
trainer.pretrain()

########### Stage 2 for pre-training (fine-tune) ##########

# load Stage 1 (Pretrain) 训练好的权重
# create model
# base_model = TranslationBaseModel.from_config("config/base_model_adaLNZero_384d_8h_10l_7c.yaml").cuda(rank)
# # create heads
# base_model.add_head(
#     "count",
#     PsiteDensityHead.create_from_model(
#         base_model,
#         d_pred_h = 256
#         ),
#     overwrite = True
# )
# checkpoint = torch.load(
#     os.path.join(
#         '/home/user/data3/rbase/translation_model/models/checkpoint/pretrain', 
#         ".".join([base_model.model_name, dataset_name, "48_0.001.best.pt"])
#         )
# )
# base_model.load_state_dict(checkpoint["model"])

# # freeze encoder except adaptive layer normalization
# keywords_to_unfreeze = [
#     "adaln_modulation",   # AdaLN 模块名称
#     "cell_embed",         # 细胞类型词嵌入层
#     "heads.count"         # 丰度预测头
# ]
# base_model = freeze_encoder_for_finetuning(base_model, trainable_keywords=keywords_to_unfreeze)
# base_model = DDP(
#     base_model,
#     device_ids=[rank],
#     output_device=rank,
#     find_unused_parameters=True # freeze parameters
# )

# # trainer
# epoch_num = 5
# ft_trainer = PretrainingTrainer(
#     model = base_model,
#     dataset = train_dataset,
#     val_dataset = val_dataset,
#     dataset_name = dataset_name,
#     batch_size = 12,
#     checkpoint_dir = '/home/user/data3/rbase/translation_model/models/checkpoint/finetune',
#     log_dir = '/home/user/data3/rbase/translation_model/models/log/finetune',
#     world_size = world_size,
#     rank = rank,
#     resume = True,
#     save_every = 1,
#     epoch_num = epoch_num,
#     mask_value = 0,
#     mask_perc = {"count": 1, "cell": 0.2},
#     learning_rate = 0.0001,
#     lr_warmup_perc = 0.3,
#     accumulation_steps = 2,
#     balance_classes = True, # balanced sampling for cell types
#     beta = (0.9, 0.98),
#     epsilon = 1e-9,
#     weight_decay = 0.01
# )
# ft_trainer.pretrain()

# DDP end
dist.destroy_process_group()