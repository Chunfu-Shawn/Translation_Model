import os
import torch
import loralib as lora
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from data.RPF_counter_v3 import *
from model.translation_base_model_adaLNZero import TranslationBaseModel
from model.coding_predictor import DSConvCodingHead
from train.model_finetune import FineTuningTrainer
from data.coding_dataset import CodingDataset
from lora_utils import build_lora_model_from_pretrained, assert_all_replaced
from utils import print_param_counts

rank = int(os.environ['LOCAL_RANK'])        # torchrun 会设
world_size = int(os.environ['WORLD_SIZE'])  # 一般等于 nproc_per_node
torch.cuda.set_device(rank)
dist.init_process_group('nccl', rank=rank, world_size=world_size)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# load dataset
dataset_dir = '/home/user/data3/rbase/translation_model/data/dataset'
dataset_name = "coding_datasets_brain_fetal_6k"
train_dataset_path = os.path.join(dataset_dir, dataset_name + ".train.h5")
val_dataset_path = os.path.join(dataset_dir, dataset_name + ".valid.h5")
train_dataset = CodingDataset.from_h5(train_dataset_path, lazy=True)
val_dataset = CodingDataset.from_h5(val_dataset_path, lazy=True)

# create model and insert LoRA modules
lora_model = build_lora_model_from_pretrained(
    TranslationBaseModel.from_config("config/base_model_adaLNZero_384d_8h_10l_4c.yaml"), 
    r=16, lora_alpha=32
    ).cuda(rank)
assert_all_replaced(lora_model)

# mark only LoRA as trainable
lora.mark_only_lora_as_trainable(lora_model)

# load pretrained weights
lora_model.load_pretrained_weights(
    "../checkpoint/pretrain/base_model_adaLNZero_384d_8h_10l_4c_16ad-PsiteDensityHead.6k_depth0.1_cov0.1.40_0.001.best.pt")

# create heads
lora_model.add_head(
    "coding",
    DSConvCodingHead.create_from_model(
        lora_model,
        d_pred_h = 256,
        kernel_sizes = [7, 7]
        ),
    overwrite = True,
    requires_trg_inputs = False
)
print_param_counts(lora_model)

# wrap with DDP
lora_model = DDP(
    lora_model,
    device_ids=[rank],
    output_device=rank,
    find_unused_parameters=True
    )

# Define the pre-training trainer
epoch_num = 5
trainer = FineTuningTrainer(
        model = lora_model,
        dataset = train_dataset,
        val_dataset = val_dataset,
        batch_size = 10,
        checkpoint_dir = '/home/user/data3/rbase/translation_model/models/checkpoint',
        log_dir = '/home/user/data3/rbase/translation_model/models/log',
        world_size = world_size,
        rank = rank,
        resume = True,
        save_every = 1,
        epoch_num = epoch_num,
        learning_rate = 0.0001,
        lr_warmup_perc = 0.3,
        accumulation_steps = 1,
        beta = (0.9, 0.98),
        epsilon = 1e-9,
        weight_decay = 0.01
    )
trainer.finetune()

# DDP end
dist.destroy_process_group()