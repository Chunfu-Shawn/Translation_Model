import os
import torch
import loralib as lora
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from data.RPF_counter_v3 import *
from model.translation_base_model import TranslationBaseModel
from model.coding_predictor import CodingNetHead, DSConvCodingHead, DSDilatedConvCodingHead
from model_finetune_startstop import FineTuningTrainer
from data.dataset import TranslationDataset
from lora_utils import build_lora_model_from_pretrained, assert_all_replaced, print_param_counts

rank = int(os.environ['LOCAL_RANK'])        # torchrun 会设
world_size = int(os.environ['WORLD_SIZE'])  # 一般等于 nproc_per_node
torch.cuda.set_device(rank)
dist.init_process_group('nccl', rank=rank, world_size=world_size)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# load dataset
dataset_dir = '/home/user/data3/rbase/translation_pred/data/dataset'
dataset_name = "datasets_3_cell_types_4k"
train_dataset_path = os.path.join(dataset_dir, dataset_name + ".train.h5")
val_dataset_path = os.path.join(dataset_dir, dataset_name + ".valid.h5")
train_dataset = TranslationDataset.from_h5(train_dataset_path, lazy=True)
val_dataset = TranslationDataset.from_h5(val_dataset_path, lazy=True)

# create model and insert LoRA modules
# "config/base_model_512d_16h_12l_3c_1drop.yaml"
# "config/base_model_384d_8h_10l_3c_1drop.yaml"
# "config/base_model_256d_8h_8l_3c_1drop.yaml"
lora_model = build_lora_model_from_pretrained(
    TranslationBaseModel.from_config("config/base_model_256d_8h_8l_3c_1drop.yaml"), 
    r=4, lora_alpha=16
    ).cuda(rank)
assert_all_replaced(lora_model)

# mark only LoRA as trainable
lora.mark_only_lora_as_trainable(lora_model)

# load pretrained weights
lora_model.load_pretrained_weights("../checkpoint/pretrain/base_model_256d_8h_8l_3c_1drop.0.01_0.0005_0.2.latest.pt")

# create heads
lora_model.add_head(
    "coding",
    DSDilatedConvCodingHead.create_from_model(
        lora_model,
        d_pred_h = 128,
        trunk_layers = 2,
        task_layers = 2,
        kernel_size = 3
        ),
    overwrite = True
)
print_param_counts(lora_model)

# wrap with DDP
lora_model = DDP(
    lora_model, #torch.compile(lora_model, mode="reduce-overhead"),
    device_ids=[rank],
    output_device=rank
    )

# Define the pre-training trainer
epoch_num = 5
trainer = FineTuningTrainer(
        model = lora_model,
        dataset = train_dataset,
        val_dataset = val_dataset,
        batch_size = 18,
        checkpoint_dir = '/home/user/data3/rbase/translation_pred/models/checkpoint',
        log_dir = '/home/user/data3/rbase/translation_pred/models/log',
        world_size = world_size,
        rank = rank,
        resume = True,
        save_every = 1,
        epoch_num = epoch_num,
        learning_rate = 0.0001,
        lr_warmup_perc = 0.2,
        accumulation_steps = 3,
        beta = (0.9, 0.98),
        epsilon = 1e-9,
        weight_decay = 0.01
    )
trainer.finetune()

# DDP end
dist.destroy_process_group()