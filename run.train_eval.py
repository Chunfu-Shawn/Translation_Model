import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import torch
import torch.nn as nn
# from livelossplot import PlotLosses
from data_generate_RPF_count import *
from dataset_prepare import *
from translation_transformer import *
from RotaryEmbedding import *
from model_trainer import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.device_count()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# load file
tx_arrays_file = '/home/user/data3/rbase/translation_pred/models/lib/transcript_arrays.pkl'
RPF_count_file = '/home/user/data3/rbase/translation_pred/models/test/SRR15513158.read_count.pkl'
tx_seq_file = '/home/user/data3/rbase/translation_pred/models/lib/tx_seq.v48.pkl'
with open(tx_arrays_file, 'rb') as f:
    tx_arrays = pickle.load(f)
with open(RPF_count_file, 'rb') as f:
    RPF_count = pickle.load(f)
with open(tx_seq_file, 'rb') as f:
    tx_seq = pickle.load(f)
# split transcripts
chrom_train = ["chr" + str(i) for i in range(1,17)] + ["X"]
chrom_valid = ["chr" + str(i) for i in range(17,21)]
chrom_test = ["chr" + str(i) for i in range(21,23)] + ["Y"]
# create data loader of transcripts
batch_size = 12
trainDataLoader = BatchDataLoader(tx_seq, tx_arrays, chrom_train, batch_size, min_length=200, max_length=2000)
valDataLoader = BatchDataLoader(tx_seq, tx_arrays, chrom_valid, batch_size, min_length=200, max_length=2000)

# data embedding
train_dataset = trainDataLoader.compute_batches(RPF_count)
val_dataset = valDataLoader.compute_batches(RPF_count)
# test_dataset = pbd(RPF_count, tx_seq, tx_arrays, chrom_test, batch_size, min_length=100)

# Create model
bs, seq_len, d_input = train_dataset[0]["masked_embedding"].shape
max_seq_len = max(x['length'] for x in train_dataset)
d_model = 32
dff = 2048
heads = 4
number_of_layers = 12
encoder_transformer = nn.DataParallel(
    EncoderOnlyTransformer(d_input, d_model, dff, heads, number_of_layers, max_seq_len, PE="RoPE", flash_attn=True), 
    device_ids=[0,1]).cuda()

# Define the training parameters
lr = 0.001
beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-6
epoch_num = 100
trainer = Trainer(
    model = encoder_transformer,
    model_name = "ribomodel",
    dataset = train_dataset,
    checkpoint_dir = '/home/user/data3/rbase/translation_pred/models/checkpoint/',
    log_dir = '/home/user/data3/rbase/translation_pred/models/log/',
    batch_size = bs,
    epoch_num = epoch_num,
    learning_rate = lr,
    accumulation_steps = 5,
    beta = (beta_1, beta_2),
    epsilon = epsilon
)

# Train
trainer.pretrain(val_dataset)