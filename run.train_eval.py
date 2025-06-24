import os
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

# load datasets
## split dataset
chrom_train = ["chr" + str(i) for i in range(1,17)] + ["X"]
chrom_valid = ["chr" + str(i) for i in range(17,21)]
chrom_test = ["chr" + str(i) for i in range(21,23)] + ["Y"]
## create dataset of target transcripts
batch_size = 12
loader = BatchDatasetLoader(tx_seq, tx_arrays, chrom_train, batch_size, min_length=200, max_length=2000)
## data embedding
train_dataset = loader.compute_batches(RPF_count)
val_dataset = loader.compute_batches(RPF_count)
# test_dataset = pbd(RPF_count, tx_seq, tx_arrays, chrom_test, batch_size, min_length=100)

# Create model
bs, seq_len, d_input = train_dataset[0]["masked_embedding"].shape
max_seq_len = max(x['length'] for x in train_dataset)
d_model = 32
dff = 1024
heads = 4
number_of_layers = 12
encoder_transformer = nn.DataParallel(
    EncoderOnlyTransformer(d_input, d_model, dff, heads, number_of_layers, max_seq_len, PE="RoPE"), 
    device_ids=[0,1]
).cuda()

# Define the training parameters
lr = 0.00005
beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-9
trainer = Trainer(
    model = encoder_transformer,
    dataset = train_dataset,
    checkpoint_dir = '/home/user/data3/rbase/translation_pred/models/checkpoint/',
    batch_size = bs,
    learning_rate = lr,
    accumulation_steps = 5,
    beta = (beta_1, beta_2),
    epsilon = epsilon
)

# Train
epoch_num = 100
# liveloss = PlotLosses()
train_loss = {'epoch': [], 'batch': []}
val_loss = {'epoch': [], 'batch': []}
for epoch in range(epoch_num):
    logs = {}
    train_epoch_loss, train_batch_loss = trainer.train(epoch)
    train_loss['epoch'].append(train_epoch_loss)
    train_loss['batch'].append(train_batch_loss)
    val_epoch_loss, val_batch_loss = trainer.eval(epoch, val_dataset)
    val_loss['epoch'].append(val_epoch_loss)
    val_loss['batch'].append(val_epoch_loss)
    
    # update loss plot
    # logs['loss'] = train_epoch_loss
    # logs['val_loss'] = val_epoch_loss
    # liveloss.update(logs)
    # liveloss.draw()

# save loss data
log_dir = '/home/user/data3/rbase/translation_pred/models/log/'
with open(log_dir + "train_loss.pkl", 'wb') as f_t_loss:
    pickle.dump(train_loss, f_t_loss)
with open(log_dir + "val_loss.pkl", 'wb') as f_v_loss:
    pickle.dump(val_loss, f_v_loss)