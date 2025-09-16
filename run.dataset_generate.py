import os, json
import torch
from data.RPF_counter_v3 import *
from data.dataset_generator import DatasetGenerator
from data.dataset import TranslationDataset


# load file
lib_path = "/home/user/data3/rbase/translation_pred/models/lib"
tx_seq_file = lib_path + '/tx_seq.v48.pkl'
tx_meta_file = lib_path + '/transcript_meta.pkl'
tx_cds_file = lib_path + '/transcript_cds.pkl'
dataset_config = '/home/user/data/rbase/ribosome_profiling/Cell_line/read_count/file_dict.json'
tx_coding_emb_file = lib_path + '/transcript_start_stop_embedding.pkl'
dataset_dir = '/home/user/data3/rbase/translation_pred/data/dataset'

# load dataset config file
with open(dataset_config, "r") as f:
    datasets = json.load(f)

# split transcripts
chrom_groups = {
    "train": ["chr" + str(i) for i in range(1,18)] + ["X"],
    "valid": ["chr" + str(i) for i in range(18,21)],
    "test": ["chr21", "chr22", "Y"]
    }
len_dataset = 3
cell_types = datasets["cell_type"][:len_dataset]
read_counts = datasets["read_count"][:len_dataset]
print(cell_types)

# create data generator
max_len = 4000
dataset_grt = DatasetGenerator(tx_seq_file, tx_meta_file, tx_cds_file, 
                                chrom_groups,
                                all_cell_types = cell_types, 
                                min_length = 200, max_length = max_len,
                                motif_file_path=lib_path +"/RBP_motif_annotation.v1.tsv")

# create and save dataset
dataset_name = "datasets_" + str(len_dataset) + "_cell_types_4k"
dataset_path = os.path.join(dataset_dir, dataset_name)

dataset_grt.generate_save_dataset(
    read_counts,
    [tx_coding_emb_file] * len_dataset,
    cell_types,
    dataset_path, "h5")

# load dataset
torch.set_printoptions(threshold=np.inf)
val_dataset = TranslationDataset.from_h5(dataset_path + ".valid.h5", lazy=True)

print(val_dataset.__len__())
print(val_dataset.get_identifier(0))
for value in val_dataset.__getitem__(0):
    print(value)
