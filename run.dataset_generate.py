import os, json
from data.RPF_counter_v3 import *
from data.dataset_generator import DatasetGenerator


# load file
lib_path = "/home/user/data3/rbase/translation_model/models/lib"
tx_seq_file = lib_path + '/tx_seq.v48.pkl'
tx_meta_file = lib_path + '/transcript_meta.pkl'
tx_cds_file = lib_path + '/transcript_cds.pkl'
# /home/user/data3/yaoc/translation_model/data/file_dict.json
dataset_config = '/home/user/data3/rbase/translation_model/data/datasets_SW480_merged.json'
tx_coding_emb_file = lib_path + '/transcript_start_stop_embedding.pkl'
dataset_dir = '/home/user/data3/rbase/translation_model/data/dataset'

# load dataset config file
with open(dataset_config, "r") as f:
    datasets = json.load(f)

# split transcripts
chrom_groups = {
    "train": ["chr" + str(i) for i in range(1,21)] + ["X"],
    "test": ["chr21", "chr22", "Y"]
    }
cell_types = datasets["cell_type"]
count_files = datasets["read_count"]

len_dataset = len(count_files)
print(cell_types)
# for i in range(len(count_files)):
# create data generator
max_len = 6000
dataset_grt = DatasetGenerator(
    tx_seq_file, tx_meta_file, tx_cds_file, 
    chrom_groups, all_cell_types = cell_types, 
    min_length = 200, max_length = max_len,
    motif_file_path = lib_path +"/RBP_motif_annotation.v1.tsv"
    )

# create and save dataset
dataset_name = f"datasets_SW480_6k.depth0.05_cov0.05"
dataset_path = os.path.join(dataset_dir, dataset_name)

dataset_grt.generate_save_dataset(
    count_files,
    [tx_coding_emb_file] * len_dataset,
    cell_types,
    coverage = 0.05,
    depth = 0.05,
    keep_read_len = False,
    nor_method = "mean",
    nor_non_zero = True,
    out_path = dataset_path,
    fmt = "h5"
)
