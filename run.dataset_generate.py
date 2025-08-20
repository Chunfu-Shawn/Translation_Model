import os, json
from data.RPF_counter_v3 import *
from data.dataset_generator_v2 import DatasetGenerator
from data.dataset import TranslationDataset


# load file
lib_path = "/home/user/data3/rbase/translation_pred/models/lib"
tx_seq_file = lib_path + '/tx_seq.v48.pkl'
tx_meta_file = lib_path + '/transcript_meta.pkl'
tx_cds_file = lib_path + '/transcript_cds.pkl'
dataset_config = '/home/user/data/rbase/ribosome_profiling/Cell_line/read_count/file_dict.json'
tx_coding_emb_file = lib_path + '/transcript_coding_embedding.pkl'
dataset_dir = '/home/user/data3/rbase/translation_pred/data/dataset'

# load dataset config file
len_dataset = 4
with open(dataset_config, "r") as f:
    datasets = json.load(f)

# split transcripts
chrom_train = ["chr" + str(i) for i in range(1,17)] + ["X"]
chrom_valid = ["chr" + str(i) for i in range(17,21)]
chrom_test = ["chr" + str(i) for i in range(21,23)] + ["Y"]
tissue_types = datasets["tissue_type"][:len_dataset]
read_counts = datasets["read_count"][:len_dataset]
print(tissue_types)

# create data generator
max_len = 2000
train_dataset_grt = DatasetGenerator(tx_seq_file, tx_meta_file, tx_cds_file, chrom_train, 
                                     all_tissue_types=tissue_types, 
                                     min_length=200, max_length=max_len,
                                    motif_file_path=lib_path +"/RBP_motif_annotation.v1.tsv")
val_dataset_grt = DatasetGenerator(tx_seq_file, tx_meta_file, tx_cds_file, chrom_valid, 
                                   all_tissue_types=tissue_types, 
                                   min_length=200, max_length=max_len,
                                   motif_file_path=lib_path +"/RBP_motif_annotation.v1.tsv")
# create and save dataset
dataset_name = "datasets-" + str(len_dataset)
train_dataset_path = os.path.join(dataset_dir, dataset_name + ".train.h5")
val_dataset_path = os.path.join(dataset_dir, dataset_name + ".valid.h5")

train_dataset_grt.generate_save_dataset(
    read_counts,
    [tx_coding_emb_file] * len_dataset,
    tissue_types,
    train_dataset_path, "h5")
val_dataset_grt.generate_save_dataset(
    read_counts,
    [tx_coding_emb_file] * len_dataset,
    tissue_types,
    val_dataset_path, "h5")

# load dataset
val_dataset = TranslationDataset.from_h5(val_dataset_path)

print(val_dataset.__len__())
print(val_dataset.get_identifier(0))
for value in val_dataset.__getitem__(0):
    print(value)
