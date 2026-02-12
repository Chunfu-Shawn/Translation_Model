import sys, os
sys.path.append("/home/user/data3/rbase/translation_model/models/src")
import numpy as np
import pickle
from tqdm import tqdm

# 可选：如果想用 h5 存储
try:
    import h5py
    HAS_H5 = True
except Exception:
    HAS_H5 = False

__author__ = "Chunfu Xiao"
__version__="1.1.0"
__email__ = "xiaochunfu@126.com"


# generate embeddings and return Dataset class
class OffsetPredDatasetGenerator():
    def __init__(self, 
                 tree_index_file: str, 
                 transcript_seq_file: str, 
                 transcript_meta_file: str, 
                 transcript_cds_file: str, 
                 chrom_groups: dict,
                 all_endonucleases: list = ["RNase I", "MNase"],
                 TIS_train: bool = True,
                 TTS_train: bool = True):

        if not TIS_train and not TTS_train:
            raise ValueError("Unknown training site. Use 'TIS' or 'TTS'.")
        self.TIS_train = TIS_train
        self.TTS_train = TTS_train
        # load data
        with open(tree_index_file, 'rb') as f:
            self.tree_index = pickle.load(f)
        with open(transcript_seq_file, 'rb') as f:
            self.seq_dict = pickle.load(f)
        with open(transcript_meta_file, 'rb') as f:
            self.tx_meta = pickle.load(f)
        with open(transcript_cds_file, 'rb') as f:
            self.tx_cds = pickle.load(f)

        self.chrom_groups = chrom_groups
        self.nt_mapping = dict(zip("ACGTN", range(5))) # N means unknown
        self.endonuclease_mapping = dict(zip(all_endonucleases, range(2))) # endonuclease type
        self.filtered_tids = {group : [] for group in chrom_groups}
        self._base_seq_emb = {group : {} for group in chrom_groups}
        self.cds_start_pos = {group : {} for group in chrom_groups}
        self.cds_end_pos = {group : {} for group in chrom_groups}
        
        # transcript filter
        n_tx = {"no_start_stop_codon": 0, "overlap_other_gene": 0, "valid": 0}
        for tid, value in tqdm(self.tx_cds.items(), desc=f"Filter transcripts", mininterval=1000):
            
            # print(f"--- {tid} ---")
            # 1. having start or stop codon
            if not value["start_codon"] or not value["stop_codon"]:
                n_tx["no_start_stop_codon"] += 1
                # print("[PASS] No start codon")
                continue

            # 2. filter out start/stop codon overlapping with other genes
            chrom = self.tx_meta[tid]["chrom"]
            strand = self.tx_meta[tid]["strand"]

            ol_transcripts = []
            if TIS_train:
                tis_1base = value["cds_starts"][0] if strand == "+" else value["cds_ends"][0] # start site (genomic coordianate)
                ol_transcripts.extend([iv.data for iv in self.tree_index[chrom][tis_1base]])
            if TTS_train:
                tts_1base = value["cds_ends"][-1] if strand == "+" else value["cds_starts"][-1] # start site (genomic coordianate)
                ol_transcripts.extend([iv.data for iv in self.tree_index[chrom][tts_1base]])

            # transcript (gene) overlapping start or stop codon
            ol_transcripts = set(ol_transcripts)
            ol_genes = set(self.tx_meta[tx]["gene_id"] for tx in ol_transcripts)
            if len(ol_genes) > 1: # start codon overlap other gene
                n_tx["overlap_other_gene"] += 1
                # print(f"[PASS] Translation start site overlaps with other genes: {ol_genes}")
                continue

            # finally
            group = next((g for g, chroms in chrom_groups.items() if chrom in chroms), None)
            if group != None:
                self.filtered_tids[group].append(tid)
                self._base_seq_emb[group][tid] = self.one_hot_encode(tid)
                self.cds_start_pos[group][tid] = self.tx_cds[tid]["cds_start_pos"] - 1 # 0-based for TIS
                self.cds_end_pos[group][tid] = self.tx_cds[tid]["cds_end_pos"] # 0-based for TTS
                n_tx["valid"] += 1

        print("Number of filtered transcripts: ", n_tx)
    
    def load_pickle_file(self, file_path):
        (_, filename) = os.path.split(file_path)
        name = filename.split('.')[0]
        with open(file_path, 'rb') as f:
            dict = pickle.load(f)
        
        return dict, name

    def one_hot_encode(self, tid):
        seq = self.seq_dict[tid].upper()
        seq2 = [self.nt_mapping[i] for i in seq]
        onehot = np.eye(5)[seq2, :4] # [0, 0, 0, 0] indicates N
        return onehot
    
    def generate_save_dataset(self, 
                              count_files: list, 
                              endonucleases: list, 
                              offset_tis_limits: tuple,
                              offset_tts_limits: tuple,
                              five_end_flanking: bool = True,
                              three_end_flanking: bool = True,
                              out_path = "dataset.pkl", 
                              fmt = "pickle"):
        """
        Generate and save dataset prepared for model from multiple samples
        """

        datasets = {
            group: {
                "read_length": [],
                "enzyme_id": [],
                "flanking_seq": [],
                "offset": []
            } for group in self.chrom_groups
        }
        if len(count_files) == 0 or len(count_files) != len(endonucleases):
            raise ValueError("### No files, or files provided are not equal ! ###")
        if not five_end_flanking and not three_end_flanking:
            raise ValueError("### Flanking sequences are must requested! ###")
        
        print(f"### Load count data from {len(count_files)} samples ###")
        for i in range(len(count_files)):
            endonuclease = endonucleases[i]
            count_dict, name = self.load_pickle_file(count_files[i])

            # for target transcripts
            print(f"--- Processing dataset: {count_files[i]} ---")
            for group in datasets:
                for tid in tqdm(self.filtered_tids[group], desc=f"{group} dataset generating", mininterval=200):
                    # only for imformative transcripts
                    if tid not in count_dict:
                        continue

                    # extract flanking sequences
                    seq_emb = self._base_seq_emb[group][tid]
                    seq_len = seq_emb.shape[0]
                    tis_pos = self.cds_start_pos[group][tid] # 0-based
                    tts_pos = self.cds_end_pos[group][tid] # 0-based
                    count = count_dict[tid]
                    width = 8
                    for read_l in count:
                        for pos in count[read_l]:
                            five_end = pos - 1 # 0-based
                            three_end = five_end + read_l - 1 # 0-based
                            offset_tis = tis_pos - five_end
                            offset_tts = tts_pos - five_end
                            # enough flanking sequences
                            if (five_end - width >= 0 and three_end + width + 1 <= seq_len): # ensure 2 *(window * 2 + 1)
                                tis_train = offset_tis >= offset_tis_limits[0] and offset_tis <= offset_tis_limits[1]
                                tts_train = offset_tts >= offset_tts_limits[0] and offset_tts <= offset_tts_limits[1]
                                # select reads around TIS and TTS
                                if (self.TIS_train and tis_train) or (self.TTS_train and tts_train):
                                    # multuple RPFs simultaneously
                                    cnt = count[read_l][pos]
                                    datasets[group]["read_length"].extend([read_l] * cnt)
                                    datasets[group]["enzyme_id"].extend([self.endonuclease_mapping[endonuclease]] * cnt)
                                    five_end_flanking_seq = seq_emb[five_end - width: five_end + width + 1, :]
                                    three_end_flanking_seq = seq_emb[three_end - width: three_end + width + 1, :]
                                    # add flanking sequences
                                    if five_end_flanking and three_end_flanking:
                                        flanking_seq = [np.concatenate((five_end_flanking_seq, three_end_flanking_seq), axis=0)] * cnt
                                    elif five_end_flanking:
                                        flanking_seq = [five_end_flanking_seq] * cnt
                                    else:
                                        flanking_seq = [three_end_flanking_seq] * cnt
                                    datasets[group]["flanking_seq"].extend(flanking_seq)

                                    # offset label for training
                                    if tis_train:
                                        datasets[group]["offset"].extend([offset_tis] * cnt)
                                    else:
                                        datasets[group]["offset"].extend([offset_tts] * cnt)
                                
                print(len(datasets[group]["offset"]))
                print(datasets[group]["read_length"][0])
                print(datasets[group]["enzyme_id"][0])
                print(datasets[group]["flanking_seq"][0])
                print(datasets[group]["offset"][0])
    
        # save all datasets
        print(f"--- Saving datasets ---")
        mids = ["TIS" if self.TIS_train else "",
         "TTS" if self.TTS_train else "",
         "5fq" if five_end_flanking else "",
         "3fq" if three_end_flanking else "",]
        mid = '_'.join([x for x in mids if x])
        if fmt == "pickle":
            for group, dataset in datasets.items():
                file_path = f"{out_path}.{mid}.{group}.pkl"
                with open(file_path, "wb") as f:
                    pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Saved [{group}] dataset to {file_path} (pickle).")
        elif fmt == "h5":
            for group, dataset in datasets.items():
                file_path = f"{out_path}.{mid}.{group}.h5"
                if not HAS_H5:
                    raise ImportError("h5py not available. Install h5py to use h5 format.")
                # create h5 file，build the group by tid
                with h5py.File(file_path, "w") as f:
                    grp_root = f.create_group("samples")
                    grp_root.create_dataset("read_length", data=datasets[group]["read_length"], compression="gzip")
                    grp_root.create_dataset("enzyme_id", data=datasets[group]["enzyme_id"], compression="gzip")
                    grp_root.create_dataset("flanking_seq", data=datasets[group]["flanking_seq"], compression="gzip")
                    grp_root.create_dataset("offset", data=datasets[group]["offset"], compression="gzip")
                    # metadata attributes
                    f.attrs['n_samples'] = len(datasets[group]["read_length"])
                    f.attrs['notes'] = "Saved by DatasetGenerator.save_h5ad by Chunfu Xiao"
                print(f"Saved [{group}] dataset to {file_path} (h5).")
        else:
            raise ValueError("Unknown format. Use 'pickle' or 'h5'.")