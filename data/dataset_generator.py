import sys, os
sys.path.append("/home/user/data3/rbase/translation_model/models/src")
import numpy as np
import math
import ahocorasick
from tqdm import tqdm
from typing import Literal
from data.RPF_counter_v3 import *

# 可选：如果想用 h5 存储
try:
    import h5py
    HAS_H5 = True
except Exception:
    HAS_H5 = False

__author__ = "Chunfu Xiao"
__version__="1.1.0"
__email__ = "xiaochunfu@126.com"

def load_motifs_from_file(motif_file_path):
    motifs = []
    with open(motif_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            columns = line.split()
            if len(columns) < 3:
                continue
            motif = columns[2].strip().upper()
            if motif:
                motifs.append(motif)
    return motifs

def build_automaton(motifs):
    """Build Aho-Corasick automaton with caching"""
    
    automaton = ahocorasick.Automaton()
    for motif in motifs:
        # 跳过空字符串
        if motif:
            automaton.add_word(motif, motif)
    automaton.make_automaton()

    return automaton

# generate embeddings and return Dataset class
class DatasetGenerator():
    def __init__(self, transcript_seq_file: str, 
                 transcript_meta_file: str, 
                 transcript_cds_file: str, 
                 chrom_groups: dict, 
                 all_cell_types: list,
                 min_length=0, max_length=None, 
                 motif_file_path: str = ""):
        """
        RNA sequence dataset class, could be grouped and padded by sequence length
        
        :param transcript_seqs: 字典 {transcript_id: sequence_string}
        :param batch_size: 每个批次的大小
        :param min_length: 最小序列长度（过滤短序列）
        :param max_length: 最大序列长度（过滤长序列）
        :param padding_value: 填充值
        """

        self.data = {}
        with open(transcript_seq_file, 'rb') as f:
            self.seq_dict = pickle.load(f)
        with open(transcript_meta_file, 'rb') as f:
            self.tx_meta = pickle.load(f)
        with open(transcript_cds_file, 'rb') as f:
            self.tx_cds = pickle.load(f)

        self.chrom_groups = chrom_groups
        self.nt_mapping = dict(zip("ACGTN", range(5))) # N means unknown
        self.all_cell_types = list(set(all_cell_types))
        self.cell_type_mapping = dict(
            zip(
                self.all_cell_types, 
                range(1, len(self.all_cell_types) + 1) # Total embeddings needed = N + 1
                # mapping: token string -> index [1 .. num_classes]
                )
            )
        self.motifs_list = load_motifs_from_file(motif_file_path)
        self.motif_automaton = build_automaton(self.motifs_list)
        self.filtered_tids = {group : [] for group in chrom_groups}
        self._base_seq_emb = {group : {} for group in chrom_groups}
        
        # data cache
        for tid in tqdm(self.tx_meta, desc="Data Cache", mininterval=1000):
            seq_len = len(self.seq_dict[tid])
            if seq_len < min_length:
                continue
            if max_length and seq_len > max_length:
                continue
            # different groups
            group = next((g for g, chroms in chrom_groups.items() if self.tx_meta[tid]["chrom"] in chroms), None)
            if group != None:
                self.filtered_tids[group].append(tid)
                # seq embedding
                self._base_seq_emb[group][tid] = self.one_hot_encode(tid)
    
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

    def data_quality_eval(self, seq, counts):
        seq_l = len(seq)
        n_put_codon = math.ceil(seq_l / 3)
        total_reads = 0
        cov_put_codon = [0] * n_put_codon
        for pos, dict in counts.items():
            if pos <= seq_l:
                for _, cnt in dict.items():
                    total_reads += cnt
                    cov_put_codon[(pos-1)//3] = 1

        depth = total_reads / seq_l
        coverage = cov_put_codon.count(1)

        return depth, coverage

    def count_normalize_transcript(self, 
                                   counts: np.ndarray, 
                                   method = Literal["median", "mean", "lower_quantile"],
                                   non_zero: bool = False,
                                   eps = 1):
        # non-zero counts
        nz = counts[counts > 0]

        # median or mean of non-zero counts
        if nz.size > 0:
            if method == "mean":
                dm = np.mean(nz) + eps if non_zero else np.mean(counts) + eps
            elif method == "median":
                dm = np.median(nz) + eps if non_zero else np.median(counts) + eps
            elif method == "lower_quantile":
                dm = np.quantile(nz, 0.25) + eps if non_zero else np.quantile(counts, 0.25) + eps
            else:
                return counts
        else:
            return counts
        
        # asinh(x) ≈ x when x small, ≈ log(2x) when x large
        # asinh_counts = np.arcsinh(counts)
        log_counts = np.log1p(counts)

        # normlize by lower quantile, median or mean value
        return log_counts / np.log1p(dm)
    
    def count_embedding(
            self, seq, counts, read_len=True, 
            nor_method = Literal["median", "mean", "lower_quantile"],
            non_zero : bool = False
            ):
        seq_l = len(seq)
        # extend to shape (len(seq), 10)
        if read_len:
            raw_count = np.zeros((seq_l, 10))
            for pos, dict in counts.items():
                if pos < seq_l:
                    for read_l, cnt in dict.items():
                        raw_count[pos-1, read_l-25] = cnt
            # normalize in a transcript
            norm_count = self.count_normalize_transcript(raw_count, nor_method, non_zero)
        else:
            raw_count = np.zeros((seq_l, 1))
            for pos, dict in counts.items():
                if pos < seq_l:
                    for read_l, cnt in dict.items():
                        raw_count[pos-1, 0] = cnt
            # normalize in a transcript
            norm_count = self.count_normalize_transcript(raw_count, nor_method, non_zero)
        
        return norm_count
    
    def generate_save_dataset(
            self, 
            count_files: list = [], 
            coding_files: list = [], 
            cell_types: list = [],
            coverage: float = 0.1,
            depth: float = 0.1,
            keep_read_len: bool = False,
            nor_method = Literal["median", "mean", "lower_quantile"],
            nor_non_zero : bool = False,
            out_path="dataset.pkl", 
            fmt="pickle"):
        """
        Generate and save dataset prepared for model from multiple samples
        """

        datasets = {
            group: {
                "uuids": [],
                "tids": [],
                "seq_embs": {},
                "count_embs": [],
                "depth": [],
                "coverage": [],
                "coding_embs" : [],
                "cds_start_pos": [],
                "cds_end_pos": [],
                "motif_occs": [],
                "cell_type_idxs" : []
            } for group in self.chrom_groups
        }
        if len(count_files) == 0 or len(count_files) != len(coding_files) or len(coding_files) != len(cell_types):
            print("### No files, or files provided are not equal ! ###")
            return False
        
        print(f"### Load count and coding data from {len(count_files)} samples ###")
        # add shared sequence embeddings
        for group in datasets:
            datasets[group]["seq_embs"]= {tid: np.float32(seq_emb) for tid, seq_emb in self._base_seq_emb[group].items()} #  [L, 4]
        
        for i in range(len(count_files)):
            count_dict, _ = self.load_pickle_file(count_files[i])
            coding_dict, _ = self.load_pickle_file(coding_files[i])
            cell_type = cell_types[i]

            if not cell_type in self.cell_type_mapping:
                print(f"### Cell type of dataset {count_files[i]} is not in list ###")

            # for target transcripts
            print(f"--- Processing dataset: {count_files[i]} ---")
            for group in datasets:
                for tid in tqdm(self.filtered_tids[group], desc=f"{group} dataset generating", mininterval=200):
                    # only for imformative transcripts
                    if tid not in count_dict.keys():
                        continue
                    
                    # exclude tx with only start or stop codon
                    if tid not in coding_dict.keys():
                        continue

                    seq_upper = self.seq_dict[tid].upper()
                    counts = count_dict[tid]

                    # retain informative tx
                    d, c = self.data_quality_eval(seq_upper, counts)
                    if d < depth or c < coverage:
                        continue

                    count_emb = self.count_embedding(seq_upper, counts, keep_read_len, nor_method, nor_non_zero)

                    # motif positions
                    motif_occs = []
                    for end_idx, motif in self.motif_automaton.iter_long(seq_upper):
                        start_idx = end_idx - len(motif) + 1
                        motif_occs.append((start_idx, end_idx + 1))  # (start, end) where end is exclusive 0-based

                    # cell type index
                    cell_type_idx = self.cell_type_mapping.get(cell_type, 0)

                    # add data
                    datasets[group]["uuids"].append("-".join([tid, cell_type, str(i)]))
                    datasets[group]["tids"].append(tid)
                    datasets[group]["count_embs"].append(np.float32(count_emb)) #  [L, 10/1]
                    datasets[group]["depth"].append(np.float16(d))
                    datasets[group]["coverage"].append(np.float16(c))
                    datasets[group]["coding_embs"].append(np.float32(coding_dict[tid]))
                    datasets[group]["cds_start_pos"].append(np.int16(self.tx_cds[tid]["cds_start_pos"]))
                    datasets[group]["cds_end_pos"].append(np.int16(self.tx_cds[tid]["cds_end_pos"]))
                    datasets[group]["motif_occs"].append(list(motif_occs))
                    datasets[group]["cell_type_idxs"].append(np.int16(cell_type_idx))
    
        # save all datasets
        print(f"--- Saving datasets ---")
        if fmt == "pickle":
            for group, dataset in datasets.items():
                file_path = f"{out_path}.{group}.pkl"
                with open(file_path, "wb") as f:
                    pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Saved [{group}] dataset to {file_path} (pickle).")
        elif fmt == "h5":
            for group, dataset in datasets.items():
                file_path = f"{out_path}.{group}.h5"
                if not HAS_H5:
                    raise ImportError("h5py not available. Install h5py to use h5 format.")
                # create h5 file，build the group by tid
                with h5py.File(file_path, "w") as f:
                    grp_root = f.create_group("samples")
                    for i, uuid in enumerate(dataset["uuids"]):
                        g = grp_root.create_group(uuid)
                        g.attrs["tid"] = dataset["tids"][i]
                        g.attrs['cell_type_idx'] = dataset["cell_type_idxs"][i]
                        g.attrs['cds_start_pos'] = dataset["cds_start_pos"][i]
                        g.attrs['cds_end_pos'] = dataset["cds_end_pos"][i]
                        g.attrs['motif_occ'] = dataset["motif_occs"][i]
                        g.attrs['depth'] = dataset["depth"][i]
                        g.attrs['coverage'] = dataset["coverage"][i]
                        g.create_dataset("count_emb", data=dataset["count_embs"][i], compression="gzip")
                        g.create_dataset("coding_emb", data=dataset["coding_embs"][i], compression="gzip")
                    # sequence embedding
                    grp_seq = f.create_group("sequences")
                    for tid, seq_emb in dataset["seq_embs"].items():
                        grp_seq.create_dataset(tid, data=seq_emb, compression="gzip")
                    # metadata attributes
                    f.attrs['n_samples'] = i + 1
                    f.attrs['notes'] = "Saved by DatasetGenerator.save_h5ad by Chunfu Xiao"
                print(f"Saved [{group}] dataset to {file_path} (h5).")
        else:
            raise ValueError("Unknown format. Use 'pickle' or 'h5'.")
        