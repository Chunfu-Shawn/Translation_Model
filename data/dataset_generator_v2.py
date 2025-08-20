import sys, os
sys.path.append("/home/user/data3/rbase/translation_pred/models/src")
import numpy as np
from tqdm import tqdm
from data.RPF_counter_v3 import *
from data.masking_adapter import *

# 可选：如果想用 h5 存储
try:
    import h5py
    HAS_H5 = True
except Exception:
    HAS_H5 = False

__author__ = "Chunfu Xiao"
__version__="1.1.0"
__email__ = "chunfushawn@126.com"

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
                 chroms: list, 
                 all_tissue_types: list,
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

        # filter transcript with RPF reads in specific chromosomes
        self.tids = [tid for (tid, v) in self.tx_meta.items() if (v["chrom"] in chroms)]
        self.nt_mapping = dict(zip("ACGT", range(4)))
        self.all_tissue_types = list(set(all_tissue_types))
        self.tissue_mapping = dict(
            zip(
                self.all_tissue_types, 
                range(len(self.all_tissue_types))
                )
            )
        self.motifs_list = load_motifs_from_file(motif_file_path)
        self.motif_automaton = build_automaton(self.motifs_list)
        self.filtered_tids = []
        self._base_seq_emb = {}
        
        # data cache
        print("### Data Cache ###")
        for tid in tqdm(self.tids, desc="Data Cache", mininterval=1000):
            seq_len = len(self.seq_dict[tid])
            if seq_len < min_length:
                continue
            if max_length and seq_len > max_length:
                continue
            self.filtered_tids.append(tid)
            # seq embedding
            self._base_seq_emb[tid] = self.one_hot_encode(tid)
    
    def load_pickle_file(self, file_path):
        (_, filename) = os.path.split(file_path)
        name = filename.split('.')[0]
        with open(file_path, 'rb') as f:
            dict = pickle.load(f)
        
        return dict, name

    def one_hot_encode(self, tid):
        seq = self.seq_dict[tid].upper()
        seq2 = [self.nt_mapping[i] for i in seq]
        onehot = np.eye(4)[seq2]
        return onehot


    def tissue_type_encode(self, tissue_type):

        return self.tissue_mapping[tissue_type]
        

    def count_normalize_transcript(self, counts: np.ndarray, method="lower_quantile", eps=1):
        # non-zero counts
        nz = counts[counts > 0]

        # median or mean of non-zero counts
        if nz.size > 0:
            if method == "lower_quantile":
                dm = np.quantile(nz, 0.25)
            elif method == "median":
                dm = np.median(nz)
            elif method == "mean":
                dm = np.mean(nz)
            else:
                return counts
        else:
            return counts
        
        # asinh(x) ≈ x when x small, ≈ log(2x) when x large
        # asinh_counts = np.arcsinh(counts)
        log_counts = np.log1p(counts)

        # normlize by lower quantile, median or mean value
        return log_counts / np.log1p(dm)
    
    def count_embedding(self, count_dict, tid, read_len=False, offset=12):
        # add general offset of 12 to align RPF footprint and sequence

        seq = self.seq_dict[tid]
        # extend to shape (len(seq), 10)
        raw_count = np.zeros((len(seq), 10)) if read_len else np.zeros((len(seq), 1))
        if tid in count_dict:
            count = count_dict[tid]
            for read_l in count:
                for pos in count[read_l]:
                    raw_count[pos-1, read_l-25] = count[read_l][pos]
            # normalize in a transcript
            norm_count = self.count_normalize_transcript(raw_count)

            # shift backrward for 12 nt
            return np.concatenate((np.zeros((offset, 10)), norm_count[:-offset,:]), axis=0)
        else:
            return raw_count
    
    def generate_save_dataset(self, count_files=[], coding_files=[], tissue_type_l=[], out_path="dataset.pkl", fmt="pickle"):
        """
        Generate and save dataset prepared for model from multiple samples
        """
        dataset = {
            "uuids": [],
            "tids": [],
            "seq_embs": [],
            "count_embs": [],
            "coding_embs" : [],
            "cds_starts": [],
            "motif_occs": [],
            "tissue_idxs" : []
        }

        if len(count_files) == 0 or len(count_files) != len(coding_files) or len(coding_files) != len(tissue_type_l):
            print("### No files, or files provided are not equal ! ###")
            return False
        
        print(f"### Load count and coding data from {len(count_files)} samples ###")
        
        for i in range(len(count_files)):
            count_dict, name = self.load_pickle_file(count_files[i])
            coding_dict, _ = self.load_pickle_file(coding_files[i])
            tissue_type = tissue_type_l[i]

            if not tissue_type in self.tissue_mapping:
                print(f"### Tissue type of dataset is not in list ###")
                print(f"==> Exclude dataset: {count_files[i]}")
                return False

            # for target transcripts
            print(f"--- Processing dataset: {count_files[i]} ---")
            for tid in tqdm(self.filtered_tids, desc="Dataset generating", mininterval=200):
                # only for imformative transcripts
                if tid not in count_dict.keys():
                    continue
                
                seq_emb = self._base_seq_emb[tid]
                count_emb = self.count_embedding(count_dict, tid, True, 12)

                # motif positions
                seq_upper = self.seq_dict[tid].upper()
                motif_occs = []
                for end_idx, motif in self.motif_automaton.iter_long(seq_upper):
                    start_idx = end_idx - len(motif) + 1
                    motif_occs.append((start_idx, end_idx + 1))  # (start, end) where end is exclusive 0-based

                # tissue embedding
                tissue_idx = self.tissue_type_encode(tissue_type)

                # add data
                dataset["uuids"].append("-".join([tid, name]))
                dataset["tids"].append(tid)
                dataset["seq_embs"].append(np.float32(seq_emb)) #  [L, 4]
                dataset["count_embs"].append(np.float32(count_emb)) #  [L, 10]
                dataset["coding_embs"].append(np.float32(coding_dict[tid]))
                dataset["cds_starts"].append(np.int16(self.tx_cds[tid]["cds_start_pos"]))
                dataset["motif_occs"].append(motif_occs)
                dataset["tissue_idxs"].append(np.int16(tissue_idx))

    
        # save
        if fmt == "pickle":
            with open(out_path, "wb") as f:
                pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved dataset to {out_path} (pickle).")

        elif fmt == "h5":
            if not HAS_H5:
                raise ImportError("h5py not available. Install h5py to use h5 format.")
            # create h5 file，build the group by tid
            with h5py.File(out_path, "w") as f:
                grp_root = f.create_group("samples")
                for i, uuid in enumerate(dataset["uuids"]):
                    g = grp_root.create_group(uuid)
                    g.attrs["tid"] = dataset["tids"][i]
                    g.attrs['tissue_idx'] = int(dataset["tissue_idxs"][i])
                    g.attrs['cds_start'] = int(dataset["cds_starts"][i])
                    g.attrs['motif_occ'] = list(dataset["motif_occs"][i])
                    g.create_dataset("seq_emb", data=dataset["seq_embs"][i], compression="gzip")
                    g.create_dataset("count_emb", data=dataset["count_embs"][i], compression="gzip")
                    g.create_dataset("coding_emb", data=dataset["coding_embs"][i], compression="gzip")
                # metadata attributes
                f.attrs['n_samples'] = i + 1
                f.attrs['notes'] = "Saved by DatasetGenerator.save_h5ad by Chunfu Xiao"
            print(f"Saved dataset to {out_path} (h5).")

        else:
            raise ValueError("Unknown format. Use 'pickle' or 'h5'.")