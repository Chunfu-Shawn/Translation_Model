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


# generate embeddings and return Dataset class
class DatasetGenerator():
    def __init__(self, transcript_seq_file, transcript_meta_file, transcript_cds_file, chroms, all_tissue_types,
                 min_length=0, max_length=None, mask_perc=[0.15, 0.15], mask_value=-1, motif_file_path=""):
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
        self.masking_adapter = MaskingAdapter(self.seq_dict, mask_value, motif_file_path, mask_perc)
        self.nt_mapping = dict(zip("ACGT", range(4)))
        self.all_tissue_types = all_tissue_types
        self.tissue_mapping = dict(
            zip(
                self.all_tissue_types, 
                range(len(self.all_tissue_types))
                )
            )
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
        seq = self.seq_dict[tid]
        seq2 = [self.nt_mapping[i] for i in seq]
        onehot = np.eye(4)[seq2]
        return onehot


    def tissue_type_encode(self, tissue_type):

        return self.tissue_mapping[tissue_type]
        

    def count_normalize_transcript(self, counts: np.ndarray, method="median", eps=1):
        # non-zero counts
        # nz = counts[counts > 0]

        # median or mean of non-zero counts
        #if nz.size > 0:
        if method == "median":
            dm = np.median(counts)
        elif method == "mean":
            dm = np.mean(counts)
        else:
            return counts
        # else:
        #     return counts
        
        # asinh(x) ≈ x when x small, ≈ log(2x) when x large
        asinh_counts = np.arcsinh(counts)

        # normlize by median or mean value
        return (asinh_counts) / (np.arcsinh(dm) + eps)
    
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
            return np.concatenate((np.zeros((12, 10)), norm_count[:-12,:]), axis=0)
        else:
            return raw_count
    
    def generate_save_dataset(self, count_files=[], coding_files=[], tissue_type_l=[], out_path="dataset.pkl", fmt="pickle"):
        """
        Generate and save dataset prepared for model from multiple samples
        """
        dataset = {
            "uuids": [],
            "tids": [],
            "embs": [],
            "masked_embs": [],
            "emb_masks" : [],
            "coding_embs" : [],
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
                print("### Tissue type of dataset is not in list ###")
                return False

            # for target transcripts
            for tid in tqdm(self.filtered_tids, desc="Dataset generating", mininterval=200):
                # only for imformative transcripts
                if tid not in count_dict.keys():
                    continue
                
                seq_emb = self._base_seq_emb[tid]
                count_emb = self.count_embedding(count_dict, tid, True, 12)
                # combined_emb = np.concatenate((seq_emb, count_emb), axis=-1)

                # mask embedding for learning
                ## True indicate this is a token need to be blocked and predicted
                masked_emb, emb_mask = self.masking_adapter.get_random_mask_function(
                    self.seq_dict[tid], 
                    [seq_emb, count_emb],
                    self.tx_cds[tid]
                    )
                
                # tissue embedding
                tissue_idx = self.tissue_type_encode(tissue_type)

                # add data
                dataset["uuids"].append("_".join([tid, name]))
                dataset["tids"].append(tid)
                dataset["embs"].append(np.concatenate((seq_emb, count_emb), axis=-1, dtype=np.float32)) #  [L, 4 + 10]
                dataset["masked_embs"].append(np.concatenate(masked_emb, axis=-1, dtype=np.float32)) # [L, 4 + 10]
                dataset["emb_masks"].append(np.stack(emb_mask, axis=1, dtype=np.bool)) # [L, 1 + 1]
                dataset["coding_embs"].append(np.float32(coding_dict[tid]))
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
                    g.attrs["tids"] = dataset["tids"][i]
                    g.attrs['tissue_idx'] = int(dataset["tissue_idxs"][i])
                    g.create_dataset("embs", data=dataset["embs"][i], compression="gzip")
                    g.create_dataset("masked_embs", data=dataset["masked_embs"][i], compression="gzip")
                    g.create_dataset("emb_mask", data=dataset["emb_masks"][i], compression="gzip")
                    g.create_dataset("coding_emb", data=dataset["coding_embs"][i], compression="gzip")
                # metadata attributes
                f.attrs['n_samples'] = i + 1
                f.attrs['notes'] = "Saved by DatasetGenerator.save_h5ad by Chunfu Xiao"
            print(f"Saved dataset to {out_path} (h5).")

        else:
            raise ValueError("Unknown format. Use 'pickle' or 'h5'.")