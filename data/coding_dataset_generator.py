import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from data.RPF_counter_v3 import *

# 可选：如果想用 h5 存储
try:
    import h5py
    HAS_H5 = True
except Exception:
    HAS_H5 = False


def build_ms_label_dict(ms_file):
    """
    将质谱鉴定表格转化为以 tid 为键的快速查询字典。
    处理同一转录本上有多个鉴定产物的情况。
    """
    # 统一去掉版本号
    ms_df = pd.read_csv(ms_file)
    ms_df['tid_clean'] = ms_df['tid'].apply(lambda x: str(x).split('.')[0])
    
    ms_dict = {}
    for _, row in ms_df.iterrows():
        tid = row['tid_clean']
        if tid not in ms_dict:
            ms_dict[tid] = []
        
        ms_dict[tid].append({
            'start_pos': int(row['Start_pos']),
            'end_pos': int(row['End_pos'])
        })
        
    print(f"Built MS lookup dictionary for {len(ms_dict)} unique transcripts.")
    return ms_dict


class CodingDatasetGenerator(Dataset):
    def __init__(self,
                 transcript_seq_file: str,
                 transcript_meta_file: str,
                 target_transcripts: list,
                 chrom_groups: dict, 
                 all_cell_types: list,
                 min_length=0, max_length=None,):
        """
        :param tids: 用于训练/验证的转录本 ID 列表
        :param seq_dict: {tid: sequence} 字典，用于获取长度 L
        :param feature_dict: {tid: array(L, D)} 你的特征矩阵 (如 P-site, TE, 序列One-hot)
        :param ms_dict: 通过 build_ms_label_dict 生成的质谱 Ground Truth 字典
        """

        with open(transcript_seq_file, 'rb') as f:
            self.seq_dict = pickle.load(f)
        with open(transcript_meta_file, 'rb') as f:
            self.tx_meta = pickle.load(f)
        self.target_transcripts = list(target_transcripts)
        self.chrom_groups = chrom_groups
        self.nt_mapping = dict(zip("ACGTN", range(5)))
        self.all_cell_types = list(set(all_cell_types))
        self.cell_type_mapping = dict(zip(self.all_cell_types, range(1, len(self.all_cell_types) + 1)))
        self.filtered_tids = {group : [] for group in chrom_groups}
        self._base_seq_emb = {group : {} for group in chrom_groups}
        
        for tid in tqdm(self.target_transcripts, desc="Data Cache", mininterval=1000):
            seq_len = len(self.seq_dict[tid])
            if seq_len < min_length:
                continue
            if max_length and seq_len > max_length:
                continue
            group = next((g for g, chroms in chrom_groups.items() if self.tx_meta[tid]["chrom"] in chroms), None)
            if group != None:
                self.filtered_tids[group].append(tid)
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
    
    def count_embedding(
            self, seq, counts, 
            read_len=True
            ):
        seq_l = len(seq)
        # extend to shape (len(seq), 10)
        if read_len:
            raw_count = np.zeros((seq_l, 10))
            for pos, dict in counts.items():
                if pos < seq_l:
                    for read_l, cnt in dict.items():
                        raw_count[pos-1, read_l-25] = cnt
        else:
            raw_count = np.zeros((seq_l, 1))
            for pos, dict in counts.items():
                if pos < seq_l:
                    for read_l, cnt in dict.items():
                        raw_count[pos-1, 0] = cnt
        
        return raw_count

    def coding_embedding(self, tid, ms_dict):
        # 1. 获取基础信息
        clean_tid = tid.split('.')[0]
        seq_len = len(self.seq_dict[tid])
        
        # 2. 初始化全 0 的 Label 矩阵
        # 形状为 (L, 2)，channel 0 为 TIS, channel 1 为 TTS
        labels = np.zeros((seq_len, 2), dtype=np.float32)
        
        # 3. 如果该转录本在质谱中有鉴定到，激活对应坐标
        if clean_tid in ms_dict:
            for orf in ms_dict[clean_tid]:
                # 转换为 0-based 坐标
                s_idx = orf['start_pos'] - 1 # pos of 1st nt for start codon
                e_idx = orf['end_pos'] - 3 # pos of 1st nt for stop codon
                
                # 替代直接赋值 1.0 的平滑逻辑：
                # 给目标位点赋 1.0，向外扩散递减 (如 0.8, 0.4, 0.1)
                window = [0.1, 0.3, 0.7, 1.0, 0.7, 0.3, 0.1]
                offset = len(window) // 2  # 3
                
                # 处理 TIS 平滑
                for w_i, val in enumerate(window):
                    target_s = s_idx - offset + w_i
                    if 0 <= target_s < seq_len:
                        labels[target_s, 0] = max(labels[target_s, 0], val)
                
                # 处理 TTS 平滑
                for w_i, val in enumerate(window):
                    target_e = e_idx - offset + w_i
                    if 0 <= target_e < seq_len:
                        labels[target_e, 1] = max(labels[target_e, 1], val)

                # 安全性检查：防止越界
                # if 0 <= s_idx < seq_len:
                #     labels[s_idx, 0] = 1.0  # TIS 正样本
                
                # if 0 <= e_idx < seq_len:
                #     labels[e_idx, 1] = 1.0  # TTS 正样本
        
        return labels
    
    def generate_save_dataset(
            self, 
            ms_res_files: list = [],
            ribo_count_files: list = [],
            cell_types: list = [],
            keep_read_len: bool = False,
            out_path="dataset.pkl", 
            fmt="pickle"):
        """
        Generate and save dataset prepared for model from multiple samples
        """

        datasets = {
            group: {
                "uuids": [], "tids": [], "seq_embs": {}, 
                "coding_embs": [], "count_embs": [], "cell_types" : []
            } for group in self.chrom_groups
        }

        if len(ms_res_files) == 0 or len(ms_res_files) != len(ribo_count_files) or len(ribo_count_files) != len(cell_types):
            print("### No files, or files provided are not equal ! ###")
            return False
        
        print(f"### Load coding labels and Ribo-seq data from {len(ms_res_files)} samples ###")
        # add shared sequence embeddings
        for group in datasets:
            datasets[group]["seq_embs"]= {tid: np.float32(seq_emb) for tid, seq_emb in self._base_seq_emb[group].items()} #  [L, 4]

        for i in range(len(ms_res_files)):
            ms_dict = build_ms_label_dict(ms_res_files[i])
            ribo_count_dict, _ = self.load_pickle_file(ribo_count_files[i])
            cell_type = cell_types[i]

            for group in datasets:
                for tid in tqdm(self.filtered_tids[group], desc=f"{group} [{cell_type}]"):
                    if tid not in ribo_count_dict:
                        continue
                    
                    seq_upper = self.seq_dict[tid].upper()
                    counts = ribo_count_dict[tid]

                    coding_emb = self.coding_embedding(tid, ms_dict)
                    count_emb = self.count_embedding(seq_upper, counts, keep_read_len)

                    datasets[group]["uuids"].append("-".join([tid, cell_type, str(i)]))
                    datasets[group]["tids"].append(tid)
                    datasets[group]["coding_embs"].append(np.float32(coding_emb))
                    datasets[group]["count_embs"].append(np.float32(count_emb))
                    datasets[group]["cell_types"].append(str(cell_type))
    
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
                        g.attrs['cell_type'] = dataset["cell_types"][i]
                        g.create_dataset("coding_emb", data=dataset["coding_embs"][i], compression="gzip")
                        g.create_dataset("count_emb", data=dataset["count_embs"][i], compression="gzip")
                    # sequence embedding
                    grp_seq = f.create_group("sequences")
                    for tid, seq_emb in dataset["seq_embs"].items():
                        grp_seq.create_dataset(tid, data=seq_emb, compression="gzip")
                    # metadata attributes
                    f.attrs['n_samples'] = i + 1
                    f.attrs['notes'] = "Saved by CodingDatasetGenerator.save_h5ad by Chunfu Xiao"
                print(f"Saved [{group}] dataset to {file_path} (h5).")
        else:
            raise ValueError("Unknown format. Use 'pickle' or 'h5'.")