import sys, os
sys.path.append("/home/user/data3/rbase/translation_model/models/src")
import numpy as np
import math
import json
import ahocorasick
import torch
from tqdm import tqdm
from typing import Literal
from data.RPF_counter_v3 import *
from scipy.stats import linregress

# 可选：如果想用 h5 存储
try:
    import h5py
    HAS_H5 = True
except Exception:
    HAS_H5 = False

__author__ = "Chunfu Xiao"
__version__="1.2.0"
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
                 species: str,
                 min_length=0, max_length=None, 
                 motif_file_path: str = ""):
        
        with open(transcript_seq_file, 'rb') as f:
            self.seq_dict = pickle.load(f)
        with open(transcript_meta_file, 'rb') as f:
            self.tx_meta = pickle.load(f)
        with open(transcript_cds_file, 'rb') as f:
            self.tx_cds = pickle.load(f)

        self.chrom_groups = chrom_groups
        self.species = species
        self.nt_mapping = dict(zip("ACGTN", range(5)))
        
        self.motifs_list = load_motifs_from_file(motif_file_path)
        self.motif_automaton = build_automaton(self.motifs_list)
        self.filtered_tids = {group : [] for group in chrom_groups}
        self._base_seq_emb = {group : {} for group in chrom_groups}
        
        for tid in tqdm(self.tx_meta, desc="Data Cache", mininterval=1000):
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
    
    def parse_and_winsorize_to_array(self, count_dict, keep_read_len, percentile=99.5):
        """
        将原始字典转换为 Dense NumPy 矩阵并原位截断。
        从此往后，整个数据流彻底告别 dict.items() 的 for 循环！
        """
        arr_dict = {}
        for tid, counts in count_dict.items():
            if tid not in self.seq_dict or not counts:
                continue
            
            seq_l = len(self.seq_dict[tid])
            
            # 1. 解析字典构建矩阵
            if keep_read_len:
                arr = np.zeros((seq_l, 10), dtype=np.float32)
                for pos, read_dict in counts.items():
                    if pos <= seq_l:
                        if isinstance(read_dict, dict):
                            for r_len, cnt in read_dict.items():
                                if 25 <= r_len <= 34 and cnt > 0:
                                    arr[pos-1, r_len-25] = cnt
                        else:
                            arr[pos-1, 0] = read_dict
            else:
                arr = np.zeros((seq_l, 1), dtype=np.float32)
                for pos, read_dict in counts.items():
                    if pos <= seq_l:
                        val = sum(read_dict.values()) if isinstance(read_dict, dict) else read_dict
                        if val > 0:
                            arr[pos-1, 0] = val
                            
            # 2. 矩阵级 Winsorization
            nz_mask = arr > 0
            if np.any(nz_mask):
                cap_val = np.percentile(arr[nz_mask], percentile)
                np.clip(arr, a_min=None, a_max=cap_val, out=arr) # 原位修改内存，极速
                
            arr_dict[tid] = arr
            
        return arr_dict
    
    def data_quality_eval(self, seq_l, arr, cds_s, cds_e, has_valid_cds):
        """
        全向量化质控，计算CDS区域或者全长的depth和coverage
        """
        if has_valid_cds:
            start_idx = cds_s - 1  # 0-based index
            end_idx = cds_e
        else:
            start_idx = 0
            end_idx = seq_l

        region_l = end_idx - start_idx
        if region_l <= 0:
            return np.array([]), 0.0, 0.0, 0.0, 0.0

        # 将 (L, 10) 压缩为 (L,) 的 1D 数组
        pos_counts_1d = arr.sum(axis=1) if arr.ndim == 2 else arr.flatten()
        region_arr = pos_counts_1d[start_idx:end_idx] # 瞬间提取目标区间

        depth_full = pos_counts_1d.sum() / seq_l
        depth = region_arr.sum() / region_l

        # 利用 Numpy 步长切片快速计算 Frame 0 reads 占比
        if has_valid_cds and region_arr.sum() > 0:
            frame0_ratio = region_arr[0::3].sum() / region_arr.sum()
        else:
            frame0_ratio = 0.0

        # 向量化 Coverage 计算 (每 3 个核苷酸一组求和，看有几个组大于0)
        n_put_codon = math.ceil(region_l / 3)
        pad_len = (3 - region_l % 3) % 3
        region_arr_padded = np.pad(region_arr, (0, pad_len)) if pad_len > 0 else region_arr
        
        codons = region_arr_padded.reshape(-1, 3).sum(axis=1)
        coverage = np.count_nonzero(codons) / n_put_codon if n_put_codon > 0 else 0

        return pos_counts_1d, depth, depth_full, coverage, frame0_ratio
    
    def compute_sample_te(self, ribo_arr_dict, rna_count_dict, depth_threshold, coverage_threshold, rpm_threshold=1, periodicity=0.33):
        """
        接收已经 Winsorized 过的 numpy dict, 计算 TE。
        """
        rpf_totals, rna_totals, rpf_depth, rpf_cov = {}, {}, {}, {}

        sample_total_rna_reads = sum(count for count in rna_count_dict.values())
        if sample_total_rna_reads == 0:
            sample_total_rna_reads = 1.0 # 防止极端情况下的除零错误

        for tid, ribo_arr in ribo_arr_dict.items():
            if tid not in rna_count_dict: 
                continue

            total_reads_rna = rna_count_dict[tid]
            # 计算当前转录本的 RPM，如果不满足阈值直接跳过
            rpm = (total_reads_rna / sample_total_rna_reads) * 1e6
            if rpm < rpm_threshold:
                continue

            has_valid_cds = False
            cds_info = self.tx_cds.get(tid, {'cds_start_pos':-1, 'cds_end_pos':-1})
            cds_s, cds_e = cds_info.get('cds_start_pos', -1), cds_info.get('cds_end_pos', -1)
            if cds_s != -1 and (cds_e - cds_s) > 45:
                has_valid_cds = True

            seq_l = len(self.seq_dict[tid])
            
            pos_1d_ribo, d_ribo, d_ribo_full, c_ribo, f0_ratio = self.data_quality_eval(seq_l, ribo_arr, cds_s, cds_e, has_valid_cds)

            # 阈值过滤
            if has_valid_cds:
                # 在过滤条件中增加 f0_ratio <= 0.33 判断
                if d_ribo < min(2, depth_threshold * 3.0) or c_ribo < min(1, coverage_threshold * 3.0) or f0_ratio <= periodicity: 
                    continue
            else:
                if d_ribo < depth_threshold or c_ribo < coverage_threshold: 
                    continue
                
            rpf_depth[tid], rpf_cov[tid] = d_ribo_full, c_ribo

            # 切片求和
            if has_valid_cds:
                # Python slice 是左闭右开，[cds_s+6 : cds_e-9] 完美对应你之前的 valid_s 和 valid_e
                valid_s_idx = cds_s + 6
                valid_e_idx = cds_e - 9 
                
                total_reads_ribo = pos_1d_ribo[valid_s_idx : valid_e_idx].sum()
            else:
                total_reads_ribo = pos_1d_ribo.sum()


            rpf_totals[tid] = total_reads_ribo + 10.0 
            rna_totals[tid] = total_reads_rna + 10.0
            
        if not rpf_totals or not rna_totals: return {}, {}, {}

        # CLR 与 线性回归
        tids = list(rpf_totals.keys())
        rpf_clr = np.log(list(rpf_totals.values())) - np.mean(np.log(list(rpf_totals.values())))
        rna_clr = np.log(list(rna_totals.values())) - np.mean(np.log(list(rna_totals.values())))
        
        slope, intercept, _, _, _ = linregress(rna_clr, rpf_clr)
        
        te_dict = {t: rpf_clr[i] - (slope * rna_clr[i] + intercept) for i, t in enumerate(tids)}
        return te_dict, rpf_depth, rpf_cov

    def count_embedding(self, arr, te_residual):
        """
        arr 已经是被 parse_and_winsorize_to_array 组装且净化好的矩阵了。
        """
        nz_mask = arr > 0
        nz_mean = np.mean(arr[nz_mask]) if np.any(nz_mask) else 1.0
        
        scaled_count = np.log1p((arr / nz_mean) * np.exp(te_residual))
        
        return scaled_count

    def generate_save_dataset(
            self,
            dataset_config: list, 
            depth: float = 0.0,
            coverage: float = 0.0,
            rpm: float = 1.0,
            expr_dict_path: str = None,
            keep_read_len: bool = False,
            out_path="dataset.h5"
            ):
        """
        Generate and save dataset prepared for model from multiple samples using config dicts.
        If `expr_dict_path` is provided, fetches the specific continuous vector for that cell_type.
        """

        datasets = {
            group: {
                "uuids": [], "tids": [], "seq_embs": {}, "count_embs": [], 
                "rpf_depth": [], "rpf_coverage": [], "te_val": [],
                "cds_start_pos": [], "cds_end_pos": [],
                "motif_occs": [], "cell_types": [], 
                "cell_expr_dict" : {}
            } for group in self.chrom_groups
        }
        
        if not dataset_config:
            print("### Dataset config is empty! ###")
            return False

        all_cell_types = [item["cell_type"] for item in dataset_config]
        # all_cell_types = list(dict.fromkeys(raw_cell_types))
        
        # ==========================================
        # Pre-load the Cell Expression Dictionary
        # ==========================================
        cell_expr_mapping = {}
        mean_expr_vector = None
        if expr_dict_path and os.path.exists(expr_dict_path):
            print(f"### Loading continuous expression profiles from: {expr_dict_path} ###")
            if expr_dict_path.endswith('.pt'):
                # Load via PyTorch
                loaded_dict = torch.load(expr_dict_path, map_location="cpu")
                # Convert to fp16 numpy for lightweight saving
                cell_expr_mapping = {k: v.numpy().astype(np.float16) for k, v in loaded_dict.items()}
            elif expr_dict_path.endswith('.pkl'):
                with open(expr_dict_path, 'rb') as f:
                    cell_expr_mapping = pickle.load(f)
            else:
                raise ValueError("Unsupported expr_dict format. Use .pt or .pkl")
            
            # Calculate the mean vector for fallbacks
            all_vecs = list(cell_expr_mapping.values())
            mean_expr_vector = np.mean(all_vecs, axis=0).astype(np.float16) if all_vecs else None
        else:
            print("### WARNING: No expr_dict_path provided or file missing. Expression vectors will not be saved. ###")

        print(f"### Load count, transcriptome and coding data from {len(dataset_config)} samples ###")
        print(f"### Species: {self.species} ###")
        print(f"### Cell types: {all_cell_types} ###")
        
        # add shared sequence embeddings
        for group in datasets:
            datasets[group]["seq_embs"]= {tid: np.float32(seq_emb) for tid, seq_emb in self._base_seq_emb[group].items()} #  [L, 4]

        # Iterate over dataset_config
        for i, config_item in enumerate(dataset_config):
            cell_type = config_item["cell_type"]
            ribo_file = config_item["read_count"]
            rna_file = config_item["rna_count"]

            ribo_count_dict, _ = self.load_pickle_file(ribo_file)
            rna_count_dict, _ = self.load_pickle_file(rna_file)

            print(f"--- Vectorizing and Winsorizing arrays for {cell_type} ---")
            ribo_arr_dict = self.parse_and_winsorize_to_array(ribo_count_dict, keep_read_len)

            print(f"--- Computing CLR and Regression TE for {cell_type} ---")
            sample_te_dict, rpf_depth_dict, rpf_cov_dict = self.compute_sample_te(
                ribo_arr_dict, rna_count_dict, depth, coverage, rpm
                )

            # ==========================================
            # Resolve the expression vector for the current cell_type
            # ==========================================
            current_expr_vector = None
            if cell_expr_mapping:
                current_expr_vector = cell_expr_mapping.get(cell_type, mean_expr_vector)
            
            if current_expr_vector is not None:
                for group in datasets:
                    if cell_type not in datasets[group]["cell_expr_dict"]:
                        datasets[group]["cell_expr_dict"][cell_type] = current_expr_vector

            for group in datasets:
                for tid in tqdm(self.filtered_tids[group], desc=f"{group} [{cell_type}]"):
                    # contain translation
                    if tid not in ribo_arr_dict or tid not in sample_te_dict:
                        continue

                    # with or without both defined start/stop codon 
                    if self.tx_cds[tid]['start_codon'] != self.tx_cds[tid]['stop_codon']:
                        continue

                    # count array scaling
                    arr = ribo_arr_dict[tid]
                    te_val = sample_te_dict[tid]
                    count_emb = self.count_embedding(arr, te_val)

                    # motif positions
                    seq_upper = self.seq_dict[tid].upper()
                    motif_occs = []
                    for end_idx, motif in self.motif_automaton.iter_long(seq_upper):
                        start_idx = end_idx - len(motif) + 1
                        motif_occs.append((start_idx, end_idx + 1))  # (start, end) where end is exclusive 0-based

                    datasets[group]["uuids"].append("-".join([tid, cell_type, str(i)]))
                    datasets[group]["tids"].append(tid)
                    datasets[group]["count_embs"].append(np.float32(count_emb))
                    datasets[group]["rpf_depth"].append(np.float32(rpf_depth_dict[tid]))
                    datasets[group]["rpf_coverage"].append(np.float32(rpf_cov_dict[tid]))
                    datasets[group]["te_val"].append(np.float32(te_val))
                    datasets[group]["cds_start_pos"].append(np.int16(self.tx_cds[tid]['cds_start_pos']))
                    datasets[group]["cds_end_pos"].append(np.int16(self.tx_cds[tid]['cds_end_pos']))
                    datasets[group]["motif_occs"].append(list(motif_occs))
                    datasets[group]["cell_types"].append(str(cell_type))
    
        # save all datasets
        print(f"--- Saving datasets ---")
        for group, dataset in datasets.items():
            file_path = f"{out_path}.{group}.h5"
            if not HAS_H5:
                raise ImportError("h5py not available. Install h5py to use h5 format.")
            # create h5 file，build the group by tid
            with h5py.File(file_path, "w") as f:
                # 1. Save Samples
                grp_root = f.create_group("samples")
                for i, uuid in enumerate(dataset["uuids"]):
                    g = grp_root.create_group(uuid)
                    g.attrs["tid"] = dataset["tids"][i]
                    g.attrs['species'] = str(self.species)
                    g.attrs['cell_type'] = dataset["cell_types"][i]
                    g.attrs['cds_start_pos'] = dataset["cds_start_pos"][i]
                    g.attrs['cds_end_pos'] = dataset["cds_end_pos"][i]
                    g.attrs['motif_occ'] = dataset["motif_occs"][i] #json.dumps(dataset["motif_occs"][i]) 
                    g.attrs['rpf_depth'] = dataset["rpf_depth"][i]
                    g.attrs['rpf_coverage'] = dataset["rpf_coverage"][i]
                    g.attrs['te_val'] = dataset["te_val"][i]
                    g.create_dataset("count_emb", data=dataset["count_embs"][i], compression="gzip")

                # 2. save sequence embedding
                grp_seq = f.create_group("sequences")
                for tid, seq_emb in dataset["seq_embs"].items():
                    grp_seq.create_dataset(tid, data=seq_emb, compression="gzip")
                
                # 3. Save Global Cell Expression Dict
                if dataset["cell_expr_dict"]:
                    grp_expr = f.create_group("cell_exprs")
                    for ct, expr_vec in dataset["cell_expr_dict"].items():
                        grp_expr.create_dataset(ct, data=expr_vec, compression="gzip")

                # 4. metadata attributes
                cell_types_counts = {}
                for ct in dataset["cell_types"]:
                    cell_types_counts[ct] = cell_types_counts.get(ct, 0) + 1

                f.attrs['cell_type_counts'] = json.dumps(cell_types_counts)
                f.attrs['n_samples'] = len(dataset["uuids"])
                f.attrs['notes'] = "Saved by DatasetGenerator with Optimized Global cell_exprs"

            print(f"Saved [{group}] dataset to {file_path} (h5).")