import os
import re
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Optional
from tqdm import tqdm

from eval.save_prediction_results import _prepare_prediction_dataloader
from task.translation_metrics import compute_pif, compute_uniformity, compute_dropoff
from utils import unwrap_model, clean_up_memory

# =================================================================
# 工具函数: Fasta 解析
# =================================================================
def read_fasta(file_path: str) -> Dict[str, str]:
    """读取 Fasta 文件并返回 {tid: sequence} 字典"""
    seq_dict = {}
    curr_tid = ""
    curr_seq = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if curr_tid:
                    seq_dict[curr_tid] = "".join(curr_seq)
                # 提取 > 后面的 ID，通常以空格分隔取第一部分
                curr_tid = line[1:].split()[0]
                curr_seq = []
            else:
                curr_seq.append(line.upper())
        if curr_tid:
            seq_dict[curr_tid] = "".join(curr_seq)
    print(f"Loaded {len(seq_dict)} sequences from {file_path}")
    return seq_dict

# =================================================================
# 步骤 1 & 2 & 3: Fasta 序列处理与潜在 ORF 挖掘
# =================================================================
class SequenceORFBuilder:
    def __init__(self, seq_dict: Dict[str, str]):
        self.seq_dict = seq_dict

    def find_all_orfs(self, sequence: str, start_codons: List[str], stop_codons: List[str], min_len: int) -> List[Dict]:
        orfs = []
        seq_len = len(sequence)
        priority_map = {codon: i for i, codon in enumerate(start_codons)}

        for start_motif in start_codons:
            for match in re.finditer(f'(?=({start_motif}))', sequence):
                s_pos = match.start()
                for i in range(s_pos + 3, seq_len - 2, 3):
                    if sequence[i:i+3] in stop_codons:
                        o_len = i + 3 - s_pos
                        if o_len >= min_len:
                            orfs.append({
                                'start': s_pos + 1,  # 1-based
                                'end': i + 3,        # 1-based including stop codon
                                'length': o_len,
                                'start_codon': start_motif,
                                'priority': priority_map.get(start_motif, 99)
                            })
                        break # 找到同框终止密码子立即停止
        return orfs

    def build_and_collapse_orfs(self, 
                                start_codons: List[str] = ["ATG", "CTG", "GTG", "TTG"], 
                                stop_codons: List[str] = ["TGA", "TAA", "TAG"], 
                                min_len: int = 60,
                                out_file: str = "potential_orfs.csv") -> pd.DataFrame:
        all_results = []
        for tid, seq in tqdm(self.seq_dict.items(), desc="Extracting Potential ORFs"):
            tid_clean = str(tid).split('|')[0]
            orfs = self.find_all_orfs(seq, start_codons, stop_codons, min_len)
            for orf in orfs:
                orf['tid'] = tid_clean
                all_results.append(orf)
                
        if not all_results:
            print("No ORFs found!")
            return pd.DataFrame()
            
        df = pd.DataFrame(all_results)
        
        # Collapse 逻辑：按 tid, end 分组；优先级升序，长度降序
        df_sorted = df.sort_values(by=['tid', 'end', 'priority', 'length'], 
                                   ascending=[True, True, True, False])
        df_collapsed = df_sorted.drop_duplicates(subset=['tid', 'end'], keep='first')
        
        df_collapsed = df_collapsed[['tid', 'start', 'end', 'length', 'start_codon']]
        df_collapsed.to_csv(out_file, index=False)
        print(f"Collapsed raw ORFs down to {len(df_collapsed)} unique ORFs.")
        return df_collapsed

# =================================================================
# 步骤 4: 零样本推理 Dataset 与预测引擎
# =================================================================
class DeNovoSequenceDataset(Dataset):
    """
    轻量级内存 Dataset，专为仅仅只有 RNA 序列的从头 (De Novo) ORF 预测设计。
    完美兼容 TranslationDataset 的输出结构和 self.lengths 属性。
    """
    def __init__(self, seq_dict: Dict[str, str], cell_type: str, cell_type_mapping: Dict[str, int]):
        self.tids = list(seq_dict.keys())
        self.seq_dict = seq_dict
        self.cell_type = cell_type
        self.cell_type_idx = cell_type_mapping.get(cell_type, 0)
        self.nt_mapping = dict(zip("ACGTN", range(5)))
        
        self.uuids = []
        self.seq_embs = []
        self.lengths = []
        
        # 预先进行 One-hot 编码并记录长度，支持 BucketSampler
        for tid in tqdm(self.tids, desc="Encoding DeNovo Dataset"):
            seq = self.seq_dict[tid].upper()
            self.lengths.append(len(seq))
            
            # 清理 PACBIO 冗余 ID
            tid_clean = str(tid).split('|')[0]
            if tid_clean.startswith('ENS') and '.' in tid_clean:
                tid_clean = tid_clean.split('.')[0]
                
            uuid = f"{tid_clean}-{self.cell_type}-DeNovo"
            self.uuids.append(uuid)
            
            seq_idx = [self.nt_mapping.get(nt, 4) for nt in seq]
            seq_emb = np.eye(5)[seq_idx, :4]
            self.seq_embs.append(seq_emb)
            
        self.n_samples = len(self.lengths)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        uuid = self.uuids[idx]
        cell_type_idx = torch.tensor(self.cell_type_idx, dtype=torch.long)
        seq_emb = torch.from_numpy(self.seq_embs[idx]).float()
        
        # 动态生成占位的 Count 矩阵 (纯 0)
        count_emb = torch.zeros((self.lengths[idx], 1), dtype=torch.float32)
        
        # 【关键修复】返回空字典 meta_info 作为占位，保持与 TranslationDataset 严格一致的 5 元素输出
        meta_info = {}
        
        return uuid, cell_type_idx, meta_info, seq_emb, count_emb


def collate_fn_denovo(batch):
    """专门配套 DeNovoDataset 的组装函数，输出严格的 6 元素 Tuple"""
    uuids, cell_idxs, meta_infos, seq_embs, count_embs = zip(*batch)
    lengths = [s.shape[0] for s in seq_embs]
    
    seq_padded = pad_sequence(seq_embs, batch_first=True, padding_value=-1)
    count_padded = pad_sequence(count_embs, batch_first=True, padding_value=-1)
    cell_idxs = torch.stack(cell_idxs)
    
    return uuids, cell_idxs, meta_infos, seq_padded, count_padded, lengths

# =================================================================
# 步骤 5: 翻译信号评估
# =================================================================
class TranslationSignalEvaluator:
    def __init__(self, pred_pkl: str, orf_csv: str, out_dir: str = "./"):
        print(f"Loading predictions from {pred_pkl}...")
        with open(pred_pkl, 'rb') as f:
            self.pred_data = pickle.load(f)
            
        print(f"Loading ORF annotations from {orf_csv}...")
        self.orf_df = pd.read_csv(orf_csv)
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def evaluate(self, thresholds: tuple = (0.5, 0.7, 0.7), suffix: str = "de_novo") -> pd.DataFrame:
        p_thr, u_thr, d_thr = thresholds
        results = []
        
        orf_dict = {}
        for _, row in self.orf_df.iterrows():
            tid = row['tid']
            if tid not in orf_dict:
                orf_dict[tid] = []
            orf_dict[tid].append({
                'start': row['start'], 'end': row['end'],
                'length': row['length'], 'start_codon': row['start_codon']
            })

        for uuid, content in tqdm(self.pred_data.items(), desc="Evaluating Signals"):
            parts = str(uuid).split("-")
            tid_clean = parts[0].split('|')[0]
            cell_type = parts[1] if len(parts) > 1 else "Unknown"

            if tid_clean not in orf_dict:
                continue
                
            pred = content['pred'] 
            
            for orf in orf_dict[tid_clean]:
                s_idx, e_idx = orf['start'] - 1, orf['end']
                if e_idx > len(pred):
                    continue
                    
                cds_sig_real = np.expm1(pred[s_idx:e_idx])
                mean_intensity = np.mean(cds_sig_real)
                
                if mean_intensity <= 0:
                    pif, uni, drop = 0.0, 0.0, 0.0
                else:
                    pif = compute_pif(cds_sig_real)
                    uni = compute_uniformity(cds_sig_real[0::3])
                    drop = compute_dropoff(pred, s_idx, e_idx) 

                results.append({
                    'tid': tid_clean, 'cell_type': cell_type,
                    'start_codon': orf['start_codon'],
                    'start_pos': orf['start'], 'end_pos': orf['end'],
                    'length': orf['length'], 'Mean_Intensity': mean_intensity,
                    'PIF': pif, 'Uniformity': uni, 'Drop_off': drop,
                    'is_passed': (pif >= p_thr and uni >= u_thr and drop >= d_thr)
                })

        if not results:
            print("No ORFs evaluated.")
            return pd.DataFrame()

        df_res = pd.DataFrame(results)
        out_csv = os.path.join(self.out_dir, f"evaluated_orfs_{suffix}.csv")
        df_res.to_csv(out_csv, index=False)
        passed_df = df_res[df_res['is_passed'] == True]
        
        print(f"High Confidence Translations (Passed): {len(passed_df)} / {len(df_res)}")
        return passed_df


# =================================================================
# 主集成引擎: TranslationMinerPipeline
# =================================================================
class TranslationMinerPipeline:
    def __init__(self, 
                 model: torch.nn.Module, 
                 fasta_file: str, 
                 cell_type: str, 
                 out_dir: str = "./miner_results",
                 batch_size: int = 32):
        self.model = model
        self.fasta_file = fasta_file
        self.cell_type = cell_type
        self.out_dir = out_dir
        self.batch_size = batch_size
        
        os.makedirs(self.out_dir, exist_ok=True)
        base_name = os.path.basename(fasta_file).split('.')[0]
        self.orf_csv_path = os.path.join(self.out_dir, f"{base_name}_potential_orfs.csv")
        self.pred_pkl_path = os.path.join(self.out_dir, f"{base_name}_signal_preds.pkl")
        
    def run_inference(self, seq_dict: Dict[str, str], rank: Optional[int] = None, world_size: Optional[int] = None):
        self.model.eval()
        base_model = unwrap_model(self.model)
        device = base_model.device
        cell_mapping = getattr(base_model, "cell_type_mapping", {})
        
        # 1. 实例化新版 Dataset
        dataset = DeNovoSequenceDataset(seq_dict, self.cell_type, cell_mapping)
        
        # 2. 调用你优化的 DataLoder 构建函数
        dataloader, run_rank, run_world_size = _prepare_prediction_dataloader(
            dataset, collate_fn_denovo, num_samples=None, batch_size=self.batch_size,
            rank=rank, world_size=world_size
        )
        
        saved_data = {}
        iterator = tqdm(dataloader, desc=f"[Rank {run_rank}] Miner Infer") if (run_rank == 0 or run_world_size == 1) else dataloader
        
        with torch.no_grad():
            for batch in iterator:
                # 【关键修复】精准解包 6 个元素，绝不产生维度错乱
                b_uuids, b_cell_idxs, b_meta, b_seq, b_count, b_lengths = batch
                
                b_seq = b_seq.to(device)
                b_cell_idxs = b_cell_idxs.to(device)
                b_count = b_count.to(device)
                
                src_mask = (b_seq[:, :, 0] != -1)
                
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    out = base_model.predict(b_seq, b_count, b_cell_idxs, src_mask, head_names=["count"])
                
                # 转为概率
                probs_batch = torch.sigmoid(out["count"])
                
                # 解析并存入嵌套字典
                for i, uuid in enumerate(b_uuids):
                    valid_len = b_lengths[i]
                    pred_sample = probs_batch[i, :valid_len].cpu().numpy().astype(np.float16)
                    tid = str(uuid).split('-')[0]
                    
                    if self.cell_type not in saved_data:
                        saved_data[self.cell_type] = {}
                    saved_data[self.cell_type][tid] = pred_sample
                    
        total_preds = sum(len(tids) for tids in saved_data.values())
        print(f"[Rank {run_rank}] Saving {total_preds} predictions to {self.pred_pkl_path}")
        with open(self.pred_pkl_path, 'wb') as f:
            pickle.dump(saved_data, f)
            
        clean_up_memory()

    def run(self, 
            start_codons=["ATG", "CTG", "GTG", "TTG"], 
            stop_codons=["TGA", "TAA", "TAG"], 
            min_len=60, 
            thresholds=(0.5, 0.7, 0.7)):
        """执行完整 Pipeline"""
        print(f"\n[1/4] Reading Fasta File: {self.fasta_file}")
        seq_dict = read_fasta(self.fasta_file)
        
        print("\n[2/4] Building and Collapsing ORFs...")
        orf_builder = SequenceORFBuilder(seq_dict)
        orf_builder.build_and_collapse_orfs(
            start_codons=start_codons, stop_codons=stop_codons, 
            min_len=min_len, out_file=self.orf_csv_path
        )
        
        print("\n[3/4] Running Deep Learning Translation Prediction...")
        self.run_inference(seq_dict)
        
        print("\n[4/4] Evaluating Translation Signals & Filtering...")
        evaluator = TranslationSignalEvaluator(self.pred_pkl_path, self.orf_csv_path, self.out_dir)
        final_df = evaluator.evaluate(thresholds=thresholds, suffix=self.cell_type)
        
        print("\n🎉 Pipeline Execution Completed Successfully!")
        return final_df