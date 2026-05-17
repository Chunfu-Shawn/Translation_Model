import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
# =================================================================
# [MODIFIED] 引入 List 和 Union 以支持多种类型输入
# =================================================================
from typing import Dict, Optional, List, Union
from tqdm import tqdm

from eval.save_prediction_results import _prepare_prediction_dataloader
from utils import unwrap_model, clean_up_memory

# =================================================================
# 工具函数: Fasta 解析
# =================================================================
def read_fasta(file_paths: Union[str, List[str]]) -> Dict[str, str]:
    """读取单个或多个 Fasta 文件并返回合并后的 {tid: sequence} 字典"""
    if isinstance(file_paths, str):
        file_paths = [file_paths]
        
    seq_dict = {}
    total_files = len(file_paths)
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"[Warning] Fasta file not found: {file_path}. Skipping...")
            continue
            
        print(f"Reading Fasta File: {file_path}")
        curr_tid = ""
        curr_seq = []
        file_seq_count = 0
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if curr_tid:
                        seq_dict[curr_tid] = "".join(curr_seq)
                        file_seq_count += 1
                    # 提取 > 后面的 ID，通常以空格分隔取第一部分
                    curr_tid = line[1:].split()[0]
                    curr_seq = []
                else:
                    curr_seq.append(line.upper())
                    
            if curr_tid:
                seq_dict[curr_tid] = "".join(curr_seq)
                file_seq_count += 1
                
        print(f"  -> Loaded {file_seq_count} sequences from this file.")
        
    print(f"✅ Successfully loaded a total of {len(seq_dict)} unique sequences from {total_files} file(s).")
    return seq_dict

# =================================================================
# 零样本推理 Dataset (保持不变)
# =================================================================
class DeNovoSequenceDataset(Dataset):
    """
    轻量级内存 Dataset，专为仅含有 RNA 序列的翻译图谱预测设计。
    """
    def __init__(self, 
                 seq_dict: Dict[str, str], 
                 species: str,
                 cell_type: str, 
                 cell_expr_vector: np.ndarray,
                 min_len: int = 200,
                 max_len: int = 20000):
        self.tids = list(seq_dict.keys())
        self.seq_dict = seq_dict
        self.species = species
        self.cell_type = cell_type
        self.cell_expr_vector = np.array(cell_expr_vector, dtype=np.float32)
        
        self.nt_mapping = dict(zip("ACGTN", range(5)))
        self.min_len = min_len
        self.max_len = max_len  # 设置最大支持长度限制
        
        self.uuids = []
        self.seq_embs = []
        self.lengths = []
        
        # 预先进行 One-hot 编码并记录长度
        for tid in tqdm(self.tids, desc="Encoding Sequences"):
            seq = self.seq_dict[tid].upper()
            
            # 如果序列长度超过模型限制，进行丢弃
            if len(seq) > self.max_len or len(seq) < self.min_len:
                continue
                
            self.lengths.append(len(seq))
            
            # 清理 ID，防止带版本号或复合格式影响后续映射
            tid_clean = str(tid).split('|')[0]
            # if tid_clean.startswith('ENS') and '.' in tid_clean:
            #     tid_clean = tid_clean.split('.')[0]
                
            uuid = f"{tid_clean}-{self.species}-{self.cell_type}-Prediction"
            self.uuids.append(uuid)
            
            seq_idx = [self.nt_mapping.get(nt, 4) for nt in seq]
            seq_emb = np.eye(5)[seq_idx, :4]
            self.seq_embs.append(seq_emb)
            
        self.n_samples = len(self.lengths)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        uuid = self.uuids[idx]
        species = str(self.species)
        cell_expr_tensor = torch.from_numpy(self.cell_expr_vector)
        seq_emb = torch.from_numpy(self.seq_embs[idx]).float()
        
        # 动态生成占位的 Count 矩阵 (纯 0)，长度将匹配截断后的长度
        count_emb = torch.zeros((self.lengths[idx], 1), dtype=torch.float32)
        
        # 返回空字典 meta_info 作为占位
        meta_info = {}
        
        return uuid, species, cell_expr_tensor, meta_info, seq_emb, count_emb

def collate_fn_denovo(batch):
    """专门配套的组装函数"""
    uuids, species, cell_exprs, meta_infos, seq_embs, count_embs = zip(*batch)
    lengths = [s.shape[0] for s in seq_embs]
    
    seq_padded = pad_sequence(seq_embs, batch_first=True, padding_value=-1)
    count_padded = pad_sequence(count_embs, batch_first=True, padding_value=-1)
    species_list = list(species)
    cell_exprs = torch.stack(cell_exprs)
    
    return uuids, species_list, cell_exprs, meta_infos, seq_padded, count_padded, lengths

# =================================================================
# 主引擎: 仅预测翻译图谱
# =================================================================
class TranslationProfilePredictor:
    def __init__(self, 
                 model: torch.nn.Module, 
                 fasta_files: Union[str, List[str]]):
        self.model = model
        self.fasta_files = fasta_files

        print(f"\nInitializing Fasta parsing pipeline...")
        # 传入单一字符串或列表都可以被正确解析和合并
        self.seq_dict = read_fasta(self.fasta_files)

    def run(
            self,
            species: str,
            cell_type: str,
            cell_expr_vector: np.ndarray, 
            target_tids: Optional[list] = None, 
            out_dir: str = "./results",
            suffix: str = "results",
            min_len: int = 200,
            max_len: int = 20000,
            batch_size: int = 32,
            rank: Optional[int] = None, 
            world_size: Optional[int] = None):
        """
        执行 Fasta 读取与预测。
        如果传入了 target_tids (list)，则仅预测存在于该列表中的转录本序列。
        """

        os.makedirs(out_dir, exist_ok=True)
        pred_pkl_path = os.path.join(out_dir, f"predictions_count.{self.model.model_name}.{suffix}.pkl")

        # ========================================================
        # 目标转录本过滤逻辑
        # ========================================================
        if target_tids is not None:
            target_set = set(target_tids)
            filtered_seq_dict = {}
            for tid, seq in self.seq_dict.items():
                # 同样清理 Fasta 头里的版本号或管道符
                clean_tid = str(tid).split('|')[0] #.split('.')[0]
                if clean_tid in target_set:
                    filtered_seq_dict[tid] = seq
            
            print(f"Filtered Fasta: Keeping {len(filtered_seq_dict)} sequences matching target Tids "
                  f"(out of {len(self.seq_dict)} total).")
            seq_dict = filtered_seq_dict
            
            if not seq_dict:
                print("Warning: No matching sequences found! Please check if your Tids match the Fasta headers.")
                return None
        else:
            seq_dict = self.seq_dict
                
        print("\nRunning Deep Learning Translation Prediction...")
        self.model.eval()
        base_model = unwrap_model(self.model)
        device = base_model.device
        
        # 构建 Dataset 和 DataLoader
        dataset = DeNovoSequenceDataset(seq_dict, species, cell_type, cell_expr_vector, min_len, max_len)
        dataloader, run_rank, run_world_size = _prepare_prediction_dataloader(
            dataset, collate_fn_denovo, num_samples=None, batch_size=batch_size,
            rank=rank, world_size=world_size
        )
        
        saved_data = {cell_type: {}}
        iterator = tqdm(dataloader, desc=f"Predicting") if (run_rank == 0 or run_world_size == 1) else dataloader
        
        with torch.no_grad():
            for batch in iterator:
                b_uuids, species_list, b_cell_exprs, b_meta, b_seq, b_count, b_lengths = batch
                
                b_cell_exprs = b_cell_exprs.to(device)
                b_seq = b_seq.to(device)
                b_count = b_count.to(device)
                
                src_mask = (b_seq[:, :, 0] != -1)
                
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    out = base_model.predict(
                        seq_batch=b_seq, 
                        count_batch=b_count, 
                        species=species_list,
                        expr_vector=b_cell_exprs,
                        src_mask=src_mask, 
                        head_names=["count"])
                
                probs_batch = out["count"]
                
                # 解析并存入目标字典
                for i, uuid in enumerate(b_uuids):
                    valid_len = b_lengths[i]
                    pred_sample = probs_batch[i, :valid_len].cpu().numpy().astype(np.float16)
                    # 恢复 tid
                    tid = str(uuid).split('-')[0]
                    saved_data[cell_type][tid] = pred_sample
                    
        total_preds = len(saved_data[cell_type])
        print(f"Saving {total_preds} predictions to {pred_pkl_path}")
        with open(pred_pkl_path, 'wb') as f:
            pickle.dump(saved_data, f)
            
        clean_up_memory()
        print("🎉 Translation Profile Prediction Completed Successfully!")
        
        return pred_pkl_path