import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Optional
from tqdm import tqdm

from eval.save_prediction_results import _prepare_prediction_dataloader
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
# 零样本推理 Dataset
# =================================================================
class DeNovoSequenceDataset(Dataset):
    """
    轻量级内存 Dataset，专为仅含有 RNA 序列的翻译图谱预测设计。
    """
    def __init__(self, 
                 seq_dict: Dict[str, str], 
                 cell_type: str, 
                 cell_type_mapping: Dict[str, int], 
                 min_len: int = 200,
                 max_len: int = 20000):
        self.tids = list(seq_dict.keys())
        self.seq_dict = seq_dict
        self.cell_type = cell_type
        self.cell_type_idx = cell_type_mapping.get(cell_type, 0)
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
            if tid_clean.startswith('ENS') and '.' in tid_clean:
                tid_clean = tid_clean.split('.')[0]
                
            uuid = f"{tid_clean}-{self.cell_type}-Prediction"
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
        
        # 动态生成占位的 Count 矩阵 (纯 0)，长度将匹配截断后的长度
        count_emb = torch.zeros((self.lengths[idx], 1), dtype=torch.float32)
        
        # 返回空字典 meta_info 作为占位
        meta_info = {}
        
        return uuid, cell_type_idx, meta_info, seq_emb, count_emb

def collate_fn_denovo(batch):
    """专门配套的组装函数"""
    uuids, cell_idxs, meta_infos, seq_embs, count_embs = zip(*batch)
    lengths = [s.shape[0] for s in seq_embs]
    
    seq_padded = pad_sequence(seq_embs, batch_first=True, padding_value=-1)
    count_padded = pad_sequence(count_embs, batch_first=True, padding_value=-1)
    cell_idxs = torch.stack(cell_idxs)
    
    return uuids, cell_idxs, meta_infos, seq_padded, count_padded, lengths

# =================================================================
# 主引擎: 仅预测翻译图谱
# =================================================================
class TranslationProfilePredictor:
    def __init__(self, 
                 model: torch.nn.Module, 
                 fasta_file: str):
        self.model = model
        self.fasta_file = fasta_file

        print(f"\nReading Fasta File: {self.fasta_file}")
        self.seq_dict = read_fasta(self.fasta_file)

    def run(
            self,
            cell_type: str, 
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
        # [NEW] 目标转录本过滤逻辑
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
        cell_mapping = getattr(base_model, "cell_type_mapping", {})
        
        # 构建 Dataset 和 DataLoader
        dataset = DeNovoSequenceDataset(seq_dict, cell_type, cell_mapping, min_len, max_len)
        dataloader, run_rank, run_world_size = _prepare_prediction_dataloader(
            dataset, collate_fn_denovo, num_samples=None, batch_size=batch_size,
            rank=rank, world_size=world_size
        )
        
        saved_data = {cell_type: {}}
        iterator = tqdm(dataloader, desc=f"Predicting") if (run_rank == 0 or run_world_size == 1) else dataloader
        
        with torch.no_grad():
            for batch in iterator:
                b_uuids, b_cell_idxs, b_meta, b_seq, b_count, b_lengths = batch
                
                b_seq = b_seq.to(device)
                b_cell_idxs = b_cell_idxs.to(device)
                b_count = b_count.to(device)
                
                src_mask = (b_seq[:, :, 0] != -1)
                
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    out = base_model.predict(b_seq, b_count, b_cell_idxs, src_mask, head_names=["count"])
                
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