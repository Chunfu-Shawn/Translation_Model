import torch
import numpy as np
import os
import pickle
from tqdm import tqdm
from typing import Union, List, Optional, Dict
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from utils import clean_up_memory
from train.distributed_bucket_sampler import DistributedBucketSampler 

def save_prediction_results(
    model, 
    dataset, 
    num_samples: int = 200, 
    batch_size: int = 16,
    out_dir: str = "./results", 
    suffix: str = "",
    rank: Optional[int] = None,
    world_size: Optional[int] = None
):
    """
    运行模型预测并保存结果 (兼容单机和分布式环境)。
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    # --- 0. 自动处理分布式参数 ---
    # 如果用户没有显式传入 rank/world_size，则根据当前环境自动判断
    if torch.distributed.is_initialized():
        if rank is None:
            rank = torch.distributed.get_rank()
        if world_size is None:
            world_size = torch.distributed.get_world_size()
    else:
        # 非分布式环境 (Notebook 或 单卡脚本)
        print("Distributed environment not detected. Running in single-process mode.")
        if rank is None:
            rank = 0
        if world_size is None:
            world_size = 1

    # --- 1. Determine file name ---
    if world_size > 1:
        file_name = f"predictions_seq.{model.model_name}.{suffix}.rank{rank}.pkl"
    else:
        file_name = f"predictions_seq.{model.model_name}.{suffix}.pkl"
        
    save_path = os.path.join(out_dir, file_name)
    print(f"[Rank {rank}] Predictions will be saved to: {save_path}")

    # --- 2. Select samples ---
    all_indices = np.arange(len(dataset))
    
    if num_samples is not None and len(all_indices) > num_samples:
        np.random.seed(42) 
        target_indices = np.random.choice(all_indices, num_samples, replace=False)
    else:
        target_indices = all_indices
        
    print(f"[Rank {rank}] Selected {len(target_indices)} samples.")
    
    # Create Subset
    subset = Subset(dataset, target_indices)

    # --- 3. Prepare Bucket Sampler ---
    # 获取 Subset 对应的长度
    if hasattr(dataset, "lengths"):
        subset_lengths = [dataset.lengths[i] for i in target_indices]
    else:
        # Fallback: 手动计算长度
        print("Calculating lengths manually...")
        subset_lengths = [len(dataset[i][3]) for i in target_indices]

    # 初始化 Sampler (关键：显式传入 num_replicas 和 rank)
    sampler = DistributedBucketSampler(
        lengths=subset_lengths,
        batch_size=batch_size,
        num_replicas=world_size,  # 这里的 world_size 现在保证不是 None
        rank=rank,                # 这里的 rank 现在保证不是 None
        shuffle=False, 
        drop_last=False
    )

    # --- 4. Define Collate Function with UUID ---
    def collate_fn(batch):
        # 提取 UUID (假设 item[0] 是 UUID 字符串)
        uuids = [item[0] for item in batch] 
        cell_types = [str(item[0].split("-")[1]) if "-" in str(item[0]) else "Unknown" for item in batch]
        
        # 提取 Meta Info
        cds_starts = [item[2].get("cds_start_pos", -1) for item in batch]
        cds_ends = [item[2].get("cds_end_pos", -1) for item in batch]
        depths = [item[2].get("depth", None) for item in batch]
        coverages = [item[2].get("coverage", None) for item in batch]

        # 提取 Tensors
        seq_embs = [item[3] for item in batch]
        count_embs = [item[4] for item in batch]
        
        lengths = [s.shape[0] for s in seq_embs]
        
        # Padding
        seq_padded = pad_sequence(seq_embs, batch_first=True, padding_value=-1)
        count_padded = pad_sequence(count_embs, batch_first=True, padding_value=-1)
        
        return uuids, cell_types, seq_padded, count_padded, lengths, cds_starts, cds_ends, depths, coverages

    # --- 5. DataLoader ---
    dataloader = DataLoader(
        subset, 
        batch_sampler=sampler, # 使用 sampler
        collate_fn=collate_fn,
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True
    )

    # --- 6. Inference Loop ---
    saved_data = {} 
    
    # 如果是 rank 0 且单机运行，显示进度条；多机环境下避免打印混乱可根据需要调整
    iterator = tqdm(dataloader, desc=f"[Rank {rank}] Inferencing") if (rank == 0 or world_size == 1) else dataloader

    for batch_data in iterator:
        b_uuids, b_cell_types, b_seq, b_count, b_lengths, b_cds_starts, b_cds_ends, b_depths, b_coverages = batch_data
        
        b_seq = b_seq.to(model.device)
        b_count = b_count.to(model.device)
        
        masked_batch = torch.zeros_like(b_count)
        src_mask = (b_seq[:, :, 0] != -1)
        
        with torch.no_grad():
            out = model.predict(b_seq, masked_batch, b_cell_types, src_mask, head_names=["count"])
            pred_batch = out["count"]

        # Save per sample
        for i, uuid in enumerate(b_uuids):
            valid_len = b_lengths[i]
            
            # Ground Truth & Pred
            gt = b_count[i, :valid_len].cpu().numpy().astype(np.float16)
            pred_sample = pred_batch[i, :valid_len].squeeze().cpu().numpy().astype(np.float16)
            
            # CDS Info
            raw_start = b_cds_starts[i]
            raw_end = b_cds_ends[i]
            cds_info = {'start': int(raw_start), 'end': int(raw_end)} if (raw_start != -1) else None

            saved_data[uuid] = {
                'truth': gt,
                'pred': pred_sample,
                'cds_info': cds_info,
                'depth': b_depths[i],
                'coverage': b_coverages[i]
            }

    # --- 7. Save ---
    print(f"[Rank {rank}] Saving {len(saved_data)} transcripts to disk...")
    with open(save_path, 'wb') as f:
        pickle.dump(saved_data, f)

    clean_up_memory()
    print(f"[Rank {rank}] Done.")
    return save_path