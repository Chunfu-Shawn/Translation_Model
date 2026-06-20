import torch
import numpy as np
import os
import pickle
from tqdm import tqdm
from typing import Union, List, Optional, Dict
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from utils import clean_up_memory, unwrap_model
from train.distributed_bucket_sampler import DistributedBucketSampler 

def _prepare_prediction_dataloader(
    dataset, 
    collate_fn, 
    num_samples: Optional[int], 
    batch_size: int, 
    rank: Optional[int] = None, 
    world_size: Optional[int] = None
):
    """
    处理单机/分布式环境判定、随机子集采样和 DataLoader 构建。
    """
    # --- 1. 自动处理分布式参数 ---
    if torch.distributed.is_initialized():
        rank = rank if rank is not None else torch.distributed.get_rank()
        world_size = world_size if world_size is not None else torch.distributed.get_world_size()
    else:
        rank = rank if rank is not None else 0
        world_size = world_size if world_size is not None else 1

    # --- 2. 样本子集选择 ---
    all_indices = np.arange(len(dataset))
    if num_samples is not None and len(all_indices) > num_samples:
        np.random.seed(42) 
        target_indices = np.random.choice(all_indices, num_samples, replace=False)
    else:
        target_indices = all_indices
        
    print(f"[Rank {rank}] Selected {len(target_indices)} samples for inference.")
    subset = Subset(dataset, target_indices)

    # --- 3. 准备 Bucket Sampler ---
    if hasattr(dataset, "lengths"):
        subset_lengths = [dataset.lengths[i] for i in target_indices]
    else:
        # Fallback: 在两种 dataset 中，seq_emb 都在索引 3 的位置
        print(f"[Rank {rank}] Calculating lengths manually...")
        subset_lengths = [dataset[i][4].shape[0] for i in target_indices]

    sampler = DistributedBucketSampler(
        lengths=subset_lengths,
        batch_size=batch_size,
        num_replicas=world_size, 
        rank=rank,               
        shuffle=False, 
        drop_last=False
    )

    # --- 4. 构建 DataLoader ---
    dataloader = DataLoader(
        subset, 
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True
    )
    
    return dataloader, rank, world_size


def save_count_predictions(
    model, 
    dataset, 
    num_samples: int = 200, 
    batch_size: int = 16,
    out_dir: str = "./results", 
    suffix: str = "count",
    rank: Optional[int] = None,
    world_size: Optional[int] = None
):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    base_model = unwrap_model(model)

    # --- Collate Function for Translation Dataset ---
    def collate_fn_count(batch):
        # 解包: uuid, cell_types, cell_type_idx, meta_info, seq_emb, count_emb
        uuids, species, cell_types, expr_vectors, meta_infos, seq_embs, count_embs = zip(*batch)
        lengths = [s.shape[0] for s in seq_embs]
        
        seq_padded = pad_sequence(seq_embs, batch_first=True, padding_value=-1)
        count_padded = pad_sequence(count_embs, batch_first=True, padding_value=-1)
        species_list = list(species)
        cell_types = list(cell_types)
        expr_vectors = torch.stack(expr_vectors)
        
        return uuids, species_list, cell_types, expr_vectors, meta_infos, seq_padded, count_padded, lengths

    # 获取 DataLoader
    dataloader, run_rank, run_world_size = _prepare_prediction_dataloader(
        dataset, collate_fn_count, num_samples, batch_size, rank, world_size
    )

    # 确定文件名
    model_name = getattr(base_model, "model_name", "model")
    file_name = f"predictions_count.{model_name}.{suffix}"
    file_name += f".rank{run_rank}.pkl" if run_world_size > 1 else ".pkl"
    save_path = os.path.join(out_dir, file_name)

    saved_data = {} 
    iterator = tqdm(dataloader, desc=f"[Rank {run_rank}] Infer Count") if (run_rank == 0 or run_world_size == 1) else dataloader

    # --- Inference Loop ---
    for batch_data in iterator:
        b_uuids, b_species, b_cell_types, b_expr_vectors, b_meta, b_seq, b_count, b_lengths = batch_data
        
        b_seq = b_seq.to(base_model.device)
        
        # Count head 推理通常需要全零的 count 输入作为占位符
        masked_batch = torch.zeros_like(b_count).to(base_model.device)
        src_mask = (b_seq[:, :, 0] != -1)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = base_model.predict(
                    seq_batch=b_seq, 
                    count_batch=masked_batch,
                    species=b_species,
                    expr_vector=b_expr_vectors, 
                    src_mask=src_mask, 
                    head_names=["count"]
                    )
            pred_batch = out["count"]

        # 解析并存储 (修改为嵌套字典结构)
        for i, uuid in enumerate(b_uuids):
            valid_len = b_lengths[i]
            # 仅提取 pred
            if isinstance(pred_batch, dict):
                pred = pred_batch["profile"][i, :valid_len].squeeze().cpu().numpy().astype(np.float16)
            else:
                pred = pred_batch[i, :valid_len].squeeze().cpu().numpy().astype(np.float16)
            
            # 解析 UUID 获取 tid
            parts = str(uuid).split('-')
            tid = parts[0]
            
            # =================================================================
            # [MODIFIED] 直接使用 dataset 提供的干净 cell_type，不再依赖 uuid 解析
            # =================================================================
            cell_type = str(b_cell_types[i])

            # 嵌套字典赋值
            if cell_type not in saved_data:
                saved_data[cell_type] = {}
            saved_data[cell_type][tid] = pred

    # 统计预测结果数量
    total_preds = sum(len(tids) for tids in saved_data.values())
    print(f"[Rank {run_rank}] Saving {total_preds} Count predictions across {len(saved_data)} cell types to {save_path}")
    
    with open(save_path, 'wb') as f:
        pickle.dump(saved_data, f)
    
    clean_up_memory()

    return save_path