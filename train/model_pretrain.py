import os
import time
import json
import math
import contextlib
from typing import Union, Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from utils import unwrap_model
from data.translation_dataset import TranslationDataset
from train.distributed_balanced_bucket_sampler import DistributedBucketSampler
from train.masking_adapter import BatchMaskingAdapter, get_dynamic_mask_ratio

torch.set_printoptions(profile="full")

class TensorEncoder(json.JSONEncoder):
    """Custom JSON encoder to automatically parse and convert PyTorch Tensors."""
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist() if obj.numel() > 1 else float(obj.item())
        return super().default(obj)

def create_lr_lambda(total_steps: int, warmup_steps: int = 0, min_eta: float = 1e-4):
    """Linear warmup then cosine decay lambda for LambdaLR."""
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return max(min_eta, float(current_step) / float(max(1, warmup_steps)))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(max(min_eta, decay))
    return lr_lambda


def default_no_weight_decay(name: str) -> bool:
    """
    Heuristic to decide whether to disable weight decay for a parameter name.
    Common rule: bias, LayerNorm weights, and embedding weights typically do not use weight decay.
    """
    name = name.lower()
    if name.endswith(".bias"):
        return True
    if "layernorm" in name or "layer_norm" in name or ".ln" in name:
        return True
    if ".embedding" in name or "embed" in name:
        return True
    return False


class PretrainingTrainer:
    """
    Pretraining trainer class with Continuous Expression Vector Support and Noise Injection.
    """
    def __init__(
        self,
        model: nn.Module,
        dataset_paths: Union[str, List[str]],
        val_dataset_paths: Union[str, List[str]],
        dataset_name: str,
        batch_size: int,
        checkpoint_dir: str,
        log_dir: str,
        world_size: int,
        rank: int,
        resume: bool = True,
        mask_value: float = 0,
        print_progress_every: int = 50,
        save_every: int = 5,
        epoch_num: int = 100,
        patience: int = 8,
        mask_perc: dict = {"count": 0.3, "species": 0.15, "cell": 0.15},
        expr_noise_std: float = 0.1, # Standard deviation of Gaussian noise to inject (10% of Z-score variance)
        learning_rate: float = 1e-5,
        lr_warmup_perc: float = 0.2,
        accumulation_steps: int = 5,
        balance_classes: bool = False,
        beta: Tuple[float, float] = (0.9, 0.98),
        epsilon: float = 1e-9,
        weight_decay: float = 0.01
    ):
        # basic attributes
        self.model = model
        self.model_name = unwrap_model(model).model_name
        self.rank = rank
        self.world_size = world_size
        self.resume = resume
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.patience = patience
        self.patience_counter = 0
        self.ac_steps = accumulation_steps
        self.lr = learning_rate
        self.lr_warmup_perc = lr_warmup_perc
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self._print_progress_every = print_progress_every
        self._save_every = save_every

        # device for this process
        self.device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")

        # masking and sampler (reuse user's adapters)
        self.masking_adapter = BatchMaskingAdapter(mask_value)
        self.mask_perc = mask_perc
        self.current_mask_range = self.mask_perc.get("count", (0, 0.2))
        self.balance_classes = balance_classes

        # Build dataset and sampler dynamically
        self.dataset, self.sampler = self._build_dataset_and_sampler(dataset_paths, is_train=True)
        self.val_dataset, self.val_sampler = self._build_dataset_and_sampler(val_dataset_paths, is_train=False)

        # Expression logic: Calculate global mean vector across all datasets
        self.expr_noise_std = expr_noise_std
        self.cell_mean_expr = {}
        self._cache_expression_means()

        # counts and scheduling
        self.steps_per_epoch = len(self.sampler)  # per-process mini-batches per epoch
        self._total_steps = max(1, int(self.epoch_num * self.steps_per_epoch // max(1, self.ac_steps)))
        
        # loss criterions
        self.dynamcis_criterion = nn.SmoothL1Loss(reduction="none", beta=1)  #nn.MSELoss(reduction="none")
        self.te_criterion = nn.MSELoss(reduction="none")

        # build optimizer & scheduler & scaler
        self.optimizer = self._build_optimizer(lr=self.lr, betas=self.beta, eps=self.epsilon, weight_decay=self.weight_decay)
        warmup_steps = int(self.lr_warmup_perc * self._total_steps)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=create_lr_lambda(total_steps=self._total_steps, warmup_steps=warmup_steps, min_eta=1e-4)
        )
        self.scaler = torch.amp.GradScaler() 

        # logging & checkpointing
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.model_full_name = f"{self.model_name}.{dataset_name}.{self.batch_size*self.ac_steps*self.world_size}_{self.lr}"
        self.training_epoch_data: List[Dict[str, Any]] = []
        self.training_batch_data: List[Dict[str, Any]] = []

        # bookkeeping for resume
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        if self.resume:
            self._maybe_load_checkpoint()

        if self.rank == 0:
            print(f"[Trainer] model_trainer_name={self.model_full_name}")
            print(f"[Trainer] device={self.device}, steps_per_epoch={self.steps_per_epoch}, total_steps={self._total_steps}")
            print(f"[Trainer] mask_perc={self.mask_perc}")
            print(f"[Trainer] Train Datasets: {len(self.dataset)} samples. Eval Datasets: {len(self.val_dataset)} samples.")

    # =========================================================
    # Dataset Builder Method
    # =========================================================
    def _build_dataset_and_sampler(
        self, 
        paths: Union[str, List[str]], 
        is_train: bool
    ) -> Tuple[ConcatDataset, DistributedBucketSampler]:
        """
        Dynamically build a ConcatDataset and its corresponding BucketSampler.
        """
        if isinstance(paths, str):
            paths = [paths]
            
        datasets = []
        for path in paths:
            ds = TranslationDataset.from_h5(path, lazy=True) 
            datasets.append(ds)
            
        combined_dataset = ConcatDataset(datasets)
        
        # {species: {cell_type: expr_array}}
        combined_dataset.global_expr_dict = {}
        
        all_lengths = []
        all_cell_types = []
        
        for ds in combined_dataset.datasets:
            all_lengths.extend(ds.lengths)
            
            if hasattr(ds, 'cell_types') and hasattr(ds, 'species'):
                # 1. 为 Sampler 构建跨物种的类别标签
                if self.balance_classes:
                    combined_types = [f"{sp}_{ct}" for sp, ct in zip(ds.species, ds.cell_types)]
                    all_cell_types.extend(combined_types)
                
                # 2. 极速提取独一无二的 (species, cell_type) 组合
                if hasattr(ds, 'cell_expr_dict') and ds.cell_expr_dict:
                    unique_pairs = set(zip(ds.species, ds.cell_types))
                    
                    for sp, ct in unique_pairs:
                        if ct in ds.cell_expr_dict:
                            if sp not in combined_dataset.global_expr_dict:
                                combined_dataset.global_expr_dict[sp] = {}
                            # 填入表达向量（不同 h5 文件中相同的组织表达谱通常一致，直接覆盖即可）
                            combined_dataset.global_expr_dict[sp][ct] = ds.cell_expr_dict[ct]
                
        sampler = DistributedBucketSampler(
            lengths=all_lengths,
            batch_size=self.batch_size,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=is_train,
            drop_last=is_train,
            balance_classes=self.balance_classes if is_train else False, 
            cell_types=all_cell_types if (self.balance_classes and is_train) else None
        )
        
        return combined_dataset, sampler
        
        
    # =========================================================
    # Internal Method for Caching Expression Means
    # =========================================================
    def _cache_expression_means(self):
        """
        基于预构建的 global_expr_dict，极速计算并缓存跨物种的细胞特异性平均表达向量。
        彻底移除了无生物学意义的“物种均值”和“全局均值”。
        """
        # 直接获取挂载在 ConcatDataset 上的全局字典
        global_expr_dict = getattr(self.dataset, 'global_expr_dict', {})
        
        # 1. 遍历精简后的字典 (层级为 species -> cell_type -> vector)
        for sp, ct_dict in global_expr_dict.items():
            for ct, vec in ct_dict.items():
                # 按细胞类型收集 (跨物种合并)
                if ct not in self.cell_mean_expr:
                    self.cell_mean_expr[ct] = []
                self.cell_mean_expr[ct].append(vec)
                        
        # 2. 将收集到的列表聚合为均值 Tensor
        for ct in self.cell_mean_expr:
            ct_mean = np.mean(self.cell_mean_expr[ct], axis=0)
            self.cell_mean_expr[ct] = torch.from_numpy(ct_mean).float()
            
        # 3. 打印精简后的日志
        if self.cell_mean_expr:
            if self.rank == 0:
                print(f"[Trainer] Cached {len(self.cell_mean_expr)} cross-species cell means.")
        else:
            if self.rank == 0:
                print("[Trainer] WARNING: No expression vectors found. Masked cell types will strictly fallback to absolute zeros.")

    # -------------------------
    # optimizer / param grouping
    # -------------------------
    def _get_param_groups(self):
        """
        Collect parameters that require gradients and split them into two groups:
        - with weight decay (default)
        - without weight decay (bias, LayerNorm, embeddings)
        This avoids applying weight decay to normalization and bias parameters.
        """
        model_unwrapped = unwrap_model(self.model)
        decay_params = []
        no_decay_params = []
        for name, p in model_unwrapped.named_parameters():
            if not p.requires_grad:
                continue
            if default_no_weight_decay(name):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        groups = []
        if len(decay_params) > 0:
            groups.append({"params": decay_params, "weight_decay": self.weight_decay})
        if len(no_decay_params) > 0:
            groups.append({"params": no_decay_params, "weight_decay": 0.0})
        return groups
    
    def _build_optimizer(self, lr: float, betas: Tuple[float, float], eps: float, weight_decay: float):
        """
        Construct AdamW optimizer using parameter groups returned by _get_param_groups.
        """
        groups = self._get_param_groups()
        if len(groups) == 0:
            raise RuntimeError("No trainable parameters found. Check requires_grad flags.")
        opt = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)
        return opt

    # -------------------------
    # checkpoint helpers
    # -------------------------
    def _checkpoint_paths(self) -> Tuple[str, str]:
        latest = os.path.join(self.checkpoint_dir, self.model_full_name + ".latest.pt")
        best = os.path.join(self.checkpoint_dir, self.model_full_name + ".best.pt")
        return latest, best

    def save_checkpoint(self, epoch: int, is_best: bool):
        """
        Save the unwrapped model state_dict, optimizer, scheduler, scaler and bookkeeping.
        Only rank 0 should call this or it will save multiple times.
        """
        latest, best = self._checkpoint_paths()
        state = {
            "epoch": epoch,
            "model": unwrap_model(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_epoch_data": self.training_epoch_data,
            "training_batch_data": self.training_batch_data,
        }
        torch.save(state, latest)
        if is_best:
            torch.save(state, best)
        if self.rank == 0:
            print(f"[Trainer] Saved checkpoint to {latest} (best={is_best})")

    def _maybe_load_checkpoint(self):
        """
        Try to load the latest checkpoint if it exists.
        """
        latest, _ = self._checkpoint_paths()
        if os.path.isfile(latest):
            if self.rank == 0:
                print(f"[Trainer] Loading checkpoint {latest}")
            map_loc = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt = torch.load(latest, map_location=map_loc)
            # restore model weights into unwrapped model
            model_unwrapped = unwrap_model(self.model)
            model_unwrapped.load_state_dict(ckpt["model"], strict=True)
            self.start_epoch = int(ckpt.get("epoch", 0))

            bvl = ckpt.get("best_val_loss", float('inf'))
            if isinstance(bvl, torch.Tensor):
                self.best_val_loss = float(bvl.sum().item())
            else:
                self.best_val_loss = bvl

            # attempt to restore optimizer/scheduler/scaler if present
            try:
                if "optimizer" in ckpt and ckpt["optimizer"] is not None:
                    self.optimizer.load_state_dict(ckpt["optimizer"])
                if "scheduler" in ckpt and ckpt["scheduler"] is not None and self.scheduler is not None:
                    self.scheduler.load_state_dict(ckpt["scheduler"])
                if "scaler" in ckpt and ckpt["scaler"] is not None:
                    self.scaler.load_state_dict(ckpt["scaler"])
                if "training_epoch_data" in ckpt:
                    self.training_epoch_data = ckpt["training_epoch_data"]
                if "training_batch_data" in ckpt:
                    self.training_batch_data = ckpt["training_batch_data"]
            except Exception as e:
                if self.rank == 0:
                    print(f"[Trainer] Warning: could not fully restore optimizer/scheduler/scaler: {e}")
            if self.rank == 0:
                print(f"[Trainer] Resumed from epoch {self.start_epoch}")
    
    # -------------------------
    # masking + collate pipeline
    # -------------------------
    def collate_mask_pad_batch_to_cuda(self, batch, is_eval=False):
        """
        Collate function matching user's masking behavior.
        Receives expr_vectors directly from the dataset.
        """
        # Unpack matching TranslationDataset.__getitem__ outputs
        _, species, cell_types, expr_vectors, meta_info, seq_embs, count_embs = zip(*batch)

        species_list = list(species)
        cell_types = list(cell_types)
        expr_batch = torch.stack(expr_vectors) # [B, d_expr]
        
        cds_starts = [meta.get("cds_start_pos", -1) for meta in meta_info]
        cds_stops = [meta.get("cds_stop_pos", -1) for meta in meta_info]
        motif_occs = [meta.get("motif_occs", []) for meta in meta_info]

        seq_embs_padded = pad_sequence(seq_embs, batch_first=True, padding_value=-1)
        count_embs_padded = pad_sequence(count_embs, batch_first=True, padding_value=-1)

        pad_masks = (seq_embs_padded != -1)[:, :, 0].squeeze(-1)
        B, L = seq_embs_padded.shape[:2]

        # count masking
        count_emb_masks = torch.zeros((B, L), dtype=torch.bool)
        if "count" in self.mask_perc:
            count_embs_masked, count_emb_masks = self.masking_adapter.get_random_masked_batch(
                embeddings=count_embs_padded, 
                cds_starts=cds_starts, 
                occs=motif_occs, 
                pad_mask=pad_masks, 
                mask_perc_range=self.current_mask_range, 
                full_mask_perc=1 if is_eval else 0.0
            )
        else:
            count_embs_masked = count_embs_padded.clone()

        # ==========================================
        # Species Masking
        # ==========================================
        species_mask = torch.zeros(B, dtype=torch.bool)
        if "species" in self.mask_perc and not is_eval:
            species_mask = torch.rand(B) < self.mask_perc.get("species", 0)
            species_list = [
                "unknown" if mask_flag else sp 
                for sp, mask_flag in zip(species_list, species_mask)
            ]

        # ==========================================
        # Cell Type Masking 
        # ==========================================
        cell_mask = torch.zeros(B, dtype=torch.bool)
        if "cell" in self.mask_perc and not is_eval:
            cell_mask = torch.rand(B) < self.mask_perc.get("cell", 0)
            cell_types_masked = [
                "unknown" if mask_flag else ct 
                for ct, mask_flag in zip(cell_types, cell_mask)
            ]

        # ==========================================
        # Logical Matrix for Expression Vector Masking
        # ==========================================
        for i in range(B):
            s_masked = species_mask[i].item()
            c_masked = cell_mask[i].item()
            
            ct = cell_types[i]
            
            if c_masked:
                expr_batch[i] = 0.0
                
            elif s_masked and not c_masked:
                if ct in self.cell_mean_expr:
                    expr_batch[i] = self.cell_mean_expr[ct].to(expr_batch.dtype)
                else:
                    expr_batch[i] = 0.0

        cds_masks = torch.zeros((B, L), dtype=torch.bool)
        for i in range(B):
            s = cds_starts[i]
            e = cds_stops[i]
            if s != -1 and e != -1 and e > s:
                e_clip = min(e, L)
                cds_masks[i, s - 1: e_clip] = True
            else:
                cds_masks[i, :] = True

        return (
            species_list,
            cell_types,
            cell_mask,
            expr_batch,
            seq_embs_padded,
            count_embs_padded,
            count_embs_masked,
            count_emb_masks,
            cds_masks,
            pad_masks
        )
    
    def count_task_criterion(
        self,
        result: Dict[str, torch.Tensor],
        count_raw_emb: torch.Tensor, 
        count_emb_masks: torch.Tensor,
        cds_masks: torch.Tensor,
        cds_weight_factor: float = 1.5
    ) -> torch.Tensor:
        """
        Calculates the joint loss combining Token-level Micro Loss and Frame-aware Macro Loss.
        Operating purely in linear scale (No Log/ReLU transformation).
        """
        pred = result["count"].float()  # (B, L, d_count) 
        count_raw_emb = count_raw_emb.float()
        
        # ---------------------------------------------------------
        # 动态生成 CDS 权重矩阵
        # CDS 区域权重为 1.5，UTR 区域为 1.0
        # ---------------------------------------------------------
        token_weights = torch.where(cds_masks, cds_weight_factor, 1.0).to(pred.device)
        
        # 计算加权后的有效长度 (CDS 算作多个 token) 以便后续做均值
        weighted_masks = count_emb_masks.float() * token_weights
        norm_lengths = weighted_masks.sum(dim=1)
        
        # ==========================================
        # 1. Micro Loss (Token-level Shape Constraint)
        # ==========================================
        if pred.shape[2] == 1:
            loss_all = self.dynamcis_criterion(pred.squeeze(-1), count_raw_emb.squeeze(-1)) 
            # 乘上 mask 并叠加上 CDS 专属权重
            loss_all = loss_all * count_emb_masks.float() * token_weights
        else:
            loss_all = self.dynamcis_criterion(pred, count_raw_emb) 
            loss_all = loss_all * count_emb_masks.unsqueeze(-1).float() * token_weights.unsqueeze(-1)
            
        if loss_all.dim() == 3:
            loss_per_seq = loss_all.sum(dim=[1, 2])
        else:
            loss_per_seq = loss_all.sum(dim=1)

        safe_lengths = torch.clamp(norm_lengths.to(loss_per_seq.device), min=1.0)
        per_sample_micro_loss = loss_per_seq / safe_lengths
        
        # ==========================================
        # 2. Macro Loss (Frame-aware TE Constraint via Linear-MSE)
        # ==========================================
        B, L = cds_masks.shape
        device = pred.device
        
        # Dynamically parse the start of the CDS to construct accurate Frame Masks.
        _, start_indices = cds_masks.max(dim=1)
        
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        relative_positions = positions - start_indices.unsqueeze(1)
        
        # Isolate the 3 reading frames within the CDS boundaries
        f0_mask = cds_masks & (relative_positions % 3 == 0)
        f1_mask = cds_masks & (relative_positions % 3 == 1)
        f2_mask = cds_masks & (relative_positions % 3 == 2)
        
        frame_masks = [f0_mask, f1_mask, f2_mask]
        
        # Unify shape handling for single or multi-channel RPF density (linear)
        if pred.shape[2] == 1:
            p_val = pred.squeeze(-1)
            t_val = count_raw_emb.squeeze(-1)
        else:
            p_val = pred.sum(dim=-1)
            t_val = count_raw_emb.sum(dim=-1)
            
        frame_mse_losses = []
        
        # Iterate through the 3 reading frames to compute independent Linear-MSE losses
        for f_mask in frame_masks:
            # Mask intersection
            target_eval_mask = count_emb_masks.to(device) & f_mask
            
            p_sum = (p_val * target_eval_mask.float()).sum(dim=1)
            t_sum = (t_val * target_eval_mask.float()).sum(dim=1)
            
            t_lengths = target_eval_mask.sum(dim=1).float()
            safe_t_lengths = torch.clamp(t_lengths, min=1.0)
            
            p_mean = p_sum / safe_t_lengths
            t_mean = t_sum / safe_t_lengths
            
            # Compute MSE loss
            f_loss = self.te_criterion(p_mean, t_mean)
            frame_mse_losses.append(f_loss)

        # Average the loss across all 3 frames
        per_sample_macro_loss = (frame_mse_losses[0] + frame_mse_losses[1] + frame_mse_losses[2]) / 3.0

        # ==========================================
        # 3. Fusion
        # ==========================================
        alpha = 2
        # Combine Micro local-shape constraint and Macro global-scale constraint
        total_sample_loss = per_sample_micro_loss + alpha * per_sample_macro_loss
        
        # Return the batch mean
        loss = total_sample_loss.mean()

        return loss

    
    # -------------------------
    # epoch-level train / eval
    # -------------------------
    def train_epoch(self, epoch: int) -> Tuple[torch.Tensor, List]:
        self.model.train()
        dataloader = DataLoader(
            self.dataset, 
            batch_sampler=self.sampler, 
            num_workers=5,
            prefetch_factor=5, 
            persistent_workers=True, 
            pin_memory=True,
            collate_fn=lambda s: self.collate_mask_pad_batch_to_cuda(s, is_eval=False)
        )
        total_loss = torch.zeros(1).to(self.device)
        local_loss = []
        batch_num = len(dataloader)
        start_time = time.time()
        self.sampler.set_epoch(epoch) 

        if self.rank == 0:
            pbar = tqdm(total = batch_num, desc=f"Epoch {epoch+1} train")

        for batch_idx, batch_data in enumerate(dataloader):
            # Unpack the batch
            species_list, _, _, expr_batch, \
            seq_embs_padded, \
            count_embs_padded, count_embs_masked, count_emb_masks, cds_masks, pad_masks = batch_data

            # Move tensors to CUDA
            expr_batch = expr_batch.cuda(non_blocking=True)
            seq_embs_padded = seq_embs_padded.cuda(non_blocking=True)
            count_embs_masked = count_embs_masked.cuda(non_blocking=True)
            count_embs_padded = count_embs_padded.cuda(non_blocking=True)
            count_emb_masks = count_emb_masks.cuda(non_blocking=True)
            cds_masks = cds_masks.cuda(non_blocking=True)
            pad_masks = pad_masks.cuda(non_blocking=True)

            # ==========================================
            # Inject Gaussian Noise into Expression Vectors
            # Done on GPU for maximum speed, only during training
            # ==========================================
            if self.expr_noise_std > 0:
                # Generate noise in float32 for numerical stability, then cast to match expr_batch (fp16)
                noise = torch.randn_like(expr_batch, dtype=torch.float32) * self.expr_noise_std
                expr_batch = expr_batch + noise.to(expr_batch.dtype)

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                # Pass the dynamically generated expr_vector to the model
                outputs = self.model(
                    seq_batch=seq_embs_padded, 
                    count_batch=count_embs_masked, 
                    species=species_list,
                    expr_vector=expr_batch,      # Feed continuous tensor directly
                    src_mask=pad_masks, 
                    head_names=["count"]
                )
                
                loss = self.count_task_criterion(outputs, count_embs_padded, count_emb_masks, cds_masks)
                acc_loss = loss / self.ac_steps

            do_sync = ((batch_idx + 1) % self.ac_steps == 0)

            context = (self.model.no_sync() if not do_sync else contextlib.nullcontext())
            with context:
                self.scaler.scale(acc_loss).backward()

            if do_sync:
                # gradient clipping
                self.scaler.unscale_(self.optimizer) # 解开 scale 以计算真实梯度 norm
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0) # 裁剪梯度
                self.scaler.step(self.optimizer) # 步进优化器并更新scalar
                self.scaler.update() 
                self.scheduler.step()
                self.optimizer.zero_grad() 

            total_loss += loss.detach()
            local_loss.append([float(loss)])

            if self.rank == 0 and (batch_idx + 1) % self._print_progress_every == 0:
                pbar.update(self._print_progress_every)
                print(f"\tloss: {loss}")

        if self.rank == 0:
            pbar.close()

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM) 
        gathered_losses = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_losses, local_loss)
        mean_epoch_losses = total_loss/float(batch_num * self.world_size)
        
        end_time = time.time()
        print(f'Epoch {epoch+1} training time: {end_time - start_time} seconds, mean loss: {mean_epoch_losses}')

        return mean_epoch_losses, gathered_losses
    

    def eval_epoch(self, epoch: int) -> Tuple[torch.Tensor, List]:
        self.model.eval() 
        val_loader = DataLoader(
            self.val_dataset, 
            batch_sampler=self.val_sampler, 
            num_workers=5,
            prefetch_factor=5, 
            persistent_workers=True, 
            pin_memory=True,
            collate_fn=lambda s: self.collate_mask_pad_batch_to_cuda(s, is_eval=True)
        )
        total_loss = torch.zeros(1).to(self.device)
        local_loss = []
        batch_num = len(val_loader)
        start_time = time.time()
        self.val_sampler.set_epoch(epoch) 
    
        if self.rank == 0:
            pbar = tqdm(total = batch_num, desc=f"Epoch {epoch+1} evaluate")

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                
                species_list, _, _, expr_batch, \
                seq_embs_padded, \
                count_embs_padded, count_embs_masked, count_emb_masks, cds_masks, pad_masks = batch_data

                # Move tensors to CUDA
                expr_batch = expr_batch.cuda(non_blocking=True)
                seq_embs_padded = seq_embs_padded.cuda(non_blocking=True)
                count_embs_masked = count_embs_masked.cuda(non_blocking=True)
                count_embs_padded = count_embs_padded.cuda(non_blocking=True)
                count_emb_masks = count_emb_masks.cuda(non_blocking=True)
                cds_masks = cds_masks.cuda(non_blocking=True)
                pad_masks = pad_masks.cuda(non_blocking=True)

                # NOTE: We DO NOT add noise during eval_epoch. Model sees pure expression profile.
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = unwrap_model(self.model).predict(
                        seq_batch=seq_embs_padded, 
                        count_batch=count_embs_masked, 
                        species=species_list,
                        expr_vector=expr_batch,
                        src_mask=pad_masks, 
                        head_names=["count"],
                        move_inputs_to_device=False # Already moved manually
                    )
                    loss = self.count_task_criterion(outputs, count_embs_padded, count_emb_masks, cds_masks)

                total_loss += loss.detach()
                local_loss.append([float(loss)])

                if self.rank == 0 and (batch_idx + 1) % self._print_progress_every == 0:
                    pbar.update(self._print_progress_every) 
                    print(f"\tloss: {loss}")

        if self.rank == 0:
            pbar.close()

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        gathered_losses = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_losses, local_loss)
        mean_epoch_losses = total_loss/float(batch_num * self.world_size)

        end_time = time.time()
        print(f'Epoch {epoch+1} evaluating time: {end_time - start_time} seconds, mean loss: {mean_epoch_losses}')

        return mean_epoch_losses, gathered_losses
    

    # -------------------------
    # orchestrate pretraining
    # -------------------------
    def pretrain(self):
        """
        Main training loop orchestrator. Iterates epochs, runs train/eval,
        saves checkpoints and logs losses.
        """
        start_epoch = int(self.start_epoch)
        for epoch in range(start_epoch, self.epoch_num):
            if self.rank == 0:
                print(f"[Trainer] === Starting epoch {epoch+1}/{self.epoch_num} ===")

            # update mask ratio
            if isinstance(self.mask_perc["count"], (tuple, list)):
                if len(self.mask_perc["count"]) == 2:
                    start = self.mask_perc["count"][0]
                    end = self.mask_perc["count"][1]
                total_epoch = math.floor(self.epoch_num * self.lr_warmup_perc)
                current_mask_range = get_dynamic_mask_ratio(epoch, total_epoch, start, end)
                self.current_mask_range = current_mask_range
            if self.rank == 0:
                print(f"[Trainer] === Epoch {epoch+1} Curriculum: Mask Ratio Range set to {self.current_mask_range}")

            train_loss, train_batch_loss = self.train_epoch(epoch)
            val_loss, val_batch_loss = self.eval_epoch(epoch)

            # update best and bookkeeping (only for active heads/mask task)
            val_loss_float = float(val_loss.item())
            is_best = val_loss_float < self.best_val_loss
            
            # --- [NEW] Early Stopping Update Logic ---
            if is_best:
                self.best_val_loss = val_loss_float
                self.patience_counter = 0 
            else:
                self.patience_counter += 1
                if self.rank == 0:
                    print(f"[Trainer] Early Stopping Counter: {self.patience_counter} / {self.patience}")

            self.training_epoch_data.append({"epoch": epoch+1, "train_loss": float(train_loss.item()), "valid_loss": val_loss})
            self.training_batch_data.append({"epoch": epoch+1, 
                                             "train_batch_loss": [x for x in train_batch_loss], 
                                             "valid_loss": [x for x in val_batch_loss]})

            # save on master periodically
            if self.rank == 0 and (epoch + 1) % self._save_every == 0:
                self.save_checkpoint(epoch + 1, is_best)
    
                with open(os.path.join(self.log_dir, self.model_full_name + ".epoch_data.json"), "w") as f:
                    json.dump(self.training_epoch_data, f, cls=TensorEncoder)
                
                with open(os.path.join(self.log_dir, self.model_full_name + ".batch_data.json"), "w") as f:
                    json.dump(self.training_batch_data, f, cls=TensorEncoder)

            # --- [NEW] Early Stopping Trigger ---
            if self.patience_counter >= self.patience:
                if self.rank == 0:
                    print(f"\n[Trainer] 🛑 Early stopping triggered! Validation loss did not improve for {self.patience} consecutive epochs. Stopping at epoch {epoch+1}.")
                    
                    # Optional: Force a final save before exiting
                    self.save_checkpoint(epoch + 1, is_best=False)
                    with open(os.path.join(self.log_dir, self.model_full_name + ".epoch_data.json"), "w") as f:
                        json.dump(self.training_epoch_data, f, cls=TensorEncoder)
                    with open(os.path.join(self.log_dir, self.model_full_name + ".batch_data.json"), "w") as f:
                        json.dump(self.training_batch_data, f, cls=TensorEncoder)
                        
                break