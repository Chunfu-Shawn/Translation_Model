import os
import time
import json
import math
import contextlib
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from utils import unwrap_model
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
        dataset,
        val_dataset,
        dataset_name,
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
        mask_perc: dict = {"count": 0.3, "cell": 0.1},
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
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.rank = rank
        self.world_size = world_size
        self.resume = resume
        self.batch_size = batch_size
        self.epoch_num = epoch_num
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

        # expression logic: extract mean vector from dataset directly
        self.expr_noise_std = expr_noise_std
        self.mean_expr_vector = None
        
        if hasattr(self.dataset, 'cell_expr_dict') and len(self.dataset.cell_expr_dict) > 0:
            all_vecs = list(self.dataset.cell_expr_dict.values())
            self.mean_expr_vector = torch.from_numpy(np.mean(all_vecs, axis=0)).float()
            if self.rank == 0:
                print(f"[Trainer] Calculated mean expression vector from {len(all_vecs)} cell profiles in dataset.")
        else:
            if self.rank == 0:
                print("[Trainer] WARNING: No cell_expr_dict found in dataset. Unknown cells will use batch-level mean.")

        # samplers that produce per-rank batches
        self.balance_classes = balance_classes
        self.sampler = DistributedBucketSampler(
            lengths=self.dataset.lengths,
            batch_size=self.batch_size,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=True,
            balance_classes=self.balance_classes,
            cell_types=getattr(self.dataset, 'cell_types', None)
        )
        self.val_sampler = DistributedBucketSampler(
            lengths=self.val_dataset.lengths,
            batch_size=self.batch_size,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            drop_last=False,
            balance_classes=self.balance_classes,
            cell_types=getattr(self.val_dataset, 'cell_types', None)
        )

        # counts and scheduling
        self.steps_per_epoch = len(self.sampler)  # per-process mini-batches per epoch
        self._total_steps = max(1, int(self.epoch_num * self.steps_per_epoch // max(1, self.ac_steps)))
        
        # loss criterions
        self.count_criterion = nn.SmoothL1Loss(reduction="none", beta=1.0) 

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
    def collate_mask_pad_batch_to_cuda(self, batch, full_mask=0.15):
        """
        Collate function matching user's masking behavior.
        Receives expr_vectors directly from the dataset.
        """
        # Unpack matching TranslationDataset.__getitem__ outputs
        _, cell_types, expr_vectors, meta_info, seq_embs, count_embs = zip(*batch)

        cell_types = list(cell_types)
        expr_batch = torch.stack(expr_vectors) # [B, d_expr]
        # print(cell_types)
        # print(expr_batch.sum(dim=1))
        
        cds_starts = [meta.get("cds_start_pos", -1) for meta in meta_info]
        motif_occs = [meta.get("motif_occs", []) for meta in meta_info]

        seq_embs_padded = pad_sequence(seq_embs, batch_first=True, padding_value=-1)
        count_embs_padded = pad_sequence(count_embs, batch_first=True, padding_value=-1)
        pad_masks = (seq_embs_padded != -1)[:, :, 0].squeeze(-1)
        B, L = seq_embs_padded.shape[:2]

        # seq masking
        seq_emb_masks = torch.zeros((B, L), dtype=torch.bool)
        if "seq" in self.mask_perc:
            seq_embs_masked, seq_emb_masks = self.masking_adapter.get_random_masked_batch(
                embeddings=seq_embs_padded, 
                cds_starts=cds_starts, 
                occs=motif_occs, 
                pad_mask=pad_masks, 
                mask_perc_range=self.mask_perc.get("seq", 0)
            )
        else:
            seq_embs_masked = seq_embs_padded.clone()

        # count masking
        count_emb_masks = torch.zeros((B, L), dtype=torch.bool)
        if "count" in self.mask_perc:
            count_embs_masked, count_emb_masks = self.masking_adapter.get_random_masked_batch(
                embeddings=count_embs_padded, 
                cds_starts=cds_starts, 
                occs=motif_occs, 
                pad_mask=pad_masks, 
                mask_perc_range=self.current_mask_range, 
                full_mask_perc=full_mask
            )
        else:
            count_embs_masked = count_embs_padded.clone()

        # ==========================================
        # Cell Type Masking (Fallback to Mean Vector)
        # ==========================================
        cell_mask = torch.zeros(B, dtype=torch.bool)
        
        if "cell" in self.mask_perc:
            probs = torch.rand(B)
            cell_mask = probs < self.mask_perc.get("cell", 0)
            
            if cell_mask.any():
                if self.mean_expr_vector is not None:
                    # Apply mean vector to masked cells
                    expr_batch[cell_mask] = self.mean_expr_vector.to(expr_batch.dtype)
                else:
                    # Fallback to current batch mean if global mean is missing
                    expr_batch[cell_mask] = expr_batch.mean(dim=0).to(expr_batch.dtype)

        return (
            cell_types,              # original cell string (ground truth for logging)
            expr_batch,              # Provided directly from dataset unpacking
            cell_mask,               # bool: which samples were masked
            seq_embs_padded,
            seq_embs_masked,
            seq_emb_masks,
            count_embs_padded,
            count_embs_masked,
            count_emb_masks,
            pad_masks
        )
    
    def count_task_criterion(
            self,
            result: Dict[str, torch.Tensor],
            count_raw_emb: torch.Tensor, 
            count_emb_masks: torch.Tensor,
            eps: float = 1e-6) -> torch.Tensor:
        """
        Compute per-task losses and return a tensor [seq_loss, count_loss, cell_loss].
        result is expected to be a dict mapping 'seq','count','cell' to model outputs.
        """
        count_pred = result["count"]  # (B, L, 10 or 1)
        
        num_masked_count_token = count_emb_masks.sum()
        if count_pred.shape[2] == 1:
            ## for summarized density
            count_loss_all = self.count_criterion(count_pred.squeeze(-1), count_raw_emb.squeeze(-1)) * count_emb_masks.float()
            
        else:
            ## for RPF density for all read length
            count_loss_all = self.count_criterion(count_pred, count_raw_emb) * count_emb_masks.unsqueeze(-1).float()

        return count_loss_all.sum() / (num_masked_count_token + eps)

    
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
            collate_fn=lambda s: self.collate_mask_pad_batch_to_cuda(s)
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
            cell_types, expr_batch, cell_mask, \
            seq_embs_padded, seq_embs_masked, seq_emb_masks, \
            count_embs_padded, count_embs_masked, count_emb_masks, pad_masks = batch_data

            # Move tensors to CUDA
            expr_batch = expr_batch.cuda(non_blocking=True)
            seq_embs_masked = seq_embs_masked.cuda(non_blocking=True)
            count_embs_masked = count_embs_masked.cuda(non_blocking=True)
            count_embs_padded = count_embs_padded.cuda(non_blocking=True)
            count_emb_masks = count_emb_masks.cuda(non_blocking=True)
            pad_masks = pad_masks.cuda(non_blocking=True)

            # ==========================================
            # Inject Gaussian Noise into Expression Vectors
            # Done on GPU for maximum speed, only during training
            # ==========================================
            if self.expr_noise_std > 0:
                # Generate noise in float32 for numerical stability, then cast to match expr_batch (fp16)
                noise = torch.randn_like(expr_batch, dtype=torch.float32) * self.expr_noise_std
                expr_batch = expr_batch + noise.to(expr_batch.dtype)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # Pass the dynamically generated expr_vector to the model
                outputs = self.model(
                    seq_batch=seq_embs_masked, 
                    count_batch=count_embs_masked, 
                    expr_vector=expr_batch,      # Feed continuous tensor directly
                    src_mask=pad_masks, 
                    head_names=["count"]
                )
                
                loss = self.count_task_criterion(outputs, count_embs_padded, count_emb_masks)
                acc_loss = loss / self.ac_steps

            do_sync = ((batch_idx + 1) % self.ac_steps == 0)

            context = (self.model.no_sync() if not do_sync else contextlib.nullcontext())
            with context:
                self.scaler.scale(acc_loss).backward()

            if do_sync:
                self.scaler.step(self.optimizer)
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
            collate_fn=lambda s: self.collate_mask_pad_batch_to_cuda(s, full_mask=1)
        )
        total_loss = torch.zeros(1).to(self.device)
        local_loss = []
        batch_num = len(val_loader)
        start_time = time.time()
        self.val_sampler.set_epoch(epoch) 
    
        if self.rank == 0:
            pbar = tqdm(total = batch_num, desc=f"Epoch {epoch+1} evaluate")

        for batch_idx, batch_data in enumerate(val_loader):
            
            cell_types, expr_batch, cell_mask, \
            seq_embs_padded, seq_embs_masked, seq_emb_masks, \
            count_embs_padded, count_embs_masked, count_emb_masks, pad_masks = batch_data

            # Move tensors to CUDA
            expr_batch = expr_batch.cuda(non_blocking=True)
            seq_embs_masked = seq_embs_masked.cuda(non_blocking=True)
            count_embs_masked = count_embs_masked.cuda(non_blocking=True)
            count_embs_padded = count_embs_padded.cuda(non_blocking=True)
            count_emb_masks = count_emb_masks.cuda(non_blocking=True)
            pad_masks = pad_masks.cuda(non_blocking=True)

            # NOTE: We DO NOT add noise during eval_epoch. Model sees pure expression profile.
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = unwrap_model(self.model).predict(
                    seq_batch=seq_embs_masked, 
                    count_batch=count_embs_masked, 
                    expr_vector=expr_batch,   
                    src_mask=pad_masks, 
                    head_names=["count"],
                    move_inputs_to_device=False # Already moved manually
                )
                loss = self.count_task_criterion(outputs, count_embs_padded, count_emb_masks)

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
            if is_best:
                self.best_val_loss = val_loss_float

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