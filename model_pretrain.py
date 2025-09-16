import os
import time
import json
import math
import contextlib
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from utils import unwrap_model
from distributed_bucket_sampler import DistributedBucketSampler
from data.masking_adapter_for_batch import BatchMaskingAdapter


def create_lr_lambda(total_steps: int, warmup_steps: int = 0, min_eta: float = 1e-4):
    """
    Linear warmup then cosine decay lambda for LambdaLR.
    """
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
    Pretraining trainer class.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train. May already be wrapped in DDP externally.
    model_name : str
        A short name for the model (used for checkpoint/log file names).
    dataset, val_dataset : Dataset
        Training and validation datasets. Each dataset is expected to have `.lengths` used by
        DistributedBucketSampler in this code.
    batch_size : int
        Per-process batch size for DataLoader (sampler is distributed-aware).
    checkpoint_dir : str
        Directory where checkpoints are saved.
    log_dir : str
        Directory for training logs (json).
    world_size : int
        Number of distributed processes (gpus).
    rank : int
        Current process rank (0 for master).
    resume : bool
        Whether to try to resume from an existing checkpoint.
    epoch_num : int
        Number of epochs to train.
    accumulation_steps : int
        Gradient accumulation steps.
    mask_value, mask_perc, mask_learn_task_warmup_perc:
        Masking-related hyperparameters used by the user's masking adapter.
    learning_rate, lr_warmup_perc, beta, epsilon, weight_decay:
        Optimizer and scheduler settings.
    print_progress_every, save_every:
        Logging/checkpoint frequency.
    """
    def __init__(
        self,
        model: nn.Module,
        dataset,
        val_dataset,
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
        mask_learn_task_warmup_perc: dict = {"seq": 0, "count": 0.3, "cell": 0.3},
        mask_perc: dict = {"seq": 0.1, "count": 0.3, "cell": 0.15},
        learning_rate: float = 1e-5,
        lr_warmup_perc: float = 0.2,
        accumulation_steps: int = 5,
        beta: Tuple[float, float] = (0.9, 0.98),
        epsilon: float = 1e-9,
        weight_decay: float = 0.01,
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
        self.mask_learn_task_warmup_perc = mask_learn_task_warmup_perc

        # samplers that produce per-rank batches
        self.sampler = DistributedBucketSampler(
            lengths=self.dataset.lengths,
            batch_size=self.batch_size,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=True
        )
        self.val_sampler = DistributedBucketSampler(
            lengths=self.val_dataset.lengths,
            batch_size=self.batch_size,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            drop_last=False
        )

        # counts and scheduling
        self.steps_per_epoch = len(self.sampler)  # per-process mini-batches per epoch
        self._total_steps = max(1, int(self.epoch_num * self.steps_per_epoch // max(1, self.ac_steps)))
        
        # loss criterions
        self.classify_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.count_criterion = nn.SmoothL1Loss(reduction="none", beta=0.5) # nn.MSELoss(reduction="none")

        # build optimizer & scheduler & scaler
        self.optimizer = self._build_optimizer(lr=self.lr, betas=self.beta, eps=self.epsilon, weight_decay=self.weight_decay)
        warmup_steps = int(self.lr_warmup_perc * self._total_steps)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=create_lr_lambda(total_steps=self._total_steps, warmup_steps=warmup_steps, min_eta=1e-4)
        )
        self.scaler = torch.amp.GradScaler() # for gradiant scaling in autocasting

        # logging & checkpointing
        self.checkpoint_dir = os.path.join(checkpoint_dir, "pretrain")
        self.log_dir = os.path.join(log_dir, "pretrain")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.model_full_name = f"{self.model_name}.{self.weight_decay}_{self.lr}_{self.lr_warmup_perc}"
        self.training_epoch_data: List[Dict[str, Any]] = []
        self.training_batch_data: List[Dict[str, Any]] = []

        # bookkeeping for resume
        self.start_epoch = 0
        self.best_val_loss = torch.tensor([np.inf] * len(self.mask_perc))  # seq, count, cell default
        if self.resume:
            self._maybe_load_checkpoint()

        if self.rank == 0:
            print(f"[Trainer] device={self.device}, steps_per_epoch={self.steps_per_epoch}, total_steps={self._total_steps}")
            print(f"[Trainer] mask_perc={self.mask_perc}, mask_warmup={self.mask_learn_task_warmup_perc}")

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
            "best_val_loss": self.best_val_loss.tolist(),
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
            self.best_val_loss = torch.tensor(ckpt.get("best_val_loss", self.best_val_loss.tolist()))
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
    def get_active_heads(self, epoch: int) -> List[str]:
        """
        Determine which heads/tasks are active at the given epoch according to the warmup schedule.
        """
        heads: List[str] = []
        if "seq" in self.mask_perc and epoch >= self.mask_learn_task_warmup_perc.get("seq", np.inf) * self.epoch_num:
            heads.append("seq")
        if "count" in self.mask_perc and epoch >= self.mask_learn_task_warmup_perc.get("count", np.inf) * self.epoch_num:
            heads.append("count")
        if "cell" in self.mask_perc and epoch >= self.mask_learn_task_warmup_perc.get("cell", np.inf) * self.epoch_num:
            heads.append("cell")
        if self.rank == 0:
            print(f"[Trainer] Activate heads: {heads}")
        return heads
    
    def collate_mask_pad_batch_to_cuda(self, batch, active_heads):
        """
        Collate function matching user's masking behavior.
        Input batch entries expected as:
          (cell_idx, cds_start, motif_occs, seq_emb, count_emb, coding_emb)
        Returns many tensors; calling code is responsible to move them to device.
        """
        cell_idxs, cds_starts, motif_occs, seq_embs, count_embs, _ = zip(*batch)

        # convert cell indices
        cell_idxs = torch.tensor(cell_idxs, dtype=torch.long)

        # pad sequences
        seq_embs_padded = pad_sequence(seq_embs, batch_first=True, padding_value=-1)
        count_embs_padded = pad_sequence(count_embs, batch_first=True, padding_value=-1)

        # pad mask where True means valid token
        pad_masks = (seq_embs_padded != -1)[:, :, 0].squeeze(-1)

        B, L = seq_embs_padded.shape[:2]

        # seq masking
        seq_emb_masks = torch.zeros((B, L), dtype=torch.bool)
        if "seq" in active_heads:
            seq_embs_masked, seq_emb_masks = self.masking_adapter.get_random_masked_batch(
                embeddings=seq_embs_padded,
                cds_starts=cds_starts,
                occs=motif_occs,
                pad_mask=pad_masks,
                mask_perc=self.mask_perc.get("seq", 0)
            )
        else:
            seq_embs_masked = seq_embs_padded.clone()

        # count masking
        count_emb_masks = torch.zeros((B, L), dtype=torch.bool)
        if "count" in active_heads:
            count_embs_masked, count_emb_masks = self.masking_adapter.get_random_masked_batch(
                embeddings=count_embs_padded,
                cds_starts=cds_starts,
                occs=motif_occs,
                pad_mask=pad_masks,
                mask_perc=self.mask_perc.get("count", 0)
            )
        else:
            count_embs_masked = count_embs_padded.clone()

        # cell masking (mask some samples' cell id)
        cell_mask = torch.zeros(B, dtype=torch.bool)
        if "cell" in active_heads:
            probs = torch.rand(B)
            cell_mask = probs < self.mask_perc.get("cell", 0)
            mask_idx = unwrap_model(self.model).num_cells if hasattr(unwrap_model(self.model), "num_cells") else -1
            cell_idxs_masked = cell_idxs.clone()
            cell_idxs_masked[cell_mask] = mask_idx
        else:
            cell_idxs_masked = cell_idxs.clone()

        return (
            cell_idxs,               # original cell idx (ground truth) for loss calculation
            cell_idxs_masked,        # use masked cell idx for prompt construction
            cell_mask,               # bool: which samples were masked (for cell loss)
            seq_embs_padded,
            seq_embs_masked,
            seq_emb_masks,
            count_embs_padded,
            count_embs_masked,
            count_emb_masks,
            pad_masks
        )
    

    # -------------------------
    # multi-task loss & DWA
    # -------------------------
    def multi_task_criterion(self, active_heads, result: Dict[str, torch.Tensor],
                             seq_raw_emb: torch.Tensor, seq_emb_masks: torch.Tensor,
                             count_raw_emb: torch.Tensor, count_emb_masks: torch.Tensor,
                             cell_idx: torch.Tensor, cell_mask: torch.Tensor,
                             eps: float = 1e-6) -> torch.Tensor:
        """
        Compute per-task losses and return a tensor [seq_loss, count_loss, cell_loss].
        result is expected to be a dict mapping 'seq','count','cell' to model outputs.
        """
        device = self.device
        losses = {head: torch.tensor(0.0, device=device) for head in self.mask_perc}

        if "seq" in active_heads:
            seq_pred = result["seq"]  # shape: (B, L, 4) or similar
            num_masked_seq_token = seq_emb_masks.sum()
            seq_loss_all = -(torch.nn.functional.log_softmax(seq_pred, dim=-1) * seq_raw_emb * seq_emb_masks.unsqueeze(-1).float())
            losses["seq"] = seq_loss_all.sum() / (num_masked_seq_token + eps)

        if "count" in active_heads:
            count_pred = result["count"]  # (B, L, 10 or 1)
            num_masked_count_token = count_emb_masks.sum()
            if count_pred.shape[2] == 1:
                ## for summarized density
                count_loss_all = self.count_criterion(count_pred.squeeze(-1), count_raw_emb.sum(dim=-1)) * count_emb_masks.float()
            else:
                ## for RPF density for all read length
                count_loss_all = self.count_criterion(count_pred, count_raw_emb) * count_emb_masks.unsqueeze(-1).float()
            losses["count"] = count_loss_all.sum() / (num_masked_count_token + eps)

        if "cell" in active_heads:
            cell_logits = result["cell"]  # (B, num_cells)
            masked_idx = cell_mask.nonzero(as_tuple=False).view(-1)
            if masked_idx.numel() > 0:
                sel_logits = cell_logits[masked_idx]
                sel_labels = cell_idx[masked_idx].to(torch.long)
                losses["cell"] = self.classify_criterion(sel_logits, sel_labels)

        return torch.stack([loss for _, loss in losses.items()], dim=0)

    
    def dynamic_weight_average(self, epoch: int, T: float = 2.0, eps: float = 1e-8) -> torch.Tensor:
        """
        Dynamic Weight Averaging (DWA) heuristic: adapt task weights based on recent training losses.
        """
        device = self.device
        base = torch.tensor([1.0 * i + 1.0 for i in range(len(self.mask_perc))][::-1], dtype=torch.float32, device=device)
        if epoch > 1 and len(self.training_epoch_data) >= 2:
            prev = torch.tensor(self.training_epoch_data[-1]["train_loss"], dtype=torch.float32, device=device)
            prev2 = torch.tensor(self.training_epoch_data[-2]["train_loss"], dtype=torch.float32, device=device)
            w_i = prev / (prev2 + eps)
            if len(w_i) == len(base):
                loss_weight = base * torch.nn.functional.softmax(w_i / T, dim=-1)
            else:
                raise RuntimeError("Numer of mask tasks are not equal to number of losses to resumed!")
        else:
            loss_weight = base
        if self.rank == 0:
            print(f"[Trainer] loss_weight: {loss_weight}")
        return loss_weight

    
    # -------------------------
    # epoch-level train / eval
    # -------------------------
    def train_epoch(self, epoch: int, active_heads: list) -> Tuple[torch.Tensor, List]:
        """
        Run one training epoch on the train dataset for this rank/process.
        Returns mean losses (tensor) and gathered per-batch losses from all ranks.
        """
        self.model.train()
        dataloader = DataLoader(
            self.dataset,
            batch_sampler=self.sampler,
            num_workers=5,
            prefetch_factor=5,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=lambda s: self.collate_mask_pad_batch_to_cuda(s, active_heads)
        )
        total_losses = torch.zeros(len(self.mask_perc)).to(self.device)
        local_losses = []
        batch_num = len(dataloader)
        start_time = time.time()
        self.sampler.set_epoch(epoch) # DDP
        loss_weight = self.dynamic_weight_average(epoch, T=2.0) # DWA

        if self.rank == 0:
            pbar = tqdm(total = batch_num, desc=f"Epoch {epoch+1} train")

        for batch_idx, batch_data in enumerate(dataloader):
            # move tensors to device
            cell_idxs, cell_idxs_masked, cell_mask, \
                seq_embs_padded, seq_embs_masked, seq_emb_masks, \
                    count_embs_padded, count_embs_masked, count_emb_masks,\
                        pad_masks = [data.cuda(non_blocking=True) for data in batch_data]

            # run the forward pass with autocasting
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.model(seq_embs_masked, count_embs_masked, cell_idxs_masked, pad_masks, active_heads)
                # multi-task learning losses
                losses = self.multi_task_criterion(active_heads, outputs,
                                                   seq_embs_padded, seq_emb_masks, 
                                                   count_embs_padded, count_emb_masks,
                                                   cell_idxs, cell_mask)
                ## mean loss (dynamic weight average) for accumulation
                acc_losses = losses / self.ac_steps
                # compute a scalar loss for backward that respects loss_weight
                # normalize loss_weight to sum=1 to avoid scaling issues across different numbers of tasks
                scalar_for_backward = (acc_losses * (loss_weight / loss_weight.sum())).sum()

            # to synchronize and update in this batch ?
            do_sync = ((batch_idx + 1) % self.ac_steps == 0)

            # if not in last step，with no_sync to skip this turn of all-reduce
            context = (
                self.model.no_sync() # forbid the all-reduce in DDP
                if not do_sync
                else contextlib.nullcontext()  # null context
            )
            with context:
                # scales loss and back propagation accumulatively
                self.scaler.scale(scalar_for_backward).backward()

            # synchronic gradient accumulation
            if do_sync:
                self.scaler.step(self.optimizer) # update parameters
                self.scaler.update() # update the scale for next iteration
                self.scheduler.step() # update learning rate
                self.optimizer.zero_grad() # reset gradient

            total_losses += losses.detach()
            local_losses.append([float(l.item()) for l in losses])

            if self.rank == 0 and (batch_idx + 1) % self._print_progress_every == 0:
                pbar.update(self._print_progress_every)
                print(f"loss: {losses}")

        if self.rank == 0:
            pbar.close()

        # sync and gather
        dist.all_reduce(total_losses, op=dist.ReduceOp.SUM) # total of seq_loss and count_loss
        gathered_losses = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_losses, local_losses)
        mean_epoch_losses = total_losses/float(batch_num * self.world_size)

        # time
        end_time = time.time()
        print(f'Epoch {epoch+1} training time: {end_time - start_time} seconds, mean loss: {mean_epoch_losses}')

        return mean_epoch_losses, gathered_losses
    

    def eval_epoch(self, epoch: int, active_heads: list) -> Tuple[torch.Tensor, List]:
        """
        Run one evaluation epoch (no gradient).
        """
        self.model.eval() # set eval mode
        val_loader = DataLoader(
            self.val_dataset,
            batch_sampler=self.val_sampler,
            num_workers=5,
            prefetch_factor=5,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=lambda s: self.collate_mask_pad_batch_to_cuda(s, active_heads)
        )
        total_losses = torch.zeros(len(self.mask_perc)).to(self.device)
        local_losses = []
        batch_num = len(val_loader)
        start_time = time.time()
        self.val_sampler.set_epoch(epoch) # DDP
    
        if self.rank == 0:
                pbar = tqdm(total = batch_num, desc=f"Epoch {epoch+1} evaluate")

        for batch_idx, batch_data in enumerate(val_loader):
            cell_idxs, cell_idxs_masked, cell_mask, \
            seq_embs_padded, seq_embs_masked, seq_emb_masks, \
                count_embs_padded, count_embs_masked, count_emb_masks,\
                    pad_masks = [data.cuda(non_blocking=True) for data in batch_data]

                # run the forward pass with autocasting
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = unwrap_model(self.model).predict(seq_embs_masked, count_embs_masked, cell_idxs_masked, pad_masks, active_heads)
                # multi-task learning losses
                losses = self.multi_task_criterion(active_heads, outputs,
                                                seq_embs_padded, seq_emb_masks, 
                                                count_embs_padded, count_emb_masks,
                                                cell_idxs, cell_mask)

            total_losses += losses.detach()
            local_losses.append([float(l.item()) for l in losses])

            if self.rank == 0 and (batch_idx + 1) % self._print_progress_every == 0:
                pbar.update(self._print_progress_every)
                print(f"loss: {losses}") 

        if self.rank == 0:
            pbar.close()

        # all-reduce and gather
        dist.all_reduce(total_losses, op=dist.ReduceOp.SUM)
        gathered_losses = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_losses, local_losses)
        mean_epoch_losses = total_losses/float(batch_num * self.world_size)

        # time
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

            active_heads = self.get_active_heads(epoch)
            train_loss, train_batch_loss = self.train_epoch(epoch, active_heads)
            val_loss, val_batch_loss = self.eval_epoch(epoch, active_heads)

            # update best and bookkeeping (only for active heads/mask task)
            is_best = torch.sum(val_loss[:len(active_heads)]) < torch.sum(self.best_val_loss[:len(active_heads)])
            if is_best:
                self.best_val_loss = val_loss.detach().cpu()

            self.training_epoch_data.append({"epoch": epoch+1, "train_loss": train_loss.tolist(), "valid_loss": val_loss.tolist()})
            self.training_batch_data.append({"epoch": epoch+1, 
                                             "train_batch_loss": [x for proc in train_batch_loss for x in proc], 
                                             "valid_loss": [x for proc in val_batch_loss for x in proc]})

            # save on master periodically
            if self.rank == 0 and (epoch + 1) % self._save_every == 0:
                self.save_checkpoint(epoch + 1, is_best)
                with open(os.path.join(self.log_dir, self.model_full_name + ".epoch_data.json"), "w") as f:
                    json.dump(self.training_epoch_data, f)
                with open(os.path.join(self.log_dir, self.model_full_name + ".batch_data.json"), "w") as f:
                    json.dump(self.training_batch_data, f)