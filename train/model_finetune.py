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
    if "layernorm" in name or "layer_norm" in name or ".ln" in name or '.norm' in name:
        return True
    if ".embedding" in name or "embed" in name:
        return True
    return False

def collate_pad_batch_to_cuda(batch):
        """
        Collate functiond
        Input batch entries expected as:
          (cell_idx, cds_start, motif_occs, seq_emb, count_emb, coding_emb)
        Returns many tensors; calling code is responsible to move them to device.
        """
        cell_idxs, _, _, seq_embs, count_embs, coding_embs = zip(*batch)

        # convert cell indices
        cell_idxs = torch.tensor(cell_idxs, dtype=torch.long)

        # pad sequences
        seq_embs_padded = pad_sequence(seq_embs, batch_first=True, padding_value=-1)
        count_embs_padded = pad_sequence(count_embs, batch_first=True, padding_value=-1)
        coding_embs_padded = pad_sequence(coding_embs, batch_first=True, padding_value=-1)

        # pad mask where True means valid token
        pad_masks = (seq_embs_padded != -1)[:, :, 0].squeeze(-1)

        # print(seq_embs_padded.shape, coding_embs_padded.shape)

        return (
            cell_idxs, 
            seq_embs_padded,
            count_embs_padded,
            coding_embs_padded,
            pad_masks
        )

class FineTuningTrainer:
    """
    Fine-tuning trainer class.

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
        print_progress_every: int = 50,
        save_every: int = 5,
        epoch_num: int = 100,
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
        self._print_progress_every = print_progress_every
        self._save_every = save_every

        # train hyperparams
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.ac_steps = accumulation_steps
        self.lr = learning_rate
        self.lr_warmup_perc = lr_warmup_perc
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        # device for this process
        self.device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")

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
        self.bceloss_in_orf = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(20))
        self.alpha = 0.5
        self.gamma = 2.0

        # build optimizer & scheduler & scaler
        self.optimizer = self._build_optimizer(lr=self.lr, betas=self.beta, eps=self.epsilon, weight_decay=self.weight_decay)
        warmup_steps = int(self.lr_warmup_perc * self._total_steps)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=create_lr_lambda(total_steps=self._total_steps, warmup_steps=warmup_steps, min_eta=1e-4)
        )
        self.scaler = torch.amp.GradScaler() # for gradiant scaling in autocasting

        # logging & checkpointing
        self.checkpoint_dir = os.path.join(checkpoint_dir, "finetune")
        self.log_dir = os.path.join(log_dir, "finetune")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.model_full_name = f"{self.model_name}.{self.weight_decay}_{self.lr}_{self.lr_warmup_perc}"
        self.training_epoch_data: List[Dict[str, Any]] = []
        self.training_batch_data: List[Dict[str, Any]] = []

        # bookkeeping for resume
        self.start_epoch = 0
        self.best_val_loss = torch.tensor([np.inf])  # seq, count, cell default
        if self.resume:
            self._maybe_load_lora_checkpoint()
        else:
            if self.rank == 0:
                print(f"[Trainer] Train from scratch")

        if self.rank == 0:
            print(f"[Trainer] device={self.device}, steps_per_epoch={self.steps_per_epoch}, total_steps={self._total_steps}")
    
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
            if default_no_weight_decay(name) or ("lora" in name.lower()):
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
        latest = os.path.join(self.checkpoint_dir, self.model_full_name + ".lora.latest.pt")
        best = os.path.join(self.checkpoint_dir, self.model_full_name + ".lora.best.pt")
        return latest, best
    

    def save_lora_and_heads(self, epoch: str, is_best: bool):
        """
        Save only LoRA params and heads. This produces a small checkpoint used for distribution / deployment.
        We select parameters whose name contains 'lora' or belong to the ModuleDict 'heads'.
        """
        latest, best = self._checkpoint_paths()
        model_unwrapped = unwrap_model(self.model)
        model_lora = {}
        for name, param in model_unwrapped.state_dict().items():
            n = name.lower()
            # include LoRA params
            if "lora" in n:
                model_lora[name] = param.cpu()
            # include heads by prefix (common pattern: 'heads.' moduledict)
            elif name.startswith("heads.") or ".heads." in name:
                model_lora[name] = param.cpu()
            # include any parameter named 'coding' or 'coding_head'
            elif "coding" in n or "count_head" in n or "cell_head" in n:
                model_lora[name] = param.cpu()
        state = {
            "epoch": epoch,
            "model": model_lora,
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
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
            print(f"[Trainer] Saved LoRA+heads checkpoint ({len(model_lora)} tensors) to {latest}")

    def load_lora_and_heads(self, path: str, strict: bool = False, map_location: Optional[str] = None):
        """
        Load a LoRA-only checkpoint (or any partial checkpoint) into the model.
        This will update matching keys and leave other params untouched.
        """
        if path is None:
            return
        map_loc = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=map_loc)
        sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        model_unwrapped = unwrap_model(self.model)
        res = model_unwrapped.load_state_dict(sd, strict=strict)
        if self.rank == 0:
            print(f"[Trainer] Loaded LoRA+heads checkpoint: missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}")
        return res


    def _maybe_load_lora_checkpoint(self, strict: bool = False):
        """
        Try to load the latest checkpoint if it exists.
        """
        latest, _ = self._checkpoint_paths()
        if os.path.isfile(latest):
            if self.rank == 0:
                print(f"[Trainer] Loaded LoRA+heads checkpoint: {latest}")
            map_loc = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt = torch.load(latest, map_location=map_loc)
            # restore model weights into unwrapped model
            unwrap_model(self.model).load_lora_and_heads(latest, map_loc, strict)

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
        else:
            if self.rank == 0:
                print(f"[Trainer] Train from scratch")
    

    # -------------------------
    # Loss utilities
    # -------------------------
    @staticmethod
    def focal_loss_multi_channel_from_logits(
            logits: torch.Tensor,   # (bs, L, C)  C=2 for start+stop
            targets: torch.Tensor,  # (bs, L, C)
            mask: torch.Tensor,     # (bs, L) bool
            alpha: Optional[float] = 0.25,
            gamma: float = 2.0,
    ) -> torch.Tensor:
        """
        Vectorized focal loss over channels (bs, L, C).
        NOTE: Do NOT pass pos_weight into binary_cross_entropy_with_logits here,
              because we rely on p_t = exp(-bce) trick. Use alpha (per-class scalar) instead.
        Returns scalar (mean over masked positions and channels, normalized).
        """
        bs, L, C = logits.shape
        # cast once
        mask_f = mask.float()  # (bs, L)
        # compute bce per-element for all channels in one call
        # flatten last dim to let function treat it elementwise
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')   # (bs, L, C)
        # p_t = exp(-bce) trick (valid when no pos_weight used)
        p_t = torch.exp(-bce)  # (bs, L, C)

        modulating_factor = (1.0 - p_t) ** gamma  # (bs, L, C)

        alpha_factor = targets * alpha + (1.0 - targets) * (1.0 - alpha)  # (bs, L, C)
        loss = alpha_factor * modulating_factor * bce  # (bs, L, C)

        # apply mask -> expand mask to channels
        loss = loss * mask_f.unsqueeze(-1)  # (bs, L, C)
        denom = mask_f.sum().clamp_min(1.0) * C  # normalize by tokens * channels
        # return start and stop loss seperately
        return loss[:, :, 0].sum() / denom, loss[:, :, 1].sum() / denom
        
        
    def binary_loss_with_mask(self,
                              logits: torch.Tensor,
                              targets: torch.Tensor,
                              mask: torch.Tensor) -> torch.Tensor:
        """
        Compute BCEWithLogitsLoss per-element, apply mask (bool, same shape) and normalize by mask.sum()
        logits, targets, mask: (bs, L)
        """
        per_elem = self.bceloss_in_orf(logits, targets)  # (bs, L)
        # apply mask: mask is bool where True = valid token
        per_elem = per_elem * mask.float()
        total = per_elem.sum()
        denom = mask.sum().float().clamp_min(1.0)
        return total / denom  # normalized mean over valid tokens
    

    def compute_coding_loss(self,
                            outputs: torch.Tensor,
                            targets: torch.Tensor,
                            pad_mask: torch.Tensor,
                            w_start: float = 1.0,
                            w_stop: float = 1.0,
                            w_inorf: float = 1.0) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        """
        logits_dict: {'start','stop','in_orf'} logits (bs,L)
        targets_dict: same keys but targets in {0,1} float tensors (bs,L)
        pad_mask: bool tensor (bs,L) True = valid token
        pos_weight: should be equal to (positive examples) / (negative examples). > 1 increase recall, < 1 increase precision
        returns: total_loss, dict of component losses
        """
        output_logits = outputs["coding"]

        # combined focal for start+stop (no pos_weight here; use alpha instead)
        start_loss, stop_loss = self.focal_loss_multi_channel_from_logits(
            logits = output_logits[:,:,0:2],
            targets = targets[:,:,0:2],
            mask = pad_mask,
            alpha = self.alpha,
            gamma = self.gamma
        )
        # start_loss = self.focal_loss_from_logits(output_logits[:,:,0], targets[:,:,0], pad_mask, alpha=0.25, gamma=2)
        # stop_loss  = self.focal_loss_from_logits(output_logits[:,:,1], targets[:,:,1], pad_mask, alpha=0.25, gamma=2)
        inorf_loss = self.binary_loss_with_mask(output_logits[:,:,2], targets[:,:,2], pad_mask)

        total = w_start * start_loss + w_stop * stop_loss + w_inorf * inorf_loss
        return total, [w_start * start_loss, w_stop * stop_loss, w_inorf * inorf_loss]

    
    # -------------------------
    # epoch-level train / eval
    # -------------------------
    def train_epoch(self, epoch: int) -> Tuple[torch.Tensor, List]:
        """
        Run one training epoch on the train dataset for this rank/process.
        Returns mean losses (tensor) and gathered per-batch losses from all ranks.
        """
        self.model.train()
        dataloader = DataLoader(
            self.dataset,
            batch_sampler=self.sampler,
            num_workers=3,
            prefetch_factor=3,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=lambda s: collate_pad_batch_to_cuda(s)
        )
        total_loss = torch.zeros(1).to(self.device)
        local_losses = []
        batch_num = len(dataloader)
        start_time = time.time()
        self.sampler.set_epoch(epoch) # DDP

        if self.rank == 0:
            pbar = tqdm(total = batch_num, desc=f"Epoch {epoch+1} train")

        for batch_idx, batch_data in enumerate(dataloader):
            # move tensors to device
            cell_idxs, seq_embs_padded, count_embs_padded, \
                coding_embs_padded, pad_masks = [data.cuda(non_blocking=True) for data in batch_data]

            # run the forward pass with autocasting
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.model(seq_embs_padded, count_embs_padded, cell_idxs, pad_masks, ["coding"])
                # three labels learning losses
                loss, losses = self.compute_coding_loss(outputs, coding_embs_padded, pad_masks)
                ## mean loss (dynamic weight average) for accumulation
                acc_losses = loss / self.ac_steps

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
                self.scaler.scale(acc_losses).backward()

            # synchronic gradient accumulation
            if do_sync:
                self.scaler.step(self.optimizer) # update parameters
                self.scaler.update() # update the scale for next iteration
                self.scheduler.step() # update learning rate
                self.optimizer.zero_grad() # reset gradient

            total_loss += loss.detach()
            local_losses.append([float(l.item()) for l in losses])

            if self.rank == 0 and (batch_idx + 1) % self._print_progress_every == 0:
                pbar.update(self._print_progress_every)
                print(f"loss: {losses}")

        if self.rank == 0:
            pbar.close()

        # sync and gather
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM) # total of seq_loss and count_loss
        gathered_losses = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_losses, local_losses)
        mean_epoch_losses = total_loss/float(batch_num * self.world_size)

        # time
        end_time = time.time()
        print(f'Epoch {epoch+1} training time: {end_time - start_time} seconds, mean loss: {mean_epoch_losses}')

        return mean_epoch_losses, gathered_losses
    

    def eval_epoch(self, epoch: int) -> Tuple[torch.Tensor, List]:
        """
        Run one evaluation epoch (no gradient).
        """
        self.model.eval() # set eval mode
        val_loader = DataLoader(
            self.val_dataset,
            batch_sampler=self.val_sampler,
            num_workers=3,
            prefetch_factor=3,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=lambda s: collate_pad_batch_to_cuda(s)
        )
        total_loss = torch.zeros(1).to(self.device)
        local_losses = []
        batch_num = len(val_loader)
        start_time = time.time()
        self.val_sampler.set_epoch(epoch) # DDP
    
        if self.rank == 0:
                pbar = tqdm(total = batch_num, desc=f"Epoch {epoch+1} evaluate")

        for batch_idx, batch_data in enumerate(val_loader):
            # move tensors to device
            cell_idxs, seq_embs_padded, count_embs_padded, \
                coding_embs_padded, pad_masks = [data.cuda(non_blocking=True) for data in batch_data]

            # run the forward pass with autocasting
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = unwrap_model(self.model).predict(seq_embs_padded, count_embs_padded, cell_idxs, pad_masks, ["coding"])
                # three labels learning losses
                loss, losses = self.compute_coding_loss(outputs, coding_embs_padded, pad_masks)

            total_loss += loss.detach()
            local_losses.append([float(l.item()) for l in losses])

            if self.rank == 0 and (batch_idx + 1) % self._print_progress_every == 0:
                pbar.update(self._print_progress_every)
                print(f"loss: {losses}") 

        if self.rank == 0:
            pbar.close()

        # all-reduce and gather
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        gathered_losses = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_losses, local_losses)
        mean_epoch_losses = total_loss/float(batch_num * self.world_size)

        # time
        end_time = time.time()
        print(f'Epoch {epoch+1} evaluating time: {end_time - start_time} seconds, mean loss: {mean_epoch_losses}')

        return mean_epoch_losses, gathered_losses
    

    # -------------------------
    # orchestrate finetuning
    # -------------------------
    def finetune(self):
        """
        Main training loop orchestrator. Iterates epochs, runs train/eval,
        saves checkpoints and logs losses.
        """
        start_epoch = int(self.start_epoch)
        for epoch in range(start_epoch, self.epoch_num):
            if self.rank == 0:
                print(f"[Trainer] === Starting epoch {epoch+1}/{self.epoch_num} ===")

            train_loss, train_batch_loss = self.train_epoch(epoch)
            val_loss, val_batch_loss = self.eval_epoch(epoch)

            # update best and bookkeeping (only for active heads/mask task)
            is_best = val_loss < self.best_val_loss.to(self.device)
            if is_best:
                self.best_val_loss = val_loss.detach().cpu()

            self.training_epoch_data.append({"epoch": epoch+1, "train_loss": float(train_loss), "valid_loss": float(val_loss)})
            self.training_batch_data.append({"epoch": epoch+1, 
                                             "train_batch_loss": [x for proc in train_batch_loss for x in proc], 
                                             "valid_loss": [x for proc in val_batch_loss for x in proc]})

            # save on master periodically
            if self.rank == 0 and (epoch + 1) % self._save_every == 0:
                # save small LoRA+heads checkpoint too
                self.save_lora_and_heads(epoch, is_best)
                # write logs
                with open(os.path.join(self.log_dir, self.model_full_name + ".epoch_data.json"), "w") as f:
                    json.dump(self.training_epoch_data, f)
                with open(os.path.join(self.log_dir, self.model_full_name + ".batch_data.json"), "w") as f:
                    json.dump(self.training_batch_data, f)

         # final save
        if self.rank == 0:
            print("[Trainer] Finetuning finished. Final saving...")
            self.save_lora_and_heads(epoch, is_best)

