import contextlib
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from distributed_bucket_sampler import DistributedBucketSampler
from data.masking_adapter_for_batch import *


def create_lr_lambda(total_steps, warmup_steps=0, min_eta=1e-4):
    def lr_lambda(current_step):
        # linear warmup
        if current_step < warmup_steps:
            return max(min_eta, float(current_step) / float(max(1, warmup_steps)))
        # cosine annealing
        progress = float(current_step - warmup_steps) / float(total_steps - warmup_steps)
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return float(max(min_eta, decay))

    return lr_lambda

class PretrainingTrainer:
    def __init__(self,
                 model,
                 model_name: str,
                 dataset,
                 val_dataset,
                 batch_size,
                 checkpoint_dir: None,
                 log_dir: None,
                 world_size: int,
                 rank,
                 resume = True,
                 mask_value=-1,
                 print_progress_every: int = 50,
                 save_every: int = 5,
                 epoch_num = 100,
                 mask_learn_task_warmup_perc: dict = {"seq": 0, "count": 0.3, "cell": 0.3},
                 mask_perc: dict = {"seq": 0.1, "count": 0.3, "cell": 0.15},
                 learning_rate: float = 0.00001,
                 lr_warmup_perc: float = 0.2,
                 accumulation_steps: int = 5,
                 beta: list = (0.9, 0.98),
                 epsilon: float = 1e-9,
                 ):
        
        self.model = model
        self.model_name = model_name
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.coding_params = list(self.model.module.coding_head.parameters())
        self.coding_param_ids = {id(p) for p in self.coding_params}
        self.main_params = [p for p in self.model.module.parameters() if id(p) not in self.coding_param_ids]
        self.rank = rank
        self.world_size = world_size
        self.resume = resume
        self.batch_size = batch_size
        self.masking_adapter = BatchMaskingAdapter(mask_value)
        self.sampler = DistributedBucketSampler(
            lengths = dataset.lengths,
            batch_size = self.batch_size,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=True
        )
        self.val_sampler = DistributedBucketSampler(
            lengths = val_dataset.lengths,
            batch_size = self.batch_size,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=True
        )
        self.batch_num = len(dataset)
        self.epoch_num = epoch_num
        self.current_epoch = 0
        self.ac_steps = accumulation_steps
        self._total_steps = int(self.epoch_num * self.batch_num / self.ac_steps)
        self.lr_warmup_perc = lr_warmup_perc
        self.mask_learn_task_warmup_perc = mask_learn_task_warmup_perc
        self._print_progress_every = print_progress_every
        self._save_every = save_every
        self.checkpoint_dir = os.path.join(checkpoint_dir, "pretrain")
        self.log_dir = log_dir
        self.num_losses = len(mask_learn_task_warmup_perc.keys())
        self.mask_perc = mask_perc
        self.classify_criterion = nn.CrossEntropyLoss(reduction="mean") # for cell type
        self.count_criterion = nn.SmoothL1Loss(reduction="none", beta=1) # nn.MSELoss(reduction="none") # for count 
        self.lr = learning_rate
        self.optimizer = torch.optim.AdamW(
            self.main_params, # train for main parameters first
            lr=self.lr,
            betas=beta,
            eps = epsilon,
            weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer = self.optimizer,
            lr_lambda = create_lr_lambda(
                total_steps = self._total_steps,
                warmup_steps = int(self.lr_warmup_perc * self._total_steps),
                min_eta = 1e-4
            ))
        self.scaler = torch.amp.GradScaler() # for gradiant scaling in autocasting
        self.model_full_name = '.'.join([model_name, str(self.lr), str(self.lr_warmup_perc), str(self.ac_steps)])
        self.training_epoch_data = []
        self.training_batch_data = []

        # freeze the parameters of coding head
        for p in self.coding_params:
            p.requires_grad = False
        print('==> Frost the coding head, fine-tune it after pretraining')
        print(f'==> Mask learning task: {mask_learn_task_warmup_perc}')
        print(f'==> Mask percentage: {mask_perc}')
    
    def get_active_heads(self, epoch):
        heads = []
        if "seq" in self.mask_perc and epoch >= self.mask_learn_task_warmup_perc.get("seq", np.inf) * self.epoch_num:
            heads.append("seq")
        if "count" in self.mask_perc and epoch >= self.mask_learn_task_warmup_perc.get("count", np.inf) * self.epoch_num:
            heads.append("count")
        if "cell" in self.mask_perc and epoch >= self.mask_learn_task_warmup_perc.get("cell", np.inf) * self.epoch_num:
            heads.append("cell")
        print(f'--> Activate heads: {heads}')
        return heads
    
    def collate_mask_pad_batch_to_cuda(self, batch, active_heads):
        """
        batch_list: list, length is equal to the batch_size of DataLoader.
        batch_list[0] is a dict, including:
        'seq_embeddings', 'count_embeddings', 'masked_embedding', 'pad_mask', 'pred_mask'
        return Tensor values
        """
        # batch elements: (cell_idx, cds_start, motif_occs, seq_emb, count_emb, coding_emb)
        cell_idxs, cds_starts, motif_occs, seq_embs, count_embs, coding_embs = zip(*batch)

        # 1. pad sequence for [:, seq_len, :]
        cell_idxs = torch.tensor(cell_idxs, dtype=torch.long) # int
        seq_embs_padded = pad_sequence(seq_embs, batch_first=True, padding_value=-1)
        count_embs_padded = pad_sequence(count_embs, batch_first=True, padding_value=-1)
        # coding_embs_padded = pad_sequence(coding_embs, batch_first=True, padding_value=-1)

        # 2. pad_mask (True indicate this is a valid value
        pad_masks = (seq_embs_padded != -1)[:, :, 0].squeeze(-1)

        # 3. seq embedding masking for learning (True indicate this is a masked token
        B, L = seq_embs_padded.shape[:2]
        seq_emb_masks = torch.zeros((B, L), dtype=torch.bool)
        if "seq" in active_heads:
            seq_embs_masked, seq_emb_masks = self.masking_adapter.get_random_masked_batch(
                embeddings = seq_embs_padded,
                cds_starts = cds_starts,
                occs = motif_occs,
                pad_mask = pad_masks,
                mask_perc = self.mask_perc.get("seq", 0)
                )
        else:
            seq_embs_masked = seq_embs_padded.clone()

        # 4. count masking
        count_emb_masks = torch.zeros((B, L), dtype=torch.bool)
        if "count" in active_heads:
            count_embs_masked, count_emb_masks = self.masking_adapter.get_random_masked_batch(
                embeddings = count_embs_padded,
                cds_starts = cds_starts,
                occs = motif_occs,
                pad_mask = pad_masks,
                mask_perc = self.mask_perc.get("count", 0)
                )
        else:
            count_embs_masked = count_embs_padded.clone()
        
        # 5. cell type masking
        cell_mask = torch.zeros(B, dtype=torch.bool)  # True indicate the cell idx of this sample was masked
        if "cell" in active_heads:
            # randomly mask cell type of some samples
            probs = torch.rand(B)
            cell_mask = probs < self.mask_perc.get("cell", 0)
            # replace true cell idx as mask_idx
            mask_idx = self.model.module.num_tissues if hasattr(self.model, "module") else self.model.num_tissues
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
        

    def multi_task_criterion(self, active_heads, result: list,
                   seq_raw_emb: torch.Tensor, seq_emb_masks: torch.Tensor,
                   count_raw_emb: torch.Tensor, count_emb_masks: torch.Tensor,
                   cell_idx: torch.Tensor, cell_mask: torch.Tensor,
                   eps=1e-6):
        """
        result: dictionary mapping result to active_heads, e.g. {"seq","count","cell" -> seq_pred, count_pred, cell_logits}
        cell_idx: original cell labels (bs,)
        cell_mask: bool (bs,) True means this sample's cell was masked and should be predicted
        Returns: tensor([seq_loss, count_loss, cell_loss])  length == self.num_losses
        """
        # get outputs by name mapping: map result list to names using active_heads available in scope.
        # but simpler: the trainer when calling this function will pass the outputs in fixed order:
        # assume order [seq_pred, count_pred, cell_logits] when all active. We'll do safe extraction.

        # find seq and count outputs
        seq_pred = None; count_pred = None; cell_logits = None
        # length-based approach: if result[0] is tensor with same last dim as seq_raw_emb dim -> seq

        # seq loss (same as you had)
        if "seq" in active_heads:
            seq_pred = result["seq"]
            num_masked_seq_token = seq_emb_masks.sum()
            seq_loss_all = -(F.log_softmax(seq_pred, dim=-1) * seq_raw_emb * seq_emb_masks.unsqueeze(-1).float())
            seq_loss = seq_loss_all.sum() / (num_masked_seq_token + eps)
        else:
            seq_loss = torch.tensor(0.0, device=seq_raw_emb.device)

        # count loss
        if "count" in active_heads:
            count_pred = result["count"]
            num_masked_count_token = count_emb_masks.sum()
            count_loss_all = self.count_criterion(count_pred, count_raw_emb) * count_emb_masks.unsqueeze(-1).float()
            count_loss = count_loss_all.sum() / (num_masked_count_token + eps)
        else:
            count_loss = torch.tensor(0.0, device=seq_raw_emb.device)

        # cell loss (only for masked samples)
        if "cell" in active_heads:
            cell_logits = result["cell"]  # (bs, num_cells)
            mask_idx = cell_mask.nonzero(as_tuple=False).view(-1)
            if mask_idx.numel() == 0:
                cell_loss = torch.tensor(0.0, device=seq_raw_emb.device)
            else:
                # select logits and true labels
                sel_logits = cell_logits[mask_idx]  # (n_mask, num_cells)
                sel_labels = cell_idx[mask_idx].to(torch.long)
                cell_loss = self.classify_criterion(sel_logits, sel_labels)
        else:
            cell_loss = torch.tensor(0.0, device=seq_raw_emb.device)

        return torch.stack([seq_loss, count_loss, cell_loss], dim=0)

    
    def dynamic_weight_average(self, epoch, T = 2.0, eps = 1e-8):
        # 'End-To-End Multi-Task Learning With Attention'
        device = torch.device(f'cuda:{self.rank}') if torch.cuda.is_available() else torch.device('cpu')
        if epoch > 1:
            prev = torch.tensor(
                self.training_epoch_data[epoch-1]["train_loss"],
                dtype=torch.float32, device=device
            )
            prev2 = torch.tensor(
                self.training_epoch_data[epoch-2]["train_loss"],
                dtype=torch.float32, device=device
            )
            w_i = prev / (prev2 + eps)
            loss_weight = torch.tensor([5, 2, 1], device=device) * F.softmax(w_i / T, dim=-1) # default weight for seq, count and cell is 5, 1
        else:
            loss_weight = torch.tensor([5, 2, 1], device=device)  # torch.ones(self.num_losses, device=device) # torch.ones_like(losses).cuda()
        
        if self.rank == 0:
            print("loss_weight: ", loss_weight)

        return loss_weight


    def save_checkpoint(self, state, is_best: bool):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # save checkpoint
        latest_path = os.path.join(self.checkpoint_dir, self.model_full_name + ".latest.pt")
        torch.save(state, latest_path)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, self.model_name + ".best.pt")
            torch.save(state, best_path)

    
    def train(self, epoch):
        self.model.train() # set train mode
        active_heads = self.get_active_heads(epoch)
        print(f"--- Active heads of model: {active_heads} ---")
        dataloader = DataLoader(
            self.dataset,
            batch_sampler=self.sampler,
            num_workers=3,
            prefetch_factor=3,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=lambda s: self.collate_mask_pad_batch_to_cuda(s, active_heads)
        )
        total_losses = torch.zeros(self.num_losses).cuda()
        local_losses = []
        batch_num = len(dataloader)
        start_time = time.time()
        self.sampler.set_epoch(epoch) # DDP
        loss_weight = self.dynamic_weight_average(epoch, 2.0) # DWA

        if self.rank == 0:
            pbar = tqdm(total = batch_num, desc=f"Epoch {epoch+1} for training")

        for batch_idx, batch_data in enumerate(dataloader):
            cell_idxs, cell_idxs_masked, cell_mask, \
                seq_embs_padded, seq_embs_masked, seq_emb_masks, \
                    count_embs_padded, count_embs_masked, count_emb_masks,\
                        pad_masks = [data.cuda(non_blocking=True) for data in batch_data]

            # run the forward pass with autocasting
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                output = self.model(seq_embs_masked, count_embs_masked, cell_idxs_masked, pad_masks, active_heads)
                # multi-task learning losses
                losses = self.multi_task_criterion(active_heads, output,
                                                   seq_embs_padded, seq_emb_masks, 
                                                   count_embs_padded, count_emb_masks,
                                                   cell_idxs, cell_mask)
                ## mean loss (dynamic weight average) for accumulation
                acc_losses = losses / self.ac_steps

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
                self.scaler.scale(acc_losses).backward(gradient = loss_weight)

            # synchronic gradient accumulation
            if do_sync:
                self.scaler.step(self.optimizer) # update parameters
                self.scaler.update() # update the scale for next iteration
                self.scheduler.step() # update learning rate
                self.optimizer.zero_grad() # reset gradient

            total_losses += losses.detach()
            local_losses.append([l.item() for l in losses])

            if self.rank == 0 and (batch_idx + 1) % self._print_progress_every == 0:
                pbar.update(self._print_progress_every)
                print(losses)

        if self.rank == 0:
            pbar.close()

        # gather loss
        dist.all_reduce(total_losses, op=dist.ReduceOp.SUM) # total of seq_loss and count_loss
        
        gathered_losses = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_losses, local_losses)
        mean_epoch_losses = total_losses/(batch_num * self.world_size)

        # time
        end_time = time.time()
        print(f'Epoch {epoch+1} training time: {end_time - start_time} seconds, mean loss: {mean_epoch_losses}')

        return mean_epoch_losses, gathered_losses
    

    def eval(self, epoch):
        self.model.eval() # set eval mode
        active_heads = self.get_active_heads(epoch)
        val_loader = DataLoader(
            self.val_dataset,
            batch_sampler=self.val_sampler,
            num_workers=3,
            prefetch_factor=3,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=lambda s: self.collate_mask_pad_batch_to_cuda(s, active_heads)
        )
        total_losses = torch.zeros(self.num_losses).cuda()
        local_losses = []
        batch_num = len(val_loader)
        start_time = time.time()
        self.val_sampler.set_epoch(epoch) # DDP
    
        with torch.no_grad():
            if self.rank == 0:
                pbar = tqdm(total = batch_num, desc=f"Epoch {epoch+1} for evaluating")

            for batch_idx, batch_data in enumerate(val_loader):
                cell_idxs, cell_idxs_masked, cell_mask, \
                seq_embs_padded, seq_embs_masked, seq_emb_masks, \
                    count_embs_padded, count_embs_masked, count_emb_masks,\
                        pad_masks = [data.cuda(non_blocking=True) for data in batch_data]

                 # run the forward pass with autocasting
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    output = self.model(seq_embs_masked, count_embs_masked, cell_idxs_masked, pad_masks, active_heads)
                    # multi-task learning losses
                    losses = self.multi_task_criterion(active_heads, output,
                                                    seq_embs_padded, seq_emb_masks, 
                                                    count_embs_padded, count_emb_masks,
                                                    cell_idxs, cell_mask)

                total_losses += losses.detach()
                local_losses.append([l.item() for l in losses])

                if self.rank == 0 and (batch_idx + 1) % self._print_progress_every == 0:
                    pbar.update(self._print_progress_every)        

        if self.rank == 0:
            pbar.close()

        # all-reduce and gather
        dist.all_reduce(total_losses, op=dist.ReduceOp.SUM)
        gathered_losses = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_losses, local_losses)
        mean_epoch_losses = total_losses/(batch_num * self.world_size)

        # time
        end_time = time.time()
        print(f'Epoch {epoch+1} evaluating time: {end_time - start_time} seconds, mean loss: {mean_epoch_losses}')

        return mean_epoch_losses, gathered_losses
    

    def pretrain(self):
        start_epoch = 0
        best_val_loss = torch.tensor([np.inf for _ in range(self.num_losses)] )

        # resume training
        if self.resume:
            ckpt_path = os.path.join(self.checkpoint_dir, self.model_full_name + ".latest.pt")
            if os.path.isfile(ckpt_path):
                print(f"==> Loading checkpoint '{ckpt_path}' in GPU {self.rank}")
                # resume model
                ckpt = torch.load(ckpt_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
                start_epoch = ckpt['epoch']
                best_val_loss = torch.tensor(ckpt.get('best_val_loss', best_val_loss))
                self.model.module.load_state_dict(ckpt['model'])
                self.optimizer.load_state_dict(ckpt['optimizer'])
                if self.scheduler and ckpt.get('scheduler') is not None:
                    self.scheduler.load_state_dict(ckpt['scheduler'])
                
                # resume loss data
                log_epoch_path = os.path.join(self.log_dir, self.model_full_name + ".epoch_data.json")
                log_batch_path = os.path.join(self.log_dir, self.model_full_name + ".batch_data.json")
                if os.path.isfile(log_epoch_path):
                    with open(log_epoch_path, 'r') as f_epoch:
                        self.training_epoch_data = json.load(f_epoch)
                    with open(log_batch_path, 'r') as f_batch:
                        self.training_batch_data = json.load(f_batch)

                print(f"==> Resuming from epoch {start_epoch}, best_val_loss={best_val_loss} in GPU {self.rank}")
            else:
                print(f"==> No checkpoint found, training from scratch in GPU {self.rank}.")
        else:
            print(f"==> Training from scratch in GPU {self.rank}.")


        # start training
        for epoch in range(start_epoch, self.epoch_num):
            print(f"==> Training start from epoch {epoch + 1} in GPU {self.rank}")
            
            # --- train epoch
            train_epoch_loss, train_batch_loss = self.train(epoch)
            # --- validate epoch
            val_epoch_loss, val_batch_loss = self.eval(epoch)
            
            # save checkpoints and loss
            is_best = torch.sum(val_epoch_loss) < torch.sum(best_val_loss)
            best_val_loss = val_epoch_loss if is_best else best_val_loss

            # details
            epoch_data = {
                "epoch": epoch + 1,
                "train_loss": train_epoch_loss.tolist(),
                "valid_loss": val_epoch_loss.tolist()
            }
            batch_data = {
                "epoch": epoch + 1,
                "train_batch_loss": [x for proc in train_batch_loss for x in proc],
                "valid_loss": [x for proc in val_batch_loss for x in proc]
            }
            self.training_epoch_data.append(epoch_data)
            self.training_batch_data.append(batch_data)

            # save in main process
            if self.rank == 0 and (epoch + 1) % self._save_every == 0:
                # cpt
                print(f"==> Save checkpoints, best validation loss is {best_val_loss}")
                checkpoint = {
                    'epoch': epoch + 1,
                    'model': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'best_val_loss': best_val_loss.tolist(),
                }
                self.save_checkpoint(checkpoint, is_best)
                # loss
                with open(os.path.join(self.log_dir, self.model_full_name + ".epoch_data.json"), 'w') as f_epoch:
                    json.dump(self.training_epoch_data, f_epoch)
                    f_epoch.write('\n')
                with open(os.path.join(self.log_dir, self.model_full_name + ".batch_data.json"), 'w') as f_batch:
                    json.dump(self.training_batch_data, f_batch)
                    f_batch.write('\n')