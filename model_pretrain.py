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
from torch.utils.data import DataLoader, distributed
import torch.multiprocessing as mp
from tqdm import tqdm
from distributed_bucket_sampler import DistributedBucketSampler
# from RPF_counter import *
# from transcript_exon_index import *
# mp.set_start_method('spawn', force=True)


def collate_pad_pretrain_batch_to_cuda(batch):
    """
    batch_list: list, length is equal to the batch_size of DataLoader.
    batch_list[0] is a dict, including:
      'seq_embeddings', 'count_embeddings', 'masked_embedding', 'pad_mask', 'pred_mask'
    return Tensor values
    """
    # 取出唯一的元素
    raw_embs, inps, emb_masks, coding_embs, tissue_idxs = zip(*batch)
    # 1. same input length
    # lens = [tensor.size(0) for tensor in inps]

    # 2. pad sequence for [:, seq_len, :]
    raw_embs_padded = pad_sequence(raw_embs, batch_first=True, padding_value=0.0)
    inps_padded = pad_sequence(inps, batch_first=True, padding_value=-1)
    emb_masks_padded = pad_sequence(emb_masks, batch_first=True, padding_value=False)
    tissue_idxs = torch.tensor(tissue_idxs, dtype=torch.long) # int

    # 3. pad_mask (True indicate this is a padding need to be blocked)
    pad_masks = (inps_padded == -1)[:, :, 0].squeeze(-1)

    return (
        raw_embs_padded, 
        inps_padded, 
        emb_masks_padded, 
        pad_masks,
        tissue_idxs
    )

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
                 print_progress_every: int = 50,
                 save_every: int = 5,
                 epoch_num = 100,
                 learning_rate: float = 0.00001,
                 warmup_perc: float = 0.2,
                 accumulation_steps: int = 5,
                 beta: list = (0.9, 0.98),
                 epsilon: float = 1e-9,
                 ):
        
        self.model = model
        self.model_name = model_name
        self.coding_params = list(self.model.module.coding_head.parameters())
        self.coding_param_ids = {id(p) for p in self.coding_params}
        self.main_params = [p for p in self.model.module.parameters() if id(p) not in self.coding_param_ids]
        self.rank = rank
        self.world_size = world_size
        self.resume = resume
        self.batch_size = batch_size
        self.sampler = DistributedBucketSampler(
            lengths = dataset.lengths,
            batch_size = self.batch_size,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=True
        )
        self.dataloader = DataLoader(
            dataset,
            batch_sampler=self.sampler,
            num_workers=1,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=collate_pad_pretrain_batch_to_cuda
        )
        self.val_sampler = DistributedBucketSampler(
            lengths = val_dataset.lengths,
            batch_size = self.batch_size,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_sampler=self.val_sampler,
            num_workers=1,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=collate_pad_pretrain_batch_to_cuda
        )
        self.batch_num = len(dataset)
        self.epoch_num = epoch_num
        self.current_epoch = 0
        self.ac_steps = accumulation_steps
        self._total_steps = int(self.epoch_num * self.batch_num / self.ac_steps)
        self.warmup_perc = warmup_perc
        self._print_progress_every = print_progress_every
        self._save_every = save_every
        self.checkpoint_dir = os.path.join(checkpoint_dir, "pretrain")
        self.log_dir = log_dir
        self.num_losses = 3
        self.classify_criterion = nn.CrossEntropyLoss(reduction="mean") # for ATCG
        self.count_criterion = nn.SmoothL1Loss(reduction="none", beta=1) # for count
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
                warmup_steps = int(self.warmup_perc * self._total_steps),
                min_eta = 1e-4
            ))
        self.scaler = torch.amp.GradScaler() # for gradiant scaling in autocasting
        self.model_full_name = '.'.join([model_name, str(self.lr), str(self.warmup_perc), str(self.ac_steps)])
        self.training_epoch_data = []
        self.training_batch_data = []

        # freeze the parameters of coding head
        for p in self.coding_params:
            p.requires_grad = False
        print('==> Frost the coding head, fine-tune it after pretraining')
        

    def multi_task_criterion(self, 
                       result: torch.Tensor,
                       tissue_logits: torch.Tensor,
                       raw_embs: torch.Tensor, 
                       tissue_idx: torch.Tensor,
                       emb_masks: torch.Tensor,
                       eps = 1e-6):
        # split tensor for separate loss calculation
        seq_pred_emb, count_pred_emb = result.split([4, 10], dim=-1) # shape: (batch_size, seq_len, 1)
        seq_raw_emb, count_raw_emb = raw_embs.split([4, 10], dim=-1) # shape: (batch_size, seq_len, 1)

        # masked token was indicated by [True], raw token by [False]
        seq_token_mask, count_token_mask = emb_masks.split(1, dim=-1) # shape: (batch_size, seq_len, 1)
        num_masked_seq_token = seq_token_mask.sum() # number of masked token: batch_size * seq_len(True)
        num_masked_count_token = count_token_mask.sum()

        # for ACTG prediction (already logSoftmax)
        seq_loss_all = -(seq_pred_emb * seq_raw_emb * seq_token_mask.float()) # all CrossEntropy loss of masked token
        seq_loss = seq_loss_all.sum() / (num_masked_seq_token + eps) # self.seq_criterion(result[:,:,:4].view(-1, 4), target_seq.view(-1))  # mean loss

        # for normlized count prediction
        count_loss_all = self.count_criterion(count_pred_emb, count_raw_emb) * count_token_mask.float() # all loss of masked token
        count_loss = count_loss_all.sum() / (num_masked_count_token + eps) # mean loss

        # for tissue prediction
        tissue_loss = self.classify_criterion(tissue_logits, tissue_idx)

        return torch.stack([seq_loss, count_loss, tissue_loss], dim=0) # torch.mul(losses, loss_weight) # torch.sqrt((seq_loss + eps) * (count_loss + eps))
    

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
            loss_weight = torch.tensor([5, 1, 1], device=device) * F.softmax(w_i / T, dim=-1) # default weight for seq, count and tissue is 5, 1, 1
        else:
            loss_weight = torch.tensor([5, 1, 1], device=device)  # torch.ones(self.num_losses, device=device) # torch.ones_like(losses).cuda()
        
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
        total_losses = torch.zeros(self.num_losses).cuda()
        local_losses = []
        batch_num = len(self.dataloader)
        start_time = time.time()
        self.sampler.set_epoch(epoch) # DDP
        loss_weight = self.dynamic_weight_average(epoch, 2.0) # DWA

        if self.rank == 0:
            pbar = tqdm(total = batch_num, desc=f"Epoch {epoch+1} for training")

        for batch_idx, batch_data in enumerate(self.dataloader):
            raw_embs, inps, emb_masks, pad_masks, tissue_idx = [data.cuda(non_blocking=True) for data in batch_data]

            # run the forward pass with autocasting
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                output, tissue_pred = self.model(inps, tissue_idx, pad_masks)
                # multi-task learning losses
                losses = self.multi_task_criterion(output, tissue_pred, raw_embs, tissue_idx, emb_masks)
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
        total_losses = torch.zeros(self.num_losses).cuda()
        local_losses = []
        batch_num = len(self.val_loader)
        start_time = time.time()
        self.val_sampler.set_epoch(epoch) # DDP
    
        with torch.no_grad():
            if self.rank == 0:
                pbar = tqdm(total = batch_num, desc=f"Epoch {epoch+1} for evaluating")

            for batch_idx, batch_data in enumerate(self.val_loader):
                raw_emb, inp, emb_mask, pad_mask, tissue_idx = [data.cuda(non_blocking=True) for data in batch_data]

                 # run the forward pass with autocasting
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    output, tissue_pred = self.model(inp, tissue_idx, pad_mask)
                    losses = self.multi_task_criterion(output, tissue_pred, raw_emb, tissue_idx, emb_mask)

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
        best_val_loss = torch.tensor([np.inf] * self.num_losses)

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