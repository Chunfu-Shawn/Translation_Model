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


def collate_pad_finetune_batch_to_cuda(batch):
    """
    batch_list: list, length is equal to the batch_size of DataLoader.
    batch_list[0] is a dict, including:
      'seq_embeddings', 'count_embeddings', 'masked_embedding', 'pad_mask', 'pred_mask'
    return Tensor values
    """
    raw_embs, inps, pred_masks, coding_embs = zip(*batch)

    # 1. same input length
    # lens = [tensor.size(0) for tensor in inps]

    # 2. pad sequence for [:, seq_len, :]
    raw_embs_padded = pad_sequence(raw_embs, batch_first=True, padding_value=-1)
    coding_embs_padded = pad_sequence(coding_embs, batch_first=True, padding_value=0.0)

    # 3. pad_mask (True indicate this is a padding need to be blocked)
    pad_masks = (raw_embs_padded == -1)[:, :, 0].squeeze(-1)

    return (
        raw_embs_padded, 
        coding_embs_padded,
        pad_masks
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

class FineTuneTrainer:
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
                 full_model_epoch_perc = 0.5,
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
            prefetch_factor=1,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=collate_pad_finetune_batch_to_cuda
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
            prefetch_factor=1,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=collate_pad_finetune_batch_to_cuda
        )
        self.batch_num = len(dataset)
        self.epoch_num = epoch_num
        self.full_model_epoch_perc = full_model_epoch_perc
        self.current_epoch = 0
        self.ac_steps = accumulation_steps
        self._total_steps = int(self.epoch_num * self.batch_num / self.ac_steps)
        self.warmup_perc = warmup_perc
        self._print_progress_every = print_progress_every
        self._save_every = save_every
        self.cpt_pretrain_dir = os.path.join(checkpoint_dir, "pretrain")
        self.cpt_finetune_dir = os.path.join(checkpoint_dir, "finetune")
        self.log_dir = log_dir
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean") # for 3-frames
        self.lr = learning_rate
        self.optimizer = torch.optim.AdamW(
            self.coding_params, # train for coding parameters first
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
        self.full_fine_tune = False

        # freeze the main parameters
        for p in self.main_params:
            p.requires_grad = False
        print('==> Frost the main model, fine-tune the coding head first')
    
    def coding_criterion(self,
                         output_embs: torch.Tensor, 
                         coding_embs: torch.Tensor, 
                         reg_w = 1):

        coding_loss = self.criterion(output_embs, coding_embs)
        # ensure continuous coding prediction
        variation_penalty = torch.abs(output_embs[:, 1:, :] - output_embs[:, :-1, :]).mean()
        
        return coding_loss + reg_w * variation_penalty


    def save_checkpoint(self, state, is_best: bool):
        os.makedirs(self.cpt_finetune_dir, exist_ok=True)
        # save checkpoint
        latest_path = os.path.join(self.cpt_finetune_dir, self.model_full_name + ".latest.pt")
        torch.save(state, latest_path)
        if is_best:
            best_path = os.path.join(self.cpt_finetune_dir, self.model_name + ".best.pt")
            torch.save(state, best_path)

    
    def train(self, epoch):
        self.model.train() # set train mode
        total_loss = torch.zeros(1).cuda()
        local_loss = []
        batch_num = len(self.dataloader)
        start_time = time.time()
        self.sampler.set_epoch(epoch) # DDP

        if self.rank == 0:
            pbar = tqdm(total = batch_num, desc=f"Epoch {epoch+1} for training")

        for batch_idx, batch_data in enumerate(self.dataloader):
            raw_emb, coding_embs, pad_mask = [data.cuda(non_blocking=True) for data in batch_data]

            # run the forward pass with autocasting
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                output = self.model(raw_emb, pad_mask, head_name = "coding")
                # two task loss
                loss = self.coding_criterion(output, coding_embs)
                ## mean loss (dynamic weight average) for accumulation
                acc_loss = loss / self.ac_steps

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
                self.scaler.scale(acc_loss).backward()

            # synchronic gradient accumulation
            if do_sync:
                self.scaler.step(self.optimizer) # update parameters
                self.scaler.update() # update the scale for next iteration
                self.scheduler.step() # update learning rate
                self.optimizer.zero_grad() # reset gradient

            total_loss += loss.detach()
            local_loss.append(loss.detach())

            if self.rank == 0 and (batch_idx + 1) % self._print_progress_every == 0:
                pbar.update(self._print_progress_every)
                print(loss)

        if self.rank == 0:
            pbar.close()

        # gather loss
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM) # total of seq_loss and count_loss
        
        gathered_loss = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_loss, local_loss)
        mean_epoch_loss = total_loss/(batch_num * self.world_size)

        # time
        end_time = time.time()
        print(f'Epoch {epoch+1} training time: {end_time - start_time} seconds, mean loss: {mean_epoch_loss}')

        return mean_epoch_loss, gathered_loss
    

    def eval(self, epoch):
        self.model.eval() # set eval mode
        total_loss = torch.zeros(1).cuda()
        local_loss = []
        batch_num = len(self.val_loader)
        start_time = time.time()
        self.val_sampler.set_epoch(epoch) # DDP
    
        with torch.no_grad():
            if self.rank == 0:
                pbar = tqdm(total = batch_num, desc=f"Epoch {epoch+1} for evaluating")

            for batch_idx, batch_data in enumerate(self.val_loader):
                raw_emb, coding_embs, pad_mask = [data.cuda(non_blocking=True) for data in batch_data]

                 # run the forward pass with autocasting
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    output = self.model(raw_emb, pad_mask, head_name = "coding")
                    loss = self.coding_criterion(output, coding_embs)

                total_loss += loss.detach()
                local_loss.append(loss.detach())

                if self.rank == 0 and (batch_idx + 1) % self._print_progress_every == 0:
                    pbar.update(self._print_progress_every)        

        if self.rank == 0:
            pbar.close()

        # all-reduce and gather
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        gathered_loss = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_loss, local_loss)
        mean_epoch_loss = total_loss/(batch_num * self.world_size)

        # time
        end_time = time.time()
        print(f'Epoch {epoch+1} evaluating time: {end_time - start_time} seconds, mean loss: {mean_epoch_loss}')

        return mean_epoch_loss, gathered_loss
    

    def fine_tune(self):
        start_epoch = 0
        best_val_loss = torch.tensor(np.inf)
        
        # resume training
        ckpt_path = os.path.join(self.cpt_finetune_dir, self.model_full_name + ".latest.pt")
        if self.resume and os.path.isfile(ckpt_path):
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
            print(f"==> Start fine-tuning from scratch in GPU {self.rank}.")
            # load best pretrain parameter
            best_ckpt_path = os.path.join(self.cpt_pretrain_dir, self.model_full_name + ".best.pt")
            print(f"--> Import the best pretrain model from {best_ckpt_path} in GPU {self.rank}.")
            if os.path.isfile(best_ckpt_path):
                ckpt = torch.load(best_ckpt_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
                self.model.module.load_state_dict(ckpt['model'])


        # start training
        for epoch in range(start_epoch, self.epoch_num):
            print(f"==> Training start from epoch {epoch + 1} in GPU {self.rank}")
            
            # defrost full model and fine-tune
            if not self.full_fine_tune and epoch + 1 >= self.full_model_epoch_perc * self.epoch_num:
                for p in self.main_params:
                    p.requires_grad = True
                # add main parameters in optimizer 
                self.optimizer.add_param_group({'params': self.main_params})
                self.full_fine_tune = True
                print(f"[rank{self.rank}] defrost main model, start to fine-tune it")

            # --- train epoch
            train_epoch_loss, train_batch_loss = self.train(epoch)
            # --- validate epoch
            val_epoch_loss, val_batch_loss = self.eval(epoch)
            
            # save checkpoints and loss
            is_best = val_epoch_loss < best_val_loss
            best_val_loss = val_epoch_loss if is_best else best_val_loss

            # details
            epoch_data = {
                "epoch": epoch + 1,
                "train_loss": train_epoch_loss.item(),
                "valid_loss": val_epoch_loss.item()
            }
            batch_data = {
                "epoch": epoch + 1,
                "train_batch_loss": [x.item() for proc in train_batch_loss for x in proc],
                "valid_loss": [x.item() for proc in val_batch_loss for x in proc]
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

        