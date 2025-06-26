import json
import math
import os
import numpy as np
import torch
import torch.nn as nn
import time
from livelossplot import PlotLosses
from tqdm import tqdm

def create_lr_lambda(total_steps, warmup_steps=0, min_eta=1e-4):
    def lr_lambda(current_step):
        # linear warmup
        if current_step < warmup_steps:
            return max(min_eta, float(current_step) / float(max(1, warmup_steps)))
        # cosine annealing
        progress = float(current_step - warmup_steps) / float(total_steps - warmup_steps)
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return max(min_eta, decay)
    
    return lr_lambda

class Trainer:
    def __init__(self,
                 model,
                 dataset,
                 checkpoint_dir: None,
                 log_dir: None,
                 model_name = "ribomodel",
                 print_progress_every: int = 50,
                 save_every: int = 5,
                 batch_size: int = 8,
                 epoch_num = 100,
                 learning_rate: float = 0.00001,
                 accumulation_steps: int = 5,
                 beta: list = (0.9, 0.98),
                 epsilon: float = 1e-9,
                 ):
        
        self.model = model
        self.model_name = model_name,
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_num = len(dataset)
        self.epoch_num = epoch_num
        self.current_epoch = 0
        self._total_steps = self.epoch_num * self.batch_num
        self._warmup_steps = 0.2 * self._total_steps
        self._print_progress_every = print_progress_every
        self._save_every = save_every
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.criterion = nn.CrossEntropyLoss() # nn.MSELoss()
        self.accumulation_steps = accumulation_steps
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            betas=beta,
            eps = epsilon,
            weight_decay=0.015)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer = self.optimizer,
            lr_lambda = create_lr_lambda(
                total_steps = self._total_steps,
                warmup_steps = self._warmup_steps,
                min_eta = 1e-4
            ))
        
    def token_loss(self, result: torch.Tensor, raw_emb: torch.Tensor, inverse_token_mask: torch.Tensor):
        inv_token_mask = torch.from_numpy(inverse_token_mask).unsqueeze(-1)
        # predicted masked output
        result = result.masked_fill(inv_token_mask, 0).to(torch.float32)
        # target masked output
        target = raw_emb.masked_fill(inv_token_mask, 0)

        return self.criterion(result, target)


    def save_checkpoint(self, state, model_name, is_best: bool):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # save checkpoint
        latest_path = os.path.join(self.checkpoint_dir, model_name, '.latest.pt')
        torch.save(state, latest_path)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, model_name, '.best.pt')
            torch.save(state, best_path)

    
    def train(self, epoch):
        train_loss = []
        total_loss = 0
        batch_num = len(self.dataset)
        start_time = time.time()
        with tqdm(total = batch_num) as pbar:
            pbar.set_description(f"Epoch {epoch+1}, training...")
            for index in range(batch_num):
                tids, length, seq_emb, count_emb, padding_mask, masked_emb, seq_mask = self.dataset[index].values()

                # get model output
                padding_mask = torch.from_numpy(padding_mask).cuda()
                inp = torch.from_numpy(masked_emb).to(torch.float32).cuda()
                output = self.model(inp, padding_mask).cpu()

                # calculate loss
                raw_emb = torch.from_numpy(np.concatenate((seq_emb, count_emb), axis=2)).to(torch.float32)
                loss = self.token_loss(output, raw_emb, seq_mask)
                
                # gradient accumulation
                loss = loss / self.accumulation_steps ## mean loss
                loss.backward() # back propagation
                if index % self.accumulation_steps == 0: # update first batch
                    self.optimizer.step() # update parameters
                    self.optimizer.zero_grad() # reset gradient
                self.scheduler.step() # update learning rate

                if index % self._print_progress_every == 0:
                    # elapsed = time.gmtime(time.time() - prev)
                    # print(f"Epoch {epoch + 1}, Batch {index + 1}, Loss: {loss/np.sum(seq_mask)}")
                    pbar.update(self._print_progress_every)
                    pbar.set_postfix(loss=loss.item()/np.sum(seq_mask))
                total_loss += loss.item()
                train_loss.append(loss.item())
                
                # free up memory
                del output, loss
                torch.cuda.empty_cache()

        # training time
        end_time = time.time()
        print(f'Epoch {epoch+1} training time: {end_time - start_time} seconds')

        return total_loss/batch_num, train_loss
    
    def eval(self, epoch, val_dataset):
        val_loss = []
        total_loss = 0
        batch_num = len(val_dataset)
        start_time = time.time()
        with torch.no_grad():
            with tqdm(total = batch_num) as pbar:
                pbar.set_description(f"Epoch {epoch+1}, evaluating...")
                for index in range(batch_num):
                    tids, length, seq_emb, count_emb, padding_mask, masked_emb, seq_mask = val_dataset[index].values()
                    # get model output
                    padding_mask = torch.from_numpy(padding_mask).cuda()
                    inp = torch.from_numpy(masked_emb).to(torch.float32).cuda()
                    output = self.model(inp, padding_mask).cpu()
                    # calculate loss
                    raw_emb = torch.from_numpy(np.concatenate((seq_emb, count_emb), axis=2)).to(torch.float32)
                    loss = self.token_loss(output, raw_emb, seq_mask)
                    loss = loss / self.accumulation_steps # loss compared with train steps

                    if index % self._print_progress_every == 0:
                        # elapsed = time.gmtime(time.time() - prev)
                        # print(f"Epoch {epoch + 1}, Batch {index + 1}, Loss: {loss/np.sum(seq_mask)}")
                        pbar.update(self._print_progress_every)
                        pbar.set_postfix(loss=loss.item()/np.sum(seq_mask))
                    total_loss += loss.item()
                    val_loss.append(loss.item())
                    
                    # free up memory
                    del output, loss
                    torch.cuda.empty_cache()
        # evaluate running time
        end_time = time.time()
        print(f'Epoch {epoch+1} evaluating time: {end_time - start_time} seconds')
                
        return total_loss/batch_num, val_loss
    
    def pretrain(self, val_dataset):
        liveloss = PlotLosses()

        # resume training
        start_epoch = 0
        best_val_loss = float('inf')
        ckpt_path = self.checkpoint_dir + "ribomodel.latest.pt"
        if os.path.isfile(self.checkpoint_dir + "ribomodel.latest.pt"):
            print(f"==> Loading checkpoint '{ckpt_path}'")
            ckpt = torch.load(ckpt_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            start_epoch = ckpt['epoch']
            best_val_loss = ckpt.get('best_val_loss', best_val_loss)

            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optim'])
            if self.scheduler and ckpt.get('scheduler') is not None:
                self.scheduler.load_state_dict(ckpt['scheduler'])
            print(f"==> Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        else:
            print("==> No checkpoint found, training from scratch.")
        
        # resume loss data
        if os.path.isfile(self.log_dir + "training_epoch_data.json"):
            with open(self.log_dir + "training_epoch_data.json", 'r') as f_epoch:
                training_epoch_data = json.load(f_epoch)
            with open(self.log_dir + "training_batch_data.json", 'r') as f_batch:
                training_batch_data = json.load(f_batch)
        else:
            training_epoch_data = []
            training_batch_data = []

        # start training
        for epoch in range(start_epoch, self.epoch_num):
            print(f"==> Training start from epoch {epoch + 1}")
            logs = {}
            train_epoch_loss, train_batch_loss = self.train(epoch)
            val_epoch_loss, val_batch_loss = self.eval(epoch, val_dataset)
            
            # update loss plot
            logs['loss'] = train_epoch_loss
            logs['val_loss'] = val_epoch_loss
            liveloss.update(logs)
            liveloss.draw()
            
            # save checkpoints and loss
            is_best = val_epoch_loss < best_val_loss
            best_val_loss = min(best_val_loss, val_epoch_loss)
            checkpoint = {
                'epoch': epoch + 1,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }
            epoch_data = {
                "epoch": epoch + 1,
                "train_loss": train_epoch_loss,
                "valid_loss": val_epoch_loss
            }
            batch_data = {
                "epoch": epoch + 1,
                "train_loss": train_batch_loss,
                "valid_loss": val_batch_loss
            }
            training_epoch_data.append(epoch_data)
            training_batch_data.append(batch_data)
            if (epoch + 1) % self._save_every == 0:
                # cpt
                self.save_checkpoint(checkpoint, "ribomodel", is_best)
                # loss
                with open(self.log_dir + "training_epoch_data.json", 'w') as f_epoch:
                    json.dump(training_epoch_data, f_epoch)
                    f_epoch.write('\n')
                with open(self.log_dir + "training_batch_data.json", 'w') as f_batch:
                    json.dump(training_batch_data, f_batch)
                    f_batch.write('\n')