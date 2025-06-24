import numpy as np
import torch
import torch.nn as nn
import time
from tqdm import tqdm

class Trainer:
    def __init__(self,
                 model,
                 dataset,
                 checkpoint_dir: None,
                 print_progress_every: int = 50,
                 batch_size: int = 8,
                 learning_rate: float = 0.00001,
                 accumulation_steps: int = 5,
                 beta: list = (0.9, 0.98),
                 epsilon: float = 1e-9
                 ):
        
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.current_epoch = 0
        self._print_progress_every = print_progress_every
        self.checkpoint_dir = checkpoint_dir
        self.criterion = nn.CrossEntropyLoss() # nn.MSELoss()
        self.accumulation_steps = accumulation_steps
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            betas=beta,
            weight_decay=0.015)
        
    def token_loss(self, result: torch.Tensor, raw_emb: torch.Tensor, inverse_token_mask: torch.Tensor):
        inv_token_mask = torch.from_numpy(inverse_token_mask).unsqueeze(-1)
        # predicted masked output
        result = result.masked_fill(inv_token_mask, 0).to(torch.float32)
        # target masked output
        target = raw_emb.masked_fill(inv_token_mask, 0)

        return self.criterion(result, target)
    
    def train(self, epoch):
        train_loss = []
        total_loss = 0
        batch_num = len(self.dataset)
        start_time = time.time()
        with tqdm(total = batch_num) as pbar:
            pbar.set_description(f"Epoch {epoch+1}, training...")
            for index in range(batch_num):
                tids, length, seq_emb, count_emb, padding_mask, masked_emb, seq_mask = self.dataset[index].values()
                self.optimizer.zero_grad()

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
                if (index + 1) % self.accumulation_steps == 0:
                    self.optimizer.step() # update parameters
                    self.optimizer.zero_grad() # reset gradient

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