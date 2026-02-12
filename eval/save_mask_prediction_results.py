import torch
import numpy as np
import os
import pickle
from tqdm import tqdm
from typing import Union, List, Optional, Dict
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from utils import clean_up_memory

def save_prediction_results(
    model, 
    dataset, 
    mask_ratios: Optional[Union[float, List[float]]] = None, 
    num_samples: int = 200, 
    batch_size: int = 16,
    out_dir: str = "./results", 
    suffix: str = ""
):
    """
    Run model predictions in batches and save the results as a Pickle file.
    
    Args:
        model: Trained model.
        dataset: Test dataset.
        mask_ratios: (Optional) Custom mask ratios.
                     - None: Default np.linspace(0.1, 1.0, 10).
                     - float: Single ratio, e.g., 0.5.
                     - list: List of ratios, e.g., [0.2, 0.5, 0.8].
        num_samples: Number of samples to select.
        batch_size: Batch size for inference.
        out_dir: Output directory.
        suffix: Filename suffix.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    
    # --- 1. Handle mask_ratios parameter ---
    if mask_ratios is None:
        # Default: 10 points from 0.1 to 1.0
        ratios = np.round(np.linspace(0.1, 1.0, 10), 1)
    elif isinstance(mask_ratios, (float, int)):
        # Convert single value to list
        ratios = [float(mask_ratios)]
    else:
        # Use provided list
        ratios = mask_ratios
        
    print(f"Target Mask Ratios: {ratios}")

    # --- 2. Determine file name ---
    file_name = f"predictions_{model.model_name}.{suffix}.pkl"
    save_path = os.path.join(out_dir, file_name)
    print(f"Predictions will be saved to: {save_path}")

    # --- 3. Select samples ---
    all_indices = np.arange(len(dataset))
    if len(all_indices) > num_samples:
        np.random.seed(42)
        target_indices = np.random.choice(all_indices, num_samples, replace=False)
    else:
        target_indices = all_indices
        
    print(f"Selected {len(target_indices)} samples for prediction generation.")
    
    # Create a Subset for the target indices
    subset = Subset(dataset, target_indices)

    # --- 4. Define Collate Function for Batching ---
    def collate_fn(batch):
        """
        Pads sequences to the longest in the batch.
        Assumes dataset items are tuples/lists where:
        idx 0: cell_idx
        idx 1: cds_info_tuple (start, end)
        idx 3: seq_emb
        idx 4: count_emb
        """
        # Unzip the batch
        # We also need original indices to retrieve UUIDs later if needed, 
        # but since we are using a Subset, we will map them via UUIDs.
        
        # Collect raw data
        # Note: 'item' is a tuple returned by __getitem__ of the dataset
        cell_idxs = [torch.tensor([item[0]]) for item in batch]
        cds_starts = [item[1]["cds_start_pos"] for item in batch]
        cds_ends = [item[1]["cds_end_pos"] for item in batch]
        seq_embs = [item[3] for item in batch]
        count_embs = [item[4] for item in batch]
        
        # Get original lengths for un-padding later
        lengths = [s.shape[0] for s in seq_embs]
        
        # Pad sequences (Batch First)
        # Assuming -1 is the padding value for embeddings based on previous context
        seq_padded = pad_sequence(seq_embs, batch_first=True, padding_value=-1)
        count_padded = pad_sequence(count_embs, batch_first=True, padding_value=-1)
        
        cell_stack = torch.stack(cell_idxs).squeeze(1) # (B,)
        
        return cell_stack, seq_padded, count_padded, lengths, cds_starts, cds_ends

    # --- 5. Create DataLoader ---
    dataloader = DataLoader(
        subset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True
    )

    # --- 6. Initialize result dictionary ---
    saved_data = {} 
    
    # Track which global index (within target_indices) we are processing
    global_ptr = 0

    # --- 7. Batch Inference Loop ---
    for batch_data in tqdm(dataloader, desc="Generating predictions (Batch)"):
        b_cell_idxs, b_seq, b_count, b_lengths, b_cds_starts, b_cds_ends = batch_data
        
        # Move to device
        b_cell_idxs = b_cell_idxs.to(model.device)
        b_seq = b_seq.to(model.device)
        b_count = b_count.to(model.device)
        
        batch_current_size = b_seq.shape[0]
        
        # Retrieve metadata for this batch
        batch_uuids = []
        batch_cds_infos = []
        
        for i in range(batch_current_size):
            # Get real index from target_indices
            real_idx = target_indices[global_ptr + i]
            
            # Get UUID
            try:
                uuid = dataset.uuids[real_idx]
            except:
                uuid = str(real_idx)
            batch_uuids.append(uuid)
            
            # Parse CDS info
            raw_start = b_cds_starts[i]
            raw_end = b_cds_ends[i]
            if raw_start != -1 and raw_end != -1:
                batch_cds_infos.append({'start': int(raw_start), 'end': int(raw_end)})
            else:
                batch_cds_infos.append(None)
                
        # Update global pointer
        global_ptr += batch_current_size

        # --- Initialize storage for this batch ---
        # We need to store ground truth first
        for i, uuid in enumerate(batch_uuids):
            # Extract unpadded ground truth
            valid_len = b_lengths[i]
            gt = b_count[i, :valid_len].cpu().numpy().astype(np.float16)
            
            saved_data[uuid] = {
                'truth': gt, # Ground Truth (unpadded)
                'cds_info': batch_cds_infos[i],
                'ratios': {} 
            }

        # --- Iterate over Ratios ---
        for ratio in ratios:
            # Clone count for masking
            masked_batch = b_count.clone()
            
            # We need to store mask indices for each sample to save them later
            batch_mask_pos_list = []
            
            # Apply masking per sample (because lengths and mask positions vary)
            for i in range(batch_current_size):
                valid_len = b_lengths[i]
                
                # Identify valid positions (ignoring padding)
                # In padded tensor, valid positions are 0 to valid_len
                # Additionally check for internal missing data if represented by -1
                valid_indices = torch.where(b_seq[i, :valid_len, 0] != -1)[0].cpu().numpy()
                
                n_mask = int(len(valid_indices) * ratio)
                if ratio >= 0.99: n_mask = len(valid_indices)
                
                mask_pos = np.array([], dtype=np.int32)
                
                if n_mask > 0:
                    mask_pos = np.random.choice(valid_indices, n_mask, replace=False)
                    # Apply mask (set to 0.0)
                    masked_batch[i, mask_pos, :] = 0.0
                
                batch_mask_pos_list.append(mask_pos)
            
            # Create src_mask for the model (True for valid tokens, False for padding)
            # Assuming b_seq padding value is -1
            src_mask = (b_seq[:, :, 0] != -1)
            
            # --- Model Inference (Batch) ---
            with torch.no_grad():
                out = model.predict(b_seq, masked_batch, b_cell_idxs, src_mask, head_names=["count"])
                pred_batch = out["count"] # (B, L, D)
            
            # --- Save results per sample ---
            for i, uuid in enumerate(batch_uuids):
                valid_len = b_lengths[i]
                
                # Slice prediction to original length
                pred_sample = pred_batch[i, :valid_len].squeeze().cpu().numpy().astype(np.float16)
                mask_indices = batch_mask_pos_list[i].astype(np.int32)
                
                saved_data[uuid]['ratios'][ratio] = {
                    'pred': pred_sample,
                    'mask_indices': mask_indices
                }

    # --- 8. Save to Disk ---
    print(f"Saving {len(saved_data)} transcripts to disk...")
    with open(save_path, 'wb') as f:
        pickle.dump(saved_data, f)

    # -- claer memory
    clean_up_memory()
    print("Done.")
    return save_path