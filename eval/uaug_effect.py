import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from eval.calculate_te import calculate_morf_mean_signal


def get_samples_5utr_clean_starts(
        dataset, top_n=50, utr_len_range=[300, 2000], check_region_len=300, 
        target_cell_type=None):
    """
    Filter samples within a specific 5'UTR length range and clean the background.
    
    Cleaning rules:
    Within `check_region_len` upstream of the CDS Start, if ATG, CTG, or GTG are found,
    mutate the 3rd base to C (index 1) to eliminate upstream initiation.
    Stop codons are no longer considered in this version.
    
    Args:
        dataset: Dataset object
        top_n: Number of samples to return
        utr_len_range: [min_len, max_len] for 5'UTR length filtering
        check_region_len: Range upstream of the start codon to clean
        target_cell_type: (Optional) String representing the cell type to filter by.
    """
    candidates = []
    min_len, max_len = utr_len_range
    
    # Update print message to reflect cell type filtering
    if target_cell_type:
        print(f"Scanning dataset (UTR len: {min_len}-{max_len}, Cell type: {target_cell_type}) and cleaning 5'UTR backgrounds (Range: -{check_region_len}nt)...")
    else:
        print(f"Scanning dataset (UTR len: {min_len}-{max_len}) and cleaning 5'UTR backgrounds (Range: -{check_region_len}nt)...")
    
    # Base mapping: A=0, C=1, G=2, T=3
    # Target rule: mutate codon[2] (the 3rd base) to 1 (C)
    targets = {
        (0, 3, 2), # ATG -> ATC
        (1, 3, 2), # CTG -> CTC
        (2, 3, 2)  # GTG -> GTC
    }
    
    # One-hot vector for 'C' substitution
    C_VECTOR = np.array([0, 1, 0, 0], dtype=np.float32)

    for i in tqdm(range(len(dataset))):
        try:
            # Unpack based on the new dataset structure
            item = dataset[i]
            uuid, species, cell_type, expr_vector, meta_info, seq_emb, count_emb = item
            
            # =================================================================
            # [NEW] Skip if the sample does not match the target cell type
            # =================================================================
            if target_cell_type is not None and cell_type != target_cell_type:
                continue
            
            # Extract CDS boundaries safely from meta_info
            cds_s = int(meta_info.get("cds_start_pos", -1)) if isinstance(meta_info, dict) else getattr(meta_info, "cds_start_pos", -1)
            cds_e = int(meta_info.get("cds_end_pos", -1)) if isinstance(meta_info, dict) else getattr(meta_info, "cds_end_pos", -1)
            
            if cds_s == -1 or cds_e == -1: 
                continue
                
            # Convert to 0-based index
            start = max(0, cds_s - 1)
            end = cds_e
            
            # 1. Filter by 5'UTR length
            if not (min_len <= start <= max_len):
                continue
            
            # 2. Extract Numpy Embedding
            if isinstance(seq_emb, torch.Tensor):
                seq_emb_np = seq_emb.cpu().numpy()
            else:
                seq_emb_np = seq_emb.copy()
                
            # Ensure shape is (Length, 4)
            if seq_emb_np.shape[0] == 4 and seq_emb_np.shape[1] > 4:
                seq_emb_np = seq_emb_np.T 

            # Extract expr_vector safely to numpy array for downstream processing
            if isinstance(expr_vector, torch.Tensor):
                expr_np = expr_vector.cpu().numpy()
            else:
                expr_np = np.array(expr_vector)

            # 3. Define the scanning region (upstream of CDS start)
            region_end = start
            region_start = max(0, start - check_region_len)
            
            # Extract sequence indices (0, 1, 2, 3) for quick motif matching
            upstream_indices = np.argmax(seq_emb_np[region_start : region_end], axis=1)
            
            # 4. Sliding window scan & replace (5' -> 3')
            mutation_count = 0
            
            for k in range(len(upstream_indices) - 2):
                codon = tuple(upstream_indices[k : k+3])
                
                if codon in targets:
                    # Target found: mutate the 3rd base (index k+2) to 'C'
                    mutation_pos = region_start + k + 2
                    seq_emb_np[mutation_pos] = C_VECTOR
                    
                    # Update indices array to prevent overlapping logic errors
                    upstream_indices[k+2] = 1 
                    mutation_count += 1

            # 5. Store valid candidate
            candidates.append({
                'index': i,
                'cell_type': cell_type,
                'expr_vector': expr_np, 
                'utr5_len': start,
                'morf_start': start,
                'morf_end': end,
                'uuid': str(uuid), # Ensure UUID is a string
                'seq_emb': seq_emb_np, # The cleaned embedding
                'mutations_made': mutation_count
            })
            
        except Exception as e:
            continue
            
    # Sort descending by 5'UTR length and select Top N
    candidates.sort(key=lambda x: x['utr5_len'], reverse=True)
    selected = candidates[:top_n]
    
    if len(selected) > 0:
        print(f"Selected {len(selected)} transcripts.")
        print(f"5'UTR Length Range: {selected[-1]['utr5_len']} - {selected[0]['utr5_len']} nt")
        total_muts = sum(s['mutations_made'] for s in selected)
        print(f"Total background start codons neutralized in check regions: {total_muts}")
    else:
        print("Warning: No transcripts met the criteria.")
    
    return selected


# ==============================================================================
# Helper Dataset for Batched Mutants
# ==============================================================================
class MutantDataset(Dataset):
    def __init__(self, mutant_records):
        self.records = mutant_records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        return (
            record['uuid'],
            record['distance'],
            record['codon'],
            record['frame'],
            record['te_wt'],
            record['m_start'],
            record['m_end'],
            torch.from_numpy(record['seq_emb']), 
            torch.from_numpy(record['expr_vector']) 
        )

def collate_fn_mutants(batch):
    uuids, distances, codons, frames, te_wts, m_starts, m_ends, seq_embs, expr_vectors = zip(*batch)
    
    lengths = [s.shape[0] for s in seq_embs]
    seq_padded = pad_sequence(seq_embs, batch_first=True, padding_value=-1)
    expr_padded = torch.stack(expr_vectors)
    
    return uuids, distances, codons, frames, te_wts, m_starts, m_ends, seq_padded, expr_padded, lengths


# ==============================================================================
# Main Evaluator Class
# ==============================================================================
class uStartCodonEvaluatorEmb:
    def __init__(self, model, out_dir="."):
        self.device = model.device 
        self.out_dir = out_dir
        self.model = model
        
        self.BASES_MAP = {
            'A': np.array([1, 0, 0, 0], dtype=np.float32),
            'C': np.array([0, 1, 0, 0], dtype=np.float32),
            'G': np.array([0, 0, 1, 0], dtype=np.float32),
            'T': np.array([0, 0, 0, 1], dtype=np.float32),
            'N': np.array([0, 0, 0, 0], dtype=np.float32)
        }

    # =================================================================
    # [MODIFIED] Added logic to embed a strong Kozak context specifically for ATG
    # =================================================================
    def inject_start_codon(self, base_emb, morf_start, distance, codon='ATG'):
        new_emb = base_emb.copy()
        codon_pos = morf_start - distance
        
        # Ensure we have enough space for the Kozak context (-3 upstream, +4 downstream)
        if codon_pos < 3 or codon_pos + 3 >= len(new_emb):
            return None
            
        # 1. Insert the main codon (ATG, CTG, or GTG)
        for i, base in enumerate(codon):
            new_emb[codon_pos + i] = self.BASES_MAP[base]
            
        # 2. Add optimal Kozak context (ACC[ATG]G) ONLY if the codon is ATG
        if codon == 'ATG':
            # -3: A, -2: C, -1: C
            new_emb[codon_pos - 3] = self.BASES_MAP['A']
            new_emb[codon_pos - 2] = self.BASES_MAP['C']
            new_emb[codon_pos - 1] = self.BASES_MAP['C']
            # +4: G
            new_emb[codon_pos + 3] = self.BASES_MAP['G']
            
        return new_emb

    def _predict_batch(self, seq_tensor, expr_tensor):
        """Internal helper to call model.predict and clean up memory."""
        self.model.eval()
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                masked_batch = torch.zeros((seq_tensor.shape[0], seq_tensor.shape[1], 1), device=self.device)
                src_mask = (seq_tensor[:, :, 0] != -1)
                
                out_dict = self.model.predict(
                    seq_batch=seq_tensor, 
                    count_batch=masked_batch,
                    expr_vector=expr_tensor, 
                    src_mask=src_mask,
                    head_names=["count"]
                )
                pred_tensor = out_dict['count']
                
            pred_np = np.expm1(pred_tensor.cpu().numpy().astype(np.float32))
            
            del out_dict, pred_tensor, masked_batch, src_mask
            
        return pred_np

    def evaluate_scanning(self, samples, max_distance=50, batch_size=32, num_samples: int = None, 
                          suffix: str = "", save_csv: bool = True):
        """
        Global Batch Scanning using DataLoader.
        Now handles data saving right after evaluation completes.
        """
        os.makedirs(self.out_dir, exist_ok=True)
        
        # --- 1. Sub-sample selection ---
        if num_samples is not None and len(samples) > num_samples:
            np.random.seed(42)
            selected_indices = np.random.choice(len(samples), num_samples, replace=False)
            eval_samples = [samples[i] for i in selected_indices]
            print(f"Selected {num_samples} samples out of {len(samples)} for inference.")
        else:
            eval_samples = samples
            
        distances = list(range(3, max_distance + 1))
        codons = ['ATG', 'CTG', 'GTG']
        
        # --- 2. Phase 1: Pre-calculate all WT Baselines ---
        print(">>> Phase 1: Calculating WT baselines...")
        wt_te_dict = {} 
        
        for sample in tqdm(eval_samples, desc="WT Predict"):
            uuid = sample.get('uuid', 'unknown')
            seq_emb = sample['seq_emb']
            expr_vec = sample['expr_vector']
            m_start = sample['morf_start']
            m_end = sample['morf_end']
            
            seq_t = torch.from_numpy(seq_emb).unsqueeze(0).to(self.device)
            expr_t = torch.from_numpy(expr_vec).unsqueeze(0).float().to(self.device)
            
            pred_wt = self._predict_batch(seq_t, expr_t)[0]
            te_wt = calculate_morf_mean_signal(pred_wt, m_start, m_end)
            wt_te_dict[uuid] = te_wt
            
            del seq_t, expr_t

        # --- 3. Phase 2: Generate all mutant records ---
        print(f">>> Phase 2: Generating all mutant combinations (Max Dist: {max_distance})...")
        mutant_records = []

        for sample in tqdm(eval_samples, desc="Generating Mutants"):
            uuid = sample.get('uuid', 'unknown')
            te_wt = wt_te_dict.get(uuid, 0)
            
            if te_wt < 1e-6: 
                continue 

            expr_vector = sample['expr_vector']
            base_emb = sample['seq_emb']
            m_start = sample['morf_start'] 
            m_end = sample['morf_end']

            for dist in distances:
                frame_status = "In-frame" if dist % 3 == 0 else "Out-frame"
                
                for codon in codons:
                    emb_mut = self.inject_start_codon(base_emb, m_start, distance=dist, codon=codon)
                    if emb_mut is not None:
                        mutant_records.append({
                            'uuid': uuid,
                            'distance': dist, 
                            'codon': codon, 
                            'frame': frame_status,
                            'te_wt': te_wt,
                            'm_start': m_start,
                            'm_end': m_end,
                            'seq_emb': emb_mut,
                            'expr_vector': expr_vector
                        })

        # --- 4. Phase 3: Global DataLoader Prediction ---
        total_mutants = len(mutant_records)
        print(f">>> Phase 3: Predicting {total_mutants} generated mutant sequences using DataLoader...")
        
        mutant_dataset = MutantDataset(mutant_records)
        dataloader = DataLoader(
            mutant_dataset, 
            batch_size=batch_size, 
            collate_fn=collate_fn_mutants,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        results = []
        for batch_data in tqdm(dataloader, desc="Mutant Inference"):
            b_uuids, b_dists, b_codons, b_frames, b_te_wts, b_m_starts, b_m_ends, b_seq, b_expr, b_lengths = batch_data
            
            b_seq = b_seq.to(self.device)
            b_expr = b_expr.float().to(self.device)
            
            try:
                batch_preds = self._predict_batch(b_seq, b_expr)
                
                for j in range(len(b_uuids)):
                    valid_len = b_lengths[j]
                    pred_single = batch_preds[j, :valid_len]
                    
                    te_mut = calculate_morf_mean_signal(pred_single, b_m_starts[j], b_m_ends[j])
                    
                    results.append({
                        'UUID': b_uuids[j],
                        'Distance': b_dists[j], 
                        'Codon': b_codons[j], 
                        'Frame': b_frames[j],
                        'Relative_TE': te_mut / b_te_wts[j]
                    })
            except Exception as e:
                print(f"Error in inference batch: {e}. Skipping.")
                
            torch.cuda.empty_cache()
            
        df_results = pd.DataFrame(results)
        
        if save_csv and not df_results.empty:
            raw_csv = os.path.join(self.out_dir, f"uStartCodon_scanning_raw.{suffix}.csv" if suffix else "uStartCodon_scanning_raw.csv")
            df_results.to_csv(raw_csv, index=False)
            print(f"Saved raw scanning data to {raw_csv}")
            
            agg_df = df_results.groupby(['Distance', 'Codon'])['Relative_TE'].mean().reset_index()
            agg_csv = os.path.join(self.out_dir, f"uStartCodon_scanning_aggregated.{suffix}.csv" if suffix else "uStartCodon_scanning_aggregated.csv")
            agg_df.to_csv(agg_csv, index=False)
            print(f"Saved aggregated scanning data to {agg_csv}")
                
        return df_results

    def plot_scanning_effect(self, df, suffix=""):
        """
        Pure plotting function: Does not save CSV data.
        Highly restores the reference illustration: Dual top-bottom panels, 
        color coding, and vertical dashed lines marking the in-frame positions.
        """
        if df.empty: return
        import os
        import matplotlib.pyplot as plt
        
        os.makedirs(self.out_dir, exist_ok=True)
        
        # Aggregate to calculate the mean effect at each position for plotting
        plot_df = df.groupby(['Distance', 'Codon'])['Relative_TE'].mean().reset_index()
        plot_df['Position'] = -plot_df['Distance']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6), sharex=True, sharey=True)
        
        # Color configuration
        color_atg = "#4C72B0"  
        color_ctg = "#4C72B0"  
        color_gtg = "#C44E52"  
        
        # --- Top Panel: ATG ---
        df_atg = plot_df[plot_df['Codon'] == 'ATG']
        ax1.plot(df_atg['Position'], df_atg['Relative_TE'], color=color_atg, linewidth=2.5, label='AUG')
        ax1.set_ylabel("Relative CDS translation signal", fontsize=14)
        ax1.legend(loc='upper center', frameon=False, fontsize=12)
        
        # --- Bottom Panel: CTG & GTG ---
        df_ctg = plot_df[plot_df['Codon'] == 'CTG']
        df_gtg = plot_df[plot_df['Codon'] == 'GTG']
        ax2.plot(df_ctg['Position'], df_ctg['Relative_TE'], color=color_ctg, linewidth=2.5, label='CUG')
        ax2.plot(df_gtg['Position'], df_gtg['Relative_TE'], color=color_gtg, linewidth=2.5, label='GUG')
        ax2.set_ylabel("Relative CDS translation signal", fontsize=14)
        ax2.set_xlabel("5' UTR position relative to CDS", fontsize=14)
        ax2.legend(loc='upper center', ncol=2, frameon=False, fontsize=12)
        
        # --- Aesthetics and In-frame markers ---
        min_pos = plot_df['Position'].min()
        max_pos = plot_df['Position'].max()
        in_frame_positions = [p for p in range(int(min_pos), int(max_pos)+1) if p % 3 == 0]
        
        for ax in [ax1, ax2]:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.tick_params(width=1.5, length=6, labelsize=12, direction='out')
            
            # =================================================================
            # [NEW] 设定 Y 轴上限为 1.1，为 Legend 留出充足空间
            # =================================================================
            # ax.set_ylim(ymax=1.05, ymin=0.77)
            
            # =================================================================
            # [NEW] 动态过滤 Y 轴刻度，只保留 <= 1.0 的刻度线
            # =================================================================
            yticks = ax.get_yticks()
            filtered_yticks = [y for y in yticks if y <= 1.0]
            ax.set_yticks(filtered_yticks)
            
            # Baseline at 1.0 (Wild-Type reference line)
            ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, zorder=0, alpha=0.8)
            
            for p in in_frame_positions:
                ax.axvline(x=p, color='black', linestyle='--', linewidth=1, zorder=0, alpha=0.3)
        
        plt.tight_layout()
        
        save_file = os.path.join(self.out_dir, f"uStartCodon_scanning_effect.{suffix}.pdf" if suffix else "uStartCodon_scanning_effect.pdf")
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Saved periodic scanning plot to {save_file}")
        plt.close()