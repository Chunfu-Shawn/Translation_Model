import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from plotnine import *
from scipy.stats import spearmanr, pearsonr
from eval.calculate_te import calculate_morf_mean_signal

# --- Feature Extraction ---
# =================================================================
# [MODIFIED] Added target_cell_type parameter
# =================================================================
def extract_length_vs_te_data(preds, seqs, cds_dict, target_cell_type=None):
    """
    Extract lengths for 5'UTR, CDS, 3'UTR, Total, and calculate TE.
    """
    results = []
    print(f"Extracting Sequence Lengths vs TE")
    if target_cell_type:
        print(f"Filtering for target cell type: {target_cell_type}")
    
    for cell_type, tid_dict in preds.items():
        # =================================================================
        # [NEW] Skip irrelevant cell types
        # =================================================================
        if target_cell_type is not None and cell_type != target_cell_type:
            continue
            
        print(f"  -> Processing cell type: {cell_type}")
        
        for tid, pred_signal in tqdm(tid_dict.items(), leave=False):
            # 1. Safely clean Tid
            clean_tid = str(tid).split('.')[0] if str(tid).startswith('ENST') else str(tid).split('|')[0]
            
            seq_key = tid if tid in seqs else clean_tid
            if seq_key not in seqs: 
                continue
            
            # 2. Extract CDS info from cds_dict
            cds_info = cds_dict.get(clean_tid, cds_dict.get(tid))
            if not cds_info: 
                continue
                
            cds_s = int(cds_info.get("cds_start_pos", -1)) if isinstance(cds_info, dict) else getattr(cds_info, "cds_start_pos", -1)
            cds_e = int(cds_info.get("cds_end_pos", -1)) if isinstance(cds_info, dict) else getattr(cds_info, "cds_end_pos", -1)
            
            if cds_s == -1 or cds_e == -1: 
                continue
                
            m_start = max(0, cds_s - 1) 
            m_end = cds_e
            
            seq_str = seqs[seq_key] 
            seq_len = len(seq_str)
            
            # 3. Calculate regional lengths
            len_5utr = m_start
            len_cds = m_end - m_start
            len_3utr = seq_len - m_end
            
            if len_5utr < 0 or len_cds < 0 or len_3utr < 0: 
                continue

            # 4. Calculate TE
            try:
                if hasattr(pred_signal, 'cpu'):
                    pred_signal = pred_signal.cpu().numpy()
                    
                # Restore linear space
                pred_arr = np.expm1(np.array(pred_signal).reshape(-1).astype(np.float32))
                
                if len(pred_arr) != seq_len: 
                    pred_arr = pred_arr[:seq_len]
                    
                te = calculate_morf_mean_signal(pred_arr, m_start, m_end)
                if te < 1e-6: 
                    continue
            except: 
                continue
            
            results.append({
                'UUID': f"{clean_tid}-{cell_type}",
                'Cell_type': cell_type,
                'Len_5UTR': len_5utr,
                'Len_CDS': len_cds,
                'Len_3UTR': len_3utr,
                'Len_Total': seq_len,
                'TE': te
            })
        
    return pd.DataFrame(results)

# --- 3. Plotting ---

def plot_length_correlation_faceted(df, out_dir, suffix=""):
    """
    Faceted scatter plot for Length vs TE (5'UTR, CDS, 3'UTR, Total).
    """
    if df.empty:
        print("No data to plot.")
        return
        
    # --- 1. Data Preprocessing ---
    upper_te = df['TE'].quantile(0.99)
    clean_df = df[df['TE'] <= upper_te].copy()
    
    id_vars = ['UUID', 'TE']
    value_vars = ['Len_5UTR', 'Len_CDS', 'Len_3UTR', 'Len_Total']
    
    plot_df = clean_df.melt(id_vars=id_vars, value_vars=value_vars, 
                            var_name='Region_Raw', value_name='Length')
    
    plot_df = plot_df[plot_df['Length'] > 0].copy()
    
    label_map = {
        'Len_5UTR': "5' UTR length",
        'Len_CDS': "CDS length",
        'Len_3UTR': "3' UTR length",
        'Len_Total': "Transcript length"
    }
    plot_df['Region'] = plot_df['Region_Raw'].map(label_map)
    
    region_order = ["5' UTR length", "CDS length", "3' UTR length", "Transcript length"]
    plot_df['Region'] = pd.Categorical(plot_df['Region'], categories=region_order, ordered=True)
    
    # --- 2. Calculate Correlation Stats ---
    cor_stats = []
    for region in region_order:
        sub_df = plot_df[plot_df['Region'] == region]
        if len(sub_df) < 10: continue 
        
        r_spearman, p_s = spearmanr(sub_df['Length'], sub_df['TE'])
        r_pearson, p_p = pearsonr(sub_df['Length'], sub_df['TE'])
        
        p_s_text = f"{p_s:.1e}" if p_s < 0.001 else f"{p_s:.3f}"
        p_p_text = f"{p_p:.1e}" if p_p < 0.001 else f"{p_p:.3f}"
        
        label_text = (f"Spearman R = {r_spearman:.3f} (P={p_s_text})\n"
                      f"Pearson R = {r_pearson:.3f} (P={p_p_text})")
        
        x_pos = sub_df['Length'].min()
        y_pos = sub_df['TE'].max() * 0.95
        
        cor_stats.append({
            'Region': region,
            'Label': label_text,
            'x': x_pos,
            'y': y_pos
        })
        
    cor_df = pd.DataFrame(cor_stats)
    cor_df['Region'] = pd.Categorical(cor_df['Region'], categories=region_order, ordered=True)

    # --- 3. Plotting ---
    p = (
        ggplot(plot_df, aes(x='Length', y='TE'))
        + geom_point(alpha=0.1, size=2, stroke=0, color="#2C3E50")
        + geom_smooth(method='lm', color="#E74C3C", fill="#E74C3C", alpha=0.2)
        + facet_wrap('~Region', scales='free_x', ncol=2)
        
        + geom_text(data=cor_df, mapping=aes(x='x', y='y', label='Label'),
                    ha='left', va='top', size=10, inherit_aes=False)
        
        + scale_x_log10()
        + theme_classic()
        + theme(
            figure_size=(10, 8), 
            axis_text=element_text(size=10),
            axis_title=element_text(size=12),
            strip_text=element_text(size=12),
            strip_background=element_blank(),
            panel_grid_minor=element_blank()
        )
        + labs(
            x="Length (nt, log10 scale)",
            y="Mean CDS translation signal"
        )
    )
    
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"corr_length_faceted.{suffix}.pdf")
    p.save(save_path, width=7, height=7, dpi=300, verbose=False)
    print(f"Saved faceted plot to {save_path}")

def plot_length_correlation(df, out_dir, x_col='Len_5UTR', suffix=""):
    """
    Scatter plot for Length vs TE (Single feature).
    """
    if df.empty:
        print("No data to plot.")
        return
        
    upper_te = df['TE'].quantile(0.99)
    plot_df = df[(df['TE'] <= upper_te) & (df[x_col] > 0)].copy()
    
    r_spearman, p_s = spearmanr(plot_df[x_col], plot_df['TE'])
    r_pearson, p_p = pearsonr(plot_df[x_col], plot_df['TE'])
    
    p_s_text = f"{p_s:.2e}" if p_s < 0.001 else f"{p_s:.3f}"
    p_p_text = f"{p_p:.2e}" if p_p < 0.001 else f"{p_p:.3f}"
    stats_label = (f"Spearman R = {r_spearman:.3f} (P={p_s_text})\n"
                   f"Pearson R = {r_pearson:.3f} (P={p_p_text})")
    
    label_map = {
        'Len_5UTR': "5' UTR length",
        'Len_3UTR': "3' UTR length",
        'Len_CDS': "CDS length",
        'Len_Total': "Transcript length"
    }
    x_label = label_map.get(x_col, x_col)
    
    p = (
        ggplot(plot_df, aes(x=x_col, y='TE'))
        + geom_point(alpha=0.2, size=2, stroke=0, color="#2C3E50")
        + geom_smooth(method='lm', color="#E74C3C", fill="#E74C3C", alpha=0.2)
        + annotate("text", x=plot_df[x_col].min(), y=plot_df['TE'].max()*0.95, 
                   label=stats_label, ha='left', va='top', size=10)
        + scale_x_log10()
        + theme_bw()
        + theme(
            figure_size=(8, 6),
            axis_text=element_text(size=12),
            axis_title=element_text(size=12)
        )
        + labs(
            x=f"{x_label} (nt, log10 scale)",
            y="CDS translation efficiency"
        )
    )
    
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"corr_{x_col}_vs_te.{suffix}.pdf")
    p.save(save_path, width=5, height=5, dpi=300, verbose=False)
    print(f"Saved plot to {save_path}")

# --- 4. Main Execution ---
# =================================================================
# [MODIFIED] Added target_cell_type parameter to main execution function
# =================================================================
def evaluate_rna_length_correlation(pred_pkl, seq_pkl, cds_pkl, out_dir="./results/len_eval", suffix="", target_cell_type=None):
    """
    Main entry function for Sequence Length vs Predicted TE evaluation.
    """
    print(f"Loading data for Length Analysis...")
    with open(pred_pkl, 'rb') as f: preds = pickle.load(f)
    with open(seq_pkl, 'rb') as f: seqs = pickle.load(f)
    
    # Load cds_dict to extract CDS boundaries
    print(f"Loading transcript CDS metadata from {cds_pkl}...")
    with open(cds_pkl, 'rb') as f: cds_dict = pickle.load(f)
    
    # 1. Extract
    # =================================================================
    # [MODIFIED] Pass target_cell_type to extraction function
    # =================================================================
    df_data = extract_length_vs_te_data(preds, seqs, cds_dict, target_cell_type=target_cell_type)
    
    if df_data.empty:
        print("No valid data extracted.")
        return

    # 2. Save & Plot
    os.makedirs(out_dir, exist_ok=True)
    df_data.to_csv(os.path.join(out_dir, f"length_data.{suffix}.csv"), index=False)
    
    plot_length_correlation_faceted(df_data, out_dir, suffix=suffix)