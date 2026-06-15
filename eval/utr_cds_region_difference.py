import pickle
import numpy as np
import pandas as pd
import os
from typing import Union, Dict
from tqdm import tqdm
from plotnine import *

def calculate_region_metrics(signal_array, global_start_idx, region_start, region_end, total_transcript_sum, threshold=0.01):
    """
    Calculate Proportion and Periodicity for a specific region.
    [Modified] Removed Coverage calculation as requested.
    """
    if region_end <= region_start:
        return None 
        
    region_data = signal_array[region_start:region_end]
    L = len(region_data)
    
    if L < 3: return None

    # Proportion (Sum of region / Sum of total transcript)
    if total_transcript_sum < 1e-6:
        proportion = 0.0
    else:
        proportion = np.sum(region_data) / total_transcript_sum
    
    # Periodicity
    global_indices = np.arange(region_start, region_end)
    frames = (global_indices - global_start_idx) % 3
    
    f0_sum = np.sum(region_data[frames == 0])
    region_sum = np.sum(region_data)
    
    if region_sum < 1e-6:
        periodicity = np.nan
    else:
        periodicity = f0_sum / region_sum
        
    return {
        'Proportion': proportion, 
        'Periodicity': periodicity
    }


def evaluate_region_specificity(
        truth_dataset, 
        pkl_input: Union[Dict[str, str], str], 
        out_dir: str = "./results/plots", 
        suffix: str = "",
        width: float = 5,
        height: float = 5):
    """
    Compare P-site Proportion and Periodicity across 5'UTR, CDS, and 3'UTR.
    Handles nested dictionary pkl formats: {cell_type: {tid: prediction}}
    """
    print(">>> Loading prediction files...")
    all_predictions = {}
    
    if isinstance(pkl_input, str):
        print(f"  - Loading combined predictions from: {pkl_input}")
        with open(pkl_input, 'rb') as f:
            loaded_data = pickle.load(f)
            if isinstance(loaded_data, dict):
                all_predictions = loaded_data
            else:
                raise ValueError("The provided single pickle file does not contain a dictionary.")
    elif isinstance(pkl_input, dict):
        for cell_type, pkl_path in pkl_input.items():
            print(f"  - Loading predictions for {cell_type}: {pkl_path}")
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                if cell_type in data and isinstance(data[cell_type], dict):
                    all_predictions[cell_type] = data[cell_type]
                else:
                    all_predictions[cell_type] = data
    else:
        raise TypeError("pkl_input must be either a file path (str) or a dictionary mapping (dict).")
        
    os.makedirs(out_dir, exist_ok=True)
    metrics_data = []
    
    print(f"\n>>> Analyzing region specificity...")
    
    for i in tqdm(range(len(truth_dataset))):
        uuid, species, cell_type, expr_array, meta_info, seq_emb, count_emb = truth_dataset[i]
        uuid_str = str(uuid)
        
        parts = uuid_str.split('-')
        if len(parts) < 2:
            continue
            
        tid = parts[0]
        cell_type = parts[1]
        
        if cell_type not in all_predictions:
            continue
            
        predictions = all_predictions[cell_type]
        
        lookup_tid = tid
        if lookup_tid not in predictions:
            tid_no_version = tid.split('.')[0]
            if tid_no_version in predictions:
                lookup_tid = tid_no_version
            else:
                continue
                
        pred_signal = predictions[lookup_tid]
        gt_signal = count_emb.numpy().flatten()
        
        pred_linear = np.expm1(pred_signal.astype(np.float32))
        truth_linear = np.expm1(gt_signal.astype(np.float32))
        
        pred_len = len(pred_linear)
        gt_len = len(truth_linear)
        
        cds_s = int(meta_info.get("cds_start_pos", -1))
        cds_e = int(meta_info.get("cds_end_pos", -1))
        has_cds = (cds_s != -1 and cds_e != -1)
        
        if not has_cds:
            continue
            
        start_idx = max(0, cds_s - 1)
        end_idx = cds_e + 3
        cds_len = end_idx - start_idx
        
        is_pred_full = abs(pred_len - gt_len) < abs(pred_len - cds_len)
        if not is_pred_full:
            continue
            
        min_len = min(pred_len, gt_len)
        if min_len < 2 or end_idx >= min_len: 
            continue
            
        pred_aligned = pred_linear[:min_len]
        truth_aligned = truth_linear[:min_len]
        
        total_sum_gt = np.sum(truth_aligned)
        total_sum_pred = np.sum(pred_aligned)
        
        regions = {
            '5\'UTR': (0, start_idx),
            'CDS': (start_idx, end_idx),
            '3\'UTR': (end_idx, min_len)
        }
        
        for region_name, (r_start, r_end) in regions.items():
            # [Modified] Label as 'Observation' to match the request
            m_gt = calculate_region_metrics(truth_aligned, start_idx, r_start, r_end, total_sum_gt)
            if m_gt:
                m_gt['Condition'] = 'Observation'
                m_gt['Region'] = region_name
                m_gt['UUID'] = uuid_str
                metrics_data.append(m_gt)
            
            m_pred = calculate_region_metrics(pred_aligned, start_idx, r_start, r_end, total_sum_pred)
            if m_pred:
                m_pred['Condition'] = 'Prediction'
                m_pred['Region'] = region_name
                m_pred['UUID'] = uuid_str
                metrics_data.append(m_pred)

    df = pd.DataFrame(metrics_data)
    if df.empty:
        print("Warning: No valid transcripts found for region evaluation.")
        return df
        
    csv_path = os.path.join(out_dir, f"region_specificity_stats.{suffix}.csv" if suffix else "region_specificity_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n>>> Stats saved to {csv_path}")
    
    # [Modified] Plot the classic ggplot boxplot
    plot_region_comparison(df, out_dir, suffix, width, height)
    
    return df


def plot_region_comparison(df, out_dir, suffix, w=4, h=5):
    """
    [Modified] Plots side-by-side boxplots using plotnine.
    Features classic L-shaped axes (bottom and left borders only via theme_classic).
    """
    print("\n>>> Generating L-shaped Boxplots (ggplot style)...")
    
    # 1. Prepare data (Melt format, omitting Coverage)
    plot_df = df.melt(
        id_vars=['UUID', 'Condition', 'Region'], 
        value_vars=['Proportion', 'Periodicity'],
        var_name='Metric', 
        value_name='Value'
    )
    
    plot_df = plot_df.dropna(subset=['Value'])
    
    # Enforce categorical order
    plot_df['Region'] = pd.Categorical(
        plot_df['Region'], 
        categories=['5\'UTR', 'CDS', '3\'UTR'], 
        ordered=True
    )
    plot_df['Condition'] = pd.Categorical(
        plot_df['Condition'], 
        categories=['Observation', 'Prediction'], 
        ordered=True
    )
    plot_df['Metric'] = pd.Categorical(
        plot_df['Metric'], 
        categories=['Proportion', 'Periodicity'], 
        ordered=True
    )
    
    # 2. Color Palette
    colors = {"Observation": "#A0A0A0", "Prediction": "#2C6B9A"} 
    
    # 3. Plotnine configuration
    p = (
        ggplot(plot_df, aes(x='Region', y='Value', color='Condition'))
        # Boxplot setup
        + geom_boxplot(
            fill='white',               # White inner fill
            size=0.8,                   # Thicker border
            outlier_shape=None,         # Hide outliers
            outlier_alpha=0,            # Hide outliers completely
            width=0.7,                  # Box width
            position=position_dodge(width=0.8) # Side-by-side spacing
        )
        + facet_wrap('~Metric', scales='free_y', nrow=2)
        + scale_color_manual(values=colors)
        + theme_classic() 
        + theme(
            legend_position='top',        # Legend at the top
            legend_title=element_blank(), # Remove legend title
            axis_title_x=element_blank(), # Remove x-axis title
            axis_title_y=element_blank(), # Remove y-axis title
            strip_background=element_blank(), # Remove grey background from facet labels
            strip_text=element_text(weight='bold', size=12), # Bold facet labels
            axis_text=element_text(size=11, color="black"),
            axis_line=element_line(color="black", size=1) # Ensure L-shape lines are solid black
        )
    )
    
    # Save the figure
    save_path = os.path.join(out_dir, f"region_specificity_comparison.{suffix}.pdf")
    p.save(filename=save_path, width=w, height=h, verbose=False)
    
    print(f"Comparison plot saved to {save_path}")