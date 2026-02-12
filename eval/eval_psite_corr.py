import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# --- 1. Correlation Calculation Function (无修改) ---

def calculate_transcript_correlation_from_arrays(gt_seq, pred_seq, mask_pos, threshold=1):
    """
    Calculate Pearson/Spearman correlation for a single transcript on masked positions.
    """
    if len(mask_pos) == 0:
        return np.nan, np.nan, np.nan, np.nan
        
    gt_masked = gt_seq[mask_pos].astype(np.float32)
    pred_masked = pred_seq[mask_pos].astype(np.float32)
    
    if np.expm1(gt_masked).sum() < threshold:
        return np.nan, np.nan, np.nan, np.nan
    
    r_p, p_val_p = pearsonr(gt_masked, pred_masked)
    r_s, p_val_s = spearmanr(gt_masked, pred_masked)
    
    return r_p, p_val_p, r_s, p_val_s

# --- 2. Visualization Function (无修改) ---

def plot_correlation_trend_violin_custom(df, save_path, yvalue="Pearson_R"):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 7))
    
    sns.violinplot(
        x='Mask_Ratio', y=yvalue, data=df, color='white', inner="box",
        linewidth=1.2, width=0.7, cut=0, saturation=0.75, ax=ax
    )
    plt.setp(ax.collections, edgecolor="#34495e")
    
    median_line_length = 0.25
    median_line_width = 2.5
    median_color = '#e74c3c'

    ratios = sorted(df['Mask_Ratio'].unique())
    medians = df.groupby('Mask_Ratio')[yvalue].median()

    for x_coord, ratio in enumerate(ratios):
        if ratio in medians:
            med_val = medians[ratio]
            ax.plot([x_coord - median_line_length, x_coord + median_line_length], 
                    [med_val, med_val], color=median_color, linewidth=median_line_width, 
                    zorder=5, solid_capstyle='round')

    sns.pointplot(
        x='Mask_Ratio', y=yvalue, data=df, estimator=np.median, color=median_color,  
        markers='', linestyles='--', errorbar=None, ax=ax, zorder=4, alpha=0.5
    )

    plt.title('Transcript-wise Prediction Correlation vs. Input Mask Ratio')
    plt.xlabel('Mask Ratio (Input Corruption)')
    plt.ylabel(f'{yvalue} Correlation (R)')
    plt.ylim(-0.1, 1.05) 
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=median_color, lw=2, label='Median (Horizontal Line)'),
        Line2D([0], [0], color=median_color, linestyle='--', label='Median Trend'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', label='Individual Transcripts', markersize=6),
        Line2D([0], [0], color='#34495e', lw=2, label='Density (Violin)')
    ]
    ax.legend(handles=legend_elements, loc='lower left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Violin plot saved to {save_path}")

# --- 3. Main Evaluation Function ---
def evaluate_position_wise_correlation(pkl_path, target_ratios=None, target_cell=None, out_dir="./results", suffix=""):
    """
    Load predictions from pickle, calculate Pearson or Spearman R for all ratios, and plot trend.
    Args:
        target_ratios: (list or float) 指定要分析的 mask_ratio。如果为 None，则分析所有。
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")
        
    if target_ratios is not None:
        if isinstance(target_ratios, (float, int)):
            target_ratios = [target_ratios]
        target_ratios = set(target_ratios)
        print(f"Targeting specific mask ratios: {sorted(list(target_ratios))}")
    if target_cell is not None:
        if isinstance(target_cell, (float, int)):
            target_cell = [target_cell]
        target_cell = set(target_cell)
        print(f"Targeting specific cell types: {sorted(list(target_cell))}")

    print(f"Loading data from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    os.makedirs(out_dir, exist_ok=True)
    
    all_results = []
    
    print(f"Processing {len(data)} transcripts...")
    
    # Iterate over all transcripts
    for uuid, sample in tqdm(data.items(), desc="Calculating Correlations"):
        cell_type = uuid.split("-")[1]
        if target_cell is not None and cell_type not in target_cell:
            continue

        truth = sample['truth'].reshape(-1)
        ratios_dict = sample.get('ratios', None)
        if ratios_dict is None: continue

        # Iterate over all available ratios
        for ratio, ratio_data in ratios_dict.items():
            
            if target_ratios is not None:
                if ratio not in target_ratios:
                    continue

            prediction = ratio_data['pred'].reshape(-1)
            mask_indices = ratio_data['mask_indices']
            
            r_p, p_val_p, r_s, p_val_s = calculate_transcript_correlation_from_arrays(
                truth, prediction, mask_indices, threshold=10 
            )
            
            if not np.isnan(r_p):
                all_results.append({
                    'UUID': uuid,
                    'Mask_Ratio': ratio,
                    'Pearson_R': r_p,
                    'Pearson_P_Value': p_val_p,
                    'Spearman_R': r_s,
                    'Spearman_P_Value': p_val_s
                })
    
    df = pd.DataFrame(all_results)
    
    if df.empty:
        print("No valid correlations found. (Check thresholds or if target_ratios match the data).")
        return None

    csv_name = f"psite_correlation_results.{suffix}.csv" if suffix else "psite_correlation_results.csv"
    csv_path = os.path.join(out_dir, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"Correlation summary saved to {csv_path}")
    
    print("\n=== Pearson or Spearman R Median by Ratio ===")
    print(df.groupby('Mask_Ratio')['Pearson_R'].median())
    print(df.groupby('Mask_Ratio')['Spearman_R'].median())
    
    plot_name = f"psite_correlation_trend.{suffix}.pdf" if suffix else "psite_correlation_trend.pdf"
    plot_path = os.path.join(out_dir, plot_name)
    
    plot_correlation_trend_violin_custom(df, plot_path)
    
    return df