import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# --- 1. Basic Calculation Function ---

def get_frame0_fraction_from_array(data_log, cds_start, cds_end=None, threshold=1):
    """
    Calculate Frame 0 fraction in the CDS region from Numpy Array.
    """
    # Restore Linear Space
    data_linear = np.expm1(data_log)
    
    # Slice CDS region
    end_idx = cds_end if cds_end is not None else len(data_linear)
    end_idx = min(end_idx, len(data_linear))
    
    # Check bounds
    if cds_start >= end_idx:
        return np.nan

    cds_data = data_linear[cds_start:end_idx]
    
    # Truncate to multiple of 3
    n_bases = len(cds_data)
    trim_len = (n_bases // 3) * 3
    
    if trim_len == 0:
        return np.nan
        
    cds_data = cds_data[:trim_len]
    
    # Check total reads threshold
    total_signal = cds_data.sum()
    if total_signal < threshold:
        return np.nan
    
    # Reshape & Calculate
    codons = cds_data.reshape(-1, 3)
    frame_sums = codons.sum(axis=0) 
    
    # Avoid division by zero
    if frame_sums.sum() == 0:
        return np.nan
        
    return frame_sums[0] / frame_sums.sum()

# --- 2. Visualization Function (Customized Violin) ---

def plot_periodicity_trend_complex(df, save_path):
    """
    Complex plot:
    X-axis: Mask Ratio
    Y-axis: Frame 0 Fraction
    Style: 
      - Violin with colored border and WHITE fill.
      - Colored trend lines.
    """
    # Filter data
    plot_df = df[df['Condition'].isin(['Masked truth', 'Prediction'])]
    
    # Set plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 7))
    
    # Define colors
    # Border colors for Violins
    # Gray for Masked Input, Blue for Prediction
    color_map = {'Masked truth': '#7f8c8d', 'Prediction': '#3498db'}
    
    # Trend line colors (Can be same or slightly darker/different)
    trend_colors = {'Masked truth': '#555555', 'Prediction': '#2980b9'}

    # 1. Draw Violin Plot (The base)
    # inner=None to remove internal boxplot
    ax = sns.violinplot(
        x='Mask_Ratio', 
        y='Frame 0 Fraction', 
        hue='Condition', 
        data=plot_df, 
        palette=color_map,
        inner='box',
        linewidth=1.5,
        cut=0,          # Limit range to data bounds
        saturation=0.75
    )

    # Seaborn returns a PolyCollection. We iterate and modify facecolor.
    for collection in ax.collections:
        # Set facecolor to white
        collection.set_facecolor('white')
        # Ensure alpha is 1 (opaque white) to cover grid lines if needed, or <1 for transparecy
        collection.set_alpha(1) 
        # Note: Edgecolor is already set by the 'palette' in violinplot
    
    # 3. Layer: Trend Line (Connecting Medians)
    sns.pointplot(
        x='Mask_Ratio', 
        y='Frame 0 Fraction', 
        hue='Condition', 
        data=plot_df, 
        estimator=np.median, 
        errorbar=None, 
        markers='',
        scale=0.8,
        linestyles=[':', '--'], # Dotted for truth, Solid for Pred
        palette=trend_colors, 
        dodge=0.4,      # Match violin dodge width
        ax=ax,
        zorder=10       # Put on top
    )

    # Reference Line (Random Baseline)
    plt.axhline(0.333, linestyle='--', color='#e74c3c', alpha=0.5, label='Random (0.33)', zorder=0)
    
    # Clean up Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=trend_colors['Masked truth'], linestyle=':', marker='o', label='Masked Input (Median)'),
        Line2D([0], [0], color=trend_colors['Prediction'], linestyle='-', marker='D', label='Prediction (Median)'),
        Line2D([0], [0], color='white', markeredgecolor=color_map['Masked truth'], marker='o', label='Masked Input (Dist)'),
        Line2D([0], [0], color='white', markeredgecolor=color_map['Prediction'], marker='o', label='Prediction (Dist)'),
    ]
    plt.legend(handles=legend_elements, loc='lower left', title='Condition')
    
    plt.title('Periodicity Recovery vs. Input Mask Ratio (CDS Only)')
    plt.xlabel('Mask Ratio')
    plt.ylabel('Frame 0 Fraction')
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Violin plot saved to {save_path}")

# --- 3. Core Evaluation Loop (From Pickle) ---
def evaluate_periodicity(pkl_path, out_dir="./results", suffix=""):
    """
    Load predictions from pickle, calculate Frame 0 Fraction for all ratios, and plot trend.
    Compare: Masked Ground Truth vs. Prediction
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")
        
    print(f"Loading data from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    os.makedirs(out_dir, exist_ok=True)
    
    all_results = []
    
    print(f"Processing {len(data)} transcripts...")
    
    # Iterate over all transcripts
    for uuid, sample in tqdm(data.items(), desc="Calculating Periodicity"):
        # Load Ground Truth
        truth = sample['truth'].reshape(-1).astype(np.float32)
        
        # Get CDS Info
        cds_info = sample.get('cds_info', None)
        if cds_info is None:
            continue
        if cds_info['start'] == -1 or cds_info['end'] == -1:
            continue
            
        # Parse start/end (Assuming 0-based indexing in pickle)
        # If your pickle stores 1-based, adjust here: e.g., cds_info['start'] - 1
        cds_start = cds_info['start'] - 1
        cds_end = cds_info['end']
        
        # Iterate over all available ratios
        # Structure: sample['prediction'][ratio] = {'pred': ..., 'mask_indices': ...}
        for ratio, ratio_data in sample['ratios'].items():
            prediction = ratio_data['pred'].reshape(-1).astype(np.float32)
            mask_indices = ratio_data['mask_indices']
            
            # --- Reconstruct Masked Input (Virtual) ---
            # We don't have the masked input array stored, but we have truth + mask_indices.
            # We can reconstruct it to calculate "Masked Truth F0".
            masked_truth = truth.copy()
            if len(mask_indices) > 0:
                masked_truth[mask_indices] = 0.0 # Simulate masking
            
            # --- Calculate Metrics ---
            
            # 1. Masked Input F0 (To see degradation)
            f0_masked = get_frame0_fraction_from_array(masked_truth, cds_start, cds_end)
            
            # 2. Predicted F0 (To see recovery)
            f0_pred = get_frame0_fraction_from_array(prediction, cds_start, cds_end)
            
            # Record valid results
            if not np.isnan(f0_masked):
                all_results.append({
                    'UUID': uuid, 
                    'Mask_Ratio': ratio, 
                    'Condition': 'Masked truth', 
                    'Frame 0 Fraction': f0_masked
                })
            
            if not np.isnan(f0_pred):
                all_results.append({
                    'UUID': uuid, 
                    'Mask_Ratio': ratio, 
                    'Condition': 'Prediction', 
                    'Frame 0 Fraction': f0_pred
                })

    # --- 4. Save and Plot ---
    df = pd.DataFrame(all_results)
    
    if df.empty:
        print("No valid periodicity data found (check CDS info or signal threshold).")
        return None

    # Save Results CSV
    csv_name = f"periodicity_recovery.{suffix}.csv" if suffix else "periodicity_recovery.csv"
    csv_path = os.path.join(out_dir, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Draw Plot
    plot_name = f"periodicity_trend_violin.{suffix}.pdf" if suffix else "periodicity_trend_violin.pdf"
    save_path = os.path.join(out_dir, plot_name)
    
    plot_periodicity_trend_complex(df, save_path)
    
    return df

# --- Usage Example ---
if __name__ == "__main__":
    # Replace with your actual pickle path
    pkl_file = "./results/predictions_base_model.pkl" 
    
    # Run evaluation
    # evaluate_periodicity_from_pickle(pkl_file, suffix="reanalysis")