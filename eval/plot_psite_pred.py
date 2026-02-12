import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, gaussian_kde
from sklearn.metrics import mean_squared_error

class PredictionVisualizer:
    def __init__(self, pkl_path, out_dir="./results/plots"):
        """
        Initialize the visualizer by loading the pickle file.
        """
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Pickle file not found: {pkl_path}")
            
        print(f"Loading data from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        print(f"Loaded {len(self.data)} transcripts. Output dir: {self.out_dir}")

    def plot_transcript(self, uuid, mask_ratio=None, suffix=""):
        """
        Plot evaluation results for a specific transcript.
        """
        if uuid not in self.data:
            print(f"Error: UUID '{uuid}' not found in data.")
            return

        sample = self.data[uuid]
        truth = sample['truth'].reshape(-1)
        cds_info = sample.get('cds_info', None) # Get CDS info
        
        # Determine mask ratio to plot
        available_pred = list(sample['ratios'].keys())
        if mask_ratio is None:
            mask_ratio = available_pred[0]
            print(f"No mask_ratio specified, using first available: {mask_ratio}")
        elif mask_ratio not in available_pred:
            print(f"Error: Ratio {mask_ratio} not found. Available: {available_pred}")
            return

        # Retrieve prediction data
        ratio_data = sample['ratios'][mask_ratio]
        prediction = ratio_data['pred']
        mask_indices = ratio_data['mask_indices']

        # Determine plot range 
        start, end = 0, len(truth)
        
        print(f"--- Evaluating {uuid} (Ratio: {mask_ratio}) ---")
        
        # 1. Calculate Metrics on Masked Region
        if len(mask_indices) > 0:
            gt_masked = truth[mask_indices]
            pred_masked = prediction[mask_indices]
            
            # Convert float16 to float32 for stable stats calculation
            gt_masked = gt_masked.astype(np.float32)
            pred_masked = pred_masked.astype(np.float32)
            
            pcc, p_val = pearsonr(gt_masked, pred_masked)
            mse = mean_squared_error(gt_masked, pred_masked)
            print(f"Metrics (Masked Region): PCC={pcc:.4f}, MSE={mse:.4f}")
        else:
            pcc, p_val, mse = 0, 1, 0

        # 2. Generate Plots
        # Plot A: Comparison Bar Plot (Periodicity Focused)
        bar_plot_name = f"{uuid}_ratio{mask_ratio}_psite.{suffix}.pdf"
        self._plot_psite_density_bar(
            truth=truth,
            pred=prediction,
            cds_info=cds_info, # Pass CDS info
            save_path=os.path.join(self.out_dir, bar_plot_name),
            start=start,
            end=end
        )

        # Plot B: Scatter Plot
        scatter_plot_name = f"{uuid}_ratio{mask_ratio}_scatter.{suffix}.png"
        self._plot_correlation_scatter(
            truth=truth[mask_indices],
            pred=prediction[mask_indices],
            pcc=pcc,
            p_val=p_val,
            mse=mse,
            save_path=os.path.join(self.out_dir, scatter_plot_name)
        )

    def _plot_psite_density_bar(self, truth, pred, cds_info, save_path, start=0, end=None):
        """
        Plot Ground Truth vs Prediction with Frame Coloring.
        - Bars are colored by frame (0=Red, 1=Green, 2=Blue).
        - CDS region is highlighted with a gray background.
        """
        if end is None: end = len(truth)
        
        # Crop data
        x = np.arange(start, end)
        truth_seg = truth[start:end]
        pred_seg = pred[start:end]
        
        # --- Prepare Frame Colors ---
        # Palette: Frame 0 (Red), Frame 1 (Green), Frame 2 (Blue)
        frame_palette = np.array(['#e74c3c', '#2ecc71', '#3498db']) 
        
        # Calculate frame relative to CDS start
        if cds_info is not None:
            cds_start_idx = cds_info['start'] - 1
            # Calculate frame: (global_index - cds_start) % 3
            frames = (x - cds_start_idx) % 3
        else:
            # Fallback if no CDS info: just mod 3
            frames = x % 3
            
        bar_colors = frame_palette[frames]

        # --- Plotting ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 6), sharex=True)
        
        # Helper to draw CDS background
        def draw_cds_background(ax):
            if cds_info is not None:
                # Calculate intersection between current view (start, end) and CDS range
                bg_start = max(start, cds_info['start'])
                bg_end = min(end, cds_info['end'])
                
                if bg_start < bg_end:
                    ax.axvspan(bg_start, bg_end, color='#ecf0f1', alpha=0.6, zorder=0, label='CDS Region')

        # Subplot 1: Ground Truth
        draw_cds_background(ax1)
        ax1.bar(x, truth_seg, color=bar_colors, linewidth=0, edgecolor="none", width=1.0, zorder=2)
        ax1.set_ylabel("Log1p Count")
        ax1.set_title("Observed (Ground Truth)")
        ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        # Add legend for frames
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ecf0f1', label='CDS Region'),
            Patch(facecolor='#e74c3c', label='Frame 0'),
            Patch(facecolor='#2ecc71', label='Frame 1'),
            Patch(facecolor='#3498db', label='Frame 2'),
        ]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize='small')

        # Subplot 2: Prediction
        draw_cds_background(ax2)
        ax2.bar(x, pred_seg, color=bar_colors, linewidth=0, edgecolor="none", width=1.0, zorder=2)
        ax2.set_ylabel("Log1p Count")
        ax2.set_xlabel("Transcript Position (nt)")
        ax2.set_title("Model Prediction")
        ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Periodicity bar plot saved: {save_path}")

    def _plot_correlation_scatter(self, truth, pred, pcc, p_val, mse, save_path):
        """
        Scatter plot with density estimation (Gaussian KDE).
        """
        truth = truth.astype(np.float32)
        pred = pred.astype(np.float32)
        
        valid = np.isfinite(truth) & np.isfinite(pred)
        x = truth[valid]
        y = pred[valid]
        
        if len(x) < 2: return

        fig, ax = plt.subplots(figsize=(7, 6))
        
        xy = np.vstack([x, y])
        try:
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            scatter = ax.scatter(x, y, c=z, s=15, cmap='Spectral_r', alpha=0.6, edgecolors='none')
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Point Density')
        except:
            ax.scatter(x, y, c='purple', s=15, alpha=0.5, edgecolors='none')

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]), 
            np.max([ax.get_xlim(), ax.get_ylim()]), 
        ]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='y=x')
        
        ax.set_xlabel("True Value (Masked)")
        ax.set_ylabel("Predicted Value")
        ax.set_title(f"Masked Region Correlation\nR={pcc:.3f} (p={p_val:.1e}), MSE={mse:.3f}")
        ax.grid(True, linestyle='--', alpha=0.2)
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Scatter plot saved: {save_path}")