import pickle
import numpy as np
import os
import pandas as pd
from scipy.stats import pearsonr, gaussian_kde
from sklearn.metrics import mean_squared_error
from plotnine import *
import torch 

# --- Auxiliary Function: Calculate 2D Density ---
def calculate_density(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 10:
        return np.zeros_like(x)

    xy = np.vstack([x_clean, y_clean])
    try:
        z = gaussian_kde(xy)(xy)
        density_full = np.zeros_like(x, dtype=float)
        density_full[mask] = z
        return density_full
    except Exception:
        return np.zeros_like(x)


class PredictionVisualizer:
    def __init__(self, pkl_path, dataset, out_dir="./results/plots"):
        """
        Initialize the visualizer by loading prediction data and the ground truth dataset.
        """
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Pickle file not found: {pkl_path}")
            
        print(f"Loading predictions from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            self.preds_data = pickle.load(f)
            
        self.dataset = dataset
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        
        cell_count = len(self.preds_data)
        tid_count = sum(len(tids) for tids in self.preds_data.values())
        print(f"Loaded {tid_count} predictions across {cell_count} cell types. Output dir: {self.out_dir}")

        print("Building dataset UUID index...")
        self.uuid_to_idx = {str(self.dataset[i][0]): i for i in range(len(self.dataset))}
        print("Index built.")

    def plot_transcript(self, tid, cell_type, suffix="", ylim: dict = None, log_y: bool = False):
        """
        Evaluate and plot results for a specific transcript and cell type.
        :param ylim: Dictionary format, e.g., {"Observation": (0, 100), "Prediction": (0, 1.5)}
        """
        if cell_type not in self.preds_data or tid not in self.preds_data[cell_type]:
            print(f"Error: Prediction for TID '{tid}' and Cell Type '{cell_type}' not found in pkl.")
            return
            
        prediction = self.preds_data[cell_type][tid]
        prediction = prediction.reshape(-1).astype(np.float32)

        prefix = f"{tid}-{cell_type}-"
        matched_uuids = [u for u in self.uuid_to_idx.keys() if u.startswith(prefix)]
        
        if not matched_uuids:
            print(f"Error: Cannot find matching UUID starting with '{prefix}' in dataset.")
            return
            
        uuid = matched_uuids[0]
        dataset_idx = self.uuid_to_idx[uuid]
                
        print(f"--- Evaluating {uuid} (TID: {tid}, Cell: {cell_type}) ---")

        sample = self.dataset[dataset_idx]
        meta = sample[4]
        count_emb = sample[6]
        
        if isinstance(count_emb, torch.Tensor):
            truth = count_emb.detach().cpu().numpy()
        else:
            truth = count_emb
            
        if len(truth.shape) > 1 and truth.shape[1] > 1:
            truth = np.sum(truth, axis=1)
        elif len(truth.shape) > 1 and truth.shape[1] == 1:
            truth = truth.flatten()
            
        truth = truth.astype(np.float32)

        min_len = min(len(truth), len(prediction))
        truth = truth[:min_len]
        prediction = prediction[:min_len]

        cds_start = int(meta.get('cds_start_pos', -1))
        cds_end = int(meta.get('cds_end_pos', -1))
        cds_info = {'start': cds_start, 'end': cds_end} if cds_start != -1 else None
        
        valid_mask = np.isfinite(truth) & np.isfinite(prediction)
        if np.sum(valid_mask) > 1:
            pcc, p_val = pearsonr(truth[valid_mask], prediction[valid_mask])
            mse = mean_squared_error(truth[valid_mask], prediction[valid_mask])
        else:
            pcc, p_val, mse = 0, 1, 0

        bar_plot_name = f"{uuid}_psite.{suffix}.pdf"
        self._plot_psite_density_bar_plotnine(
            truth=truth,
            pred=prediction,
            cds_info=cds_info,
            save_path=os.path.join(self.out_dir, bar_plot_name),
            pcc=pcc,   
            p_val=p_val,
            ylim=ylim,
            log_y=log_y
        )

        scatter_plot_name = f"{uuid}_scatter.{suffix}.pdf"
        self._plot_correlation_scatter_plotnine(
            truth=truth,
            pred=prediction,
            pcc=pcc,
            p_val=p_val,
            mse=mse,
            save_path=os.path.join(self.out_dir, scatter_plot_name)
        )

    def _plot_psite_density_bar_plotnine(self, truth, pred, cds_info, save_path, 
                                         pcc=None, p_val=None, ylim=None, log_y=False):
        """
        绘制 P-site 密度图，支持独立的 Y 轴限制字典和对数变换。
        精确匹配参考图的美学风格：经典主题、顶部图例、无Y轴线、严格的X轴边界。
        """
        x = np.arange(len(truth))
        cds_start_idx = (cds_info['start'] - 1) if cds_info else 0
        frames = (x - cds_start_idx) % 3
        
        df_truth = pd.DataFrame({'Pos': x, 'Density': truth, 'Frame': frames, 'Source': 'Observation'})
        df_pred = pd.DataFrame({'Pos': x, 'Density': pred, 'Frame': frames, 'Source': 'Prediction'})
        df = pd.concat([df_truth, df_pred])
        
        # 确保 Category 顺序
        source_categories = ['Observation', 'Prediction']
        df['Frame'] = df['Frame'].astype(str)
        df['Source'] = pd.Categorical(df['Source'], categories=source_categories)

        # ==========================================
        # [FIXED] 强行裁剪数据，让 ylim 在压缩坐标轴时绝对生效
        # ==========================================
        if ylim is not None and isinstance(ylim, dict):
            for src in source_categories:
                if src in ylim:
                    y_min, y_max = ylim[src]
                    mask = df['Source'] == src
                    # 把超出 y_max 的柱子像割草机一样削平
                    df.loc[mask, 'Density'] = df.loc[mask, 'Density'].clip(lower=y_min, upper=y_max)

        soft_colors = {"0": "#D73027", "1": "darkgray", "2": "#4575B4"}

        p = ggplot(df, aes(x='Pos', y='Density', fill='Frame'))

        # 添加 CDS 阴影区域
        if cds_info:
            rect_df = pd.DataFrame({
                'xmin': [cds_info['start']],
                'xmax': [cds_info['end']],
                'ymin': [-np.inf],
                'ymax': [np.inf]
            })
            p += geom_rect(
                data=rect_df,
                mapping=aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax'),
                alpha=0.1,         
                fill='gray', 
                inherit_aes=False
            )
        
        # 利用 geom_blank 实现不同分面的特定范围 (用于撑高坐标轴)
        if ylim is not None and isinstance(ylim, dict):
            blank_data = []
            for src in source_categories:
                if src in ylim:
                    y_min, y_max = ylim[src]
                    blank_data.append({'Source': src, 'Pos': 0, 'Density': y_min})
                    blank_data.append({'Source': src, 'Pos': 0, 'Density': y_max})
                    
            if blank_data:
                blank_df = pd.DataFrame(blank_data)
                blank_df['Source'] = pd.Categorical(blank_df['Source'], categories=source_categories)
                p += geom_blank(data=blank_df, mapping=aes(x='Pos', y='Density'), inherit_aes=False)
        
        p = ( 
            p + geom_col(width=1.0, size=0) 
            + facet_wrap('~Source', ncol=1, scales='free_y') 
            + scale_fill_manual(values=soft_colors, labels=['Frame 0', 'Frame 1', 'Frame 2'])
            + scale_x_continuous(expand=(0, 0), limits=(-0.5, len(truth) - 0.5))
        )
            
        # 添加文本注释 (相关系数，仅保留 R 值并使用斜体)
        if pcc is not None:
            annot_text = f"R = {pcc:.2f}"
            
            max_x = df['Pos'].max()
            
            if ylim is not None and "Prediction" in ylim:
                text_y_pos = ylim["Prediction"][1] * 0.85
            else:
                text_y_pos = df_pred['Density'].max() * 0.85
            
            annot_df = pd.DataFrame({
                'Pos': [max_x * 0.95], # 位置略微偏左一点，避免贴边太紧
                'Density': [text_y_pos], 
                'Source': ['Prediction'], 
                'Label': [annot_text]
            })
            annot_df['Source'] = pd.Categorical(annot_df['Source'], categories=source_categories)
            
            p += geom_text(
                data=annot_df,
                mapping=aes(x='Pos', y='Density', label='Label'),
                ha='right',   
                va='top',     
                size=14, 
                fontstyle='italic', # 斜体 R 以匹配参考图
                color='black',
                inherit_aes=False
            )

        y_axis_label = "Translation signal"
        
        if log_y:
            y_axis_label = "Translation signal (log1p)"
            p += scale_y_continuous(trans='log1p')

        p = (
            p + theme_classic() 
            + theme(
                figure_size=(10, 4),
                
                # --- 图例设置：顶部横向排列 ---
                legend_position="top",
                legend_direction="horizontal",
                legend_title=element_text(size=14, color='black'),
                legend_text=element_text(size=12, color='black'),
                
                # --- 分面标题：无背景，居中大号字体 ---
                strip_background=element_blank(), 
                strip_text=element_text(size=16, color='black'),
                
                # --- Y轴设定：去轴线，留刻度 ---
                axis_line_y=element_blank(),
                axis_ticks_major_y=element_line(color="black", size=0.8),
                
                # --- X轴设定：保留轴线和刻度 ---
                axis_line_x=element_line(color="black", size=0.8),
                axis_ticks_major_x=element_line(color="black", size=0.8),
                
                # --- 网格线：保留灰色主刻度网格线 ---
                panel_grid_major=element_line(color="#E0E0E0", size=0.8, alpha=0.8),
                panel_grid_minor=element_blank(),
                
                # --- 字体设定 ---
                axis_title=element_text(size=14, color='black'),
                axis_text=element_text(size=12, color='black')
            )
            + labs(x="Transcript Position (nt)", y=y_axis_label, fill="Reading frame")
        )

        p.save(save_path, verbose=False)
        print(f"Periodicity plot saved: {save_path}")

    def _plot_correlation_scatter_plotnine(self, truth, pred, pcc, p_val, mse, save_path):
        """
        Plot scatter point correlation colored by point density.
        """
        df = pd.DataFrame({'True': truth, 'Predicted': pred})
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        df['density'] = calculate_density(df['True'].values, df['Predicted'].values)
        df = df.sort_values(by='density')

        p = (
            ggplot(df, aes(x='True', y='Predicted', color='density'))
            + geom_point(alpha=0.7, size=1.5, stroke=0, show_legend=False)
            + scale_color_cmap(cmap_name='magma')
            + theme_classic()
            + theme(
                figure_size=(5, 5),
                axis_ticks_major_y=element_blank(),
                panel_border=element_rect(color="black", fill=None, size=1)
            )
            + labs(
                title=f"Correlation: R={pcc:.3f}, MSE={mse:.3f}",
                x="Observation",
                y="Prediction"
            )
        )

        p.save(save_path, verbose=False)
        print(f"Scatter plot saved: {save_path}")