import pickle
import numpy as np
import os
import pandas as pd
from scipy.stats import pearsonr, gaussian_kde
from sklearn.metrics import mean_squared_error
from plotnine import *
import torch 

# --- 辅助函数：计算二维核密度 ---
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
        初始化可视化工具，加载预测数据和真实数据集。
        """
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Pickle file not found: {pkl_path}")
            
        print(f"Loading predictions from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            self.preds_data = pickle.load(f)
            
        self.dataset = dataset
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        
        # 统计加载了多少个 cell type 和 tid
        cell_count = len(self.preds_data)
        tid_count = sum(len(tids) for tids in self.preds_data.values())
        print(f"Loaded {tid_count} predictions across {cell_count} cell types. Output dir: {self.out_dir}")

        print("Building dataset UUID index...")
        self.uuid_to_idx = {str(self.dataset[i][0]): i for i in range(len(self.dataset))}
        print("Index built.")

    def plot_transcript(self, tid, cell_type, suffix="", unlog_data=True):
        """
        针对特定转录本和细胞类型绘制评估结果。
        """
        # ==========================================
        # 【修改 1】: 从双层字典结构中提取预测信号
        # ==========================================
        if cell_type not in self.preds_data or tid not in self.preds_data[cell_type]:
            print(f"Error: Prediction for TID '{tid}' and Cell Type '{cell_type}' not found in pkl.")
            return
            
        prediction = self.preds_data[cell_type][tid]
        prediction = prediction.reshape(-1).astype(np.float32)

        # ==========================================
        # 【修改 2】: 匹配 Dataset 中三段式的 UUID (tid-cell_type-order)
        # ==========================================
        prefix = f"{tid}-{cell_type}-"
        matched_uuids = [u for u in self.uuid_to_idx.keys() if u.startswith(prefix)]
        
        if not matched_uuids:
            print(f"Error: Cannot find matching UUID starting with '{prefix}' in dataset.")
            return
            
        # 默认取第一个匹配到的 order
        uuid = matched_uuids[0]
        dataset_idx = self.uuid_to_idx[uuid]
                
        print(f"--- Evaluating {uuid} (TID: {tid}, Cell: {cell_type}) ---")

        # 从 Dataset 获取 Ground Truth 和 Meta Info
        sample = self.dataset[dataset_idx]
        meta = sample[2]
        count_emb = sample[4]
        
        if isinstance(count_emb, torch.Tensor):
            truth = count_emb.detach().cpu().numpy()
        else:
            truth = count_emb
            
        if len(truth.shape) > 1 and truth.shape[1] > 1:
            truth = np.sum(truth, axis=1)
        elif len(truth.shape) > 1 and truth.shape[1] == 1:
            truth = truth.flatten()
            
        truth = truth.astype(np.float32)

        if unlog_data:
            truth = np.expm1(truth)
            prediction = np.expm1(prediction)
            
        min_len = min(len(truth), len(prediction))
        truth = truth[:min_len]
        prediction = prediction[:min_len]

        cds_start = int(meta.get('cds_start_pos', -1))
        cds_end = int(meta.get('cds_end_pos', -1))
        cds_info = {'start': cds_start, 'end': cds_end} if cds_start != -1 else None
        
        # 计算全长相关性指标
        valid_mask = np.isfinite(truth) & np.isfinite(prediction)
        if np.sum(valid_mask) > 1:
            pcc, p_val = pearsonr(truth[valid_mask], prediction[valid_mask])
            mse = mean_squared_error(truth[valid_mask], prediction[valid_mask])
        else:
            pcc, p_val, mse = 0, 1, 0

        # 生成图表
        bar_plot_name = f"{uuid}_psite.{suffix}.pdf"
        self._plot_psite_density_bar_plotnine(
            truth=truth,
            pred=prediction,
            cds_info=cds_info,
            save_path=os.path.join(self.out_dir, bar_plot_name),
            pcc=pcc,   
            p_val=p_val
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

    def _plot_psite_density_bar_plotnine(self, truth, pred, cds_info, save_path, pcc=None, p_val=None):
        """
        绘制 P-site 密度图，并在 Prediction 分面的右上角标记 R 和 P 值。
        """
        x = np.arange(len(truth))
        cds_start_idx = (cds_info['start'] - 1) if cds_info else 0
        frames = (x - cds_start_idx) % 3
        
        df_truth = pd.DataFrame({'Pos': x, 'Density': truth, 'Frame': frames, 'Source': 'Ground Truth'})
        df_pred = pd.DataFrame({'Pos': x, 'Density': pred, 'Frame': frames, 'Source': 'Prediction'})
        df = pd.concat([df_truth, df_pred])
        
        df['Frame'] = df['Frame'].astype(str)
        df['Source'] = pd.Categorical(df['Source'], categories=['Ground Truth', 'Prediction'])

        soft_colors = {"0": "#D73027", "1": "darkgray", "2": "#4575B4"}

        p = (
            ggplot(df, aes(x='Pos', y='Density', fill='Frame'))
        )

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
                alpha=0.1,          # 0.2 的透明度现在会真实生效
                fill='gray', 
                inherit_aes=False
            )
        
        p = ( 
            p + geom_col(width=1.0, size=0) 
            + facet_wrap('~Source', ncol=1, scales='free_y')
            + scale_fill_manual(values=soft_colors, labels=['Frame 0', 'Frame 1', 'Frame 2'])
        )
            
        if pcc is not None and p_val is not None:
            p_text = f"{p_val:.1e}" if p_val < 0.001 else f"{p_val:.3f}"
            annot_text = f"R = {pcc:.2f}\nP = {p_text}"
            
            # 获取 X 轴的最大值，以及 Prediction 分面 Y 轴的最大密度
            max_x = df['Pos'].max()
            max_y_pred = df_pred['Density'].max()
            
            annot_df = pd.DataFrame({
                # 乘数决定了距离边框的远近，0.98 表示在 98% 的位置，留出 2% 空隙
                'Pos': [max_x * 0.98], 
                'Density': [max_y_pred * 0.90], # 留出 10% 的顶部空隙
                'Source': ['Prediction'], 
                'Label': [annot_text]
            })
            annot_df['Source'] = pd.Categorical(annot_df['Source'], categories=['Ground Truth', 'Prediction'])
            
            p += geom_text(
                data=annot_df,
                mapping=aes(x='Pos', y='Density', label='Label'),
                ha='right',   
                va='top',     
                size=12, 
                color='black',
                inherit_aes=False
            )

        p = (
            p + theme_bw() 
            + theme(
                figure_size=(12, 4),
                strip_background=element_blank(), 
                strip_text=element_text(size=12)
            )
            + labs(x="Transcript Position (nt)", y="P-site Density", fill="Reading Frame")
        )

        p.save(save_path, verbose=False)
        print(f"Periodicity plot saved: {save_path}")

    def _plot_correlation_scatter_plotnine(self, truth, pred, pcc, p_val, mse, save_path):
        """
        绘制真实的散点图，通过点密度着色
        """
        df = pd.DataFrame({'True': truth, 'Predicted': pred})
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        df['density'] = calculate_density(df['True'].values, df['Predicted'].values)
        df = df.sort_values(by='density')

        p = (
            ggplot(df, aes(x='True', y='Predicted', color='density'))
            + geom_point(alpha=0.7, size=1.5, stroke=0, show_legend=False)
            + scale_color_cmap(cmap_name='magma')
            # + geom_abline(slope=1, intercept=0, linetype='dashed', color='red', size=0.8)
            + theme_bw()
            + theme(
                figure_size=(5, 5),
                axis_ticks_major_y=element_blank(),
                panel_border=element_rect(color="black", fill=None, size=1)
            )
            + labs(
                title=f"Correlation: R={pcc:.3f}, MSE={mse:.3f}",
                x="Ground Truth Density",
                y="Predicted Density"
            )
        )

        p.save(save_path, verbose=False)
        print(f"Scatter plot saved: {save_path}")