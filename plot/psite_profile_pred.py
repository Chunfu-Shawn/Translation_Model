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
        
        cell_count = len(self.preds_data)
        tid_count = sum(len(tids) for tids in self.preds_data.values())
        print(f"Loaded {tid_count} predictions across {cell_count} cell types. Output dir: {self.out_dir}")

        print("Building dataset UUID index...")
        self.uuid_to_idx = {str(self.dataset[i][0]): i for i in range(len(self.dataset))}
        print("Index built.")

    # ==========================================
    # [MODIFIED] 修改 ylim 的类型提示和默认行为
    # ==========================================
    def plot_transcript(self, tid, cell_type, suffix="", unlog_data=True, 
                        ylim: dict = None, log_y: bool = False):
        """
        针对特定转录本和细胞类型绘制评估结果。
        :param ylim: 字典格式，例如 {"Ground Truth": (0, 100), "Prediction": (0, 1.5)}
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
        meta = sample[3]
        count_emb = sample[5]
        
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
            ylim=ylim,    # 传递字典
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

    # ==========================================
    # [MODIFIED] 实现独立分面的 ylim 控制
    # ==========================================
    def _plot_psite_density_bar_plotnine(self, truth, pred, cds_info, save_path, 
                                         pcc=None, p_val=None, ylim=None, log_y=False):
        """
        绘制 P-site 密度图，支持独立的 Y 轴限制字典和对数变换。
        """
        x = np.arange(len(truth))
        cds_start_idx = (cds_info['start'] - 1) if cds_info else 0
        frames = (x - cds_start_idx) % 3
        
        df_truth = pd.DataFrame({'Pos': x, 'Density': truth, 'Frame': frames, 'Source': 'Ground Truth'})
        df_pred = pd.DataFrame({'Pos': x, 'Density': pred, 'Frame': frames, 'Source': 'Prediction'})
        df = pd.concat([df_truth, df_pred])
        
        # 确保 Category 顺序
        source_categories = ['Ground Truth', 'Prediction']
        df['Frame'] = df['Frame'].astype(str)
        df['Source'] = pd.Categorical(df['Source'], categories=source_categories)

        soft_colors = {"0": "#D73027", "1": "darkgray", "2": "#4575B4"}

        p = (
            ggplot(df, aes(x='Pos', y='Density', fill='Frame'))
        )

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
        
        # ==========================================
        # [NEW] 利用 geom_blank 实现不同分面的特定范围
        # ==========================================
        # 为了让每个分面有不同的上下限，我们必须使用 scales='free_y'。
        # 但是自由缩放是根据数据来的，怎么强加我们指定的边界呢？
        # 技巧：我们创建一些透明的“边界点”（geom_blank），把每个分面的坐标轴“撑开”到指定的范围。
        if ylim is not None and isinstance(ylim, dict):
            blank_data = []
            for src in source_categories:
                if src in ylim:
                    y_min, y_max = ylim[src]
                    # 添加下限占位符
                    blank_data.append({'Source': src, 'Pos': 0, 'Density': y_min})
                    # 添加上限占位符
                    blank_data.append({'Source': src, 'Pos': 0, 'Density': y_max})
                    
            if blank_data:
                blank_df = pd.DataFrame(blank_data)
                blank_df['Source'] = pd.Categorical(blank_df['Source'], categories=source_categories)
                # geom_blank 专门用于隐形扩展坐标轴
                p += geom_blank(data=blank_df, mapping=aes(x='Pos', y='Density'), inherit_aes=False)
        
        # 因为用 geom_blank 撑起了范围，所以分面必须允许自由缩放
        p = ( 
            p + geom_col(width=1.0, size=0) 
            + facet_wrap('~Source', ncol=1, scales='free_y') 
            + scale_fill_manual(values=soft_colors, labels=['Frame 0', 'Frame 1', 'Frame 2'])
        )
            
        # 添加文本注释 (相关系数)
        if pcc is not None and p_val is not None:
            p_text = f"{p_val:.1e}" if p_val < 0.001 else f"{p_val:.3f}"
            annot_text = f"R = {pcc:.2f}\nP = {p_text}"
            
            max_x = df['Pos'].max()
            
            # [MODIFIED] 智能决定文本高度
            if ylim is not None and "Prediction" in ylim:
                # 如果用户指定了 Prediction 的上限，基于上限决定高度
                text_y_pos = ylim["Prediction"][1] * 0.85
            else:
                # 否则基于真实数据的最高点
                text_y_pos = df_pred['Density'].max() * 0.90
            
            annot_df = pd.DataFrame({
                'Pos': [max_x * 0.98], 
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
                size=12, 
                color='black',
                inherit_aes=False
            )

        # ==========================================
        # [MODIFIED] 坐标轴对数变换
        # ==========================================
        y_axis_label = "P-site score"
        
        # 注意：在使用 scales='free_y' 和 geom_blank 的技巧后，
        # 不能再使用 scale_y_continuous(limits=...)，这会破坏独立分面！
        # 如果需要 log 变换，只传 trans 参数即可。
        if log_y:
            y_axis_label = "P-site score (log1p)"
            p += scale_y_continuous(trans='log1p')

        p = (
            p + theme_bw() 
            + theme(
                figure_size=(12, 4),
                strip_background=element_blank(), 
                strip_text=element_text(size=12)
            )
            + labs(x="Transcript Position (nt)", y=y_axis_label, fill="Reading Frame")
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