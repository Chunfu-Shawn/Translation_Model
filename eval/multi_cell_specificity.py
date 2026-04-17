import os
import pickle
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from tqdm import tqdm
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import *
import warnings

from eval.calculate_te import *

# 忽略警告
warnings.filterwarnings("ignore")

def extract_gt_and_cds_from_dataset(dataset):
    """
    从 TranslationDataset 提取真实序列 (GT) 和 CDS 信息。
    返回结构: { cell_type: { tid: {'gt': [...], 'cds_start': x, 'cds_end': y, 'depth': z} } }
    """
    print("Extracting Ground Truth and CDS info from dataset...")
    info_dict = defaultdict(dict)
    
    # 我们必须完整遍历 dataset，因为需要拿到每个样本的 count_emb
    for i in tqdm(range(len(dataset)), desc="Parsing Dataset"):
        # 调用 dataset 的 __getitem__ 方法获取所有需要的信息
        uuid, cell_type_idx, meta_info, seq_emb, count_emb = dataset[i]
        
        # 解析 uuid 获取 tid 和 cell_type
        # 假设 uuid 格式仍然是 'ENST000001-HeLa-xxx'
        parts = str(uuid).rsplit('-', 2)
        if len(parts) < 2: 
            continue
            
        tid = parts[0]
        cell_type = parts[1]
        
        # 将 tensor 转换为 numpy 一维数组
        if torch.is_tensor(count_emb):
            gt_array = count_emb.numpy().reshape(-1)
        else:
            gt_array = np.array(count_emb).reshape(-1)
            
        info_dict[cell_type][tid] = {
            'gt': gt_array,
            'cds_start': meta_info['cds_start_pos'],
            'cds_end': meta_info['cds_end_pos'],
            'rpf_depth': meta_info.get('rpf_depth', 0) 
        }
        
    print(f"Extraction complete for {len(info_dict)} cell types.")
    return info_dict

class MultiCellEvaluator:
    # 这里的 dataset_info_dict 传入刚才提取的双层字典
    def __init__(self, dataset, pkl_path, min_depth=1.0, min_cells=3):
        
        self.dataset_info = extract_gt_and_cds_from_dataset(dataset)
        self.pkl_path = pkl_path
        self.min_depth = min_depth
        self.min_cells = min_cells
        
        self.grouped_data = self._load_and_group_data()
        self.cell_types = self._get_all_cell_types()
        
        self.transcript_metrics_df = None 
        self.pairwise_data = None
        self._analysis_done = False

    def _load_and_group_data(self):
        print(f"Loading predictions from {self.pkl_path}...")
        with open(self.pkl_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        grouped = defaultdict(dict)
        
        # raw_data 结构为 {cell_type: {tid: predictions}}
        for cell_type, tid_dict in raw_data.items():
            for tid, pred_array in tid_dict.items():
                try:
                    # [修改核心]：检查 dataset 中是否存在对应的 (cell_type, tid)
                    if cell_type not in self.dataset_info or tid not in self.dataset_info[cell_type]:
                        continue
                        
                    ds_info = self.dataset_info[cell_type][tid]
                    
                    # 获取 CDS 和 GT
                    cds_start = ds_info['cds_start']
                    cds_end = ds_info['cds_end']
                    gt_array = ds_info['gt']
                    depth = ds_info.get('rpf_depth', 0)
                    
                    if cds_start < 0 or cds_end < 0:
                        continue
                    
                    grouped[tid][cell_type] = {
                        'pred': np.array(pred_array).reshape(-1),
                        'gt': gt_array, # 从 dataset 中来
                        'cds_start': int(cds_start),
                        'cds_end': int(cds_end),
                        'depth': float(depth)
                    }
                except Exception as e:
                    continue
                    
        print(f"Grouped into {len(grouped)} transcripts.")
        return grouped

    def _get_all_cell_types(self):
        cells = set()
        for t_data in self.grouped_data.values():
            cells.update(t_data.keys())
        return sorted(list(cells))

    # =========================================================
    # 核心方法：One-Pass Analysis
    # 一次性计算所有需要的矩阵，避免重复遍历
    # =========================================================
    def _run_global_analysis(self):
        if self._analysis_done: return

        print("Running global analysis (computing all matrices in one pass)...")
        
        # 容器初始化
        transcript_results = []
        # 使用 defaultdict 存储每对细胞的所有相关性值
        pair_stats = defaultdict(lambda: {'gt': [], 'pred': []})
        
        for tid, cells_dict in tqdm(self.grouped_data.items(), desc="Analyzing transcripts"):
            # 1. 过滤与准备数据
            available_cells = sorted(list(cells_dict.keys()))
            if len(available_cells) < self.min_cells: continue
            
            avg_depth = np.mean([cells_dict[c]['depth'] for c in available_cells])
            if avg_depth < self.min_depth: continue
            
            # 提取有效向量 (过滤方差极小的数据)
            valid_cells = []
            gt_vecs = []
            pred_vecs = []
            
            for c in available_cells:
                g = cells_dict[c]['gt']
                p = cells_dict[c]['pred']
                if np.std(g) > 1e-6 or np.std(p) > 1e-6:
                    valid_cells.append(c)
                    gt_vecs.append(g)
                    pred_vecs.append(p)
            
            if len(valid_cells) < 2: continue # 至少需要2个有效细胞
            
            # 堆叠矩阵 (N_cells x Length)
            gt_stack = np.stack(gt_vecs)
            pred_stack = np.stack(pred_vecs)
            
            # =========================================
            # 计算 1: GT 内部相关性 (GT vs GT)
            # 用途: Heatmap (Upper), Specificity (Bio_Sim)
            # =========================================
            mat_gt = np.corrcoef(gt_stack)
            
            # =========================================
            # 计算 2: Pred 内部相关性 (Pred vs Pred)
            # 用途: Heatmap (Lower)
            # =========================================
            mat_pred = np.corrcoef(pred_stack)
            
            # =========================================
            # 计算 3: Cross Correlation (Pred vs GT)
            # 用途: Specificity (Match vs Mismatch)
            # =========================================
            # 手动计算 Pred[i] vs GT[j] 矩阵
            p_mean = pred_stack.mean(axis=1, keepdims=True)
            g_mean = gt_stack.mean(axis=1, keepdims=True)
            p_centered = pred_stack - p_mean
            g_centered = gt_stack - g_mean
            
            numerator = p_centered @ g_centered.T # (N, L) @ (L, N) -> (N, N)
            p_std = np.sqrt((p_centered**2).sum(axis=1, keepdims=True))
            g_std = np.sqrt((g_centered**2).sum(axis=1, keepdims=True))
            denominator = p_std @ g_std.T
            
            mat_cross = numerator / (denominator + 1e-12)

            # =========================================
            # 数据分流 A: 存入 Transcript Metrics
            # =========================================
            # Bio Similarity (GT矩阵的上三角平均)
            upper_inds = np.triu_indices_from(mat_gt, k=1)
            bio_sim = np.nanmean(mat_gt[upper_inds])
            
            # Specificity (Cross矩阵: 对角线 vs 非对角线)
            r_match = np.nanmean(np.diag(mat_cross))
            
            mask_off = ~np.eye(len(valid_cells), dtype=bool)
            r_mismatch = np.nanmean(mat_cross[mask_off])
            
            transcript_results.append({
                'TID': tid,
                'N_Cells': len(valid_cells),
                'Avg_Depth': avg_depth,
                'Bio_Similarity': bio_sim,
                'R_Match': r_match,
                'R_Mismatch': r_mismatch,
                'Specificity_Score': r_match - r_mismatch
            })
            
            # =========================================
            # 数据分流 B: 存入 Pairwise Stats (用于 Heatmap)
            # =========================================
            n = len(valid_cells)
            for i in range(n):
                for j in range(i+1, n):
                    c1 = valid_cells[i]
                    c2 = valid_cells[j]
                    # 排序 key 保证一致性
                    if c1 > c2: c1, c2 = c2, c1
                    
                    # 收集 GT 相关性
                    if not np.isnan(mat_gt[i, j]):
                        pair_stats[(c1, c2)]['gt'].append(mat_gt[i, j])
                    
                    # 收集 Pred 相关性
                    if not np.isnan(mat_pred[i, j]):
                        pair_stats[(c1, c2)]['pred'].append(mat_pred[i, j])

        # --- 保存结果到 self ---
        self.transcript_metrics_df = pd.DataFrame(transcript_results).dropna()
        # 转换 float16 -> 32
        float16_cols = self.transcript_metrics_df.select_dtypes(include=['float16']).columns
        if len(float16_cols) > 0:
            self.transcript_metrics_df[float16_cols] = self.transcript_metrics_df[float16_cols].astype('float32')
            
        self.pairwise_data = pair_stats
        self._analysis_done = True
        print("Global analysis finished.")

    # ==========================================
    # 接口 1: 获取/保存 Specificity 结果
    # ==========================================
    def evaluate_specificity(self, out_dir="./results"):
        # 确保分析已运行
        self._run_global_analysis()
        
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "specificity_results.csv")
        self.transcript_metrics_df.to_csv(csv_path, index=False)
        print(f"Specificity results saved to {csv_path}")
        return self.transcript_metrics_df

    # ==========================================
    # 接口 2: 获取/保存 Pairwise Heatmap 数据
    # ==========================================
    def compute_pairwise_matrices(self, out_dir="./results"):
        # 确保分析已运行
        self._run_global_analysis()
        
        os.makedirs(out_dir, exist_ok=True)
        
        records = []
        for (c1, c2), val_dict in self.pairwise_data.items():
            # 计算中位数
            gt_arr = np.array(val_dict['gt'], dtype=np.float32)
            pred_arr = np.array(val_dict['pred'], dtype=np.float32)
            
            median_gt = np.nanmedian(gt_arr) if len(gt_arr) > 0 else np.nan
            median_pred = np.nanmedian(pred_arr) if len(pred_arr) > 0 else np.nan
            
            records.append({
                "Cell 1": c1,
                "Cell 2": c2,
                "GT Corr": median_gt,
                "Pred Corr": median_pred
            })
            
        df = pd.DataFrame(records)
        csv_path = os.path.join(out_dir, "cell_type_pairwise_correlation.csv")
        df.to_csv(csv_path, index=False)
        print(f"Pairwise correlation table saved to {csv_path}")
        return df

    # ==========================================
    # 绘图函数保持不变，它们只负责调用上面的接口
    # ==========================================
    def plot_specificity_vs_biosim(self, out_path="specificity_vs_biosim.pdf"):
        """Plot 3: Specificity vs Biological Similarity (Scatter + Smooth)"""
        df = self.evaluate_specificity(out_dir=os.path.dirname(out_path) or ".")
    
        r, p = pearsonr(df['Bio_Similarity'], df['Specificity_Score'])
        stats_label = (f"Pearson R = {r:.3f} (P={p:.2e})")

        p = (
            ggplot(df, aes(x='Bio_Similarity', y='Specificity_Score'))
            + geom_point(alpha=0.3, color="#2d3436", size=2, stroke=0)
            + geom_smooth(method='lm', color="#005b96", size=1)
            # + geom_hline(yintercept=0, linetype="dashed", color="gray")
            + annotate("text", x=df["Bio_Similarity"].min(), y=df['Specificity_Score'].max() * 0.95, 
                    label=stats_label, ha='left', va='top', size=10)
            + labs(x="Biological similarity in Obs.",
                   y="Specificity score of Pred.")
            # + scale_x_reverse() # 翻转X轴，左边是更相似(1.0)，右边是差异更大
            + theme_classic()
            + theme(figure_size=(4, 4))
        )
        p.save(out_path)
        print(f"Saved: {out_path}")

    def plot_specificity_vs_depth(self, out_path="specificity_vs_depth_scatter.pdf"):
        # 只需要调用 evaluate_specificity 即可，它会自动触发 _run_global_analysis
        df = self.evaluate_specificity(out_dir=os.path.dirname(out_path) or ".")
        
        df = df.copy()
        df['Log_Depth'] = np.log1p(df['Avg_Depth'])
        clean_df = df.dropna(subset=['Log_Depth', 'Specificity_Score'])
        
        if len(clean_df) < 2: return

        r, p = pearsonr(clean_df['Log_Depth'], clean_df['Specificity_Score'])
        stats_label = (f"Pearson R = {r:.3f}\n(P={p:.2e})")
        
        p = (
            ggplot(clean_df, aes(x='Log_Depth', y='Specificity_Score'))
            + geom_point(alpha=0.3, color="#2d3436", size=2, stroke=0)
            + geom_smooth(method='lm', color="#005b96", size=1)
            + annotate("text", x=clean_df["Log_Depth"].min(), y=clean_df['Specificity_Score'].max(), 
                       label=stats_label, ha='left', va='top', size=10)
            + labs(x="log(average P-site depth + 1) in Obs.", 
                   y="Specificity score of Pred.")
            + theme_classic()
            + theme(figure_size=(4, 4))
        )
        p.save(out_path)
        print(f"Saved: {out_path}")

    def plot_merged_heatmap(self, out_path="merged_heatmap.pdf"):
        # 调用 compute_pairwise_matrices 获取数据 DataFrame
        pairwise_df = self.compute_pairwise_matrices(out_dir=os.path.dirname(out_path) or ".")
        
        if len(pairwise_df) == 0: return

        # 构建查找表
        lookup_gt = {}
        lookup_pred = {}
        for _, row in pairwise_df.iterrows():
            c1, c2 = row['Cell 1'], row['Cell 2']
            lookup_gt[(c1, c2)] = row['GT Corr']
            lookup_gt[(c2, c1)] = row['GT Corr']
            lookup_pred[(c1, c2)] = row['Pred Corr']
            lookup_pred[(c2, c1)] = row['Pred Corr']

        cells = self.cell_types
        plot_data = []
        
        for i, c1 in enumerate(cells):
            for j, c2 in enumerate(cells):
                if i == j:
                    val = 1.0
                elif i < j: # Upper -> GT
                    val = lookup_gt.get((c1, c2), np.nan)
                else:       # Lower -> Pred
                    val = lookup_pred.get((c1, c2), np.nan)
                
                plot_data.append({'Cell_X': c2, 'Cell_Y': c1, 'Correlation': val})
        
        df_plot = pd.DataFrame(plot_data)
        df_plot['Cell_X'] = pd.Categorical(df_plot['Cell_X'], categories=cells)
        df_plot['Cell_Y'] = pd.Categorical(df_plot['Cell_Y'], categories=list(reversed(cells)))

        p = (
            ggplot(df_plot, aes(x='Cell_X', y='Cell_Y', fill='Correlation'))
            + geom_tile(color="white", size=0.5)
            + geom_text(aes(label='Correlation'), format_string='{:.2f}', size=8)
            + scale_fill_distiller(palette="YlGnBu", direction=-1, limits=(0, 1))
            + labs(title="Obs. correlation (Upper) vs Pred. correlation (Lower)", x="", y="")
            + theme_minimal()
            + theme(axis_text_x=element_text(rotation=45, hjust=1), figure_size=(7, 6), panel_grid=element_blank())
        )
        p.save(out_path)
        print(f"Saved: {out_path}")

    def compute_te_pairwise_matrices(self, out_dir="./results", log_transform=True):
        """
        提取所有转录本的 TE，并计算细胞类型之间的 TE 相关性矩阵
        """
        print("Calculating Transcript TE and pairwise correlations...")
        os.makedirs(out_dir, exist_ok=True)
        
        # 1. 提取所有的 TE 值
        te_records = []
        for tid, cells_dict in self.grouped_data.items():
            for cell, data in cells_dict.items():
                cds_start, cds_end = data['cds_start'], data['cds_end']
                
                gt_te = calculate_morf_mean_signal(data['gt'], cds_start, cds_end)
                pred_te = calculate_morf_mean_signal(data['pred'], cds_start, cds_end)
                
                te_records.append({
                    'TID': tid,
                    'Cell': cell,
                    'GT_TE': gt_te,
                    'Pred_TE': pred_te
                })
        
        te_df = pd.DataFrame(te_records)
        
        # [修复]：强制将 TE 数据转换为 float32，解决 Pandas Cython 底层不支持 float16 导致的 TypeError
        te_df['GT_TE'] = te_df['GT_TE'].astype('float32')
        te_df['Pred_TE'] = te_df['Pred_TE'].astype('float32')
        
        # 顺手保存一下所有转录本的 TE 原始计算值
        te_df.to_csv(os.path.join(out_dir, "transcript_TE_values.csv"), index=False)
        
        # 2. 将长表透视为宽表 (Rows: TID, Columns: Cell Types)
        # [修复 2 防御性编程]：使用 pivot_table 代替 pivot，防止极小概率出现的重复 TID 导致报错
        gt_pivot = te_df.pivot_table(index='TID', columns='Cell', values='GT_TE', aggfunc='mean')
        pred_pivot = te_df.pivot_table(index='TID', columns='Cell', values='Pred_TE', aggfunc='mean')
        
        # 3. 对数转换 (防止极高表达的管家基因主导 Pearson 相关性)
        if log_transform:
            gt_pivot = np.log1p(gt_pivot)
            pred_pivot = np.log1p(pred_pivot)
            
        # 4. 计算相关性矩阵
        gt_corr_matrix = gt_pivot.corr()
        pred_corr_matrix = pred_pivot.corr()
        
        # 5. 格式化并缓存，用于热图和 CSV
        cells = self.cell_types
        records = []
        self.te_pairwise_data = {} 
        
        for i, c1 in enumerate(cells):
            for j, c2 in enumerate(cells):
                # 检查是否在矩阵列中
                if c1 in gt_corr_matrix.columns and c2 in gt_corr_matrix.columns:
                    val_gt = gt_corr_matrix.loc[c1, c2]
                    val_pred = pred_corr_matrix.loc[c1, c2]
                else:
                    val_gt, val_pred = np.nan, np.nan
                    
                # 缓存供绘图使用 (包含双向)
                self.te_pairwise_data[(c1, c2)] = {'GT': val_gt, 'Pred': val_pred}
                
                # 只保留唯一 pair 存入 CSV
                if i < j: 
                    records.append({
                        "Cell 1": c1, "Cell 2": c2,
                        "GT TE Corr": val_gt, "Pred TE Corr": val_pred
                    })
                    
        df_corr = pd.DataFrame(records).dropna()
        csv_path = os.path.join(out_dir, "cell_type_TE_pairwise_correlation.csv")
        df_corr.to_csv(csv_path, index=False)
        print(f"TE Pairwise correlation table saved to {csv_path}")
        
        return df_corr

    def plot_te_merged_heatmap(self, out_path="te_merged_heatmap.pdf"):
        """
        绘制 TE 细胞间相关性的三角热图 (GT 上三角，Pred 下三角)
        """
        # 确保数据已计算
        if not hasattr(self, 'te_pairwise_data') or self.te_pairwise_data is None:
            self.compute_te_pairwise_matrices(out_dir=os.path.dirname(out_path) or ".")
            
        cells = self.cell_types
        plot_data = []
        
        for i, c1 in enumerate(cells):
            for j, c2 in enumerate(cells):
                if i == j:
                    val = 1.0
                elif i < j: # Upper -> GT
                    val = self.te_pairwise_data.get((c1, c2), {}).get('GT', np.nan)
                else:       # Lower -> Pred
                    val = self.te_pairwise_data.get((c1, c2), {}).get('Pred', np.nan)
                    
                plot_data.append({'Cell_X': c2, 'Cell_Y': c1, 'Correlation': val})
                
        df_plot = pd.DataFrame(plot_data)
        df_plot['Cell_X'] = pd.Categorical(df_plot['Cell_X'], categories=cells)
        df_plot['Cell_Y'] = pd.Categorical(df_plot['Cell_Y'], categories=list(reversed(cells)))

        # [贴士] 这里我换了橘红色系 (OrRd) 调色板，以便跟之前 Profile 的蓝绿色区分开
        p = (
            ggplot(df_plot, aes(x='Cell_X', y='Cell_Y', fill='Correlation'))
            + geom_tile(color="white", size=0.5)
            + geom_text(aes(label='Correlation'), format_string='{:.2f}', size=8)
            + scale_fill_distiller(palette="OrRd", direction=-1, limits=(0, 1)) 
            + labs(title="Transcript TE Correlation: Obs. (Upper) vs Pred. (Lower)", x="", y="")
            + theme_minimal()
            + theme(axis_text_x=element_text(rotation=45, hjust=1), figure_size=(7, 6), panel_grid=element_blank())
        )
        p.save(out_path)
        print(f"Saved TE Heatmap: {out_path}")

    # ==========================================
    # 5. 案例研究：Bar Chart (GT) + Line (Pred)
    # ==========================================
    def calculate_regional_specificity(self, out_dir="./results", min_cells=3, min_reads=10):
        os.makedirs(out_dir, exist_ok=True)
        results = []

        print("Calculating regional metrics and specificity...")
        
        for tid, data in tqdm(self.grouped_data.items()):
            cells = list(data.keys())
            if len(cells) < min_cells: continue
            
            ref_data = data[cells[0]]
            cds_start = ref_data['cds_start'] # 0-based
            cds_end = ref_data['cds_end']
            seq_len = len(ref_data['gt'])
            
            len_5utr = cds_start
            len_cds = cds_end - cds_start
            len_3utr = seq_len - cds_end
            
            if len_cds < 3: continue 
            
            tid_metrics = defaultdict(list)
            valid_cell_count = 0
            
            for cell in cells:
                counts = data[cell]['gt'] 
                if np.sum(counts) < min_reads: continue
                
                # [优化] 使用切片求和，无需循环
                sum_5utr = np.sum(counts[:cds_start]) if len_5utr > 0 else 0
                sum_cds  = np.sum(counts[cds_start:cds_end])
                sum_3utr = np.sum(counts[cds_end:]) if len_3utr > 0 else 0
                
                den_5utr = sum_5utr / len_5utr if len_5utr > 0 else 0
                den_cds  = sum_cds / len_cds
                den_3utr = sum_3utr / len_3utr if len_3utr > 0 else 0
                
                # [修改] 增加 eps 防止除零警告
                eps = 1e-6
                r_5 = den_5utr / (den_cds + eps)
                r_3 = den_3utr / (den_cds + eps)
                
                tid_metrics['cds_densities'].append(den_cds)
                tid_metrics['ratios_5utr'].append(r_5)
                tid_metrics['ratios_3utr'].append(r_3)
                valid_cell_count += 1
            
            if valid_cell_count < min_cells: continue
            
            # 记录结果
            results.append({
                'TID': tid,
                'Num_Cells': valid_cell_count,
                'CDS_Length': len_cds,
                'Mean_CDS_Density': np.mean(tid_metrics['cds_densities']),
                'Specificity_CDS': np.std(tid_metrics['cds_densities']),
                'Specificity_Ratio_5UTR': np.std(tid_metrics['ratios_5utr']),
                'Specificity_Ratio_3UTR': np.std(tid_metrics['ratios_3utr'])
            })
            
        df = pd.DataFrame(results)
        out_csv = os.path.join(out_dir, "transcript_regional_specificity.csv")
        df.to_csv(out_csv, index=False)
        return df

    # =========================================================================
    # Part 2: 绘制 Case Study (指定 TID 或 自动搜索)
    # =========================================================================
    def find_and_plot_case_study(self, out_dir="./results", top_k_cells=4, max_len=3000, target_tid=None):
        print("Searching for best case study...")
        
        best_case = None
        best_score = -np.inf

        # [优化] 逻辑简化，不需要先存所有 candidate 再 sort，维护一个 max 即可
        # 如果指定了 target_tid，直接取；否则遍历寻找
        
        iterator = self.grouped_data.items()
        if target_tid:
            if target_tid in self.grouped_data:
                iterator = [(target_tid, self.grouped_data[target_tid])]
            else:
                print(f"Target TID {target_tid} not found.")
                return

        for tid, data in tqdm(iterator, desc="Scanning"):
            cells = list(data.keys())
            if len(cells) < 2: continue
            
            ref_cell = cells[0]
            # gt_len = len(data[ref_cell]['gt'])
            cds_start = data[ref_cell]['cds_start']
            cds_end = data[ref_cell]['cds_end']
            
            # if gt_len > max_len or gt_len < 300: continue

            score = 0
            if target_tid is None:
                # 自动模式：计算 UTR Heterogeneity
                # utr_mask 逻辑：前段 + 后段
                # 直接切片比 boolean mask 快
                
                utr_vecs = []
                for c in cells:
                    g = data[c]['gt']
                    # 拼接 5' 和 3' UTR
                    utr_part = np.concatenate([g[:cds_start], g[cds_end:]])
                    if np.std(utr_part) > 1e-5:
                        utr_vecs.append(utr_part)
                
                if len(utr_vecs) < 2: continue
                
                # 计算 UTR 区域的一致性
                stack = np.stack(utr_vecs)
                corr_mat = np.corrcoef(stack)
                # 取上三角平均相关性
                indices = np.triu_indices_from(corr_mat, k=1)
                if len(indices[0]) == 0: continue
                
                avg_sim = np.nanmean(corr_mat[indices])
                score = 1 - avg_sim # 差异越大分数越高
            else:
                score = 100 

            if score > best_score:
                best_score = score
                best_case = {
                    'tid': tid, 'score': score, 'cells': cells,
                    'cds_start': cds_start, 'cds_end': cds_end
                }

        if not best_case:
            print("No suitable case study found.")
            return

        # 绘图准备
        best_tid = best_case['tid']
        all_cells = best_case['cells']
        cds_start = best_case['cds_start']
        cds_end = best_case['cds_end']
        
        # 选 Top K Depth Cells
        cell_depths = [(c, self.grouped_data[best_tid][c]['depth']) for c in all_cells]
        cell_depths.sort(key=lambda x: x[1], reverse=True)
        selected_cells = [x[0] for x in cell_depths[:top_k_cells]]

        print(f"Selected: {best_tid}, Score: {best_case['score']:.3f}")

        plot_data = []
        data_dict = self.grouped_data[best_tid]
        
        for cell in selected_cells:
            gt = data_dict[cell]['gt']
            pred = data_dict[cell]['pred']
            
            # Normalization
            gt_max = np.max(gt)
            pred_max = np.max(pred)
            # [修改] 防止除以0
            gt_norm = gt / gt_max if gt_max > 1e-6 else gt
            pred_norm = pred / pred_max if pred_max > 1e-6 else pred
            
            indices = np.arange(len(gt))
            
            # 使用简单的 list dict 构造 dataframe 稍微快一点
            df_gt = pd.DataFrame({'Position': indices, 'Value': gt_norm, 'Source': 'Ground Truth', 'Cell': cell})
            df_pred = pd.DataFrame({'Position': indices, 'Value': pred_norm, 'Source': 'Prediction', 'Cell': cell})
            plot_data.extend([df_gt, df_pred])

        df_plot = pd.concat(plot_data)
        df_plot['Cell'] = pd.Categorical(df_plot['Cell'], categories=selected_cells)

        # [修改] 修复 geom_rect 的数据结构和 aes 映射问题
        # 只需要一行数据即可绘制矩形，因为所有 facet 的 xmin/xmax 一样
        # 但是为了 facet_grid 能够正确分面，必须包含 Cell 列，且每个 Cell 都要有数据
        cds_rect_data = pd.DataFrame({
            'xmin': [cds_start] * len(selected_cells),
            'xmax': [cds_end] * len(selected_cells),
            'ymin': [-0.05] * len(selected_cells), 
            'ymax': [0] * len(selected_cells),
            'Cell': selected_cells
        })
        cds_rect_data['Cell'] = pd.Categorical(cds_rect_data['Cell'], categories=selected_cells)

        p = (
            ggplot()
            # [修改] aes 内部不再使用引号引用列名，除非在 mapping=aes() 之外
            + geom_rect(data=cds_rect_data,
                        mapping=aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax'),
                        fill="#f39c12", alpha=1.0) 
            + geom_col(data=df_plot[df_plot['Source'] == 'Ground Truth'], 
                       mapping=aes(x='Position', y='Value'),
                       fill="gray", alpha=1, width=1)
            + geom_line(data=df_plot[df_plot['Source'] == 'Prediction'], 
                        mapping=aes(x='Position', y='Value'),
                        color="#005b96", size=0.3, alpha=0.5)
            + facet_grid('Cell ~ .', scales="free_y")
            + labs(title=f"Case Study: {best_tid}",
                   subtitle="Gray: GT | Blue: Pred | Orange: CDS",
                   x="Position (nt)", y="Norm. Density")
            + theme_classic()
            + theme(
                strip_background=element_rect(fill="#f0f0f0"),
                figure_size=(10, 2.0 * len(selected_cells))
            )
            + coord_cartesian(ylim=(-0.05, 1.05))
        )
        
        out_path = os.path.join(out_dir, f"psite_profile_case_{best_tid}.pdf")
        p.save(out_path)
        print(f"Saved plot to: {out_path}")


    def analyze_high_variance_te_transcripts(self, out_dir="./results", top_k=100, min_cells=3, plot_top_n_cases=5):
        """
        寻找 TE 波动最大的 Top K 转录本，绘制双向层级聚类热图，并画出前 N 个 Case。
        """
        print(f"Finding Top {top_k} transcripts with highest TE variance across cells...")
        os.makedirs(out_dir, exist_ok=True)
        
        # 1. 计算所有满足条件转录本的 TE
        te_records = []
        for tid, cells_dict in self.grouped_data.items():
            if len(cells_dict) < min_cells: continue
            
            for cell, data in cells_dict.items():
                te = calculate_morf_mean_signal(data['pred'], data['cds_start'], data['cds_end'])
                te_records.append({'TID': tid, 'Cell': cell, 'TE': te})
                
        if not te_records:
            print("No valid transcripts found based on min_cells threshold.")
            return
            
        te_df = pd.DataFrame(te_records)
        
        # 2. 透视表 (Index: TID, Columns: Cell) 
        # 缺失的数据 (某细胞未表达) 填充为 0
        pivot_df = te_df.pivot_table(index='TID', columns='Cell', values='TE').fillna(0)
        
        # 3. 计算标准差 (Std) 作为差异指标
        pivot_df['TE_Std'] = pivot_df.std(axis=1)
        pivot_df['TE_Mean'] = pivot_df.mean(axis=1)
        pivot_df['TE_CV'] = pivot_df['TE_Std'] / (pivot_df['TE_Mean'] + 1e-6) # 变异系数
        
        # 4. 提取 Top K (按标准差降序)
        top_k_df = pivot_df.sort_values(by='TE_Std', ascending=False).head(top_k)
        
        # 保存 Top K 表格
        csv_path = os.path.join(out_dir, f"top_{top_k}_variable_TE_transcripts.csv")
        top_k_df.reset_index().to_csv(csv_path, index=False)
        print(f"Saved Top {top_k} list to {csv_path}")
        
        # 5. 绘制双向层级聚类热图 (Clustermap)
        # 剔除统计列，只保留表达矩阵
        heatmap_matrix = top_k_df.drop(columns=['TE_Std', 'TE_Mean', 'TE_CV'])
        
        # 对数转换以优化可视化效果 (防止极值破坏颜色映射)
        plot_matrix = np.log1p(heatmap_matrix)
        
        # 使用 seaborn 绘制聚类热图
        plt.figure(figsize=(10, 12))
        g = sns.clustermap(
            plot_matrix, 
            cmap="YlOrRd",           # 红黄色系，适合表达量
            method='ward',           # 聚类方法
            metric='euclidean',      # 距离度量
            figsize=(12, min(20, 4 + 0.15 * top_k)), # 根据基因数动态调整高度
            yticklabels=True, 
            xticklabels=True
        )
        g.fig.suptitle(f"Hierarchical Clustering of Top {top_k} Variable TEs (log1p)", y=1.02, fontsize=14)
        g.ax_heatmap.set_ylabel("Transcripts")
        g.ax_heatmap.set_xlabel("Cell Types")
        
        clustermap_path = os.path.join(out_dir, f"top_{top_k}_te_clustermap.pdf")
        g.savefig(clustermap_path, bbox_inches='tight')
        plt.close()
        print(f"Saved Clustermap to {clustermap_path}")
        
        # 6. 为排名最靠前的 N 个转录本绘制 Profile Case Study
        top_tids = top_k_df.index.tolist()[:plot_top_n_cases]
        print(f"Plotting case studies for the top {plot_top_n_cases} transcripts...")
        for tid in top_tids:
            self.plot_case_study(target_tid=tid, out_dir=out_dir, top_k_cells=min_cells)


    # =========================================================================
    # 重构: 取消寻找逻辑，直接绘图
    # =========================================================================
    def plot_case_study(self, target_tid, out_dir="./results", top_k_cells=4):
        """
        绘制指定 target_tid 的 P-site 分布。取消了之前复杂的扫描逻辑。
        """
        if target_tid not in self.grouped_data:
            print(f"Target TID {target_tid} not found in grouped data.")
            return

        os.makedirs(out_dir, exist_ok=True)
        data_dict = self.grouped_data[target_tid]
        all_cells = list(data_dict.keys())
        
        if len(all_cells) == 0:
            return
            
        # 提取 CDS 信息 (假设同一 TID 在不同细胞中一致)
        ref_cell = all_cells[0]
        cds_start = data_dict[ref_cell]['cds_start']
        cds_end = data_dict[ref_cell]['cds_end']
        
        # 筛选测序深度 Top K 的细胞用于展示，避免画出全空白的子图
        cell_depths = [(c, data_dict[c]['depth']) for c in all_cells]
        cell_depths.sort(key=lambda x: x[1], reverse=True)
        selected_cells = [x[0] for x in cell_depths[:top_k_cells]]

        plot_data = []
        for cell in selected_cells:
            gt = data_dict[cell]['gt']
            pred = data_dict[cell]['pred']
            
            # # Normalization (Max Norm)
            # gt_max = np.max(gt)
            # pred_max = np.max(pred)
            # gt_norm = gt / gt_max if gt_max > 1e-6 else gt
            # pred_norm = pred / pred_max if pred_max > 1e-6 else pred
            
            indices = np.arange(len(gt))
            
            df_gt = pd.DataFrame({'Position': indices, 'Value': gt, 'Source': 'Ground Truth', 'Cell': cell})
            df_pred = pd.DataFrame({'Position': indices, 'Value': pred, 'Source': 'Prediction', 'Cell': cell})
            plot_data.extend([df_gt, df_pred])

        df_plot = pd.concat(plot_data)
        df_plot['Cell'] = pd.Categorical(df_plot['Cell'], categories=selected_cells)

        # 构建底部的 CDS 指示条
        cds_rect_data = pd.DataFrame({
            'xmin': [cds_start] * len(selected_cells),
            'xmax': [cds_end] * len(selected_cells),
            'ymin': [-0.05] * len(selected_cells), 
            'ymax': [0] * len(selected_cells),
            'Cell': selected_cells
        })
        cds_rect_data['Cell'] = pd.Categorical(cds_rect_data['Cell'], categories=selected_cells)

        p = (
            ggplot()
            + geom_rect(data=cds_rect_data,
                        mapping=aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax'),
                        fill="#f39c12", alpha=1.0) 
            + geom_col(data=df_plot[df_plot['Source'] == 'Ground Truth'], 
                       mapping=aes(x='Position', y='Value'),
                       fill="gray", alpha=0.4, width=1)
            + geom_line(data=df_plot[df_plot['Source'] == 'Prediction'], 
                        mapping=aes(x='Position', y='Value'),
                        color="#005b96", size=0.3, alpha=1)
            + facet_grid('Cell ~ .', scales="free_y")
            + labs(title=f"Case Study: {target_tid}",
                   subtitle="Gray: GT | Blue: Pred | Orange: CDS",
                   x="Position (nt)", y="Translation profile")
            + theme_classic()
            + theme(
                strip_background=element_rect(fill="#f0f0f0"),
                figure_size=(10, 1.5 * len(selected_cells))
            )
        )
        
        out_path = os.path.join(out_dir, f"psite_profile_case_{target_tid}.pdf")
        p.save(out_path, verbose=False) # 关闭单独打印每个图的保存日志，避免刷屏