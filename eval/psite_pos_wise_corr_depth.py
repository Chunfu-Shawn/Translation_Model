import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import *
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from typing import Union, Dict # [MODIFIED] Added for type hinting
    

def load_pickle(path):
    """Helper function to load pickle files."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)

def flatten_counts(counts_dict, length):
    """
    Convert {pos: {read_len: count}} to a numpy array of the specified length.
    """
    arr = np.zeros(length, dtype=np.float32)
    total_reads = 0
    
    if counts_dict is None:
        return arr, 0

    for pos, len_dict in counts_dict.items():
        # Ensure pos is within valid range (1-based to 0-based index)
        if 1 <= pos <= length:
            count_sum = sum(len_dict.values())
            arr[pos - 1] = count_sum
            total_reads += count_sum
    
    return arr, total_reads

def calculate_psite_metrics(data_paths_dict, seq_pkl_path, out_dir, suffix=""):
    """
    Calculate correlation and depth using the actual sequence length for multiple cell types.
    
    Args:
        data_paths_dict: Dictionary with cell types as keys and arrays/lists of 2 paths as values.
                         Format: { 'CellTypeA': ['path1.pkl', 'path2.pkl'], ... }
        seq_pkl_path: Sequence file {tid: sequence_string}
        out_dir: Directory to save the output CSV.
        suffix: Optional suffix for the output filename.
        
    Returns:
        pd.DataFrame: A DataFrame containing all calculated metrics.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Load sequence data once globally for all cell types
    seq_data = load_pickle(seq_pkl_path)
    keys_seq = set(seq_data.keys())
    
    all_results = []

    # Iterate over each cell type and its corresponding datasets
    for cell_type, paths in data_paths_dict.items():
        if len(paths) != 2:
            raise ValueError(f"Expected exactly 2 paths for {cell_type}, but got {len(paths)}.")
        
        print(f"\nLoading data for {cell_type}...")
        data1 = load_pickle(paths[0])
        data2 = load_pickle(paths[1])
        
        # Get intersection of keys for this specific cell type
        keys1 = set(data1.keys())
        keys2 = set(data2.keys())
        common_tids = sorted(list(keys1 & keys2 & keys_seq))
        
        print(f"[{cell_type}] Dataset 1: {len(data1)} transcripts")
        print(f"[{cell_type}] Dataset 2: {len(data2)} transcripts")
        print(f"[{cell_type}] Intersection (Analyzable): {len(common_tids)} transcripts")

        for tid in tqdm(common_tids, desc=f"Comparing {cell_type} transcripts"):
            d1 = data1[tid]
            d2 = data2[tid]
            seq = seq_data[tid] # Get sequence

            # 1. Determine sequence length (using actual length)
            seq_len = len(seq)
            if seq_len == 0: continue

            # 2. Flatten data
            vec1, total1 = flatten_counts(d1, seq_len)
            vec2, total2 = flatten_counts(d2, seq_len)

            # 3. Calculate depth (Total Reads / Length)
            # Calculating the average depth of the two datasets
            depth = (total1 + total2) / 2 / seq_len 

            # 4. Calculate correlation
            with np.errstate(divide='ignore', invalid='ignore'):
                p_r, p_p = pearsonr(vec1, vec2)
                s_r, s_p = spearmanr(vec1, vec2)

            all_results.append({
                'Cell_type': cell_type,
                'Tid': tid,
                'Length': seq_len,
                'Reads_DS1': total1,
                'Reads_DS2': total2,
                'Depth': depth,
                'Pearson_R': p_r,
                'Pearson_P_value': p_p,
                'Spearman_R': s_r,
                'Spearman_P_value': s_p
            })

    df = pd.DataFrame(all_results)
    
    # Remove rows where correlation is NaN (usually due to a vector with 0 standard deviation)
    df = df.dropna(subset=['Pearson_R', 'Spearman_R'])
    
    # Save combined results
    csv_path = os.path.join(out_dir, f"correlation_results.{suffix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nAll results saved to {csv_path}")
    
    return df

def plot_correlation_by_depth(df, out_dir, prefix="comparison", bins=5):
    """
    根据深度分 bin 并绘制相关性分布图。
    Args:
        bins: 
            - int: 使用 pd.qcut 进行等频分箱 (Quantile)
            - list/array: 使用 pd.cut 进行绝对值分箱 (Absolute)
    """
    df_plot = df.copy()

    # 1. 深度分箱逻辑
    try:
        if isinstance(bins, int):
            # 模式 A: 相对分位数 (Quantiles)
            print(f"Binning by {bins} quantiles...")
            df_plot['Depth_Bin_Label'] = pd.qcut(df_plot['Depth'], q=bins)
            # 为了排序方便，生成一个对应的 code
            df_plot['Depth_Bin_Code'] = pd.qcut(df_plot['Depth'], q=bins, labels=False)
            xlabel_text = 'Depth Quantile (Low -> High)'
            
        elif isinstance(bins, (list, tuple, np.ndarray)):
            # 模式 B: 绝对值切割 (Absolute Cutoffs)
            # 例如: [0, 0.5, 1, 5, 10, 100]
            print(f"Binning by absolute values: {bins} ...")
            df_plot['Depth_Bin_Label'] = pd.cut(df_plot['Depth'], bins=bins, include_lowest=True)
            df_plot['Depth_Bin_Code'] = pd.cut(df_plot['Depth'], bins=bins, labels=False, include_lowest=True)
            xlabel_text = 'Read Depth Range (RPKM/Density)'
        else:
            raise TypeError("bins argument must be int or list.")
            
    except ValueError as e:
        print(f"Binning failed: {e}. Usually means not enough unique data points.")
        return

    # 移除分箱产生 NaN 的数据 (比如超出 bins 范围)
    df_plot = df_plot.dropna(subset=['Depth_Bin_Label'])

    # 2. 绘图
    plt.figure(figsize=(14, 6))

    # 定义颜色
    box_props = dict(alpha=0.6)

    # --- Pearson Plot ---
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Depth_Bin_Label', y='Pearson_R', data=df_plot, 
                showfliers=False, palette="Blues", boxprops=box_props)
    plt.title('Pearson Correlation vs Read Depth')
    plt.xlabel(xlabel_text)
    plt.ylabel('Pearson R')
    plt.ylim(-0.2, 1.1)
    plt.xticks(rotation=30, ha='right') # 旋转标签防止重叠
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # --- Spearman Plot ---
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Depth_Bin_Label', y='Spearman_R', data=df_plot, 
                showfliers=False, palette="Greens", boxprops=box_props)
    plt.title('Spearman Correlation vs Read Depth')
    plt.xlabel(xlabel_text)
    plt.ylabel('Spearman R')
    plt.ylim(-0.2, 1.1)
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"psite_depth_correlation.{prefix}.pdf")
    plt.savefig(out_path)
    # plt.show() # 如果在服务器上运行，注释掉这行
    print(f"Plot saved to {out_path}")

    # 打印统计信息
    print("\n=== Summary stats by Depth Bin ===")
    summary = df_plot.groupby('Depth_Bin_Label', observed=True)[['Pearson_R', 'Spearman_R', 'Depth', 'Reads_DS1']].agg(
        {'Pearson_R': 'median', 'Spearman_R': 'median', 'Depth': 'mean', 'Reads_DS1': 'count'}
    ).rename(columns={'Reads_DS1': 'Transcript_Count'})
    print(summary)


def calculate_correlations_multitissue(
    dataset, 
    pkl_input: Union[Dict[str, str], str], 
    output_dir: str = ".", 
    suffix: str = "",
    for_cds: bool = False
):
    """
    Calculate the correlation between predicted Ribo-seq signals and Ground Truth across multiple tissues.
    
    Args:
        dataset: PyTorch Dataset instance.
        pkl_input: Can be either:
                   1) A dictionary mapping cell_type to pickle file paths 
                   2) A single string representing the path to a combined pickle file.
        output_dir: Directory to save the output evaluation results.
        suffix: Optional suffix for the output filename.
        for_cds: If True, restricts evaluation to the CDS region.
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
            
    results = []
    
    print(f"\n>>> Evaluating transcripts in the Dataset (CDS Only: {for_cds})...")
    for i in tqdm(range(len(dataset))):
        uuid, _, cell_type, _, meta_info, _, count_emb = dataset[i]
        uuid_str = str(uuid)
        
        parts = uuid_str.split('-')
        if len(parts) < 2:
            continue
        tid = parts[0]
        
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
        
        # 3. Retrieve signals and flatten
        pred_signal = predictions[lookup_tid]
        gt_signal = count_emb.numpy().flatten()
        
        pred_len = len(pred_signal)
        gt_len = len(gt_signal)
        
        # Get CDS coordinates
        cds_s = int(meta_info.get("cds_start_pos", -1))
        cds_e = int(meta_info.get("cds_end_pos", -1))
        has_cds = (cds_s != -1 and cds_e != -1)
        
        start_idx = max(0, cds_s - 1) if has_cds else 0
        end_idx = cds_e + 3 if has_cds else gt_len
        cds_len = end_idx - start_idx
        
        # =========================================================================
        #  Robust alignment logic for Full-length vs CDS-only predictions
        # =========================================================================
        # Heuristic: Is the prediction full-length or already sliced to CDS?
        if has_cds:
            # If pred_len is closer to gt_len than to cds_len, we assume it's full-length
            is_pred_full = abs(pred_len - gt_len) < abs(pred_len - cds_len)
        else:
            is_pred_full = True # Without CDS info, assume it's full length
            
        if for_cds:
            if not has_cds or cds_len < 6:
                continue # Skip transcripts without valid CDS
                
            gt_target = gt_signal[start_idx:end_idx]
            
            if is_pred_full:
                # Prediction is full length, we need to slice it to match CDS
                safe_end = min(end_idx, pred_len)
                pred_target = pred_signal[start_idx:safe_end]
            else:
                # Prediction is ALREADY CDS-only, DO NOT slice with start_idx
                pred_target = pred_signal
                
        else: # for_cds == False
            if is_pred_full:
                # Both are full-length, compare them directly
                gt_target = gt_signal
                pred_target = pred_signal
            else:
                # [CRITICAL] User didn't request for_cds, but prediction is ONLY CDS!
                # To make biological sense, we MUST align the GT's CDS region to the prediction.
                # Otherwise, the prediction's AUG would align to the GT's 5' UTR start.
                if not has_cds:
                    continue
                gt_target = gt_signal[start_idx:end_idx]
                pred_target = pred_signal
        # =========================================================================
        
        # 4. Align lengths (take the minimum length to prevent out-of-bounds)
        min_len = min(len(pred_target), len(gt_target))
        if min_len < 2: 
            continue
            
        pred_aligned = pred_target[:min_len]
        gt_aligned = gt_target[:min_len]

        # 5. Calculate correlation and handle zero variance edge cases
        if np.std(pred_aligned) == 0 or np.std(gt_aligned) == 0:
            p_r, p_p = np.nan, np.nan
            s_r, s_p = np.nan, np.nan
        else:
            p_r, p_p = pearsonr(pred_aligned, gt_aligned)
            s_r, s_p = spearmanr(pred_aligned, gt_aligned)
            
        depth = float(meta_info.get("rpf_depth", np.nan))
        
        results.append({
            "Tid": tid,
            "Cell_type": cell_type,
            "Depth": depth,
            "Pearson_R": p_r,
            "Pearson_P_Value": p_p,
            "Spearman_R": s_r,
            "Spearman_P_Value": s_p
        })
        
    df = pd.DataFrame(results)
    
    cds_tag = "cds_only" if for_cds else "evaluation_results"
    save_filename = f"psite_corr_{cds_tag}.{suffix}.csv" if suffix else f"psite_corr_{cds_tag}.csv"
    save_path = os.path.join(output_dir, save_filename)
    
    os.makedirs(output_dir, exist_ok=True) 
    df.to_csv(save_path, sep=',', index=False, float_format='%.6g')
    
    print(f"\n>>> Evaluation complete! Successfully matched and calculated {len(df)} transcripts.")
    print(f">>> Results saved to: {save_path}")
    
    return df

def plot_scatter_depth_vs_correlation(df, out_dir, x_col="Depth", y_col="Pearson_R", suffix=".", max_points=10000):
    """
    使用 plotnine 绘制 Depth vs Correlation。
    修复了 float16 indexes are not supported 报错。
    """
    plot_name = f"depth_vs_correlation.{suffix}.pdf" if suffix else "depth_vs_correlation.pdf"
    plot_path = os.path.join(out_dir, plot_name)

    # 1. 数据清洗：Log 坐标轴不能有 <= 0 的值
    # 使用 copy() 避免 SettingWithCopyWarning
    df_plot = df[df[x_col] > 0].copy()
    
    if df_plot.empty:
        print("No positive depth data to plot.")
        return

    # plotnine/matplotlib 处理 float16 计算 scale 时会报错
    float16_cols = df_plot.select_dtypes(include=['float16']).columns
    if len(float16_cols) > 0:
        df_plot[float16_cols] = df_plot[float16_cols].astype('float32')
    
    # 确保 x_col 和 y_col 也是 float32/64 (双重保险)
    df_plot[x_col] = df_plot[x_col].astype('float32')
    df_plot[y_col] = df_plot[y_col].astype('float32')

    # Correlation
    r, p = pearsonr(df_plot[x_col], df_plot[y_col])
    stats_label = (f"Pearson R = {r:.3f} (P={p:.2e})")

    # 2. 抽样 (Downsampling)
    total_points = len(df_plot)
    if total_points > max_points:
        print(f"Downsampling plot data from {total_points} to {max_points} points...")
        df_plot = df_plot.sample(n=max_points, random_state=42)
    
    # 3. 绘图
    try:
        plot = (
            ggplot(df_plot, aes(x=x_col, y=y_col))
            # 散点
            + geom_point(alpha=0.3, color="gray", size=2, stroke=0)
            # 趋势线
            + geom_smooth(method='lm', color="#005b96", size=1.5)
            + annotate("text", x=df_plot[x_col].min(), y=df_plot[y_col].max() * 0.95, 
                    label=stats_label, ha='left', va='top', size=10)
            # 坐标轴：Log10 变换
            + scale_x_log10()
            # 限制 Y 轴范围
            # + coord_cartesian(ylim=(-0.2, 1.05))
            # 主题
            + theme_bw()
            + theme(
                text=element_text(size=12),
                plot_title=element_text( size=14)
            )
            + labs(
                title=f'Correlation vs. Sequencing Depth (n={len(df_plot)})',
                x='Sequencing Depth (Log10 Scale)',
                y=f'{y_col} Coefficient'
            )
        )

        # 4. 保存
        plot.save(plot_path, width=5, height=5, dpi=300, verbose=False)
        print(f"Plot saved to {plot_path}")
        
    except Exception as e:
        print(f"Error saving plot: {e}")
        # 打印一下数据类型以辅助调试
        print("Data types debugging:")
        print(df_plot.dtypes)

##############################
# Correlation and depth
##############################

def load_and_process_comparison_data(
    file1, name1, 
    file2, name2, 
    metric="Pearson_R", 
    target_ratio=None
):
    """
    加载两个 CSV 文件，提取原始数据，并对 Depth 进行 Log10 分箱处理。
    """
    data_list = []
    # 对 Depth 进行分箱 (Binning)
    # 这里的 bins 对应 log10 的值：0(1), 1(10), 2(100), 3(1000), 4(10000)
    # 你可以根据你的数据范围调整 bins 列表
    bins = [-np.inf, -1, -0.301, 0, 0.699, 1, np.inf]
    labels = ['<0.1', '0.1 - 0.5', '0.5 - 1', '1 - 5', '5 - 10', '>10']
    
    # 定义处理单个文件的逻辑
    def process_file(path, label):
        if not os.path.exists(path):
            print(f"[Error] File not found: {path}")
            return None
        
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[Error] Reading {path}: {e}")
            return None

        # 1. 筛选 Mask Ratio
        if target_ratio is not None and 'Mask_Ratio' in df.columns:
            df = df[df['Mask_Ratio'] == target_ratio]
        
        # 2. 检查必要列
        if metric not in df.columns or 'Depth' not in df.columns:
            print(f"[Warning] Missing '{metric}' or 'Depth' in {label}")
            return None
        
        # 3. 提取必要列并清洗
        # 确保没有 NaN 或 Inf，且 Depth > 0
        df = df[[metric, 'Depth']].dropna()
        df = df[df['Depth'] > 0].copy()
        
        # 4. 计算 Log10 Depth
        df['log_depth'] = np.log10(df['Depth'])
        df['Depth_Group'] = pd.cut(df['log_depth'], bins=bins, labels=labels)
        
        # 6. 添加组标签
        df['Source'] = label
        
        return df

    # 处理两个文件
    df1 = process_file(file1, name1)
    df2 = process_file(file2, name2)
    
    if df1 is None or df2 is None:
        return None
        
    # 合并数据
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # 移除分箱后产生 NaN 的行
    combined_df = combined_df.dropna(subset=['Depth_Group'])
    
    # 将 Depth_Group 设为有序分类变量 (保证绘图时从低到高排列)
    # 注意：我们通常希望高 Depth 在上方，或者低 Depth 在上方，可以通过 categories 顺序控制
    # 这里我们让 Depth 小的在下面 (符合直觉)
    combined_df['Depth_Group'] = pd.Categorical(
        combined_df['Depth_Group'], 
        categories=reversed(labels), # reversed 让 <1 在最下面
        ordered=True
    )
    
    return combined_df

def plot_ridge_density_comparison(df, metric="Pearson_R", out_dir="./results"):
    """
    绘制 Ridge Plot 风格的密度对比图。
    """
    if df is None or df.empty:
        print("No data to plot.")
        return

    df['Source'] = pd.Categorical(
        df['Source'], categories=["base_model (Pred. vs Obs.)", "Cross-experiment (Obs.)"], ordered=True
        )
    custom_colors = ["#3498db", "#95a5a6"] # 蓝色, 灰色

    plot = (
        ggplot(df, aes(x=metric, fill='Source', color='Source'))
        # 1. 绘制密度图
        + geom_density(alpha=0.3, size=0.3)
        
        # 2. 分面：按 Depth_Group 分行
        # scales='free_y' 允许每个深度的密度高度不同 (因为样本量可能差异巨大)
        + facet_grid('Depth_Group ~ .', scales='free_y')
        
        # 3. 坐标轴和标尺
        + scale_fill_manual(values=custom_colors)
        + scale_color_manual(values=custom_colors)
        + scale_x_continuous(
            limits=(0, 1), 
            breaks=[0, 0.25, 0.5, 0.75, 1.0],
            expand=[0, 0.01])
        
        # 4. 主题设置 (模拟 Ridge Plot 风格)
        + theme_classic()
        + theme(
            panel_spacing=0, 
            
            # 调整分面标签 (Strip) 的位置和背景
            strip_background=element_blank(),
            
            # 去掉 Y 轴刻度和网格 (Ridge Plot 通常不看具体密度值)
            axis_text_y=element_blank(),
            panel_grid_major_x=element_line(linetype="dashed", color="lightgray"),
            # axis_ticks_y=element_blank(),
            # axis_line_y=element_blank(),
            
            # 保留 X 轴
            axis_line_x=element_line(color="black"),
            
            # 图例位置
            legend_position='top',
            legend_title=element_blank(),

        )
        + labs(
            x=f"Position-wise correlation per transcript ({metric})",
            y="Sequencing Depth (Log10 Bins)"
        )
    )
    
    # 保存
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"ridge_plot_depth_comparison_{metric}.pdf")
    plot.save(save_path, width=5, height=5, dpi=300, verbose=False)
    print(f"Plot saved to: {save_path}")