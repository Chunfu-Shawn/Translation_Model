import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import *
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


class PsiteComparator:
    def __init__(self, pkl_path1, pkl_path2, seq_pkl_path):
        """
        初始化比较器。
        Args:
            pkl_path1: 数据集1 P-site counts {tid: {pos: {len: count}}}
            pkl_path2: 数据集2 P-site counts
            seq_pkl_path: 序列文件 {tid: sequence_string}
        """
        self.data1 = self._load_pickle(pkl_path1)
        self.data2 = self._load_pickle(pkl_path2)
        self.seq_data = self._load_pickle(seq_pkl_path)
        
        # 取三个文件的交集，确保都有数据
        keys1 = set(self.data1.keys())
        keys2 = set(self.data2.keys())
        keys_seq = set(self.seq_data.keys())
        
        self.common_tids = sorted(list(keys1 & keys2 & keys_seq))
        
        print(f"Dataset 1: {len(self.data1)} transcripts")
        print(f"Dataset 2: {len(self.data2)} transcripts")
        print(f"Sequence DB: {len(self.seq_data)} transcripts")
        print(f"Intersection (Analyzable): {len(self.common_tids)} transcripts")

    def _load_pickle(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _flatten_counts(self, counts_dict, length):
        """
        将 {pos: {read_len: count}} 转换为长度为 length 的 numpy 数组。
        """
        arr = np.zeros(length, dtype=np.float32)
        total_reads = 0
        
        if counts_dict is None:
            return arr, 0

        for pos, len_dict in counts_dict.items():
            # 确保 pos 在有效范围内 (1-based to 0-based index)
            if 1 <= pos <= length:
                count_sum = sum(len_dict.values())
                arr[pos - 1] = count_sum
                total_reads += count_sum
        
        return arr, total_reads

    def calculate_metrics(self, out_dir, suffix=""):
        """
        计算相关性和深度，使用真实的序列长度。
        """
        os.makedirs(out_dir, exist_ok=True)
        results = []

        for tid in tqdm(self.common_tids, desc="Comparing transcripts"):
            d1 = self.data1[tid]
            d2 = self.data2[tid]
            seq = self.seq_data[tid] # 获取序列

            # 1. 确定序列长度 (使用真实长度)
            seq_len = len(seq)
            if seq_len == 0: continue

            # 2. 展平数据
            vec1, total1 = self._flatten_counts(d1, seq_len)
            vec2, total2 = self._flatten_counts(d2, seq_len)

            # 3. 计算深度 (Total Reads / Length)
            # 这里计算的是两个数据集的平均深度，或者你可以分别保存
            depth = (total1 + total2) / 2 / seq_len 

            # 4. 计算相关性
            with np.errstate(divide='ignore', invalid='ignore'):
                p_r, p_p = pearsonr(vec1, vec2)
                s_r, s_p = spearmanr(vec1, vec2)

            results.append({
                'Transcript_ID': tid,
                'Length': seq_len,
                'Reads_DS1': total1,
                'Reads_DS2': total2,
                'Depth': depth,
                'Pearson_R': p_r,
                'Pearson_P_value': p_p,
                'Spearman_R': s_r,
                'Spearman_P_value': s_p
            })

        df = pd.DataFrame(results)
        
        # 移除相关性计算为 NaN 的行 (通常是因为某个向量标准差为0，即全0或常数)
        df = df.dropna(subset=['Pearson_R', 'Spearman_R'])
        
        # save
        csv_path = os.path.join(out_dir, f"correlation_results.{suffix}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
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


# --- 1. Correlation Calculation Function (不变) ---
def calculate_transcript_correlation_from_arrays(gt_seq, pred_seq):
    
    if np.any(np.isnan(gt_seq)) or np.any(np.isnan(pred_seq)):
        return np.nan, np.nan, np.nan, np.nan
    try:
        r_p, p_val_p = pearsonr(gt_seq, pred_seq)
        r_s, p_val_s = spearmanr(gt_seq, pred_seq)
    except:
        return np.nan, np.nan, np.nan, np.nan
    
    return r_p, p_val_p, r_s, p_val_s

# --- 2. Visualization Function (Fixed for float16 error) ---

def plot_depth_vs_correlation(df, save_path, x_col="Depth", y_col="Pearson_R", max_points=10000):
    """
    使用 plotnine 绘制 Depth vs Correlation。
    修复了 float16 indexes are not supported 报错。
    """
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
            # 坐标轴：Log10 变换
            + scale_x_log10()
            # 限制 Y 轴范围
            + coord_cartesian(ylim=(-0.2, 1.05))
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
        plot.save(save_path, width=5, height=5, dpi=300, verbose=False)
        print(f"Plot saved to {save_path}")
        
    except Exception as e:
        print(f"Error saving plot: {e}")
        # 打印一下数据类型以辅助调试
        print("Data types debugging:")
        print(df_plot.dtypes)

# --- 3. Main Evaluation Function ---
def evaluate_position_wise_pred_truth_correlation(
        pkl_path, target_cell=None, depth_threshold=0, out_dir="./results", suffix=""):
    
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")
        
    if target_cell is not None:
        if isinstance(target_cell, (float, int)):
            target_cell = [str(target_cell)]
        elif isinstance(target_cell, str):
             target_cell = {target_cell}
        else:
             target_cell = set(target_cell)
        print(f"Targeting specific cell types: {sorted(list(target_cell))}")

    print(f"Loading data from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    os.makedirs(out_dir, exist_ok=True)
    
    all_results = []
    
    print(f"Processing {len(data)} transcripts...")
    
    # Iterate over all transcripts
    for uuid, sample in tqdm(data.items(), desc="Calculating"):
        # 1. Cell Type Filtering
        try:
            cell_type = uuid.split("-")[1]
        except IndexError:
            cell_type = "Unknown"
            
        if target_cell is not None and cell_type not in target_cell:
            continue

        # 2. Data Extraction
        truth = sample['truth'].reshape(-1)
        pred = sample['pred'].reshape(-1)
        
        # Extract Depth
        depth_val = sample.get('depth', np.nan) 
        
        # 过滤深度过低的数据（如果设置了阈值）
        if pd.isna(depth_val) or depth_val < depth_threshold:
            continue
        
        # 3. Calculation
        r_p, p_val_p, r_s, p_val_s = calculate_transcript_correlation_from_arrays(truth, pred)
        
        # 4. Store Result
        if not np.isnan(r_p):
            all_results.append({
                'UUID': uuid,
                'Depth': depth_val,
                'Pearson_R': r_p,
                'Pearson_P_Value': p_val_p,
                'Spearman_R': r_s,
                'Spearman_P_Value': p_val_s
            })
    
    # --- Convert to DataFrame ---
    df = pd.DataFrame(all_results)
    
    if df.empty:
        print("No valid correlations found.")
        return None

    # --- Save CSV ---
    csv_name = f"psite_correlation_results.{suffix}.csv" if suffix else "psite_correlation_results.csv"
    csv_path = os.path.join(out_dir, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"Correlation summary saved to {csv_path}")
    
    # --- Print Summary ---
    print("\n=== Summary Stats ===")
    print(f"Total Transcripts: {len(df)}")
    print(f"Median Pearson R:  {df['Pearson_R'].median():.4f}")
    print(f"Median Spearman R: {df['Spearman_R'].median():.4f}")
    
    # --- Plotting ---
    if df['Depth'].isnull().all():
        print("Warning: 'Depth' data is all NaN. Skipping Depth vs Correlation plot.")
    else:
        plot_name = f"depth_vs_correlation.{suffix}.pdf" if suffix else "depth_vs_correlation.pdf"
        plot_path = os.path.join(out_dir, plot_name)
        # 调用新的 plotnine 绘图函数
        plot_depth_vs_correlation(df, plot_path, x_col="Depth", y_col="Pearson_R", max_points=10000)
    
    return df