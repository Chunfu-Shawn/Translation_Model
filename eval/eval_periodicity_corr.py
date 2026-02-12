import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

# --- 1. 基础计算函数 (保持不变) ---

def get_frame0_fraction_from_array(data_linear, cds_start, cds_end=None, threshold=1):
    """
    计算 CDS 区域的 Frame 0 比例 (基于 Numpy 数组)。
    """
    # 截取 CDS 区域
    end_idx = cds_end if cds_end is not None else len(data_linear)
    end_idx = min(end_idx, len(data_linear))
    
    # 简单的边界检查
    if cds_start >= end_idx: return np.nan

    cds_data = data_linear[cds_start:end_idx]
    
    # 截断到 3 的倍数
    n_bases = len(cds_data)
    trim_len = (n_bases // 3) * 3
    if trim_len == 0: return np.nan
        
    cds_data = cds_data[:trim_len]
    
    # 计算 Sum (过滤极低覆盖度)
    total_density = cds_data.sum()
    if total_density < threshold:
        return np.nan
    
    # Reshape & Calculate
    codons = cds_data.reshape(-1, 3)
    frame_sums = codons.sum(axis=0) 
    
    # 防止分母为0
    if frame_sums.sum() == 0: return 0.0

    return frame_sums[0] / frame_sums.sum()

# --- 2. 绘图函数 (修改为动态布局) ---

def plot_dynamic_scatters(df, correlations, save_path):
    """
    动态绘制散点图，根据实际的 Mask Ratio 数量自动调整子图数量。
    X轴: Prediction, Y轴: Ground Truth
    """
    ratios = sorted(df['Mask_Ratio'].unique())
    n_plots = len(ratios)
    
    if n_plots == 0:
        print("No data to plot.")
        return

    # 动态设置画布大小：每个子图宽 3.5 英寸
    fig_width = max(4, n_plots * 3.5)
    fig, axes = plt.subplots(1, n_plots, figsize=(fig_width, 3.5), sharey=True, sharex=True)
    
    # 如果只有一个子图，axes 不是列表，需要转换以便统一处理
    if n_plots == 1:
        axes = [axes]
    
    # 颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, n_plots))

    for i, ratio in enumerate(ratios):
        ax = axes[i]
        subset = df[df['Mask_Ratio'] == ratio]
        
        # 获取相关性
        r_val = correlations.get(ratio, np.nan)
        
        # 绘制散点
        ax.scatter(subset['Prediction'], subset['Ground Truth'], 
                   alpha=0.4, s=15, color=colors[i], edgecolors='white', linewidth=0.3)
        
        # 辅助线
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, zorder=0)
        ax.axhline(0.33, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(0.33, color='gray', linestyle=':', alpha=0.3)
        
        # 标题和标签
        ax.set_title(f"Ratio {ratio}\nR = {r_val:.3f}", fontsize=11)
        ax.set_xlabel("Prediction F0")
        if i == 0:
            ax.set_ylabel("Ground Truth F0")
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Scatter plots saved to {save_path}")

def plot_correlation_trend(corr_df, save_path):
    """
    画折线图：X轴=Mask Ratio, Y轴=Pearson R
    """
    if len(corr_df) < 2:
        print("Not enough points to plot trend line (need at least 2 ratios). Skipping trend plot.")
        return

    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    
    plt.plot(corr_df['Mask_Ratio'], corr_df['Pearson_R'], 
             marker='o', markersize=8, linewidth=2, color='#2c3e50', label='Pearson R')
    
    for x, y in zip(corr_df['Mask_Ratio'], corr_df['Pearson_R']):
        plt.text(x, y + 0.01, f"{y:.2f}", ha='center', fontsize=9)

    plt.title("Correlation Trend: Periodicity Prediction vs. Mask Ratio")
    plt.xlabel("Mask Ratio")
    plt.ylabel("Pearson Correlation Coefficient (R)")
    plt.ylim(0, 1.05)
    
    ratios = sorted(corr_df['Mask_Ratio'].unique())
    plt.xticks(ratios)
    
    plt.savefig(save_path)
    plt.close()
    print(f"Correlation trend plot saved to {save_path}")

# --- 3. 核心分析逻辑 (修改后) ---

def evaluate_periodicity_correlation(pkl_path, out_dir="./results/plots", suffix="", target_ratios=None):
    """
    从 Pickle 文件加载预测结果，计算周期性相关性并绘图。
    
    Args:
        pkl_path: 预测结果 pickle 文件路径
        out_dir: 图片保存路径
        suffix: 文件名后缀
        target_ratios: (list or float, optional) 指定要分析的 mask_ratio。
                       如果为 None，则分析所有。例如: [0.5, 0.75]
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")
        
    # 处理 target_ratios 格式，确保是列表
    if target_ratios is not None:
        if isinstance(target_ratios, (float, int)):
            target_ratios = [target_ratios]
        target_ratios = set(target_ratios) # 转为集合通过O(1)查找
        print(f"Targeting specific mask ratios: {sorted(list(target_ratios))}")

    print(f"Loading data from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    os.makedirs(out_dir, exist_ok=True)
    
    all_results = []
    
    print(f"Processing {len(data)} transcripts...")
    
    for uuid, sample in tqdm(data.items(), desc="Calculating Metrics"):
        # 1. 获取 Ground Truth
        gt_log = sample['truth'].reshape(-1).astype(np.float32)
        gt_linear = np.expm1(gt_log)
        
        # 2. 获取 CDS 信息
        cds_info = sample.get('cds_info', None)
        if cds_info is None: continue
            
        cds_start = cds_info['start'] - 1
        cds_end = cds_info['end']
        
        # 3. 计算 Ground Truth 的 F0
        f0_gt = get_frame0_fraction_from_array(gt_linear, cds_start, cds_end)
        if np.isnan(f0_gt): continue

        # 4. 获取预测字典
        ratios_dict = sample.get('prediction', sample.get('ratios', None))
        if ratios_dict is None: continue
            
        # 遍历 Ratios (增加过滤逻辑)
        for ratio, ratio_data in ratios_dict.items():
            # --- 新增：过滤逻辑 ---
            if target_ratios is not None:
                if ratio not in target_ratios:
                    continue
            # --------------------

            pred_log = ratio_data['pred'].reshape(-1).astype(np.float32)
            pred_linear = np.expm1(pred_log)
            
            f0_pred = get_frame0_fraction_from_array(pred_linear, cds_start, cds_end)
            
            if not np.isnan(f0_pred):
                all_results.append({
                    'UUID': uuid,
                    'Mask_Ratio': ratio,
                    'Ground Truth': f0_gt,
                    'Prediction': f0_pred
                })

    # --- 4. 汇总与计算 ---
    df = pd.DataFrame(all_results)
    
    if df.empty:
        print("No valid data found for analysis (check if target_ratios exist in the data).")
        return None, None

    csv_path = os.path.join(out_dir, f"periodicity_results.{suffix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Raw results data saved to {csv_path}")
    
    ratios = sorted(df['Mask_Ratio'].unique())
    corr_results = []
    r_dict = {} 
    
    print("\nCalculating correlations...")
    for ratio in ratios:
        sub_df = df[df['Mask_Ratio'] == ratio]
        if len(sub_df) < 5:
            r_val_p = np.nan
            print(f"Warning: Not enough data points for ratio {ratio}")
        else:
            r_val_p, p_val_p = pearsonr(sub_df['Ground Truth'], sub_df['Prediction'])
            r_val_s, p_val_s = spearmanr(sub_df['Ground Truth'], sub_df['Prediction'])
            
            corr_results.append({
                'Mask_Ratio': ratio, 
                'Pearson_R': r_val_p, 
                "Pearson_P": p_val_p,
                'Spearman_R': r_val_s,
                'Spearman_P': p_val_s
            })
            r_dict[ratio] = r_val_p
        
    corr_df = pd.DataFrame(corr_results)
    if not corr_df.empty:
        csv_path = os.path.join(out_dir, f"periodicity_correlation_results.{suffix}.csv")
        corr_df.to_csv(csv_path, index=False)
        print("\n=== Correlation Results ===")
        print(corr_df)
    
    # --- 5. 绘图 ---
    
    # 动态散点图
    plot_path_scatter = os.path.join(out_dir, f"periodicity_correlation_scatter.{suffix}.pdf")
    plot_dynamic_scatters(df, r_dict, plot_path_scatter)
    
    # 趋势图 (如果只有一个点，plot_correlation_trend 内部会自动跳过)
    if len(ratios) > 1:
        plot_path_trend = os.path.join(out_dir, f"periodicity_correlation_trend.{suffix}.pdf")
        plot_correlation_trend(corr_df, plot_path_trend)

# --- 使用示例 ---
if __name__ == "__main__":
    # 示例 1: 分析所有 ratios
    # evaluate_periodicity_correlation("your_file.pkl")

    # 示例 2: 指定分析特定的 ratios
    # evaluate_periodicity_correlation("your_file.pkl", target_ratios=[0.5, 0.75])
    pass