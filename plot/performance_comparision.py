import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import os
from scipy.stats import sem
from plotnine import *


# --- 1. Data Loading Function ---

def load_and_align_data(model_config, target_ratio=1.0, depth_threshold=0.1, corr_method="Spearman_R"):
    """
    Load and align data from CSV files.
    
    Returns:
        model_names (list): List of model names.
        psite_stats (list of dict): List containing {'mean': ..., 'sem': ...} for each model.
        period_medians (list): List of median/value for periodicity correlation for each model.
    """
    model_names = []
    
    # Store stats for P-site (Mean + SEM)
    psite_stats = []
    # Store single value/median for Periodicity
    period_medians = []

    print(f"Loading data for Depth = {depth_threshold} and Mask Ratio = {target_ratio}...")

    for model_name, paths in model_config.items():
        # --- 1. P-site Correlation (Distribution for Error Bars) ---
        # Expecting raw data with one correlation value per transcript
        df_psite = pd.read_csv(paths['psite_corr_path'])
        
        # Filter by ratio
        df_psite = df_psite[df_psite['Mask_Ratio'] == target_ratio]
        df_psite = df_psite[df_psite['Depth'] >= depth_threshold]
        
        if df_psite.empty:
            print(f"Warning: No data for {model_name} in P-site correlation file.")
            continue
            
        # Extract values (transcript-wise correlations)
        if corr_method in df_psite.columns:
            psite_values = df_psite[corr_method].dropna().values
        else:
            print(f"Error: {corr_method} column not found in {model_name} P-site file.")
            continue
            
        # Calculate Mean and Standard Error of the Mean (SEM)
        if len(psite_values) > 0:
            p_mean = np.mean(psite_values)
            p_sem = sem(psite_values)
            psite_stats.append({'mean': p_mean, 'sem': p_sem})
        else:
            psite_stats.append({'mean': np.nan, 'sem': np.nan})

        # --- 2. Periodicity Correlation (Single Value) ---
        # Expecting summary data or distribution
        df_period = pd.read_csv(paths['period_corr_path'])
        
        # Filter by ratio
        # Note: If file has 'Condition', usually filter for 'Prediction' if needed
        # Assuming simple structure based on prompt
        if 'Mask_Ratio' in df_period.columns:
            df_period = df_period[df_period['Mask_Ratio'] == target_ratio]
        
        if df_period.empty:
            print(f"Warning: No data for {model_name} in periodicity file.")
            continue
            
        # Extract value
        if corr_method in df_period.columns:
             period_val = df_period[corr_method].dropna().values
        else:
             print(f"Warning: Unknown column for periodicity in {model_name}. Skipping.")
             continue

        # Append data
        model_names.append(model_name)
        period_medians.append(period_val)

    return model_names, psite_stats, period_medians

# --- 2. Plotting Function ---

def plot_dual_axis_lineplot(model_names, psite_stats, period_vals, out_dir):
    """
    Plot Dual Y-Axis Chart using Matplotlib.
    Left Axis (Blue): P-site Correlation (Mean +/- SEM).
    Right Axis (Orange): Periodicity Correlation (Single Point).
    """
    # Set style
    plt.style.use('seaborn-v0_8-white')
    fig, ax1 = plt.subplots(figsize=(6, 6))

    n_models = len(model_names)
    indices = np.arange(n_models)
    
    # Colors
    color_psite = '#3498db'   # Blue
    color_period = '#e67e22'  # Orange
    
    # Markers loop
    available_markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
    model_markers = [available_markers[i % len(available_markers)] for i in range(n_models)]

    # Data Prep
    psite_means = [s['mean'] for s in psite_stats]
    psite_sems = [s['sem'] for s in psite_stats]

    # --- 1. Left Axis: P-site (Blue) ---
    # Connect lines
    ax1.plot(indices, psite_means, color=color_psite, linestyle='--', linewidth=2, alpha=0.5, zorder=1)
    
    # Plot Error Bars and Markers
    for i in range(n_models):
        # Marker (white fill to stand out)
        ax1.scatter(
            indices[i], psite_means[i], 
            s=80, color=color_psite, edgecolors=color_psite, linewidth=0, 
            marker=model_markers[i], zorder=3
        )
        # Error bar
        ax1.errorbar(
            indices[i], psite_means[i], yerr=psite_sems[i], 
            fmt='none', ecolor=color_psite, elinewidth=2, capsize=5, zorder=2
        )

    ax1.set_ylabel('P-site Position-wise Correlation\n(Mean ± SEM)', color=color_psite, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_psite)
    ax1.set_ylim(min(psite_means)*0.8, max(psite_means)*1.1)

    # --- 2. Right Axis: Periodicity (Orange) ---
    ax2 = ax1.twinx()
    
    # Connect lines
    ax2.plot(indices, period_vals, color=color_period, linestyle='-', linewidth=2, alpha=0.5, zorder=1)
    
    for i in range(n_models):
        ax2.scatter(
            indices[i], period_vals[i], 
            s=120, color=color_period, edgecolors='white', linewidth=1.5, 
            marker=model_markers[i], zorder=3
        )

    ax2.set_ylabel('Periodicity Transcript-wise Correlation', color=color_period, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_period)
    ax2.set_ylim(min(period_vals)*0.9, max(period_vals)*1.05)

    # --- Common ---
    ax1.set_xticks(indices)
    ax1.set_xticklabels(model_names, fontsize=11, fontweight='bold', rotation=15)
    ax1.set_xlabel('Models', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Legend
    handle1 = mlines.Line2D([], [], color=color_psite, linestyle='--', marker='o', label='P-site Corr (Left)')
    handle2 = mlines.Line2D([], [], color=color_period, linestyle='-', marker='o', label='Periodicity Corr (Right)')
    ax1.legend(handles=[handle1, handle2], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)

    plt.tight_layout()
    save_path = os.path.join(out_dir, "model_psite_dynamic_performance_comparison.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Matplotlib Plot saved to {save_path}")

##############################
# Correlation benchmark
##############################

def load_and_aggregate_multicell(data_config, target_ratio=None, depth_threshold=None, metric="Pearson_R"):
    """
    加载多细胞类型、多模型的数据，并计算统计量（中位数和SD）。
    
    Args:
        data_config (dict): 嵌套字典 
                            { 
                              "HeLa": {"ModelA": "path.csv", "Baseline": "path.csv"},
                              "HEK293": {"ModelA": "path.csv", ...}
                            }
        target_ratio (float): 需要筛选的 Mask_Ratio (如果是 Baseline 或不含该列则忽略)
        metric (str): 'Pearson_R' 或 'Spearman_R'
    
    Returns:
        pd.DataFrame: 用于 plotnine 绘图的统计汇总数据
    """
    aggregated_data = []

    print(f"--- Processing Data (Target Metric: {metric}) ---")

    for cell_type, models in data_config.items():
        for model_name, file_path in models.items():
            
            # 1. 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"[Warning] File not found: {cell_type} - {model_name}")
                continue
                
            # 2. 读取数据
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"[Error] Could not read {file_path}: {e}")
                continue
            
            # 3. 筛选数据 (Mask_Ratio)
            # 只有当 DataFrame 有 Mask_Ratio 列且用户指定了 ratio 时才筛选
            # Baseline 通常没有 Mask_Ratio，或者全部都是目标数据，因此不筛选 Baseline
            if target_ratio is not None and 'Mask_Ratio' in df.columns:
                df = df[df['Mask_Ratio'] == target_ratio]
            
            if depth_threshold is not None and 'Depth' in df.columns:
                df = df[df['Depth'] >= depth_threshold]
            
            # 4. 提取相关性数值
            if metric not in df.columns:
                print(f"[Warning] Metric '{metric}' not found in {cell_type} - {model_name}")
                continue
            
            values = df[metric].dropna().values
            
            if len(values) == 0:
                print(f"[Warning] No valid data for {cell_type} - {model_name}")
                continue

            # 5. 计算统计量：中位数 (Median) 和 SD (standard deviation)
            # 注意：用户要求点是中位数，误差棒是SD
            stat_median = np.median(values)
            stat_sd = np.std(values)
            
            aggregated_data.append({
                'Cell_Type': cell_type,
                'Model': model_name,
                'Median': stat_median,
                'SD': stat_sd,
                'N': len(values), # 样本量，可选用于检查
                'Type': 'Baseline' if 'Baseline' in model_name or 'Exp' in model_name else 'Model'
            })

    return pd.DataFrame(aggregated_data)

def plot_multicell_performance(agg_df, metric_name="Pearson Correlation", out_path="multicell_comparison.pdf"):
    """
    使用 plotnine 绘制分面图：点为中位数，误差棒为 SD。
    """
    if agg_df.empty:
        print("No data to plot.")
        return

    # --- 1. 设置因子的顺序 (可选) ---
    # 让 Baseline 排在最前面或最后面，保持图例整洁
    # models = sorted(agg_df['Model'].unique())
    agg_df['Model'] = pd.Categorical(
        agg_df['Model'], categories=["base_model", "Cross-batch", "Cross-experiment"], ordered=True
        )

    # --- 2. 定义误差棒的上下界 ---
    # Error Bar range: Median - SD 到 Median + SD
    agg_df['ymin'] = agg_df['Median'] - agg_df['SD']
    agg_df['ymax'] = agg_df['Median'] + agg_df['SD']

    # --- 3. 绘图 ---
    plot = (
        ggplot(agg_df, aes(x='Model', y='Median', color='Model'))
        # 1. 绘制误差棒 (先画误差棒，这样点会盖在误差棒上面，更好看)
        + geom_errorbar(aes(ymin='ymin', ymax='ymax'), width=0.1, size=1)
        # 2. 绘制中位数点
        + geom_point(size=4)
        + scale_color_manual(values=["#2C6B9A", "#AF804F", "#EBC67F", "#585858", "#888888", "#A9A9A9", "#D3D3D3"])
        # + scale_color_brewer(type='qual', palette='Set1') 
        + coord_cartesian(ylim=[0, 0.8])
        + facet_wrap('~Cell_Type', scales='fixed', ncol=2) 
        + theme_bw()
        + theme(
            figure_size=(10, 6),
            axis_text_x=element_blank(),
            axis_title_x=element_blank(),
            panel_grid_major_x=element_blank(),
            strip_text=element_text(size=12),  # 分面标题样式
            strip_background=element_blank(), # 分面标题背景
            legend_position='right' # 图例位置
        )
        + labs(
            y=f"P-site position-wise {metric_name} correlation (Median ± SD)",
            x="Model"
        )
    )
    plot.save(out_path, width=8, height=5, dpi=300)
    print(plot)

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