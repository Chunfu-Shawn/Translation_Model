import os
import numpy as np
import pandas as pd
from plotnine import *
import matplotlib.pyplot as plt
import pickle


GLOBAL_MODEL_ORDER = [
    "TRACE", 
    # "Encoder", "Convolution", 
    "Translatomer", "Riboformer (CDS)", "RiboMIMO (CDS)",
    "Cross-batch", "Cross-experiment"
]

GLOBAL_MODEL_COLORS = {
    "TRACE": "#2C6B9A", 
    # "Encoder": "#555555", 
    # "Convolution": "#637D96",
    "Translatomer": "#555555",
    "Riboformer (CDS)": "#777777",
    "RiboMIMO (CDS)": "#BBBBBB",
    "Cross-batch": "#AF804F",
    "Cross-experiment": "#EBC67F",
}


def load_and_aggregate_multicell(data_config, depth_threshold=None, metric="Pearson_R"):
    """
    加载多模型数据（文件内含Cell_type），并计算每个模型中每种细胞类型的平均数 (Mean)。
    
    Args:
        # --- MODIFIED: 更新了 docstring，说明现在接受数组路径 ---
        data_config (dict): {"ModelA": ["path_to_modelA_res1.tsv", "path_to_modelA_res2.tsv"], 
                             "ModelB": ["path_to_modelB_res.tsv"]}
        metric (str): 'Pearson_R' 或 'Spearman_R'
    """
    aggregated_data = []
    print(f"--- Processing Data (Target Metric: {metric}) ---")

    for model_name, file_paths in data_config.items():
        print(f"\nProcessing model: {model_name}")
        
        # --- MODIFIED START: 兼容字符串格式，确保 file_paths 始终是列表 ---
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            
        model_dfs = [] # 用于存放当前模型所有的 DataFrame
        
        # 遍历该模型对应的所有文件路径
        for file_path in file_paths:
            print(f"  Loading: {file_path}")
            # 1. 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"  [Warning] File not found: {file_path}")
                continue
                
            # 2. 读取数据 (兼容上一部分代码生成的 TSV 或常规 CSV)
            try:
                sep = '\t' if file_path.endswith(('.tsv', '.txt')) else ','
                df = pd.read_csv(file_path, sep=sep)
                model_dfs.append(df) # 将读取的 DataFrame 加入列表
            except Exception as e:
                print(f"  [Error] Could not read {file_path}: {e}")
                continue
        
        # 3. 检查是否成功加载了任何数据
        if not model_dfs:
            print(f"[Warning] No valid data loaded for {model_name}. Skipping...")
            continue
            
        # 4. 按行拼接该模型的所有 DataFrame
        combined_df = pd.concat(model_dfs, ignore_index=True)
        
        # 5. 基于合并后的 combined_df 进行过滤和列检查
        if depth_threshold is not None and 'Depth' in combined_df.columns:
            combined_df = combined_df[combined_df['Depth'] >= depth_threshold]
        
        if metric not in combined_df.columns:
            print(f"[Warning] Metric '{metric}' not found in {model_name}")
            continue
            
        cell_col = 'Cell_type'
        if cell_col not in combined_df.columns:
            print(f"[Warning] '{cell_col}' column not found in {model_name}. Skipping...")
            continue
            
        # 6. 按细胞类型分组，分别计算平均数 (Mean)
        for cell_type, group_df in combined_df.groupby(cell_col):
            values = group_df[metric].dropna().values
            
            if len(values) == 0:
                continue

            stat_mean = np.mean(values)
            
            aggregated_data.append({
                'Cell_type': cell_type, # 统一转为首字母大写规范，对齐画图逻辑
                'Model': model_name,
                'Mean': stat_mean,
                'N': len(values)
            })

    return pd.DataFrame(aggregated_data)


def plot_multicell_performance(
        agg_df, cell_types=None, 
        metric_name="Pearson Correlation", out_dir = "./",
        w = 5, h =5):
    """
    使用 plotnine 绘制 Bar + Errorbar(SEM) + Jitter Points，展示跨细胞类型的模型表现。
    """
    if agg_df.empty:
        print("No data to plot.")
        return
        
    # 为避免 SettingWithCopyWarning，先复制一份数据
    agg_df = agg_df.copy()

    agg_df['Cell_type'] = agg_df['Cell_type'].replace({'SW480': 'SW480 (Unseen)'})

    
    if cell_types:
        # 如果传入了 cell_types 过滤列表，需要同步将里面的 SW480 替换掉
        cell_types = ['SW480 (Unseen)' if ct == 'SW480' else ct for ct in cell_types]
        agg_df = agg_df[agg_df["Cell_type"].isin(cell_types)]

    # 强制转化为有序分类变量
    agg_df['Model'] = pd.Categorical(agg_df['Model'], categories=GLOBAL_MODEL_ORDER, ordered=True)
    
    # 核心修改 1：让 Unseen 细胞系排在 Categories 的最后一个
    # 这样在后续基于这个 Categorical 排序时，它就会被放到 DataFrame 的末尾
    if cell_types is None:
        cell_types = list(agg_df['Cell_type'].unique())
        
    if "SW480 (Unseen)" in cell_types:
        cell_types.remove("SW480 (Unseen)")
        cell_types.append("SW480 (Unseen)") # 追加到末尾
        
    agg_df['Cell_type'] = pd.Categorical(agg_df['Cell_type'], categories=cell_types, ordered=True)

    # 核心修改 2：根据 Categorical 的顺序对 DataFrame 进行排序
    # 因为 Unseen 在 categories 的最后，所以它的数据会被移动到 DataFrame 的尾部
    # 这样 geom_jitter 画图时最后画它，就不会被遮挡了
    agg_df = agg_df.sort_values(by=['Model', 'Cell_type'])

    # --- 3. 计算总体均值和 SEM ---
    summary_df = agg_df.groupby('Model', observed=False).agg(
        Overall_Mean=('Mean', 'mean'),
        SEM=('Mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0) 
    ).reset_index()
    
    summary_df['SEM'] = summary_df['SEM'].fillna(0)
    summary_df['ymin'] = summary_df['Overall_Mean'] - summary_df['SEM']
    summary_df['ymax'] = summary_df['Overall_Mean'] + summary_df['SEM']


    # ======================================
    #  4. 为散点图生成颜色字典，单独高亮 SW480
    # ======================================
    unique_cells = agg_df['Cell_type'].unique()
    point_colors = {ct: "#202020" for ct in unique_cells} # 默认其他细胞系为深灰
    if "SW480 (Unseen)" in point_colors:
        point_colors["SW480 (Unseen)"] = "#E74C3C" # 用显眼的亮红色高亮 Unseen

    # --- 5. 绘图 ---
    plot = (
        ggplot(mapping=aes(x='Model'))
        + geom_col(data=summary_df, mapping=aes(y='Overall_Mean', fill='Model'), width=0.7)
        + geom_errorbar(data=summary_df, mapping=aes(ymin='ymin', ymax='ymax'), width=0.2, size=0.8, color="black")
        
        # ==========================================
        # 在 jitter 中同时映射 shape 和 color 到 Cell_type
        # ==========================================
        + geom_jitter(data=agg_df, mapping=aes(y='Mean', shape='Cell_type', color='Cell_type'), 
                      width=0.2, size=3.5, alpha=0.8)
        + scale_color_manual(values=point_colors) 
        + scale_fill_manual(values=GLOBAL_MODEL_COLORS) 
        # + coord_cartesian(ylim=[0, 0.7])
        + theme_bw()
        + theme(
            axis_text_x=element_text(angle=45, hjust=1, color="black"), 
            axis_title_x=element_blank(),
            panel_grid_major_x=element_blank(),
            legend_position='right',
            legend_title=element_text(fontweight='bold')
        )
        + labs(
            y=f"Translation profile position-wise correlation",
            fill="Model",
            shape="Cell Type", 
            color="Cell Type" 
        )
    )
    
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"benchmark.translation.profile_{metric_name}_correlation.models_multicell.pdf")
    
    plot.save(save_path, width=w, height=h, dpi=300)
    print(f"Plot saved to: {save_path}")
 

# =================================================================
# Length Robustness Analysis Module
# =================================================================
def prepare_length_robustness_data(
        model_csv_dict: dict, 
        seq_pkl_path: str, 
        meta_dict: dict = None, 
        out_dir="./results/robustness", 
        suffix=""):
    """
    Step 1: Anchor samples based on TRACE model predictions.
    Load global seq_dict pickle internally to map lengths and ensure consistent evaluation scale.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # =================================================================
    # [NEW] Safely load the sequence dictionary from the pickle path
    # =================================================================
    print(f">>> Loading sequence dictionary from {seq_pkl_path}...")
    try:
        with open(seq_pkl_path, 'rb') as f:
            seq_dict = pickle.load(f)
        print(f"  -> Successfully loaded {len(seq_dict)} sequences.")
    except Exception as e:
        print(f"  [Error] Failed to load seq_dict from {seq_pkl_path}: {e}")
        return None

    print(">>> Phase 1: Reading model predictions and anchoring samples based on TRACE...")
    model_data_store = {}
    
    for model_name, csv_path in model_csv_dict.items():
        if not os.path.exists(csv_path):
            print(f"  [Warning] {csv_path} not found. Skipping {model_name}.")
            continue
        
        df = pd.read_csv(csv_path)
        if 'Spearman_R' not in df.columns:
            print(f"  [Warning] Missing 'Spearman_R' in {csv_path}. Skipping.")
            continue
            
        # Construct unique identifier (Tid, Cell_type) for filtering
        df['Sample_ID'] = df['Tid'] + "::" + df['Cell_type']
        
        # Drop rows with missing values
        df = df.dropna(subset=['Spearman_R'])
        model_data_store[model_name] = df
        
    if not model_data_store:
        print("No valid data to process.")
        return None

    # Core logic: Use TRACE as the anchor, discard samples not present in TRACE
    if "TRACE" not in model_data_store:
        print("  [Error] 'TRACE' model not found in the input dictionary. Cannot filter by TRACE.")
        return None
        
    trace_samples = set(model_data_store["TRACE"]['Sample_ID'])
    print(f"  -> Found {len(trace_samples)} valid samples in TRACE. Using these as the benchmark anchor.")

    print(">>> Phase 2: Mapping lengths globally for the anchored samples...")
    all_model_dfs = []
    
    for model_name, df in model_data_store.items():
        # Keep only samples present in TRACE
        df_filtered = df[df['Sample_ID'].isin(trace_samples)].copy()
        
        # Map Full_Length using the internally loaded seq_dict
        df_filtered['Full_Length'] = df_filtered['Tid'].map(
            lambda x: len(seq_dict[x]) if x in seq_dict else np.nan
        )
        
        if meta_dict:
            def get_cds_len(tid):
                info = meta_dict.get(tid)
                if not info: return np.nan
                cds_s = int(info.get("cds_start_pos", -1)) if isinstance(info, dict) else getattr(info, "cds_start_pos", -1)
                cds_e = int(info.get("cds_end_pos", -1)) if isinstance(info, dict) else getattr(info, "cds_end_pos", -1)
                return (cds_e - max(0, cds_s - 1)) if cds_s != -1 and cds_e != -1 else np.nan
                
            def get_utr_len(tid):
                info = meta_dict.get(tid)
                if not info: return np.nan
                cds_s = int(info.get("cds_start_pos", -1)) if isinstance(info, dict) else getattr(info, "cds_start_pos", -1)
                return max(0, cds_s - 1) if cds_s != -1 else np.nan

            df_filtered['CDS_Length'] = df_filtered['Tid'].apply(get_cds_len)
            df_filtered['5UTR_Length'] = df_filtered['Tid'].apply(get_utr_len)
        else:
            df_filtered['CDS_Length'] = np.nan
            df_filtered['5UTR_Length'] = np.nan
            
        df_filtered['Model'] = model_name
        
        # Drop auxiliary columns
        df_filtered = df_filtered.drop(columns=['Sample_ID'])
        all_model_dfs.append(df_filtered)
        
    df_final = pd.concat(all_model_dfs, ignore_index=True)
    
    before_drop = len(df_final)
    df_final = df_final.dropna(subset=['Full_Length'])
    after_drop = len(df_final)
    
    if before_drop != after_drop:
        print(f"  -> Dropped {before_drop - after_drop} samples due to missing Fasta sequence.")

    filename = f"length_robustness_data.{suffix}.csv" if suffix else "length_robustness_data.csv"
    save_path = os.path.join(out_dir, filename)
    df_final.to_csv(save_path, index=False)
    
    print(f">>> Data successfully extracted and saved to {save_path} (Total valid records: {len(df_final)})")
    return save_path


def plot_length_robustness_line_chart(
        data_path: str, 
        x_value: str = "Full_Length",
        out_dir="./results/robustness", 
        suffix="",
        bins=None,
        labels=None):
    """
    Step 3: 采用 Matplotlib 绘制上下双子图 (Subplots)。
    上方柱状图较矮且无网格线，下方折线图较高，独立 Y 轴标签。
    """
    suffix_name = f'{suffix}_{x_value}' if suffix else x_value
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data table not found: {data_path}")
        
    print(f">>> Loading data from {data_path} for line chart plotting...")
    df_final = pd.read_csv(data_path)
    
    df_final = df_final.dropna(subset=[x_value, 'Spearman_R']).copy()
    df_final = df_final[df_final[x_value] > 0]
    
    # --- 1. 动态设定分箱策略 (Binning) ---
    if bins is None or labels is None:
        if x_value == "Full_Length":
            bins = [0, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, np.inf]
            labels = ['<500', '500-1k', '1k-2k', '2k-3k', '3k-4k', '4k-5k', '4k-6k', '6k-8k', '>8k']
        else:
            bins = [0, 50, 100, 200, 500, np.inf]
            labels = ['<50', '50-100', '100-200', '200-500', '>500']
            
    df_final['Length_Bin'] = pd.cut(df_final[x_value], bins=bins, labels=labels)
    df_final = df_final.dropna(subset=['Length_Bin'])

    # --- 2. 计算样本长度分布 ---
    print(">>> Calculating sample distribution...")
    df_unique = df_final.drop_duplicates(subset=['Tid', 'Cell_type']).copy()
    dist_df = df_unique.groupby('Length_Bin', observed=False).size().reset_index(name='Count')

    # --- 3. 计算模型均值与 SEM ---
    print(">>> Aggregating means and standard errors for models...")
    summary_df = df_final.groupby(['Model', 'Length_Bin'], observed=False).agg(
        Mean_R=('Spearman_R', 'mean'),
        SEM=('Spearman_R', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0)
    ).reset_index()
    summary_df = summary_df.dropna(subset=['Mean_R'])

    model_order = [m for m in GLOBAL_MODEL_ORDER if m in summary_df['Model'].unique()]

    os.makedirs(out_dir, exist_ok=True)
    print(">>> Plotting combined custom-scaled chart with Matplotlib...")

    # =================================================================
    # Matplotlib 双子图魔法
    # =================================================================
    # 创建上下两张图，比例 1:3，共享 X 轴
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), 
                                   gridspec_kw={'height_ratios': [1, 4]}, sharex=True)
    
    # 让两张图贴紧
    plt.subplots_adjust(hspace=0.08) 
    
    x_positions = np.arange(len(labels))
    
    # ---------------------------------------------------------
    # 上方子图 (ax1) - 密度分布柱状图
    # ---------------------------------------------------------
    # 确保柱子和 x_positions 对齐
    bar_heights = [dist_df[dist_df['Length_Bin'] == lbl]['Count'].values[0] if lbl in dist_df['Length_Bin'].values else 0 for lbl in labels]
    
    ax1.bar(x_positions, bar_heights, color='darkgray', alpha=0.7, width=0.6)
    
    # =================================================================
    # [MODIFIED] 将 Transcript count 修改为 Sample count，保持两张图统计含义和尺度(Scale)的绝对一致性
    # =================================================================
    ax1.set_ylabel("Sample count", color='#606060', fontsize=11)
    
    ax1.tick_params(axis='y', colors='#606060', labelsize=10)
    ax1.grid(False) # 明确去掉上方网格线
    
    # 隐藏上方图的上边、右边和底边框
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params(axis='x', length=0) # 隐藏共享 X 轴的刻度线
    
    # ---------------------------------------------------------
    # 下方子图 (ax2) - 性能折线图
    # ---------------------------------------------------------
    for model in model_order:
        m_df = summary_df[summary_df['Model'] == model]
        if m_df.empty: continue
        
        y_vals, y_errs, valid_x = [], [], []
        for i, lbl in enumerate(labels):
            row = m_df[m_df['Length_Bin'] == lbl]
            if not row.empty:
                y_vals.append(row['Mean_R'].values[0])
                y_errs.append(row['SEM'].values[0])
                valid_x.append(i)
                
        if valid_x:
            # 高亮主推模型 TRACE (将其画在最上层 zorder=5)
            z_ord = 5 if model == "TRACE" else 3
            alpha_val = 0.9 if model == "TRACE" else 0.7
            
            ax2.errorbar(valid_x, y_vals, yerr=y_errs, fmt='-o', color=GLOBAL_MODEL_COLORS.get(model, "#C0C0C0"), 
                         linewidth=1.8, markersize=7, markerfacecolor='white', markeredgewidth=1.8,
                         capsize=4, alpha=alpha_val, label=model, zorder=z_ord)

    # 美化下方子图
    ax2.set_ylabel("Model performance (Spearman R)", fontsize=11, color="black")
    
    # 设置 X 轴类别
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(labels, rotation=30, ha='right', fontsize=10, color="black")
    x_title = f"RNA {'full length' if x_value == 'Full_Length' else '5-UTR length'} (nt)"
    ax2.set_xlabel(x_title, fontsize=12, labelpad=10)

    # 给下方的折线图加上 plotnine 风格的轻灰色网格框
    ax2.grid(True, linestyle='-', alpha=0.9, color='#E0E0E0')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # ---------------------------------------------------------
    # 全局调整与图例
    # ---------------------------------------------------------
    # 对齐左右两边的 Y 轴标签
    fig.align_ylabels([ax1, ax2])
    
    # 图例居中放在整张图的顶部
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=3, 
               frameon=False, fontsize=10, handlelength=1.5,
               bbox_transform=fig.transFigure) # 基于整个画布定位图例

    # 保存
    filename = f"robustness_combined_chart.{suffix_name}.pdf" if suffix else "robustness_combined_chart.pdf"
    plot_path = os.path.join(out_dir, filename)
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f">>> Combined chart saved to {plot_path}")
    
    return summary_df

def prepare_depth_robustness_data(
        model_csv_dict: dict, 
        out_dir="./results/robustness", 
        suffix=""):
    """
    Step 1: 直接读取带有 Depth 的 CSV。
    以 TRACE 为基准过滤样本，确保与 Length 分析拥有完全一致的 Fair Benchmark 基础。
    """
    os.makedirs(out_dir, exist_ok=True)
    print(">>> Phase 1: Reading model predictions and anchoring samples based on TRACE...")
    
    model_data_store = {}
    
    for model_name, csv_path in model_csv_dict.items():
        if not os.path.exists(csv_path):
            print(f"  [Warning] {csv_path} not found. Skipping {model_name}.")
            continue
        
        df = pd.read_csv(csv_path)
        
        if 'Depth' not in df.columns or 'Spearman_R' not in df.columns:
            print(f"  [Warning] Missing 'Depth' or 'Spearman_R' in {csv_path}. Skipping.")
            continue
        
        # 构建唯一标识 (Tid, Cell_type)
        df['Sample_ID'] = df['Tid'] + "::" + df['Cell_type']
        
        df = df.dropna(subset=['Spearman_R', 'Depth'])
        model_data_store[model_name] = df
        
    if not model_data_store:
        print("No valid data to process.")
        return None
        
    # =================================================================
    # [MODIFIED] 核心逻辑：以 TRACE 为主准星，TRACE 没有的就不要
    # =================================================================
    if "TRACE" not in model_data_store:
        print("  [Error] 'TRACE' model not found in the input dictionary. Cannot filter by TRACE.")
        return None
        
    trace_samples = set(model_data_store["TRACE"]['Sample_ID'])
    print(f"  -> Found {len(trace_samples)} valid samples in TRACE. Using these as the benchmark anchor.")

    print(">>> Phase 2: Merging data for the anchored samples...")
    all_model_dfs = []
    
    for model_name, df in model_data_store.items():
        # [MODIFIED] 仅保留在 TRACE 中存在的样本
        df_filtered = df[df['Sample_ID'].isin(trace_samples)].copy()
        df_filtered['Model'] = model_name
        df_filtered = df_filtered.drop(columns=['Sample_ID'])
        all_model_dfs.append(df_filtered)
        
    df_final = pd.concat(all_model_dfs, ignore_index=True)

    filename = f"depth_robustness_data.{suffix}.csv" if suffix else "depth_robustness_data.csv"
    save_path = os.path.join(out_dir, filename)
    df_final.to_csv(save_path, index=False)
    
    print(f">>> Data successfully extracted and saved to {save_path} (Total valid records: {len(df_final)})")
    return save_path


def plot_depth_robustness_line_chart(
        data_path: str, 
        x_value: str = "Depth",
        out_dir="./results/robustness", 
        suffix="",
        bins=None,
        labels=None):
    """
    Step 3: 采用 Matplotlib 绘制上下双子图 (Subplots)。
    基于 Reads Density (Depth) 进行分箱，评估模型在不同表达丰度下的鲁棒性。
    """
    suffix_name = f'{suffix}_{x_value}' if suffix else x_value
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data table not found: {data_path}")
        
    print(f">>> Loading data from {data_path} for line chart plotting...")
    df_final = pd.read_csv(data_path)
    
    df_final = df_final.dropna(subset=[x_value, 'Spearman_R']).copy()
    df_final = df_final[df_final[x_value] > 0]
    
    # =================================================================
    # 1. 动态设定适合 Depth (Reads Density) 的分箱策略
    # 采用类似对数的切分，适应丰度跨度大的特点
    # =================================================================
    if bins is None or labels is None:
        bins = [0, 1, 5, 10, 50, 100, np.inf]
        labels = ['<1', '1-5', '5-10', '10-50', '50-100', '>100']
            
    df_final['Depth_Bin'] = pd.cut(df_final[x_value], bins=bins, labels=labels)
    df_final = df_final.dropna(subset=['Depth_Bin'])

    # --- 2. 计算样本分布 (根据 Tid 和 Cell_type 去重，保证顶部的计数真实) ---
    print(">>> Calculating sample distribution...")
    df_unique = df_final.drop_duplicates(subset=['Tid', 'Cell_type']).copy()
    dist_df = df_unique.groupby('Depth_Bin', observed=False).size().reset_index(name='Count')

    # --- 3. 计算模型均值与 SEM ---
    print(">>> Aggregating means and standard errors for models...")
    summary_df = df_final.groupby(['Model', 'Depth_Bin'], observed=False).agg(
        Mean_R=('Spearman_R', 'mean'),
        SEM=('Spearman_R', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0)
    ).reset_index()
    summary_df = summary_df.dropna(subset=['Mean_R'])

    # 使用全局模型顺序字典进行排序过滤
    model_order = [m for m in GLOBAL_MODEL_ORDER if m in summary_df['Model'].unique()]
    # 兜底：如果有没有在字典里的模型，追加到最后
    for m in summary_df['Model'].unique():
        if m not in model_order:
            model_order.append(m)

    os.makedirs(out_dir, exist_ok=True)
    print(">>> Plotting combined custom-scaled chart with Matplotlib...")

    # =================================================================
    # Matplotlib 双子图 (保留你精心调校的美学参数)
    # =================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), 
                                   gridspec_kw={'height_ratios': [1, 4]}, sharex=True)
    
    plt.subplots_adjust(hspace=0.08) 
    x_positions = np.arange(len(labels))
    
    # ---------------------------------------------------------
    # 上方子图 (ax1) - 密度分布柱状图
    # ---------------------------------------------------------
    bar_heights = [dist_df[dist_df['Depth_Bin'] == lbl]['Count'].values[0] if lbl in dist_df['Depth_Bin'].values else 0 for lbl in labels]
    
    ax1.bar(x_positions, bar_heights, color='darkgray', alpha=0.7, width=0.6)
    
    # =================================================================
    # [MODIFIED] 与 Length 图统一标签，确保 Scale 概念一致
    # =================================================================
    ax1.set_ylabel("Sample count", color='#606060', fontsize=11)
    
    ax1.tick_params(axis='y', colors='#606060', labelsize=10)
    ax1.grid(False) 
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params(axis='x', length=0) 
    
    # ---------------------------------------------------------
    # 下方子图 (ax2) - 性能折线图
    # ---------------------------------------------------------
    for model in model_order:
        m_df = summary_df[summary_df['Model'] == model]
        if m_df.empty: continue
        
        y_vals, y_errs, valid_x = [], [], []
        for i, lbl in enumerate(labels):
            row = m_df[m_df['Depth_Bin'] == lbl]
            if not row.empty:
                y_vals.append(row['Mean_R'].values[0])
                y_errs.append(row['SEM'].values[0])
                valid_x.append(i)
                
        if valid_x:
            # 高亮主推模型 TRACE (将其画在最上层 zorder=5)
            z_ord = 5 if model == "TRACE" else 3
            alpha_val = 0.9 if model == "TRACE" else 0.7
            
            # 动态从全局字典获取颜色，如果没有则用灰色兜底
            c_color = GLOBAL_MODEL_COLORS.get(model, "#C0C0C0")
            
            ax2.errorbar(valid_x, y_vals, yerr=y_errs, fmt='-o', color=c_color, 
                         linewidth=1.8, markersize=7, markerfacecolor='white', markeredgewidth=1.8,
                         capsize=4, alpha=alpha_val, label=model, zorder=z_ord)

    # 美化下方子图
    ax2.set_ylabel("Model performance (Spearman R)", fontsize=11, color="black")
    
    # 设置 X 轴类别
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(labels, rotation=30, ha='right', fontsize=10, color="black")
    
    x_title = "RNA ribosome load density"
    ax2.set_xlabel(x_title, fontsize=12, labelpad=10)

    # 给下方的折线图加上 plotnine 风格的轻灰色网格框
    ax2.grid(True, linestyle='-', alpha=0.9, color='#E0E0E0')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # ---------------------------------------------------------
    # 全局调整与图例
    # ---------------------------------------------------------
    fig.align_ylabels([ax1, ax2])
    
    # 图例居中放在整张图的顶部 (保持你原来的设定)
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=3, 
               frameon=False, fontsize=10, handlelength=1.5,
               bbox_transform=fig.transFigure)

    # 保存
    filename = f"robustness_depth_chart.{suffix_name}.pdf" if suffix else "robustness_depth_chart.pdf"
    plot_path = os.path.join(out_dir, filename)
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f">>> Combined chart saved to {plot_path}")
    
    return summary_df