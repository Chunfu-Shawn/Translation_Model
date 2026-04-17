import os
import numpy as np
import pandas as pd
from plotnine import *
from tqdm import tqdm
from scipy.stats import linregress

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

def plot_multicell_performance(agg_df, cell_types=None, metric_name="Pearson Correlation", out_dir = "./"):
    """
    使用 plotnine 绘制 Bar + Errorbar(SEM) + Jitter Points，展示跨细胞类型的模型表现。
    """
    if agg_df.empty:
        print("No data to plot.")
        return
        
    # 为避免 SettingWithCopyWarning，先复制一份数据
    agg_df = agg_df.copy()

    agg_df['Cell_type'] = agg_df['Cell_type'].replace({'SW480': 'SW480 (Unseen)'})

    # --- 2. 设置因子的顺序 ---
    model_order = [
        "TRACE", "Encoder", "Convolution", 
        "Translatomer", "RiboMIMO (CDS)", # "Riboformer (CDS)",
        "Cross-batch", "Cross-experiment"
    ]
    
    
    if cell_types:
        # 如果传入了 cell_types 过滤列表，需要同步将里面的 SW480 替换掉
        cell_types = ['SW480 (Unseen)' if ct == 'SW480' else ct for ct in cell_types]
        agg_df = agg_df[agg_df["Cell_type"].isin(cell_types)]

    # 强制转化为有序分类变量
    agg_df['Model'] = pd.Categorical(agg_df['Model'], categories=model_order, ordered=True)
    agg_df['Cell_type'] = pd.Categorical(agg_df['Cell_type'], categories=cell_types, ordered=True)

    # --- 3. 计算总体均值和 SEM ---
    summary_df = agg_df.groupby('Model', observed=False).agg(
        Overall_Mean=('Mean', 'mean'),
        SEM=('Mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0) 
    ).reset_index()
    
    summary_df['SEM'] = summary_df['SEM'].fillna(0)
    summary_df['ymin'] = summary_df['Overall_Mean'] - summary_df['SEM']
    summary_df['ymax'] = summary_df['Overall_Mean'] + summary_df['SEM']

    color_mapping = {
        "TRACE": "#2C6B9A",
        "Encoder": "#555555",
        "Convolution": "#777777",
        "Translatomer": "#999999",
        "RiboMIMO (CDS)": "#BBBBBB",
        # "Riboformer (CDS)": "#DDDDDD",
        "Cross-batch": "#AF804F",
        "Cross-experiment": "#EBC67F"
    }

    # ======================================
    #  4. 为散点图生成颜色字典，单独高亮 SW480
    # ======================================
    unique_cells = agg_df['Cell_type'].unique()
    point_colors = {ct: "#202020" for ct in unique_cells} # 默认其他细胞系为深灰
    if "SW480 (Unseen)" in point_colors:
        point_colors["SW480 (Unseen)"] = "#E74C3C" # 用显眼的亮红色高亮 Unseen c73813 E74C3C

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
        + scale_color_manual(values=point_colors) # [NEW] 传入散点颜色配置
        + scale_fill_manual(values=color_mapping) 
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
            color="Cell Type" # [NEW] 必须保证 shape 和 color 的标题完全一致，plotnine 才会把它们合并成一个图例
        )
    )
    
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"benchmark.translation.profile_{metric_name}_correlation.models_multicell.pdf")
    
    plot.save(save_path, width=6, height=5, dpi=300)
    print(f"Plot saved to: {save_path}")
    # print(plot)


def prepare_length_robustness_data(
        dataset, 
        model_csv_dict: dict, 
        out_dir="./results/robustness", 
        suffix=""):
    """
    Step 1: 收集转录本长度和相关性并合并，保存为单独的数据表。
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. 从 Dataset 中提取长度信息
    print(">>> Extracting length info from dataset...")
    length_records = []
    
    for i in tqdm(range(len(dataset))):
        uuid, _, meta_info, seq_emb, _ = dataset[i]
        uuid_str = str(uuid)
        parts = uuid_str.split('-')
        if len(parts) < 2: continue
        
        tid, cell_type = parts[0], parts[1]
        full_len = len(seq_emb)
        cds_s = int(meta_info.get("cds_start_pos", -1))
        cds_e = int(meta_info.get("cds_end_pos", -1))
        
        if cds_s != -1 and cds_e != -1:
            cds_len = cds_e - max(0, cds_s - 1)
            five_utr_len = max(0, cds_s - 1)
        else:
            cds_len = np.nan
            five_utr_len = np.nan
            
        length_records.append({
            'Tid': tid,
            'Cell_type': cell_type,
            'Full_Length': full_len,
            '5UTR_Length': five_utr_len,
            'CDS_Length': cds_len
        })
        
    df_lengths = pd.DataFrame(length_records).drop_duplicates(subset=['Tid', 'Cell_type'])
    
    # 2. 合并不同模型的 CSV 结果
    print(">>> Merging model predictions...")
    all_model_dfs = []
    for model_name, csv_path in model_csv_dict.items():
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Skipping {model_name}.")
            continue
        
        df_corr = pd.read_csv(csv_path)
        # 合并长度特征
        df_merged = pd.merge(df_corr, df_lengths, on=['Tid', 'Cell_type'], how='inner')
        df_merged['Model'] = model_name
        all_model_dfs.append(df_merged)
        
    if not all_model_dfs:
        print("No valid data to merge.")
        return None
        
    df_final = pd.concat(all_model_dfs, ignore_index=True)
    
    # 过滤掉 NaN 的相关性
    df_final = df_final.dropna(subset=['Spearman_R', 'Full_Length'])

    # 3. 保存数据表
    filename = f"length_robustness_data.{suffix}.csv" if suffix else "length_robustness_data.csv"
    save_path = os.path.join(out_dir, filename)
    df_final.to_csv(save_path, index=False)
    
    print(f">>> Data successfully extracted and saved to {save_path}")
    return save_path


def plot_length_robustness_line_chart(
        data_path: str, 
        x_value: str = "Full_Length",
        out_dir="./results/robustness", 
        suffix="",
        bins=None,
        labels=None):
    """
    Step 3: 将连续的序列长度进行区间分箱 (Binning)，计算各区间相关性均值与 SEM，绘制带误差棒的折线图。
    """
    suffix_name = f'{suffix}_{x_value}' if suffix else x_value
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data table not found: {data_path}")
        
    print(f">>> Loading data from {data_path} for line chart plotting...")
    df_final = pd.read_csv(data_path)
    
    # 清理数据
    df_final = df_final.dropna(subset=[x_value, 'Spearman_R']).copy()
    df_final = df_final[df_final[x_value] > 0]
    
    # --- 1. 动态设定分箱策略 (Binning) ---
    if bins is None or labels is None:
        if x_value == "Full_Length":
            # 针对全长 RNA 的尺度
            bins = [0, 1000, 2000, 3000, 5000, 7000, 10000, 13000, np.inf]
            labels = ['<1k', '1k-2k', '2k-3k', '3k-5k', '5k-7k', '7k-9k', '10k-13k', '>13k']
        else:
            # 针对 5'UTR 或较短区域的尺度
            bins = [0, 50, 100, 200, 500, np.inf]
            labels = ['<50', '50-100', '100-200', '200-500', '>500']
            
    # 将长度划分为离散区间
    df_final['Length_Bin'] = pd.cut(df_final[x_value], bins=bins, labels=labels)
    df_final = df_final.dropna(subset=['Length_Bin'])
    
    # 转换为 Ordered Categorical 确保 X 轴顺序正确
    df_final['Length_Bin'] = pd.Categorical(df_final['Length_Bin'], categories=labels, ordered=True)

    # --- 2. 自定义颜色与模型排序 ---
    color_mapping = {
        "TRACE": "#2C6B9A",
        "Encoder": "#555555",
        "Convolution": "#777777",
        "Translatomer": "#999999",
        "RiboMIMO (CDS)": "#BBBBBB",
        "Cross-batch": "#AF804F",
        "Cross-experiment": "#EBC67F"
    }
    
    valid_models = [m for m in color_mapping.keys() if m in df_final['Model'].unique()]
    df_final['Model'] = pd.Categorical(df_final['Model'], categories=valid_models, ordered=True)
    
    # --- 3. 聚合计算均值与 SEM ---
    print(">>> Aggregating means and standard errors for bins...")
    summary_df = df_final.groupby(['Model', 'Length_Bin'], observed=False).agg(
        Mean_R=('Spearman_R', 'mean'),
        SEM=('Spearman_R', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0),
        N=('Spearman_R', 'count')
    ).reset_index()
    
    # 过滤掉区间内没有数据的点，避免断线
    summary_df = summary_df.dropna(subset=['Mean_R'])
    summary_df['ymin'] = summary_df['Mean_R'] - summary_df['SEM']
    summary_df['ymax'] = summary_df['Mean_R'] + summary_df['SEM']

    os.makedirs(out_dir, exist_ok=True)
    
    # --- 4. 绘制折线图 ---
    print(">>> Plotting binned line chart...")
    p_line = (
        ggplot(summary_df, aes(x='Length_Bin', y='Mean_R', color='Model', group='Model'))
        # 添加误差棒
        + geom_errorbar(aes(ymin='ymin', ymax='ymax'), width=0.15, size=0.8, alpha=0.7)
        # 添加连线
        + geom_line(size=1.2, alpha=0.7)
        # 添加数据点 (实心点)
        + geom_point(size=3, fill='white', stroke=1, alpha=1) 
        + scale_color_manual(values=color_mapping)
        + theme_bw()
        + labs(
            x=f"RNA {'full length' if x_value == 'Full_Length' else '5-UTR length'} (nt)", 
            y="Translation profile position-wise Spearman correlation (mean ± SEM)",
        )
        + theme(
            legend_position='top',
            legend_title=element_blank(),
            axis_text_x=element_text(angle=30, hjust=1, color="black"),
            axis_title_x=element_text(margin={'t': 10}),
            panel_grid_minor=element_blank()
        )
    )
    
    filename = f"robustness_line_chart.{suffix_name}.pdf" if suffix else "robustness_line_chart.pdf"
    plot_path = os.path.join(out_dir, filename)
    p_line.save(plot_path, width=5, height=5, verbose=False)
    print(f">>> Line chart saved to {plot_path}")
    
    return summary_df