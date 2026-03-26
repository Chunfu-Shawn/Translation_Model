import os
import numpy as np
import pandas as pd
import pickle
from plotnine import *
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr


def load_and_calculate_te_correlation(
        data_config, 
        ref_df, 
        metric="mORF_Mean_Density", 
        corr_method="spearman",
        out_dir="./",
        meta_pkl_path="/home/user/data3/rbase/translation_model/models/lib/transcript_meta.pkl"):
    """
    加载模型预测结果并与 Protein-to-mRNA ratio 参考表计算相关性。
    支持自动使用 meta_pkl_path 将模型结果从 Transcript ID 映射并聚合到 Gene ID 级别。
    在计算相关性前，会过滤掉 metric 和 PTR 的 1% 和 99% 异常点。
    """
    aggregated_data = []
    print(f"--- Processing Data (Target Metric: {metric}, Method: {corr_method.capitalize()}) ---")

    # ==========================================
    # 1. 识别参考表的主 ID 列
    # ==========================================
    id_cols_master = ['GeneName', 'Gid', 'Tid', 'EnsemblGeneID', 'EnsemblTranscriptID', 'EnsemblProteinID']
    available_id_cols = [c for c in id_cols_master if c in ref_df.columns]
    val_cols = [c for c in ref_df.columns if c not in available_id_cols]
    
    # 确定目标合并级别
    if any(c in available_id_cols for c in ['Tid', 'EnsemblTranscriptID']):
        ref_merge_key = [c for c in ['Tid', 'EnsemblTranscriptID'] if c in available_id_cols][0]
        target_level = 'Transcript'
    elif any(c in available_id_cols for c in ['Gid', 'EnsemblGeneID']):
        ref_merge_key = [c for c in ['Gid', 'EnsemblGeneID'] if c in available_id_cols][0]
        target_level = 'Gene'
    else:
        raise ValueError("Reference table must contain 'Tid', 'Gid', 'EnsemblTranscriptID', or 'EnsemblGeneID'.")
        
    print(f"  -> Detected target ID level in Reference: {ref_merge_key} ({target_level} Level)")

    # 宽表转长表 (Melt)
    ref_long = ref_df.melt(id_vars=available_id_cols, value_vars=val_cols, var_name='Cell_Type', value_name='PTR')
    ref_long = ref_long.dropna(subset=['PTR'])
    
    # 统一剥离版本号
    ref_long['ID_clean'] = ref_long[ref_merge_key].astype(str).str.split('.').str[0]

    # ==========================================
    # 2. 如果目标是 Gene Level，预加载映射字典
    # ==========================================
    tid2gene = {}
    if target_level == 'Gene':
        print(f"  -> Loading transcript metadata mapping from {meta_pkl_path}...")
        try:
            with open(meta_pkl_path, 'rb') as f:
                transcript_meta = pickle.load(f)
                
            for tid, info in transcript_meta.items():
                clean_tid = str(tid).split('.')[0]
                if isinstance(info, dict) and 'gene_id' in info:
                    raw_gene = info['gene_id']
                elif hasattr(info, 'gene_id'):
                    raw_gene = info.gene_id
                else:
                    raw_gene = info
                tid2gene[clean_tid] = str(raw_gene).split('.')[0]
            print(f"  -> Successfully loaded mapping for {len(tid2gene)} transcripts.")
        except Exception as e:
            raise RuntimeError(f"Failed to load or parse metadata pickle: {e}")

    # ==========================================
    # 3. 遍历模型数据、映射聚合与合并
    # ==========================================
    for model_name, file_paths in data_config.items():
        print(f"\nProcessing model: {model_name}")
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            
        # 读取模型的所有分块数据
        model_dfs = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"  [Warning] File not found: {file_path}")
                continue
            try:
                sep = '\t' if file_path.endswith(('.tsv', '.txt')) else ','
                df = pd.read_csv(file_path, sep=sep)
                model_dfs.append(df)
            except Exception as e:
                print(f"  [Error] Could not read {file_path}: {e}")
                
        if not model_dfs:
            continue
            
        combined_df = pd.concat(model_dfs, ignore_index=True)
        
        # 确定模型端包含的列
        has_tid = [c for c in ['Tid', 'EnsemblTranscriptID', 'TranscriptID'] if c in combined_df.columns]
        has_gid = [c for c in ['Gid', 'EnsemblGeneID', 'GeneID'] if c in combined_df.columns]
        
        if metric not in combined_df.columns or 'Cell_Type' not in combined_df.columns:
            print(f"  [Warning] Missing metric '{metric}' or 'Cell_Type' in {model_name}. Skipping...")
            continue

        # --------- ID 映射逻辑 ---------
        if target_level == 'Gene':
            if has_gid:
                # 模型直接就输出了 Gene ID
                combined_df['ID_clean'] = combined_df[has_gid[0]].astype(str).str.split('.').str[0]
            elif has_tid:
                # 需要通过 tid2gene 将模型的 Transcript ID 转成 Gene ID
                tid_col = has_tid[0]
                combined_df['clean_tid_temp'] = combined_df[tid_col].astype(str).str.split('.').str[0]
                combined_df['ID_clean'] = combined_df['clean_tid_temp'].map(tid2gene)
                
                unmapped = combined_df['ID_clean'].isna().sum()
                if unmapped > 0:
                    print(f"  [Info] Dropped {unmapped} records lacking Gene mapping.")
                combined_df = combined_df.dropna(subset=['ID_clean'])
            else:
                print(f"  [Warning] No suitable ID column found to map to Gene level. Skipping...")
                continue
        else:
            # 目标是 Transcript Level
            if has_tid:
                combined_df['ID_clean'] = combined_df[has_tid[0]].astype(str).str.split('.').str[0]
            else:
                print(f"  [Warning] Target is Transcript level, but model lacks Tid. Skipping...")
                continue

        # --------- 聚合与计算均值 ---------
        # 同一 Cell_Type 下，如果有多个相同 ID_clean（多转录本归属同基因），自动取均值
        combined_df_agg = combined_df.groupby(['ID_clean', 'Cell_Type'], as_index=False)[metric].mean()
        
        # --------- 与参考表合并 ---------
        merged_df = pd.merge(combined_df_agg, ref_long, on=['ID_clean', 'Cell_Type'], how='inner')
        
        if merged_df.empty:
            print(f"  [Warning] No matching IDs found between {model_name} and reference table.")
            continue
            
        print(f"  [Info] Matched {len(merged_df)} pairs for analysis.")

        # --------- 计算分组相关性 ---------
        for cell_type, group_df in merged_df.groupby('Cell_Type'):
            group_clean = group_df.dropna(subset=[metric, 'PTR'])
            
            # 【核心修改点】: 计算并过滤 1% 和 99% 的异常值
            if len(group_clean) > 0:
                p01_m = group_clean[metric].quantile(0.01)
                p99_m = group_clean[metric].quantile(0.99)
                p01_p = group_clean['PTR'].quantile(0.01)
                p99_p = group_clean['PTR'].quantile(0.99)
                
                # 仅保留在 1% ~ 99% 之间的数据
                valid_mask = (
                    (group_clean[metric] >= p01_m) & (group_clean[metric] <= p99_m) &
                    (group_clean['PTR'] >= p01_p) & (group_clean['PTR'] <= p99_p)
                )
                group_clean = group_clean[valid_mask]
            
            # 至少需要 5 个基因才能算相关性
            if len(group_clean) < 5:
                continue
                
            x = group_clean[metric].values
            y = group_clean['PTR'].values
            
            if corr_method.lower() == 'spearman':
                r_val, p_val = spearmanr(x, y)
            else:
                r_val, p_val = pearsonr(x, y)
                
            aggregated_data.append({
                'Cell_type': cell_type,
                'Model': model_name,
                'Mean': r_val,  # 依然存放相关性，适配画图函数
                'P_value': p_val,
                'N': len(group_clean)
            })

    df = pd.DataFrame(aggregated_data)
    
    os.makedirs(out_dir, exist_ok=True)
    file_suffix = f".{metric}.{corr_method}"
    save_path = os.path.join(out_dir, f"translation_efficiency_metrics{file_suffix}.csv")
    df.to_csv(save_path, index=False)
    print(f"Correlation efficiency saved to: {save_path}")

    return df


def plot_te_correlation_performance(agg_df, cell_types=None, metric_name="mORF Mean Density", 
                                    corr_method="Spearman", out_dir="./", suffix=""):
    """
    使用 plotnine 绘制 Bar + Errorbar(SEM) + Jitter Points，展示跨细胞类型的模型相关性表现。
    """
    if agg_df.empty:
        print("No data to plot.")
        return
        
    agg_df = agg_df.copy()

    # 处理特定的未见细胞系标注
    agg_df['Cell_type'] = agg_df['Cell_type'].replace({'SW480': 'SW480 (Unseen)'})

    # 设置因子的顺序
    model_order = [
        "TRACE", "Encoder", "Convolution", 
        "Translatomer", "RiboMIMO (CDS)", 
        "Raw-dataset"
    ]
    
    if cell_types:
        cell_types = ['SW480 (Unseen)' if ct == 'SW480' else ct for ct in cell_types]
        agg_df = agg_df[agg_df["Cell_type"].isin(cell_types)]

    valid_models = [m for m in model_order if m in agg_df['Model'].unique()]
    agg_df['Model'] = pd.Categorical(agg_df['Model'], categories=valid_models, ordered=True)
    if cell_types:
        agg_df['Cell_type'] = pd.Categorical(agg_df['Cell_type'], categories=cell_types, ordered=True)

    # 计算总体相关性的均值和 SEM
    summary_df = agg_df.groupby('Model', observed=False).agg(
        Overall_Mean=('Mean', 'mean'),
        SEM=('Mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0) 
    ).reset_index()
    
    summary_df['SEM'] = summary_df['SEM'].fillna(0)
    summary_df['ymin'] = summary_df['Overall_Mean'] - summary_df['SEM']
    summary_df['ymax'] = summary_df['Overall_Mean'] + summary_df['SEM']

    color_mapping = {
        "TRACE": "#2C6B9A", "Encoder": "#555555", "Convolution": "#777777",
        "Translatomer": "#999999", "RiboMIMO (CDS)": "#BBBBBB",
        "Raw-dataset": "#AF804F"
    }

    # 为散点图生成颜色字典，单独高亮 SW480
    unique_cells = agg_df['Cell_type'].unique()
    point_colors = {ct: "#202020" for ct in unique_cells}
    if "SW480 (Unseen)" in point_colors:
        point_colors["SW480 (Unseen)"] = "#E74C3C" 

    # 动态获取 Y 轴上限，防止柱状图冲出画框 (预留 20% 空间)
    y_max_data = summary_df['ymax'].max() if not summary_df.empty else 0.5
    y_limit = max(0.5, y_max_data * 1.2)

    plot = (
        ggplot(mapping=aes(x='Model'))
        + geom_col(data=summary_df, mapping=aes(y='Overall_Mean', fill='Model'), width=0.7)
        + geom_errorbar(data=summary_df, mapping=aes(ymin='ymin', ymax='ymax'), width=0.2, size=0.8, color="black")
        + geom_jitter(data=agg_df, mapping=aes(y='Mean', shape='Cell_type', color='Cell_type'), 
                      width=0.2, size=3.5, alpha=0.8)
        + scale_fill_manual(values=color_mapping) 
        + scale_color_manual(values=point_colors) 
        + coord_cartesian(ylim=[0, y_limit]) # 动态 Y 轴
        + theme_bw()
        + theme(
            axis_text_x=element_text(angle=45, hjust=1, color="black"), 
            axis_title_x=element_blank(),
            panel_grid_major_x=element_blank(),
            legend_position='right',
            legend_title=element_text(fontweight='bold')
        )
        + labs(
            y=f"{corr_method.capitalize()} correlation between prediction and PTR score",
            fill="Model",
            shape="Cell Type", 
            color="Cell Type" 
        )
    )
    
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"te_correlation_performance_{metric_name}.{suffix}.pdf")
    plot.save(save_path, width=7, height=5, dpi=300, verbose=False)
    print(f"Plot saved to: {save_path}")



def load_and_calculate_polysome_correlation(
        data_config, 
        ref_df, 
        target_cell_types=None,  # [NEW] 允许指定每个模型对应的细胞类型
        model_metric="mORF_Mean_Density", 
        ref_metric="High_vs_Low_FC", 
        corr_method="spearman",
        meta_pkl_path="/home/user/data3/rbase/translation_model/models/lib/transcript_meta.pkl"):
    """
    加载模型预测结果并与 Polysome 图谱参考表计算相关性。
    支持自定义模型指标和 Polysome 指标，并可通过 target_cell_types 过滤特定的细胞系。
    """
    aggregated_data = []
    print(f"--- Processing Data ---")
    print(f"  Model Metric : {model_metric}")
    print(f"  Ref Metric   : {ref_metric}")
    print(f"  Method       : {corr_method.capitalize()}")

    # 校验 target_cell_types 数组长度
    if isinstance(target_cell_types, list):
        if len(target_cell_types) != len(data_config):
            raise ValueError("The length of target_cell_types list must match the number of models in data_config.")

    # ==========================================
    # 1. 识别参考表的主 ID 列并提取目标指标
    # ==========================================
    id_cols_master = ['Tid', 'EnsemblTranscriptID', 'Gid', 'EnsemblGeneID', 'gene_name']
    ref_merge_key = None
    target_level = None
    
    for col in id_cols_master:
        if col in ref_df.columns:
            ref_merge_key = col
            target_level = 'Transcript' if col in ['Tid', 'EnsemblTranscriptID'] else 'Gene'
            break
            
    if not ref_merge_key:
        raise ValueError(f"Reference table must contain one of: {id_cols_master}")
        
    if ref_metric not in ref_df.columns:
        raise ValueError(f"Target reference metric '{ref_metric}' not found in the reference table!")
        
    print(f"  -> Detected target ID level in Reference: {ref_merge_key} ({target_level} Level)")

    ref_clean = ref_df[[ref_merge_key, ref_metric]].copy()
    ref_clean = ref_clean.dropna(subset=[ref_metric])
    ref_clean['ID_clean'] = ref_clean[ref_merge_key].astype(str).str.split('.').str[0]
    ref_clean = ref_clean.groupby('ID_clean', as_index=False)[ref_metric].mean()

    # ==========================================
    # 2. 预加载映射字典 (如需)
    # ==========================================
    tid2gene = {}
    if target_level == 'Gene':
        print(f"  -> Loading transcript metadata mapping from {meta_pkl_path}...")
        try:
            with open(meta_pkl_path, 'rb') as f:
                transcript_meta = pickle.load(f)
            for tid, info in transcript_meta.items():
                clean_tid = str(tid).split('.')[0]
                if isinstance(info, dict) and 'gene_id' in info:
                    raw_gene = info['gene_id']
                elif hasattr(info, 'gene_id'):
                    raw_gene = info.gene_id
                else:
                    raw_gene = info
                tid2gene[clean_tid] = str(raw_gene).split('.')[0]
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata pickle: {e}")

    # ==========================================
    # 3. 遍历模型数据、过滤细胞系、映射聚合与合并
    # ==========================================
    model_names = list(data_config.keys())
    
    for idx, (model_name, file_paths) in enumerate(data_config.items()):
        print(f"\nProcessing model: {model_name}")
        
        # --- [NEW] 解析当前模型需要保留的细胞类型 ---
        target_cell = None
        if target_cell_types is not None:
            if isinstance(target_cell_types, list):
                target_cell = target_cell_types[idx]
            elif isinstance(target_cell_types, dict):
                target_cell = target_cell_types.get(model_name)
            elif isinstance(target_cell_types, str):
                target_cell = target_cell_types # 假设全都用同一个细胞系
                
        if target_cell:
            print(f"  -> Target Cell Type filter: {target_cell}")
            
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            
        model_dfs = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"  [Warning] File not found: {file_path}")
                continue
            try:
                sep = '\t' if file_path.endswith(('.tsv', '.txt')) else ','
                df = pd.read_csv(file_path, sep=sep)
                model_dfs.append(df)
            except Exception as e:
                print(f"  [Error] Could not read {file_path}: {e}")
                
        if not model_dfs:
            continue
            
        combined_df = pd.concat(model_dfs, ignore_index=True)
        
        if model_metric not in combined_df.columns or 'Cell_Type' not in combined_df.columns:
            print(f"  [Warning] Missing metric '{model_metric}' or 'Cell_Type' in {model_name}. Skipping...")
            continue
            
        # --- [NEW] 过滤目标细胞类型 ---
        if target_cell:
            combined_df = combined_df[combined_df['Cell_Type'] == target_cell]
            if combined_df.empty:
                print(f"  [Warning] Target cell type '{target_cell}' not found in the output of {model_name}. Skipping...")
                continue

        # --------- ID 映射逻辑 ---------
        has_tid = [c for c in ['Tid', 'EnsemblTranscriptID', 'TranscriptID'] if c in combined_df.columns]
        has_gid = [c for c in ['Gid', 'EnsemblGeneID', 'GeneID'] if c in combined_df.columns]
        
        if target_level == 'Gene':
            if has_gid:
                combined_df['ID_clean'] = combined_df[has_gid[0]].astype(str).str.split('.').str[0]
            elif has_tid:
                tid_col = has_tid[0]
                combined_df['clean_tid_temp'] = combined_df[tid_col].astype(str).str.split('.').str[0]
                combined_df['ID_clean'] = combined_df['clean_tid_temp'].map(tid2gene)
                combined_df = combined_df.dropna(subset=['ID_clean'])
            else:
                continue
        else:
            if has_tid:
                combined_df['ID_clean'] = combined_df[has_tid[0]].astype(str).str.split('.').str[0]
            else:
                continue

        # --------- 聚合与合并 ---------
        combined_df_agg = combined_df.groupby(['ID_clean', 'Cell_Type'], as_index=False)[model_metric].mean()
        merged_df = pd.merge(combined_df_agg, ref_clean, on='ID_clean', how='inner')
        
        if merged_df.empty:
            print(f"  [Warning] No matching IDs found between {model_name} and Polysome reference.")
            continue
            
        print(f"  [Info] Matched {len(merged_df)} total pairs.")

        # --------- 计算分组相关性 ---------
        for cell_type, group_df in merged_df.groupby('Cell_Type'):
            group_clean = group_df.dropna(subset=[model_metric, ref_metric])
            
            if len(group_clean) < 5:
                continue
                
            x = group_clean[model_metric].values
            y = group_clean[ref_metric].values
            
            if corr_method.lower() == 'spearman':
                r_val, p_val = spearmanr(x, y)
            else:
                r_val, p_val = pearsonr(x, y)
                
            aggregated_data.append({
                'Cell_type': cell_type,
                'Model': model_name,
                'Mean': r_val,
                'P_value': p_val,
                'N': len(group_clean)
            })

    return pd.DataFrame(aggregated_data)


def plot_polysome_correlation_performance(agg_df, cell_types=None, model_metric="mORF Density", ref_metric="High vs Low FC", corr_method="Spearman", out_dir="./", suffix=""):
    """
    通用绘图函数。动态修改了 Y 轴标题以展示正在比较的指标。
    """
    if agg_df.empty:
        print("No data to plot.")
        return
        
    agg_df = agg_df.copy()
    agg_df['Cell_type'] = agg_df['Cell_type'].replace({'SW480': 'SW480 (Unseen)'})

    model_order = [
        "TRACE", "Encoder", "Convolution", 
        "Translatomer", "RiboMIMO (CDS)", 
        "Raw-dataset"
    ]
    
    if cell_types:
        cell_types = ['SW480 (Unseen)' if ct == 'SW480' else ct for ct in cell_types]
        agg_df = agg_df[agg_df["Cell_type"].isin(cell_types)]

    valid_models = [m for m in model_order if m in agg_df['Model'].unique()]
    agg_df['Model'] = pd.Categorical(agg_df['Model'], categories=valid_models, ordered=True)
    if cell_types:
        agg_df['Cell_type'] = pd.Categorical(agg_df['Cell_type'], categories=cell_types, ordered=True)

    summary_df = agg_df.groupby('Model', observed=False).agg(
        Overall_Mean=('Mean', 'mean'),
        SEM=('Mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0) 
    ).reset_index()
    
    summary_df['SEM'] = summary_df['SEM'].fillna(0)
    summary_df['ymin'] = summary_df['Overall_Mean'] - summary_df['SEM']
    summary_df['ymax'] = summary_df['Overall_Mean'] + summary_df['SEM']

    color_mapping = {
        "TRACE": "#2C6B9A", "Encoder": "#555555", "Convolution": "#777777",
        "Translatomer": "#999999", "RiboMIMO (CDS)": "#BBBBBB",
        "Raw-dataset": "#AF804F"
    }

    unique_cells = agg_df['Cell_type'].unique()
    point_colors = {ct: "#202020" for ct in unique_cells}
    if "SW480 (Unseen)" in point_colors:
        point_colors["SW480 (Unseen)"] = "#E74C3C" 

    y_max_data = summary_df['ymax'].max() if not summary_df.empty else 0.5
    y_limit = max(0.5, y_max_data * 1.2)

    plot = (
        ggplot(mapping=aes(x='Model'))
        + geom_col(data=summary_df, mapping=aes(y='Overall_Mean', fill='Model'), width=0.7)
        + geom_errorbar(data=summary_df, mapping=aes(ymin='ymin', ymax='ymax'), width=0.2, size=0.8, color="black")
        + geom_jitter(data=agg_df, mapping=aes(y='Mean', shape='Cell_type', color='Cell_type'), 
                      width=0.2, height=0, size=3.5, alpha=0.8)
        + scale_fill_manual(values=color_mapping) 
        + scale_color_manual(values=point_colors) 
        # + coord_cartesian(ylim=[0, y_limit]) 
        + theme_bw()
        + theme(
            axis_text_x=element_text(angle=45, hjust=1, color="black"), 
            axis_title_x=element_blank(),
            panel_grid_major_x=element_blank(),
            legend_position='right',
            legend_title=element_text(fontweight='bold')
        )
        + labs(
            # 动态 Y 轴标签：说明是哪个模型指标对应哪个 polysome 指标
            y=f"{corr_method.capitalize()} Corr: {model_metric} vs {ref_metric}",
            fill="Model",
            shape="Cell Type", 
            color="Cell Type" 
        )
    )
    
    os.makedirs(out_dir, exist_ok=True)
    # 将指标名加入文件名以区分保存
    safe_m_metric = model_metric.replace('_', '')
    safe_r_metric = ref_metric.replace('_', '')
    file_suffix = f".{suffix}" if suffix else ""
    save_path = os.path.join(out_dir, f"polysome_corr_{safe_m_metric}_vs_{safe_r_metric}{file_suffix}.pdf")
    
    plot.save(save_path, width=7, height=5, dpi=300, verbose=False)
    print(f"Plot saved to: {save_path}")