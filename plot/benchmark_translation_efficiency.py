import os
import numpy as np
import pandas as pd
import pickle
from plotnine import *
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr


def load_and_calculate_te_correlation(
        data_config, 
        ref_df, 
        metric="mORF_Mean_Density", 
        corr_method="spearman",
        eval_level="auto", 
        target_cell_types=None, 
        out_dir="./",
        meta_pkl_path="/home/user/data3/rbase/translation_model/models/lib/transcript_meta.pkl"
        ):
    """
    Load model prediction results and calculate correlation with the Protein-to-mRNA ratio reference table.
    Supports automatically mapping and aggregating model results from Transcript ID to Gene ID level using meta_pkl_path.
    Automatically computes the strict intersection of IDs across ALL models per Cell Type.
    [NEW] Supports filtering evaluation to a specific list of target_cell_types.
    """
    aggregated_data = []
    print(f"--- Processing Data (Target Metric: {metric}, Method: {corr_method.capitalize()}, Level: {eval_level}) ---")

    # ==========================================
    # 1. Identify the primary ID column in the reference table
    # ==========================================
    id_cols_master = ['GeneName', 'Gid', 'Tid', 'EnsemblGeneID', 'EnsemblTranscriptID', 'EnsemblProteinID']
    available_id_cols = [c for c in id_cols_master if c in ref_df.columns]
    val_cols = [c for c in ref_df.columns if c not in available_id_cols]
    
    if eval_level.lower() == "gid":
        if any(c in available_id_cols for c in ['Gid', 'EnsemblGeneID']):
            ref_merge_key = [c for c in ['Gid', 'EnsemblGeneID'] if c in available_id_cols][0]
            target_level = 'Gene'
        else:
            raise ValueError("eval_level set to 'Gid', but no Gene ID column found in reference table.")
            
    elif eval_level.lower() == "tid":
        if any(c in available_id_cols for c in ['Tid', 'EnsemblTranscriptID']):
            ref_merge_key = [c for c in ['Tid', 'EnsemblTranscriptID'] if c in available_id_cols][0]
            target_level = 'Transcript'
        else:
            raise ValueError("eval_level set to 'Tid', but no Transcript ID column found in reference table.")
            
    else:
        if any(c in available_id_cols for c in ['Tid', 'EnsemblTranscriptID']):
            ref_merge_key = [c for c in ['Tid', 'EnsemblTranscriptID'] if c in available_id_cols][0]
            target_level = 'Transcript'
        elif any(c in available_id_cols for c in ['Gid', 'EnsemblGeneID']):
            ref_merge_key = [c for c in ['Gid', 'EnsemblGeneID'] if c in available_id_cols][0]
            target_level = 'Gene'
        else:
            raise ValueError("Reference table must contain 'Tid', 'Gid', 'EnsemblTranscriptID', or 'EnsemblGeneID'.")
        
    print(f"  -> Detected target ID level in Reference: {ref_merge_key} ({target_level} Level)")

    ref_long = ref_df.melt(id_vars=available_id_cols, value_vars=val_cols, var_name='Cell_Type', value_name='PTR')
    ref_long = ref_long.dropna(subset=['PTR'])
    ref_long['ID_clean'] = ref_long[ref_merge_key].astype(str).str.split('.').str[0]

    # =================================================================
    # [NEW] 根据用户指定的细胞系进行极速过滤
    # =================================================================
    if target_cell_types is not None:
        if isinstance(target_cell_types, str):
            target_cell_types = [target_cell_types]
            
        before_ct_filter = len(ref_long)
        ref_long = ref_long[ref_long['Cell_Type'].isin(target_cell_types)]
        print(f"  -> [NEW] Filtered Reference by target cell types: {before_ct_filter} -> {len(ref_long)} records.")
        
        if ref_long.empty:
            print("  [Warning] Reference table is empty after cell type filtering. Please check if your target_cell_types match the reference columns.")
            return pd.DataFrame()

    # 确保 Reference 表内没有重复 ID (针对 Gid 包含多 Tid 的情况求 PTR 均值)
    before_agg = len(ref_long)
    ref_long = ref_long.groupby(['ID_clean', 'Cell_Type'], as_index=False)['PTR'].mean()
    if len(ref_long) < before_agg:
        print(f"  -> Averaged PTR for multiple records mapping to the same {target_level} ID ({before_agg} -> {len(ref_long)}).")

    # ==========================================
    # 2. Preload mapping dictionary if the target is Gene Level
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
    # 3. Phase 1: Load all models and collect ID sets per Cell Type
    # ==========================================
    processed_models = {}      
    cell_type_id_sets = {}     
    
    print("\n--- Phase 1: Data Loading & ID Extraction ---")
    for model_name, file_paths in data_config.items():
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
        
        has_tid = [c for c in ['Tid', 'EnsemblTranscriptID', 'TranscriptID'] if c in combined_df.columns]
        has_gid = [c for c in ['Gid', 'EnsemblGeneID', 'GeneID'] if c in combined_df.columns]
        
        if 'Cell_Type' not in combined_df.columns:
            print(f"  [Warning] Missing 'Cell_Type' in {model_name}. Skipping...")
            continue

        current_metric = metric
        if current_metric not in combined_df.columns:
            if 'TE' in combined_df.columns:
                current_metric = 'TE'
            else:
                print(f"  [Warning] Missing metric in {model_name}. Skipping...")
                continue

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

        combined_df_agg = combined_df.groupby(['ID_clean', 'Cell_Type'], as_index=False)[current_metric].mean()
        
        # Merge with Reference (由于 Reference 已经被 target_cell_types 过滤，这里的 inner join 会自动扔掉模型中不要的细胞系数据)
        merged_df = pd.merge(combined_df_agg, ref_long, on=['ID_clean', 'Cell_Type'], how='inner')
        
        if merged_df.empty:
            print(f"  [Warning] {model_name} yielded 0 matches with Reference.")
            continue
            
        processed_models[model_name] = (merged_df, current_metric)
        
        for cell_type, group in merged_df.groupby('Cell_Type'):
            if cell_type not in cell_type_id_sets:
                cell_type_id_sets[cell_type] = []
            cell_type_id_sets[cell_type].append(set(group['ID_clean']))

    # ==========================================
    # 4. Phase 2: Compute Strict Intersections
    # ==========================================
    print("\n--- Phase 2: Intersecting Common IDs across Models ---")
    intersected_ids_per_cell = {}
    expected_model_count = len(processed_models)
    
    for cell_type, sets_list in cell_type_id_sets.items():
        if len(sets_list) < expected_model_count:
            print(f"  [Warning] Cell '{cell_type}' is missing in some models. Strict intersection might be unfair, but proceeding with available models.")
        
        common_ids = set.intersection(*sets_list)
        intersected_ids_per_cell[cell_type] = common_ids
        print(f"  -> Cell '{cell_type}': Found {len(common_ids)} strictly common IDs across models.")

    # ==========================================
    # 5. Phase 3: Filter, Calculate & Log
    # ==========================================
    print("\n--- Phase 3: Calculating Fair Correlations ---")
    for model_name, (merged_df, current_metric) in processed_models.items():
        print(f"\nEvaluating: {model_name}")
        
        for cell_type, group_df in merged_df.groupby('Cell_Type'):
            valid_ids = intersected_ids_per_cell.get(cell_type, set())
            
            n_before = len(group_df)
            group_clean = group_df[group_df['ID_clean'].isin(valid_ids)].dropna(subset=[current_metric, 'PTR'])
            n_after = len(group_clean)
            
            print(f"  -> {cell_type:<15} | N Filtered: {n_before:>5} -> {n_after:>5}")
            
            if n_after < 5:
                print(f"     [Skipped] Insufficient samples ({n_after} < 5) after filtering.")
                continue
                
            x = group_clean[current_metric].values
            y = group_clean['PTR'].values
            
            if corr_method.lower() == 'spearman':
                r_val, p_val = spearmanr(x, y)
            else:
                r_val, p_val = pearsonr(x, y)
                
            aggregated_data.append({
                'Cell_type': cell_type,
                'Model': model_name,
                'Mean': r_val, 
                'P_value': p_val,
                'N': n_after 
            })

    df = pd.DataFrame(aggregated_data)
    
    os.makedirs(out_dir, exist_ok=True)
    file_suffix = f".{metric}.{corr_method}.{eval_level}"
    save_path = os.path.join(out_dir, f"translation_efficiency_metrics{file_suffix}.csv")
    df.to_csv(save_path, index=False)
    print(f"\n✅ Correlation efficiency saved to: {save_path}")

    return df


def plot_te_correlation_performance(
        agg_df, cell_types=None, metric_name="mORF Mean Density", 
        corr_method="Spearman", out_dir="./", suffix="", no_color=False):
    """
    使用 plotnine 绘制 Bar + Errorbar(SEM) + Jitter Points，展示跨细胞类型的模型相关性表现。
    """
    if agg_df.empty:
        print("No data to plot.")
        return
        
    agg_df = agg_df.copy()

    # 处理特定的未见细胞系标注
    agg_df['Cell_type'] = agg_df['Cell_type'].replace({'lung': 'lung (unseen)'})
    agg_df['Cell_type'] = pd.Categorical(
        agg_df['Cell_type'], 
        categories=["brain_cerebrum","kidney","liver","prostate","testis","lung (unseen)"], 
        ordered=True
        )

    # 设置因子的顺序
    model_order = [
        "TRACE", "Encoder", "Convolution", 
        "Optimus-5Prime", "RiboDecode", "RiboNN",
        "Raw-dataset", "Transcript-Mean", 
        "Inverse CDS length", "Inverse 5'UTR length", "Inverse 3'UTR length",
        "Inverse CDS GC%", "CAI", "Kozak score", "Inverse uAUG count"
    ]
    
    if cell_types:
        cell_types = ['lung (unseen)' if ct == 'lung' else ct for ct in cell_types]
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
        # 深度学习模型 (冷色调 + 灰度渐变)
        "TRACE": "#2C6B9A", 
        "Encoder": "#555555", 
        "Convolution": "#777777",
        "Optimus-5Prime": "#999999",
        "RiboDecode": "#BBBBBB",
        "RiboNN": "#DDDDDD",
        
        # Baseline 特征 (大地色系/暖金渐变)
        "Transcript-Mean": "#AF804F",  # 阶梯 1 (最深)
        "Inverse 5'UTR length": "#AF804F",  # 阶梯 2
        "Inverse 3'UTR length": "#B98C57",
        "Inverse CDS length": "#C3975F",   # 阶梯 3
        "Inverse CDS GC%": "#CDA367",      # 阶梯 4
        "CAI": "#D7AF6F",                  # 阶梯 5
        "Kozak score": "#E1BA77",          # 阶梯 6
        "Inverse uAUG count": "#EBC67F"    # 阶梯 7 (最浅)
    }

    # 为散点图生成颜色字典，单独高亮 lung
    unique_cells = agg_df['Cell_type'].unique()
    point_colors = {ct: "#202020" for ct in unique_cells}
    if "lung (unseen)" in point_colors:
        point_colors["lung (unseen)"] = "#E74C3C" 

    # 动态获取 Y 轴上限，防止柱状图冲出画框 (预留 20% 空间)
    y_max_data = summary_df['ymax'].max() if not summary_df.empty else 0.5
    y_limit = max(0.5, y_max_data * 1.2)

    plot = (
        ggplot(mapping=aes(x='Model'))
        + geom_col(data=summary_df, mapping=aes(y='Overall_Mean', fill='Model'), width=0.7)
        + geom_errorbar(data=summary_df, mapping=aes(ymin='ymin', ymax='ymax'), width=0.2, size=0.8, color="black")
        + geom_jitter(data=agg_df, mapping=aes(y='Mean', shape='Cell_type', color='Cell_type'), 
                      width=0.2, size=3.5, alpha=0.8)
        + scale_color_manual(values=point_colors) 
        # + coord_cartesian(ylim=[0, y_limit]) # 动态 Y 轴
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

    if not no_color:
        plot += scale_fill_manual(values=color_mapping)
    else:
        plot += scale_fill_manual(values="gray")

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"te_correlation_performance_{metric_name}.{suffix}.pdf")
    plot.save(save_path, width=6, height=5, dpi=300, verbose=False)
    print(f"Plot saved to: {save_path}")


def load_and_calculate_polysome_correlation(
        data_config: dict, 
        ref_dfs: dict,           # Accept multiple datasets in dictionary format {"Dataset_A": df_A, "Dataset_B": df_B}
        target_cell_types=None,  # [MODIFIED] Cell lines corresponding to ref_dfs; supports list, dict, or string
        model_metric="mORF_Mean_Density", 
        ref_metric="High_vs_Low_FC", 
        corr_method="spearman",
        target_ids=None,         # [NEW] Array/list of target Transcript IDs to evaluate on
        meta_pkl_path="/home/user/data3/rbase/translation_model/models/lib/transcript_meta.pkl"):
    """
    Load model prediction results and calculate correlation with multiple Polysome profiling reference tables.
    Supports intelligent metric fallback for both model outputs (TE) and reference metrics (Ribosome_Load).
    Supports assigning specific cell lines to each reference dataset for precise extraction.
    [NEW] Filters the evaluation to only include the specified `target_ids` for fair cross-model comparison.
    """
    aggregated_data = []
    print(f"--- Processing Data ---")
    print(f"  Target Model Metric : {model_metric} (Fallback: TE)")
    print(f"  Target Ref Metric   : {ref_metric} (Fallback: Ribosome_Load)")
    print(f"  Method              : {corr_method.capitalize()}")

    # [MODIFIED] Validate target_cell_types and build mapping dictionary
    target_cell_map = {}
    if target_cell_types is not None:
        if isinstance(target_cell_types, list):
            if len(target_cell_types) != len(ref_dfs):
                raise ValueError("The length of target_cell_types list must match the number of datasets in ref_dfs.")
            target_cell_map = {list(ref_dfs.keys())[i]: ct for i, ct in enumerate(target_cell_types)}
        elif isinstance(target_cell_types, dict):
            target_cell_map = target_cell_types
        elif isinstance(target_cell_types, str):
            target_cell_map = {k: target_cell_types for k in ref_dfs.keys()}

    # ==========================================
    # 1. Preprocess all Reference Datasets
    # ==========================================
    processed_refs = {}
    id_cols_master = ['Tid', 'EnsemblTranscriptID', 'Gid', 'EnsemblGeneID', 'gene_name']
    
    for ref_name, ref_df in ref_dfs.items():
        # Determine the metric used for the current dataset
        current_ref_metric = ref_metric
        if current_ref_metric not in ref_df.columns:
            if "Ribosome_Load" in ref_df.columns:
                print(f"  [Info] '{ref_metric}' not found in {ref_name}. Defaulting to 'Ribosome_Load'.")
                current_ref_metric = "Ribosome_Load"
            else:
                print(f"  [Warning] Missing metric '{ref_metric}' AND 'Ribosome_Load' in {ref_name}. Skipping this dataset.")
                continue

        # Identify the primary ID column
        ref_merge_key = next((col for col in id_cols_master if col in ref_df.columns), None)
        if not ref_merge_key:
            print(f"  [Warning] No valid ID column found in {ref_name}. Skipping...")
            continue
            
        target_level = 'Transcript' if ref_merge_key in ['Tid', 'EnsemblTranscriptID'] else 'Gene'
        
        # Extract, remove version numbers, and aggregate
        ref_clean = ref_df[[ref_merge_key, current_ref_metric]].copy()
        ref_clean = ref_clean.dropna(subset=[current_ref_metric])
        ref_clean['ID_clean'] = ref_clean[ref_merge_key].astype(str).str.split('.').str[0]
        ref_clean = ref_clean.groupby('ID_clean', as_index=False)[current_ref_metric].mean()
        
        processed_refs[ref_name] = {
            'df': ref_clean,
            'level': target_level,
            'metric': current_ref_metric
        }
        print(f"  -> Ready Reference [{ref_name}]: ID = {ref_merge_key} ({target_level}), Metric = {current_ref_metric}")

    if not processed_refs:
        raise ValueError("No valid reference datasets processed. Please check your reference data structures.")

    # ==========================================
    # 2. Preload metadata mapping if required
    # ==========================================
    tid2gene = {}
    if any(info['level'] == 'Gene' for info in processed_refs.values()):
        print(f"\n  -> Loading transcript metadata mapping from {meta_pkl_path}...")
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
    # [NEW] 2.5 Filter reference datasets using target_ids
    # ==========================================
    if target_ids is not None:
        clean_target_tids = {str(tid).split('.')[0] for tid in target_ids}
        
        for ref_name, ref_info in processed_refs.items():
            ref_level = ref_info['level']
            ref_df = ref_info['df']
            
            if ref_level == 'Gene':
                # Map target TIDs to GIDs for Gene-level references
                final_target_ids = {tid2gene[tid] for tid in clean_target_tids if tid in tid2gene}
            else:
                final_target_ids = clean_target_tids
                
            before_filter = len(ref_df)
            ref_info['df'] = ref_df[ref_df['ID_clean'].isin(final_target_ids)]
            print(f"  -> [NEW] Filtered Reference [{ref_name}] targets: {before_filter} -> {len(ref_info['df'])} records.")

    # ==========================================
    # 3. Iterate through models and align with references
    # ==========================================
    for idx, (model_name, file_paths) in enumerate(data_config.items()):
        print(f"\nProcessing model: {model_name}")
        
        # --- Read model data ---
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            
        model_dfs = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                sep = '\t' if file_path.endswith(('.tsv', '.txt')) else ','
                model_dfs.append(pd.read_csv(file_path, sep=sep))
                
        if not model_dfs:
            continue
        combined_df_raw = pd.concat(model_dfs, ignore_index=True)
        
        # --- Model metric fallback logic ---
        current_model_metric = model_metric
        if current_model_metric not in combined_df_raw.columns:
            if 'TE' in combined_df_raw.columns:
                current_model_metric = 'TE'
                print(f"  [Info] Metric '{model_metric}' not found in {model_name}. Defaulting to 'TE'.")
            else:
                print(f"  [Warning] Missing metric '{model_metric}' and 'TE' in {model_name}. Skipping...")
                continue
                
        if 'Cell_Type' not in combined_df_raw.columns:
            print(f"  [Warning] Missing 'Cell_Type' column in {model_name}. Skipping...")
            continue

        # --------- Inner Loop: Compare against each reference dataset ---------
        for ref_name, ref_info in processed_refs.items():
            ref_level = ref_info['level']
            ref_metric_name = ref_info['metric']
            ref_clean = ref_info['df']
            
            combined_df = combined_df_raw.copy()
            
            # [MODIFIED] Extract specified cell type data for the current reference dataset
            target_cell = target_cell_map.get(ref_name)
            if target_cell:
                combined_df = combined_df[combined_df['Cell_Type'] == target_cell]
                if combined_df.empty:
                    print(f"  [Warning] Target cell '{target_cell}' not found in {model_name}. Skipping {ref_name}...")
                    continue
            
            has_tid = [c for c in ['Tid', 'EnsemblTranscriptID', 'TranscriptID'] if c in combined_df.columns]
            has_gid = [c for c in ['Gid', 'EnsemblGeneID', 'GeneID'] if c in combined_df.columns]

            # Determine mapping strategy based on reference level
            if ref_level == 'Gene':
                if has_gid:
                    combined_df['ID_clean'] = combined_df[has_gid[0]].astype(str).str.split('.').str[0]
                elif has_tid:
                    combined_df['clean_tid_temp'] = combined_df[has_tid[0]].astype(str).str.split('.').str[0]
                    combined_df['ID_clean'] = combined_df['clean_tid_temp'].map(tid2gene)
                    combined_df = combined_df.dropna(subset=['ID_clean'])
                else:
                    continue
            else: # Transcript level
                if has_tid:
                    combined_df['ID_clean'] = combined_df[has_tid[0]].astype(str).str.split('.').str[0]
                else:
                    continue

            # Aggregate model data and merge with current reference dataset
            combined_df_agg = combined_df.groupby(['ID_clean', 'Cell_Type'], as_index=False)[current_model_metric].mean()
            merged_df = pd.merge(combined_df_agg, ref_clean, on='ID_clean', how='inner')
            
            if merged_df.empty:
                continue

            # Calculate correlation for each cell line
            for cell_type, group_df in merged_df.groupby('Cell_Type'):
                group_clean = group_df.dropna(subset=[current_model_metric, ref_metric_name])
                
                # Filter 1% and 99% extreme outliers
                if len(group_clean) > 0:
                    p01_m = group_clean[current_model_metric].quantile(0.01)
                    p99_m = group_clean[current_model_metric].quantile(0.99)
                    p01_p = group_clean[ref_metric_name].quantile(0.01)
                    p99_p = group_clean[ref_metric_name].quantile(0.99)
                    
                    valid_mask = (
                        (group_clean[current_model_metric] >= p01_m) & (group_clean[current_model_metric] <= p99_m) &
                        (group_clean[ref_metric_name] >= p01_p) & (group_clean[ref_metric_name] <= p99_p)
                    )
                    group_clean = group_clean[valid_mask]

                if len(group_clean) < 5:
                    continue

                x = group_clean[current_model_metric].values
                y = group_clean[ref_metric_name].values
                
                if corr_method.lower() == 'spearman':
                    r_val, p_val = spearmanr(x, y)
                else:
                    r_val, p_val = pearsonr(x, y)
                    
                # Print the number of IDs used for correlation
                print(f"  -> [{ref_name} | {cell_type}] Tids used for correlation: {len(group_clean)}")
                    
                aggregated_data.append({
                    'Dataset': ref_name,           
                    'Cell_type': cell_type,
                    'Model': model_name,
                    'Mean': r_val,
                    'P_value': p_val,
                    'N': len(group_clean)
                })

    return pd.DataFrame(aggregated_data)


def plot_polysome_correlation_heatmap(agg_df, out_dir="./", corr_method="Spearman", suffix=""):
    """
    绘制相关性热图：
    - Y 轴：不同模型与 Baseline
    - X 轴：不同的多聚核糖体数据集
    - 颜色：模型在给定数据集所有细胞系中的平均相关性。
    """
    if agg_df.empty:
        print("No data to plot.")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    agg_df = agg_df.copy()
    
    # 汇总：计算给定模型在特定数据集下的平均表现
    heatmap_df = agg_df.groupby(['Model', 'Dataset'], observed=False)['Mean'].mean().reset_index()

    # 设置 Y 轴模型顺序
    model_order = [
        "TRACE", "Encoder", "Convolution", 
        "Optimus-5Prime", "RiboDecode", "RiboNN", 
        "Raw-dataset", 
        "CDS length", "5'UTR length", "3'UTR length",
        "CDS GC%", "CAI", "Kozak score", "uAUG count"
    ]
    
    # 提取实际存在的模型
    valid_models = [m for m in model_order if m in heatmap_df['Model'].unique()]

    # 将长表转换为宽表 (Pivot)，行是 Model，列是 Dataset
    pivot_df = heatmap_df.pivot(index='Model', columns='Dataset', values='Mean')
    
    # 按照设定的顺序进行排序 (Seaborn 会自然地从上往下绘制)
    pivot_df = pivot_df.loc[valid_models]

    # 补充完整的 color_mapping
    color_mapping = {
        # 深度学习模型 (冷色调 + 灰度渐变)
        "TRACE": "#2C6B9A", "Encoder": "#555555", "Convolution": "#777777",
        "Optimus-5Prime": "#999999",
        "RiboDecode": "#BBBBBB",
        "RiboNN": "#DDDDDD",
        "Raw-dataset": "#A0A0A0",
        
        # Baseline 特征 (大地色系/暖金渐变)
        "CDS length": "#AF804F",           
        "5'UTR length": "#B98C57", 
        "3'UTR length": "#C3975F", 
        "CDS GC%": "#CDA367",      
        "CAI": "#D7AF6F",                  
        "Kozak score": "#E1BA77",          
        "uAUG count": "#EBC67F"    
    }
    
    # 提取当前数据包含的颜色列表
    row_colors = [color_mapping[m] for m in pivot_df.index]

    # 根据数据集的数量动态调整画布宽高
    num_datasets = len(pivot_df.columns)
    plot_width = max(5, num_datasets * 0.8 + 2) # 稍微加宽一点适应更大的字体
    plot_height = len(valid_models) * 0.6 + 2

    # 使用 GridSpec 分割画面：[极窄的Bar区, 宽阔的热图区]
    fig, (ax_bar, ax_heatmap) = plt.subplots(
        ncols=2,
        figsize=(plot_width, plot_height),
        gridspec_kw={'width_ratios': [0.03, 1], 'wspace': 0.02} 
    )

    # 1. 绘制左侧的 Color Bar
    bar_matrix = np.arange(len(valid_models)).reshape(-1, 1)
    bar_cmap = ListedColormap(row_colors)

    sns.heatmap(
        bar_matrix,
        cmap=bar_cmap,
        cbar=False,           # 不需要图例
        annot=False,
        linewidths=1,         # 确保高度方向与右侧热图的网格间隙 100% 对齐
        linecolor='white', 
        ax=ax_bar
    )
    
    # 将模型名称（Y轴文本）放置在这个极细的 Color Bar 左侧
    ax_bar.set_yticks(np.arange(len(valid_models)) + 0.5)
    # [MODIFIED] 同步放大 Y 轴模型的字体以保持协调
    ax_bar.set_yticklabels(valid_models, rotation=0, fontsize=14) 
    ax_bar.set_xticks([])     
    ax_bar.tick_params(left=False) 

    # 2. 绘制右侧的主热图
    main_cmap = LinearSegmentedColormap.from_list("blue_white_red", ["#2C6B9A", "#FFFFFF", "#E74C3C"])

    sns.heatmap(
        pivot_df,
        cmap=main_cmap,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 14},
        linewidths=1, 
        linecolor='white',
        # [MODIFIED] 使用 shrink 参数将 Color bar 的高度缩短为 30%
        cbar_kws={'shrink': 0.3, 'aspect': 8}, 
        ax=ax_heatmap
    )
    
    # [MODIFIED] 获取刚才画的图例对象，单独设置其 Label 大小，防止字体太小
    cbar = ax_heatmap.collections[0].colorbar
    if cbar:
        cbar.set_label(f'{corr_method} correlation coefficient', size=14)
        cbar.ax.tick_params(labelsize=12) # 调整图例刻度的字体
        cbar.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])

    # 隐藏热图的 Y 轴标签（因为已经在左侧 Color Bar 处画了）
    ax_heatmap.set_yticks([])
    ax_heatmap.set_ylabel('')
    ax_heatmap.set_xlabel('')
    
    # [MODIFIED] 调整 X 轴标签方向和字体大小 (放大至 14)
    ax_heatmap.tick_params(axis='x', rotation=45, labelsize=14, bottom=False)
    
    # 将 x 轴标签对齐方式修改为右对齐，防止被截断
    for tick in ax_heatmap.get_xticklabels():
        tick.set_horizontalalignment('right')

    # ==========================================================
    # 保存图片
    # ==========================================================
    file_suffix = f".{suffix}" if suffix else ""
    save_path = os.path.join(out_dir, f"polysome_multidataset_heatmap{file_suffix}.pdf")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to: {save_path}")