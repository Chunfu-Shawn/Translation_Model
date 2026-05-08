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
        ref_dfs: dict,           
        target_cell_types=None,  
        model_metric="mORF_Mean_Density", 
        ref_metric="High_vs_Low_FC", 
        corr_method="spearman",
        meta_pkl_path="/home/user/data3/rbase/translation_model/models/lib/transcript_meta.pkl"):
    """
    Load model prediction results and calculate correlation with multiple Polysome profiling reference tables.
    [NEW] Automatically computes the strict intersection of IDs across ALL models for each (Dataset, Cell Type) pair,
    guaranteeing that every model is evaluated on the exact same set of transcripts/genes for fair benchmarking.
    """
    aggregated_data = []
    print(f"--- Processing Data ---")
    print(f"  Target Model Metric : {model_metric} (Fallback: TE)")
    print(f"  Target Ref Metric   : {ref_metric} (Fallback: Ribosome_Load)")
    print(f"  Method              : {corr_method.capitalize()}")

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
        current_ref_metric = ref_metric
        if current_ref_metric not in ref_df.columns:
            if "Ribosome_Load" in ref_df.columns:
                print(f"  [Info] '{ref_metric}' not found in {ref_name}. Defaulting to 'Ribosome_Load'.")
                current_ref_metric = "Ribosome_Load"
            else:
                print(f"  [Warning] Missing metric '{ref_metric}' AND 'Ribosome_Load' in {ref_name}. Skipping this dataset.")
                continue

        ref_merge_key = next((col for col in id_cols_master if col in ref_df.columns), None)
        if not ref_merge_key:
            print(f"  [Warning] No valid ID column found in {ref_name}. Skipping...")
            continue
            
        target_level = 'Transcript' if ref_merge_key in ['Tid', 'EnsemblTranscriptID'] else 'Gene'
        
        ref_clean = ref_df[[ref_merge_key, current_ref_metric]].copy()
        ref_clean = ref_clean.dropna(subset=[current_ref_metric])
        ref_clean['ID_clean'] = ref_clean[ref_merge_key].astype(str).str.split('.').str[0]
        
        # Aggregate multiple TIDs targeting same GID in reference
        ref_clean = ref_clean.groupby('ID_clean', as_index=False)[current_ref_metric].mean()
        
        processed_refs[ref_name] = {
            'df': ref_clean,
            'level': target_level,
            'metric': current_ref_metric
        }
        print(f"  -> Ready Reference [{ref_name}]: ID = {ref_merge_key} ({target_level}), Metric = {current_ref_metric}")

    if not processed_refs:
        raise ValueError("No valid reference datasets processed.")

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
    # 3. Phase 1: Load Models & Collect ID Sets
    # ==========================================
    # Stores {model_name: {ref_name: merged_df}}
    processed_models = {}  
    # Stores {(ref_name, cell_type): [set(ids_model_A), set(ids_model_B), ...]}
    global_id_sets = {}    
    
    print("\n--- Phase 1: Data Loading & Initial Mapping ---")
    for idx, (model_name, file_paths) in enumerate(data_config.items()):
        print(f"Processing model: {model_name}")
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
        
        current_model_metric = model_metric
        if current_model_metric not in combined_df_raw.columns:
            if 'TE' in combined_df_raw.columns:
                current_model_metric = 'TE'
            else:
                print(f"  [Warning] Missing metric '{model_metric}' and 'TE' in {model_name}. Skipping...")
                continue
                
        if 'Cell_Type' not in combined_df_raw.columns:
            continue

        processed_models[model_name] = {}
        
        for ref_name, ref_info in processed_refs.items():
            ref_level = ref_info['level']
            ref_metric_name = ref_info['metric']
            ref_clean = ref_info['df']
            
            combined_df = combined_df_raw.copy()
            
            # Extract target cell type
            target_cell = target_cell_map.get(ref_name)
            if target_cell:
                combined_df = combined_df[combined_df['Cell_Type'] == target_cell]
                if combined_df.empty:
                    continue
            
            has_tid = [c for c in ['Tid', 'EnsemblTranscriptID', 'TranscriptID'] if c in combined_df.columns]
            has_gid = [c for c in ['Gid', 'EnsemblGeneID', 'GeneID'] if c in combined_df.columns]

            if ref_level == 'Gene':
                if has_gid:
                    combined_df['ID_clean'] = combined_df[has_gid[0]].astype(str).str.split('.').str[0]
                elif has_tid:
                    combined_df['clean_tid_temp'] = combined_df[has_tid[0]].astype(str).str.split('.').str[0]
                    combined_df['ID_clean'] = combined_df['clean_tid_temp'].map(tid2gene)
                    combined_df = combined_df.dropna(subset=['ID_clean'])
                else:
                    continue
            else: 
                if has_tid:
                    combined_df['ID_clean'] = combined_df[has_tid[0]].astype(str).str.split('.').str[0]
                else:
                    continue

            # Aggregate Model & Merge with Reference
            combined_df_agg = combined_df.groupby(['ID_clean', 'Cell_Type'], as_index=False)[current_model_metric].mean()
            merged_df = pd.merge(combined_df_agg, ref_clean, on='ID_clean', how='inner')
            
            if merged_df.empty:
                continue

            processed_models[model_name][ref_name] = (merged_df, current_model_metric, ref_metric_name)
            
            # Record mapped IDs
            for cell_type, group_df in merged_df.groupby('Cell_Type'):
                group_clean = group_df.dropna(subset=[current_model_metric, ref_metric_name])
                key = (ref_name, cell_type)
                if key not in global_id_sets:
                    global_id_sets[key] = []
                global_id_sets[key].append(set(group_clean['ID_clean']))

    # ==========================================
    # 4. Phase 2: Compute Strict Intersections
    # ==========================================
    print("\n--- Phase 2: Intersecting Common IDs across Models ---")
    intersected_ids_dict = {}
    expected_model_count = len(processed_models)
    
    for key, sets_list in global_id_sets.items():
        ref_name, cell_type = key
        if len(sets_list) < expected_model_count:
            print(f"  [Warning] Dataset '{ref_name}' (Cell: {cell_type}) missing in some models.")
        
        # Take the intersection of all sets collected for this (Dataset, Cell) pair
        common_ids = set.intersection(*sets_list)
        intersected_ids_dict[key] = common_ids
        print(f"  -> {ref_name:<20} | {cell_type:<10} | Fair IDs found: {len(common_ids)}")

    # ==========================================
    # 5. Phase 3: Filter & Calculate Correlations
    # ==========================================
    print("\n--- Phase 3: Calculating Fair Correlations ---")
    for model_name, ref_dict in processed_models.items():
        print(f"\nEvaluating: {model_name}")
        
        for ref_name, (merged_df, current_model_metric, ref_metric_name) in ref_dict.items():
            for cell_type, group_df in merged_df.groupby('Cell_Type'):
                
                valid_ids = intersected_ids_dict.get((ref_name, cell_type), set())
                group_clean_raw = group_df.dropna(subset=[current_model_metric, ref_metric_name])
                n_before = len(group_clean_raw)
                
                # Apply Strict Intersection Filter
                group_clean = group_clean_raw[group_clean_raw['ID_clean'].isin(valid_ids)]
                n_after = len(group_clean)
                
                print(f"  -> {ref_name:<15} ({cell_type}) | N Filtered: {n_before:>5} -> {n_after:>5}")

                if n_after < 5:
                    continue

                # Quantile filtering (if needed, ensure it's applied after intersection)
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
                    
                aggregated_data.append({
                    'Dataset': ref_name,           
                    'Cell_type': cell_type,
                    'Model': model_name,
                    'Mean': r_val,
                    'P_value': p_val,
                    'N': len(group_clean) # Logging the exact number of used samples
                })

    return pd.DataFrame(aggregated_data)