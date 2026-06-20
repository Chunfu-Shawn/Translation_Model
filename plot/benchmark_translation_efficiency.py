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

# =================================================================
# Global Model Order & Color Mapping 
# =================================================================
GLOBAL_MODEL_ORDER = [
    "TRACE", "Encoder", "Convolution", 
    "Optimus-5Prime", "RiboDecode", "RiboNN",
    "Raw-dataset", "Mean density", "TE scale", "Transcript-Mean", 
    "Inverse CDS length", "Inverse 5'UTR length", "Inverse 3'UTR length",
    "Inverse CDS GC%", "CAI", "Kozak score", "Inverse uAUG count"
]

GLOBAL_MODEL_COLORS = {
    # Deep Learning Models (Cool tones & Grayscale)
    "TRACE": "#2C6B9A", 
    "Encoder": "#4A4A4A", # Fallback for Encoder if present
    "Convolution": "#637D96",
    "Optimus-5Prime": "#555555",
    "RiboDecode": "#777777",
    "RiboNN": "#BBBBBB",

    # Feature Baselines (Earth tones & Warm gold gradients)
    "Raw-dataset": "#8C6D51",
    "Mean density": "#967554",
    "TE scale": "#A07E58",
    "Inverse 5'UTR length": "#8C6D51",  
    "Inverse 3'UTR length": "#B98C57",
    "Inverse CDS length": "#C3975F",   
    "Inverse CDS GC%": "#CDA367",                   
    "Kozak score": "#AF804F",
    "CAI": "#D7AF6F",      
    "Inverse uAUG count": "#EBC67F"    
}


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

    if target_cell_types is not None:
        if isinstance(target_cell_types, str):
            target_cell_types = [target_cell_types]
            
        before_ct_filter = len(ref_long)
        ref_long = ref_long[ref_long['Cell_Type'].isin(target_cell_types)]
        print(f"  -> Filtered Reference by target cell types: {before_ct_filter} -> {len(ref_long)} records.")
        
        if ref_long.empty:
            print("  [Warning] Reference table is empty after cell type filtering. Please check if your target_cell_types match the reference columns.")
            return pd.DataFrame()

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
    Plot bar chart with error bars and jitter points for TE correlations.
    Dynamically handles ANY number of Cell Types by manually cycling shapes.
    """
    if agg_df.empty:
        print("No data to plot.")
        return
        
    agg_df = agg_df.copy()

    # Replace specific unseen label
    agg_df['Cell_type'] = agg_df['Cell_type'].replace({'lung': 'lung (unseen)'})
    
    # =================================================================
    # [NEW] Dynamically determine cell types to avoid hardcoding limits
    # =================================================================
    if cell_types:
        target_cells = ['lung (unseen)' if ct == 'lung' else ct for ct in cell_types]
        agg_df = agg_df[agg_df["Cell_type"].isin(target_cells)]
    else:
        # Extract all unique cell types present in the dataframe
        target_cells = agg_df['Cell_type'].dropna().unique().tolist()
        # Ensure 'lung (unseen)' stays at the end if it exists for consistent highlighting
        if 'lung (unseen)' in target_cells:
            target_cells.remove('lung (unseen)')
            target_cells.append('lung (unseen)')

    # Apply the dynamic categories
    agg_df['Cell_type'] = pd.Categorical(
        agg_df['Cell_type'], 
        categories=target_cells, 
        ordered=True
    )

    # Use global model order
    valid_models = [m for m in GLOBAL_MODEL_ORDER if m in agg_df['Model'].unique()]
    for m in agg_df['Model'].unique():
        if m not in valid_models:
            valid_models.append(m)
            
    agg_df['Model'] = pd.Categorical(agg_df['Model'], categories=valid_models, ordered=True)

    # Aggregate summaries
    summary_df = agg_df.groupby('Model', observed=False).agg(
        Overall_Mean=('Mean', 'mean'),
        SEM=('Mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0) 
    ).reset_index()
    
    summary_df['SEM'] = summary_df['SEM'].fillna(0)
    summary_df['ymin'] = summary_df['Overall_Mean'] - summary_df['SEM']
    summary_df['ymax'] = summary_df['Overall_Mean'] + summary_df['SEM']

    # Assign global model colors
    model_colors = {}
    for m in valid_models:
        model_colors[m] = GLOBAL_MODEL_COLORS.get(m, "#C0C0C0")

    # Assign Point Colors
    point_colors = {ct: "#202020" for ct in target_cells}
    if "lung (unseen)" in point_colors:
        point_colors["lung (unseen)"] = "#E74C3C" 

    # =================================================================
    # [NEW] Create a robust shape pool to prevent IndexError
    # =================================================================
    shapes_pool = ['o', '^', 's', 'D', 'v', '*', '<', '>', 'p', 'h', '8', 'X', 'd', 'P', 'H']
    point_shapes = {ct: shapes_pool[i % len(shapes_pool)] for i, ct in enumerate(target_cells)}

    plot = (
        ggplot(mapping=aes(x='Model'))
        + geom_col(data=summary_df, mapping=aes(y='Overall_Mean', fill='Model'), width=0.7)
        + geom_errorbar(data=summary_df, mapping=aes(ymin='ymin', ymax='ymax'), width=0.2, size=0.8, color="black")
        + geom_jitter(data=agg_df, mapping=aes(y='Mean', shape='Cell_type', color='Cell_type'), 
                      width=0.2, size=3.5, alpha=0.8)
        + scale_color_manual(values=point_colors) 
        + scale_shape_manual(values=point_shapes)  # [NEW] Force manual shape assignment
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
        plot += scale_fill_manual(values=model_colors)
    else:
        # Safely map all models to gray when no_color is True
        plot += scale_fill_manual(values={m: "gray" for m in valid_models})

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"te_correlation_performance_{metric_name}.{suffix}.pdf")
    plot.save(save_path, width=max(6, len(valid_models)*0.6), height=5, dpi=300, verbose=False)
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
        
        ref_clean = ref_clean.groupby('ID_clean', as_index=False)[current_ref_metric].mean()
        
        processed_refs[ref_name] = {
            'df': ref_clean,
            'level': target_level,
            'metric': current_ref_metric
        }
        print(f"  -> Ready Reference [{ref_name}]: ID = {ref_merge_key} ({target_level}), Metric = {current_ref_metric}")

    if not processed_refs:
        raise ValueError("No valid reference datasets processed.")

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

    processed_models = {}  
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

            combined_df_agg = combined_df.groupby(['ID_clean', 'Cell_Type'], as_index=False)[current_model_metric].mean()
            merged_df = pd.merge(combined_df_agg, ref_clean, on='ID_clean', how='inner')
            
            if merged_df.empty:
                continue

            processed_models[model_name][ref_name] = (merged_df, current_model_metric, ref_metric_name)
            
            for cell_type, group_df in merged_df.groupby('Cell_Type'):
                group_clean = group_df.dropna(subset=[current_model_metric, ref_metric_name])
                key = (ref_name, cell_type)
                if key not in global_id_sets:
                    global_id_sets[key] = []
                global_id_sets[key].append(set(group_clean['ID_clean']))

    print("\n--- Phase 2: Intersecting Common IDs across Models ---")
    intersected_ids_dict = {}
    expected_model_count = len(processed_models)
    
    for key, sets_list in global_id_sets.items():
        ref_name, cell_type = key
        if len(sets_list) < expected_model_count:
            print(f"  [Warning] Dataset '{ref_name}' (Cell: {cell_type}) missing in some models.")
        
        common_ids = set.intersection(*sets_list)
        intersected_ids_dict[key] = common_ids
        print(f"  -> {ref_name:<20} | {cell_type:<10} | Fair IDs found: {len(common_ids)}")

    print("\n--- Phase 3: Calculating Fair Correlations ---")
    for model_name, ref_dict in processed_models.items():
        print(f"\nEvaluating: {model_name}")
        
        for ref_name, (merged_df, current_model_metric, ref_metric_name) in ref_dict.items():
            for cell_type, group_df in merged_df.groupby('Cell_Type'):
                
                valid_ids = intersected_ids_dict.get((ref_name, cell_type), set())
                group_clean_raw = group_df.dropna(subset=[current_model_metric, ref_metric_name])
                n_before = len(group_clean_raw)
                
                group_clean = group_clean_raw[group_clean_raw['ID_clean'].isin(valid_ids)]
                n_after = len(group_clean)
                
                print(f"  -> {ref_name:<15} ({cell_type}) | N Filtered: {n_before:>5} -> {n_after:>5}")

                if n_after < 5:
                    continue

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
                    'N': len(group_clean) 
                })

    return pd.DataFrame(aggregated_data)


def plot_polysome_correlation_heatmap(
        agg_df: pd.DataFrame, 
        out_dir: str = "./", 
        metric_name: str = "Spearman correlation coefficient",
        suffix: str = ""):
    """
    绘制模型在 Polysome 数据集上的相关性热图 (Heatmap)。
    高度还原发表级美学：左侧带有模型专属颜色的边条，使用蓝白红的发散型色带，并附带白色网格分割线。
    """
    if agg_df.empty:
        print("No data to plot.")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    plot_df = agg_df.copy()

    # =================================================================
    # 1. 构建 X 轴标签 (Dataset + Cell_type) 并透视数据
    # =================================================================
    if 'Dataset' in plot_df.columns and 'Cell_type' in plot_df.columns:
        # 智能合并标签：如果 Dataset 名称中没有包含 Cell_type，则拼在一起
        plot_df['x_label'] = plot_df.apply(
            lambda x: f"{x['Dataset']}-{x['Cell_type']}" if str(x['Cell_type']) not in str(x['Dataset']) else x['Dataset'], 
            axis=1
        )
    elif 'Dataset' in plot_df.columns:
        plot_df['x_label'] = plot_df['Dataset']
    else:
        plot_df['x_label'] = plot_df['Cell_type']

    # 生成透视表: 行是 Model, 列是 数据集, 值是 相关性 Mean
    heatmap_data = plot_df.pivot(index='Model', columns='x_label', values='Mean')

    # =================================================================
    # 2. 排序 行(Model) 和 列(Dataset)
    # =================================================================
    valid_models = [m for m in GLOBAL_MODEL_ORDER if m in heatmap_data.index]
    for m in heatmap_data.index:
        if m not in valid_models:
            valid_models.append(m)
            
    heatmap_data = heatmap_data.reindex(valid_models)
    
    # 列按字母顺序排序，保证展现整齐
    heatmap_data = heatmap_data[sorted(heatmap_data.columns)]

    # =================================================================
    # 3. 绘图与布局逻辑 (GridSpec)
    # =================================================================
    print(f"Generating Polysome Heatmap...")
    
    # 动态计算画幅宽度和高度
    width = max(8, len(heatmap_data.columns) * 0.7 + 3)
    height = max(5, len(heatmap_data.index) * 0.6 + 2)
    
    fig = plt.figure(figsize=(width, height))
    
    # 将画布分为左右两部分：极窄的左侧用于画颜色条，右侧用于画热图
    gs = fig.add_gridspec(1, 2, width_ratios=[0.03, 1], wspace=0.01)
    
    ax_colors = fig.add_subplot(gs[0])
    ax_heatmap = fig.add_subplot(gs[1])

    # =================================================================
    # [MODIFIED] 修改色带：
    # 负值端保持冷蓝(240)，正值端将大红(10)改为橙红(20)
    # s(饱和度)从90降为80，l(亮度)从50提至55，使颜色更加柔和不扎眼
    # =================================================================
    cmap = sns.diverging_palette(240, 20, s=80, l=55, center="light", as_cmap=True)

    # 绘制核心 Heatmap
    sns.heatmap(
        heatmap_data, 
        annot=True,            # 显示数值
        fmt=".2f",             # 保留两位小数
        cmap=cmap,             # 蓝白橙红色带
        center=0,              # 强制 0 为白色
        linewidths=1,          # 白色网格线宽
        linecolor='white',     # 网格线颜色
        annot_kws={"size": 14}, 
        cbar_kws={
            'label': metric_name, 
            'shrink': 0.5,     # 缩短 Colorbar，复刻原图右侧悬浮风格
            'aspect': 15       # 调整 Colorbar 粗细
        },
        ax=ax_heatmap
    )

    # 隐藏 Heatmap 自带的 Y 轴标签（我们将把它转移到左侧颜色条上）
    ax_heatmap.set_yticks([])
    ax_heatmap.set_ylabel('')
    ax_heatmap.set_xlabel('')
    
    # 调整 X 轴标签的角度和对其方式 (此处如果觉得刻度字小，也可将 fontsize 调大)
    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=12)

    # =================================================================
    # 4. 绘制左侧的 Model Color Strip
    # =================================================================
    ax_colors.set_xlim(0, 1)
    ax_colors.set_ylim(len(heatmap_data.index), 0) # 方向与 seaborn heatmap 的倒序 Y 轴保持一致
    
    # 逐行绘制带有颜色的方块
    for i, model in enumerate(heatmap_data.index):
        color = GLOBAL_MODEL_COLORS.get(model, "#C0C0C0") # 默认颜色兜底
        # 加上白色的边框线，让颜色条也具有和热图一样的切割感
        rect = plt.Rectangle((0, i), 1, 1, facecolor=color, edgecolor='white', linewidth=1)
        ax_colors.add_patch(rect)

    # 将 Y 轴的模型名称标签添加到颜色条的左侧
    ax_colors.set_yticks(np.arange(len(heatmap_data.index)) + 0.5)
    # [MODIFIED] 如果觉得模型名字太小，可将这里的 fontsize 从 12 提至 13 或 14
    ax_colors.set_yticklabels(heatmap_data.index, fontsize=13)
    ax_colors.set_xticks([])
    
    # 移除颜色条自带的所有坐标轴边框
    ax_colors.tick_params(axis='y', length=0)
    for spine in ax_colors.spines.values():
        spine.set_visible(False)

    # =================================================================
    # 5. 保存图表
    # =================================================================
    file_suffix = f".{suffix}" if suffix else ""
    save_path = os.path.join(out_dir, f"polysome_multidataset_correlation_heatmap{file_suffix}.pdf")
    
    # bbox_inches='tight' 保证长标签不会被切除
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Heatmap saved to: {save_path}")
    
    return heatmap_data



def plot_polysome_correlation_bar(
        agg_df: pd.DataFrame, 
        out_dir: str = "./", 
        metric_name: str = "Translation dynamics position-wise correlation",
        suffix: str = ""):
    """
    Plot bar chart with error bars and jitter points for Polysome correlations.
    [MODIFIED] Removed Cell_type coloring. Focus strictly on Dataset shapes.
    """

    if agg_df.empty:
        print("No data to plot.")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    plot_df = agg_df.copy()

    summary_df = plot_df.groupby('Model', observed=False).agg(
        Overall_Mean=('Mean', 'mean'),
        SEM=('Mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0)
    ).reset_index()
    
    summary_df['ymin'] = summary_df['Overall_Mean'] - summary_df['SEM']
    summary_df['ymax'] = summary_df['Overall_Mean'] + summary_df['SEM']

    # --- Use global model order ---
    valid_models = [m for m in GLOBAL_MODEL_ORDER if m in plot_df['Model'].unique()]
    for m in plot_df['Model'].unique():
        if m not in valid_models:
            valid_models.append(m)
            
    plot_df['Model'] = pd.Categorical(plot_df['Model'], categories=valid_models, ordered=True)
    summary_df['Model'] = pd.Categorical(summary_df['Model'], categories=valid_models, ordered=True)

    # --- Apply global colors dynamically ---
    model_colors = {}
    for m in valid_models:
        model_colors[m] = GLOBAL_MODEL_COLORS.get(m, "#C0C0C0")

    if 'Dataset' not in plot_df.columns:
        raise ValueError("The input DataFrame must contain a 'Dataset' column for shape mapping.")
        
    unique_datasets = plot_df['Dataset'].unique().tolist()
    plot_df['Dataset'] = pd.Categorical(plot_df['Dataset'], categories=unique_datasets, ordered=True)
    
    # =================================================================
    # [MODIFIED] Selected high-contrast, universally recognized shapes 
    # for Plotnine/Matplotlib. 
    # o=circle, ^=triangle up, s=square, D=diamond, v=triangle down, 
    # *=star, >=triangle right, <=triangle left, X=x
    # =================================================================
    shapes_pool = ['o', '^', 's', 'D', 'v', '*', '>', '<', 'X']
    dataset_shapes = {}
    for i, ds in enumerate(unique_datasets):
        dataset_shapes[ds] = shapes_pool[i % len(shapes_pool)]

    print(f"Generating Polysome Bar Chart...")
    
    p = (
        ggplot()
        + geom_col(
            data=summary_df, 
            mapping=aes(x='Model', y='Overall_Mean', fill='Model'), 
            width=0.7
        )
        + geom_errorbar(
            data=summary_df, 
            mapping=aes(x='Model', ymin='ymin', ymax='ymax'), 
            width=0.2, 
            size=0.8, 
            color="black"
        )
        # =================================================================
        # [MODIFIED] Removed 'color' mapping to Cell_type. Added a fixed dark 
        # gray color and a slight outline to make points pop against the bars.
        # =================================================================
        + geom_jitter(
            data=plot_df, 
            mapping=aes(x='Model', y='Mean', shape='Dataset'), 
            width=0.2, 
            size=3.5, 
            color="#202020",
            stroke=0.8,
            alpha=0.85
        )
        + scale_fill_manual(values=model_colors, guide=None) 
        + scale_shape_manual(values=dataset_shapes, name="Dataset") 
        + theme_bw()
        + labs(
            x="",
            y=f"{metric_name}\ncorrelation with ribosome load"
        )
        + theme(
            axis_text_x=element_text(angle=45, hjust=1, color="black"),
            axis_text_y=element_text(color="black"),
            axis_title_y=element_text(margin={'r': 10}),
            panel_grid_major_x=element_blank(), 
            legend_position="right",
            legend_title=element_text(size=13, fontweight='bold'),
            legend_text=element_text(size=11),
            figure_size=(7, 5) # Ensured figure size is properly set in theme
        )
    )

    file_suffix = f".{suffix}" if suffix else ""
    save_path = os.path.join(out_dir, f"polysome_multidataset_correlation_bar{file_suffix}.pdf")
    p.save(save_path, width=7, height=5, dpi=300, verbose=False)
    print(f"✅ Bar chart saved to: {save_path}")

    return summary_df


def load_and_calculate_silac_correlation(
        data_config: dict, 
        ref_dfs: dict,           
        target_ref_groups=None,     # Filter the 'Group' column in the SILAC reference datasets
        target_model_cells=None,    # Filter the 'Cell_Type' column in the model prediction datasets
        model_metric="mORF_Mean_Density", 
        ref_metric="Translation rate", 
        corr_method="spearman",
        meta_pkl_path="/home/user/data3/rbase/translation_model/models/lib/transcript_meta.pkl"):
    """
    Load model prediction results and calculate correlation with SILAC mass spec translation rates.
    Supports MULTIPLE CSV files per model.
    [NEW] Supports 1-to-1 nested array mapping between `target_ref_groups` and `target_model_cells`.
    """
    aggregated_data = []
    print(f"--- Processing SILAC Data ---")
    print(f"  Target Model Metric : {model_metric} (Fallback: TE)")
    print(f"  Target Ref Metric   : {ref_metric}")
    print(f"  Method              : {corr_method.capitalize()}")

    # 1. Parse target_ref_groups into dict of lists
    target_ref_group_map = {}
    if target_ref_groups is not None:
        if isinstance(target_ref_groups, dict):
            for k, v in target_ref_groups.items():
                target_ref_group_map[k] = [v] if isinstance(v, str) else list(v)
        elif isinstance(target_ref_groups, list):
            if len(target_ref_groups) != len(ref_dfs):
                raise ValueError("Length of target_ref_groups list must match the number of datasets in ref_dfs.")
            for i, ds_name in enumerate(ref_dfs.keys()):
                val = target_ref_groups[i]
                target_ref_group_map[ds_name] = [val] if isinstance(val, str) else list(val)
        else:
            raise TypeError("target_ref_groups must be a dictionary or a list.")

    # 2. Parse target_model_cells into dict of lists
    target_model_cell_map = {}
    if target_model_cells is not None:
        if isinstance(target_model_cells, dict):
            for k, v in target_model_cells.items():
                target_model_cell_map[k] = [v] if isinstance(v, str) else list(v)
        elif isinstance(target_model_cells, list):
            if len(target_model_cells) != len(ref_dfs):
                raise ValueError("Length of target_model_cells list must match the number of datasets in ref_dfs.")
            for i, ds_name in enumerate(ref_dfs.keys()):
                val = target_model_cells[i]
                target_model_cell_map[ds_name] = [val] if isinstance(val, str) else list(val)
        else:
            raise TypeError("target_model_cells must be a dictionary or a list.")

    processed_refs = {}
    id_cols_master = ['ENSEMBL ID', 'Tid', 'EnsemblTranscriptID', 'Gid', 'EnsemblGeneID', 'gene_name']
    pair_map = {} # Internal map to track [ref_name][ref_group] -> target_model_cell
    
    # ---------------------------------------------------------
    # Phase 0: Preprocess SILAC Reference Datasets & Build Pairing Map
    # ---------------------------------------------------------
    for ref_name, ref_df in ref_dfs.items():
        if ref_metric not in ref_df.columns:
            print(f"  [Warning] Missing metric '{ref_metric}' in {ref_name}. Skipping this dataset.")
            continue

        ref_merge_key = next((col for col in id_cols_master if col in ref_df.columns), None)
        if not ref_merge_key:
            print(f"  [Warning] No valid ID column found in {ref_name}. Skipping...")
            continue
            
        target_level = 'Transcript' if ref_merge_key in ['ENSEMBL ID', 'Tid', 'EnsemblTranscriptID'] else 'Gene'
        
        if 'Group' not in ref_df.columns:
            ref_clean = ref_df[[ref_merge_key, ref_metric]].copy()
            ref_clean['Group'] = "All"
        else:
            ref_clean = ref_df[[ref_merge_key, 'Group', ref_metric]].copy()
            
        ref_clean = ref_clean.rename(columns={'Group': 'Ref_Group'})
        ref_clean = ref_clean.dropna(subset=[ref_metric])
        ref_clean['ID_clean'] = ref_clean[ref_merge_key].astype(str).str.split('.').str[0]
        
        if target_ref_group_map and ref_name in target_ref_group_map:
            tgt_groups = target_ref_group_map[ref_name]
            ref_clean = ref_clean[ref_clean['Ref_Group'].isin(tgt_groups)]
            if ref_clean.empty:
                print(f"  [Warning] Specified groups {tgt_groups} not found in '{ref_name}'. Skipping...")
                continue
        
        ref_clean = ref_clean.groupby(['ID_clean', 'Ref_Group'], as_index=False)[ref_metric].mean()
        surviving_groups = ref_clean['Ref_Group'].unique().tolist()
        
        processed_refs[ref_name] = {
            'df': ref_clean,
            'level': target_level,
            'metric': ref_metric,
            'surviving_groups': surviving_groups
        }
        print(f"  -> Ready SILAC Ref [{ref_name}]: ID = {ref_merge_key} ({target_level}), Evaluated Groups = {surviving_groups}")

        # =================================================================
        # Build strict 1-to-1 pairing map based on arrays
        # =================================================================
        pair_map[ref_name] = {}
        tgt_models = target_model_cell_map.get(ref_name, [])
        
        if target_ref_group_map and ref_name in target_ref_group_map:
            user_ref_groups = target_ref_group_map[ref_name]
            if len(tgt_models) == 1:
                # Broadcast single model cell to all requested ref groups
                for rg in user_ref_groups:
                    pair_map[ref_name][rg] = tgt_models[0]
            elif len(tgt_models) == len(user_ref_groups):
                # 1-to-1 Mapping (e.g. [A, B] -> [X, Y])
                for rg, mc in zip(user_ref_groups, tgt_models):
                    pair_map[ref_name][rg] = mc
            elif len(tgt_models) > 0:
                raise ValueError(f"Length mismatch in '{ref_name}': {len(user_ref_groups)} ref groups vs {len(tgt_models)} model cells.")
        else:
            if len(tgt_models) == 1:
                for rg in surviving_groups:
                    pair_map[ref_name][rg] = tgt_models[0]
            elif len(tgt_models) > 1:
                raise ValueError(f"Cannot map {len(tgt_models)} model cells to dataset '{ref_name}' because target_ref_groups is not specified.")

    if not processed_refs:
        raise ValueError("No valid reference datasets processed.")

    # ---------------------------------------------------------
    # Load meta mapping if required
    # ---------------------------------------------------------
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

    processed_models = {}  
    global_id_sets = {}    
    
    # ---------------------------------------------------------
    # Phase 1: Data Loading & Initial Mapping
    # ---------------------------------------------------------
    print("\n--- Phase 1: Data Loading & Initial Mapping ---")
    for idx, (model_name, file_paths) in enumerate(data_config.items()):
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            
        print(f"Processing model: {model_name} (Loading {len(file_paths)} files...)")
        
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
            print(f"  [Warning] No valid files loaded for model '{model_name}'. Skipping...")
            continue
            
        combined_df_raw = pd.concat(model_dfs, ignore_index=True)
        
        current_model_metric = model_metric
        if current_model_metric not in combined_df_raw.columns:
            if 'TE' in combined_df_raw.columns:
                current_model_metric = 'TE'
            else:
                print(f"  [Warning] Missing metric '{model_metric}' and 'TE' in {model_name}. Skipping...")
                continue

        processed_models[model_name] = {}
        
        for ref_name, ref_info in processed_refs.items():
            ref_level = ref_info['level']
            ref_metric_name = ref_info['metric']
            ref_clean_full = ref_info['df']
            surviving_groups = ref_info['surviving_groups']
            
            # Map IDs uniformly first
            combined_df_mapped = combined_df_raw.copy()
            has_tid = [c for c in ['Tid', 'EnsemblTranscriptID', 'TranscriptID', 'ENSEMBL ID'] if c in combined_df_mapped.columns]
            has_gid = [c for c in ['Gid', 'EnsemblGeneID', 'GeneID'] if c in combined_df_mapped.columns]

            if ref_level == 'Gene':
                if has_gid:
                    combined_df_mapped['ID_clean'] = combined_df_mapped[has_gid[0]].astype(str).str.split('.').str[0]
                elif has_tid:
                    combined_df_mapped['clean_tid_temp'] = combined_df_mapped[has_tid[0]].astype(str).str.split('.').str[0]
                    combined_df_mapped['ID_clean'] = combined_df_mapped['clean_tid_temp'].map(tid2gene)
                    combined_df_mapped = combined_df_mapped.dropna(subset=['ID_clean'])
                else:
                    continue
            else: 
                if has_tid:
                    combined_df_mapped['ID_clean'] = combined_df_mapped[has_tid[0]].astype(str).str.split('.').str[0]
                else:
                    continue

            # =================================================================
            # Process each specific Ref_Group with its paired Model Cell_Type
            # =================================================================
            for group_name in surviving_groups:
                ref_clean_sub = ref_clean_full[ref_clean_full['Ref_Group'] == group_name]
                
                tgt_model_cell = pair_map.get(ref_name, {}).get(group_name)
                
                combined_df_sub = combined_df_mapped.copy()
                if tgt_model_cell and 'Cell_Type' in combined_df_sub.columns:
                    combined_df_sub = combined_df_sub[combined_df_sub['Cell_Type'] == tgt_model_cell]
                    if combined_df_sub.empty:
                        print(f"    [Skip] {model_name} has no Cell_Type == '{tgt_model_cell}' for Dataset '{ref_name}' Group '{group_name}'")
                        continue
                
                # Aggregate identical IDs
                combined_df_agg = combined_df_sub.groupby(['ID_clean'], as_index=False)[current_model_metric].mean()
                
                # Merge model specific cell type with ref specific group
                merged_df = pd.merge(combined_df_agg, ref_clean_sub, on='ID_clean', how='inner')
                
                if merged_df.empty:
                    continue

                # Store by group tuple
                key = (ref_name, group_name)
                processed_models[model_name][key] = (merged_df, current_model_metric, ref_metric_name)
                
                # Record valid IDs for strict intersection
                group_clean = merged_df.dropna(subset=[current_model_metric, ref_metric_name])
                if key not in global_id_sets:
                    global_id_sets[key] = []
                global_id_sets[key].append(set(group_clean['ID_clean']))

    # ---------------------------------------------------------
    # Phase 2: Compute Strict Intersections across Models
    # ---------------------------------------------------------
    print("\n--- Phase 2: Intersecting Common IDs across Models ---")
    intersected_ids_dict = {}
    expected_model_count = len(processed_models)
    
    for key, sets_list in global_id_sets.items():
        ref_name, group_name = key
        if len(sets_list) < expected_model_count:
            print(f"  [Warning] Dataset '{ref_name}' (Group: {group_name}) is missing in some models. Strict comparison may drop models.")
        
        common_ids = set.intersection(*sets_list)
        intersected_ids_dict[key] = common_ids
        print(f"  -> {ref_name:<20} | {group_name:<15} | Fair IDs found: {len(common_ids)}")

    # ---------------------------------------------------------
    # Phase 3: Calculate Fair Correlations
    # ---------------------------------------------------------
    print("\n--- Phase 3: Calculating Fair Correlations ---")
    for model_name, group_dict in processed_models.items():
        print(f"\nEvaluating: {model_name}")
        
        for (ref_name, group_name), (merged_df, current_model_metric, ref_metric_name) in group_dict.items():
            valid_ids = intersected_ids_dict.get((ref_name, group_name), set())
            group_clean_raw = merged_df.dropna(subset=[current_model_metric, ref_metric_name])
            n_before = len(group_clean_raw)
            
            group_clean = group_clean_raw[group_clean_raw['ID_clean'].isin(valid_ids)]
            n_after = len(group_clean)
            
            print(f"  -> {ref_name:<15} ({group_name}) | N Filtered: {n_before:>5} -> {n_after:>5}")

            if n_after < 5:
                continue

            # Handle outliers using 1%-99% filtering
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
                'Group': group_name,  
                'Model': model_name,
                'Mean': r_val,
                'P_value': p_val,
                'N': len(group_clean) 
            })

    return pd.DataFrame(aggregated_data)

# =================================================================
# 2. 柱状图可视化 (SILAC Correlation Bar)
# =================================================================
def plot_silac_correlation_bar(
        agg_df: pd.DataFrame, 
        out_dir: str = "./", 
        metric_name: str = "Model Translation Prediction",
        suffix: str = ""
        ):
    """
    Plot bar chart with error bars and jitter points for SILAC translation rate correlations.
    Points are mapped to Dataset via shapes. Group variance is aggregated.
    """
    if agg_df.empty:
        print("No data to plot.")
        return
    
    file_suffix = f".{suffix}" if suffix else ""
    os.makedirs(out_dir, exist_ok=True)

    agg_df.to_csv(out_dir + f"silac_correlation{file_suffix}.csv")
    plot_df = agg_df.copy()

    summary_df = plot_df.groupby('Model', observed=False).agg(
        Overall_Mean=('Mean', 'mean'),
        SEM=('Mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0)
    ).reset_index()
    
    summary_df['ymin'] = summary_df['Overall_Mean'] - summary_df['SEM']
    summary_df['ymax'] = summary_df['Overall_Mean'] + summary_df['SEM']

    valid_models = [m for m in GLOBAL_MODEL_ORDER if m in plot_df['Model'].unique()]
    for m in plot_df['Model'].unique():
        if m not in valid_models:
            valid_models.append(m)
            
    plot_df['Model'] = pd.Categorical(plot_df['Model'], categories=valid_models, ordered=True)
    summary_df['Model'] = pd.Categorical(summary_df['Model'], categories=valid_models, ordered=True)

    model_colors = {m: GLOBAL_MODEL_COLORS.get(m, "#C0C0C0") for m in valid_models}

    unique_datasets = plot_df['Dataset'].unique().tolist()
    plot_df['Dataset'] = pd.Categorical(plot_df['Dataset'], categories=unique_datasets, ordered=True)
    
    # 采用高对比度形状用于区分不同的 Dataset
    shapes_pool = ['o', '^', 's', 'D', 'v', '*', '>', '<', 'X']
    dataset_shapes = {ds: shapes_pool[i % len(shapes_pool)] for i, ds in enumerate(unique_datasets)}

    print(f"Generating SILAC Correlation Bar Chart...")
    
    p = (
        ggplot()
        + geom_col(
            data=summary_df, 
            mapping=aes(x='Model', y='Overall_Mean', fill='Model'), 
            width=0.7
        )
        + geom_errorbar(
            data=summary_df, 
            mapping=aes(x='Model', ymin='ymin', ymax='ymax'), 
            width=0.2, 
            size=0.8, 
            color="black"
        )
        + geom_jitter(
            data=plot_df, 
            mapping=aes(x='Model', y='Mean', shape='Dataset'), 
            width=0.2, 
            size=3.5, 
            color="#202020",
            stroke=0.8,
            alpha=0.85
        )
        + scale_fill_manual(values=model_colors, guide=None) 
        + scale_shape_manual(values=dataset_shapes, name="SILAC Dataset") 
        + theme_bw()
        + labs(
            x="",
            y=f"{metric_name} correlation with SILAC Translation Rate"
        )
        + theme(
            axis_text_x=element_text(angle=45, hjust=1, color="black"),
            axis_text_y=element_text(color="black"),
            axis_title_y=element_text(margin={'r': 10}),
            panel_grid_major_x=element_blank(), 
            legend_position="right",
            legend_title=element_text(size=13, fontweight='bold'),
            legend_text=element_text(size=11),
            figure_size=(6, 5) 
        )
    )

    save_path = os.path.join(out_dir, f"silac_correlation_bar{file_suffix}.pdf")
    p.save(save_path, width=6, height=5, dpi=300, verbose=False)
    print(f"✅ Bar chart saved to: {save_path}")

    return summary_df