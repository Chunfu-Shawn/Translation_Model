#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import torch
import json
import sys

def load_id_mapping(mapping_json_path):
    """
    Loads the pre-computed anchor mapping (Ensembl ID -> Anchor ID).
    """
    try:
        with open(mapping_json_path, 'r') as f:
            anchor_to_native = json.load(f)
            
        id_mapping = {}
        for anchor, species_dict in anchor_to_native.items():
            if 'Human' in species_dict:
                id_mapping[species_dict['Human']] = anchor
        return id_mapping
    except Exception as e:
        print(f"Error loading mapping JSON: {e}")
        sys.exit(1)

def generate_cell_env_expr_dict(
    counts_file, 
    ref_order_path, 
    mapping_json_path, 
    quant_level='transcript', # [新增]: 指示输入 count 矩阵的级别
    tx2gene_file=None,        # 当 quant_level='transcript' 时使用
    min_tpm_threshold=0.0, 
    output_pt_path=None
):
    """
    Generates personalized, Z-scored expression vectors directly in memory.
    Supports both direct Gene-level inputs and Transcript-level inputs with on-the-fly RPK aggregation.
    """
    print(f"\n[ExprBuilder] Generating expression array from: {counts_file}")
    print(f"[ExprBuilder] Input Quantification Level: {quant_level.upper()}")
    
    # 1. Load Reference Order
    try:
        with open(ref_order_path, 'r') as f:
            reference_anchor_ids = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"[ExprBuilder] Error loading reference order: {e}")
        sys.exit(1)

    # 2. Load Global ID Mapping (Ensembl Gene -> Anchor)
    id_mapping = load_id_mapping(mapping_json_path)

    # 3. Read featureCounts Matrix
    try:
        df = pd.read_csv(counts_file, sep='\t', comment='#')
    except Exception as e:
        print(f"[ExprBuilder] Error reading counts file: {e}")
        sys.exit(1)
        
    if len(df.columns) <= 6:
        print("[ExprBuilder] Error: featureCounts file lacks sample columns.")
        sys.exit(1)
        
    bam_cols = df.columns[6:]
    rename_dict = {col: os.path.basename(col).split('.')[0].replace('_uniq.sorted', '').replace('.bam', '') for col in bam_cols}
    df = df.rename(columns=rename_dict)
    sample_cols = list(rename_dict.values())
    
    # [遵守修正记录] 严格剔除 ID 的版本号后缀
    df['Clean_ID'] = df['Geneid'].astype(str).str.split('.').str[0]
    
    # 4. Determine Gene-Level Grouping Target
    if quant_level == 'transcript':
        if not tx2gene_file:
            print("[ExprBuilder] Error: tx2gene_file is required when quant_level is 'transcript'.")
            sys.exit(1)
            
        print(f"[ExprBuilder] Loading Tx-to-Gene mapping from: {tx2gene_file}")
        try:
            tx2gene_df = pd.read_csv(tx2gene_file, sep='\t')
            col_tx = tx2gene_df.columns[0]
            col_gene = tx2gene_df.columns[1]
            
            # 剔除映射表里的版本号后缀
            tx_keys = tx2gene_df[col_tx].astype(str).str.split('.').str[0]
            gene_vals = tx2gene_df[col_gene].astype(str).str.split('.').str[0]
            tx2gene_map = dict(zip(tx_keys, gene_vals))
            
            df['Target_Gene_ID'] = df['Clean_ID'].map(tx2gene_map)
            unmapped = df['Target_Gene_ID'].isna().sum()
            if unmapped > 0:
                print(f"[Warning] {unmapped} transcripts could not be mapped to genes and will be dropped.")
            df = df.dropna(subset=['Target_Gene_ID']).copy()
            
        except Exception as e:
            print(f"[ExprBuilder] Error processing Tx-to-Gene mapping: {e}")
            sys.exit(1)
            
    elif quant_level == 'gene':
        # 输入已经是基因水平，Clean_ID 直接作为目标靶点
        df['Target_Gene_ID'] = df['Clean_ID']
        if tx2gene_file:
            print("[ExprBuilder] Warning: tx2gene_file provided but ignored since quant_level is 'gene'.")
            
    else:
        print(f"[ExprBuilder] Error: Invalid quant_level '{quant_level}'. Choose 'transcript' or 'gene'.")
        sys.exit(1)

    # 5. Calculate RPK & Aggregate to Gene Level
    print(f"[ExprBuilder] Calculating true RPK and assembling Gene-level TPM...")
    rpk_df = pd.DataFrame({'Target_Gene_ID': df['Target_Gene_ID']})
    for col in sample_cols:
        rpk_df[col] = df[col] / (df['Length'] / 1000.0)

    # 无论是 transcript 输入还是 gene 输入，在这里进行 groupby 都是安全的。
    # 对于 gene 输入，大部分情况是一对一映射（无须求和的单行分组），天然兼容。
    gene_rpk_df = rpk_df.groupby('Target_Gene_ID')[sample_cols].sum()

    # 6. Calculate TPM and Robust Z-score from Gene RPKs
    gene_tpm_df = pd.DataFrame(index=gene_rpk_df.index)
    zscore_cols = []
    
    for col in sample_cols:
        rpk = gene_rpk_df[col]
        scaling_factor = rpk.sum() / 1e6  
        tpm = rpk / scaling_factor
        
        log_tpm = np.log2(tpm + 1.0)
        
        active_mask = tpm > min_tpm_threshold
        if active_mask.sum() > 0:
            active_mean = log_tpm[active_mask].mean()
            active_std = log_tpm[active_mask].std()
        else:
            active_mean = log_tpm.mean()
            active_std = log_tpm.std()
        
        z_score = (log_tpm - active_mean) / (active_std + 1e-8)
        z_col_name = f"{col}_Zscore"
        gene_tpm_df[z_col_name] = z_score
        zscore_cols.append(z_col_name)

    # 7. Map to Ortholog Anchor IDs
    gene_tpm_df = gene_tpm_df.reset_index()
    gene_tpm_df['Anchor_ID'] = gene_tpm_df['Target_Gene_ID'].map(id_mapping)
    aligned_df = gene_tpm_df.dropna(subset=['Anchor_ID']).copy()
    
    # 8. 核心阵列覆盖率雷达 (Coverage Radar)
    covered_anchors = set(aligned_df['Anchor_ID'].unique())
    ref_anchors_set = set(reference_anchor_ids)
    
    found_anchors = covered_anchors.intersection(ref_anchors_set)
    missing_anchors = ref_anchors_set.difference(covered_anchors)
    
    total_ref = len(ref_anchors_set)
    found_ref = len(found_anchors)
    coverage_pct = (found_ref / total_ref * 100) if total_ref > 0 else 0
    
    print(f"\n=============================================")
    print(f" 🎯 Anchor Gene Expression Coverage Report")
    print(f"=============================================")
    print(f" -> Total Anchors Required by TRACE    : {total_ref}")
    print(f" -> Anchors Found in Input Count Matrix: {found_ref}")
    print(f" -> Global Vector Integrity            : {coverage_pct:.2f}%")
    
    if len(missing_anchors) > 0:
        print(f" -> Missing Anchors (Top 5 examples): {list(missing_anchors)[:5]}...")
    print(f"=============================================\n")
    
    if coverage_pct < 50.0:
        print("[Warning] INTEGRITY ALERT! Your vector integrity is below 50%.")
        print("          TRACE translation models perform poorly with sparse expression inputs.")
        print("          Please ensure your featureCounts file contains global genomic expression, not just filtered subsets.")

    # 如果有多个 Ensembl ID 映射到了同一个 Anchor ID，取平均值
    grouped_zscore = aligned_df.groupby('Anchor_ID')[zscore_cols].mean()

    # 9. Align to Reference Coordinate System
    final_df = grouped_zscore.reindex(reference_anchor_ids).fillna(0.0)
    
    # 10. Pack into Dictionary
    expr_dict = {}
    for col, z_col in zip(sample_cols, zscore_cols):
        expr_dict[col] = torch.tensor(final_df[z_col].values, dtype=torch.float16)
        
    print(f"[ExprBuilder] ✅ Generated vectors for {len(expr_dict)} samples.")
    
    if output_pt_path:
        os.makedirs(os.path.dirname(output_pt_path) or '.', exist_ok=True)
        torch.save(expr_dict, output_pt_path)
        print(f"[ExprBuilder] Saved dict to: {output_pt_path}")
        
    return expr_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate patient-specific Z-scored expression vectors.")
    parser.add_argument("-c", "--counts_file", required=True, help="Path to featureCounts matrix")
    parser.add_argument("-r", "--ref_order", required=True, help="Path to reference anchor order list")
    parser.add_argument("-m", "--mapping_json", required=True, help="Path to species mapping JSON")
    
    # [新增] 引入 quant_level 命令行参数
    parser.add_argument("-q", "--quant_level", required=True, choices=['transcript', 'gene'], help="Level of quantification in the counts_file ('transcript' or 'gene')")
    parser.add_argument("-t", "--tx2gene", default=None, help="Path to Transcript-to-Gene mapping TSV (Required if quant_level is 'transcript')")
    
    parser.add_argument("-o", "--output_pt", required=True, help="Path to save output .pt dictionary")
    parser.add_argument("--min_tpm", type=float, default=0.0, help="Minimum TPM to consider a gene active")
    args = parser.parse_args()
    
    generate_cell_env_expr_dict(
        counts_file=args.counts_file,
        ref_order_path=args.ref_order,
        mapping_json_path=args.mapping_json,
        quant_level=args.quant_level,
        tx2gene_file=args.tx2gene,
        output_pt_path=args.output_pt,
        min_tpm_threshold=args.min_tpm
    )