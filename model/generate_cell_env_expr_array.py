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

def generate_patient_expr_dict(
    counts_file, 
    ref_order_path, 
    mapping_json_path, 
    min_tpm_threshold=0.0, 
    output_pt_path=None
):
    """
    Generates personalized, Z-scored expression vectors directly in memory.
    Can be imported and called by other scripts.
    """
    print(f"\n[ExprBuilder] Generating expression array from: {counts_file}")
    
    # 1. Load Reference Order
    try:
        with open(ref_order_path, 'r') as f:
            reference_anchor_ids = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"[ExprBuilder] Error loading reference order: {e}")
        sys.exit(1)

    # 2. Load ID Mapping
    id_mapping = load_id_mapping(mapping_json_path)

    # 3. Read featureCounts
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
    
    df['Clean_Geneid'] = df['Geneid'].str.split('.').str[0]
    
    # 4. Calculate TPM and Robust Z-score
    zscore_cols = []
    for col in sample_cols:
        rpk = df[col] / (df['Length'] / 1000.0)
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
        df[z_col_name] = z_score
        zscore_cols.append(z_col_name)

    # 5. Map to Ortholog Anchor IDs
    df['Anchor_ID'] = df['Clean_Geneid'].map(id_mapping)
    aligned_df = df.dropna(subset=['Anchor_ID']).copy()
    grouped_zscore = aligned_df.groupby('Anchor_ID')[zscore_cols].mean()

    # 6. Align to Reference Coordinate System
    final_df = grouped_zscore.reindex(reference_anchor_ids).fillna(0.0)
    
    # 7. Pack into Dictionary
    expr_dict = {}
    for col, z_col in zip(sample_cols, zscore_cols):
        expr_dict[col] = torch.tensor(final_df[z_col].values, dtype=torch.float16)
        
    print(f"[ExprBuilder] ✅ Generated vectors for samples: {list(expr_dict.keys())}")
    
    # Optionally save to disk if path is provided
    if output_pt_path:
        os.makedirs(os.path.dirname(output_pt_path) or '.', exist_ok=True)
        torch.save(expr_dict, output_pt_path)
        print(f"[ExprBuilder] Saved dict to: {output_pt_path}")
        
    return expr_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate patient-specific Z-scored expression vectors.")
    parser.add_argument("-c", "--counts_file", required=True)
    parser.add_argument("-r", "--ref_order", required=True)
    parser.add_argument("-m", "--mapping_json", required=True)
    parser.add_argument("-o", "--output_pt", required=True)
    parser.add_argument("--min_tpm", type=float, default=0.0)
    args = parser.parse_args()
    
    generate_patient_expr_dict(
        counts_file=args.counts_file,
        ref_order_path=args.ref_order,
        mapping_json_path=args.mapping_json,
        output_pt_path=args.output_pt,
        min_tpm_threshold=args.min_tpm
    )