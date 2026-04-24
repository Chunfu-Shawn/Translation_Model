import os
import pandas as pd
import numpy as np
import torch
import json

def prepare_ortholog_mapping(ortho_csv_path):
    """
    Reads BioMart ortholog table and resolves 1:many, many:1, and many:many 
    relationships using a greedy confidence-based approach to construct a strict 1:1:1 mapping.
    """
    print(f"Loading ortholog table from: {ortho_csv_path}")
    # Assuming the file is tab-separated. Change to sep=',' if it is a standard CSV.
    df = pd.read_csv(ortho_csv_path, sep='\t') 

    # Define column names (ensure these exactly match your file headers)
    h_id = 'Gene stable ID'
    mac_id = 'Macaque gene stable ID'
    mac_conf = 'Macaque orthology confidence [0 low, 1 high]'
    mus_id = 'Mouse gene stable ID'
    mus_conf = 'Mouse orthology confidence [0 low, 1 high]'

    # ---------------------------------------------------------
    # 1. Independently resolve Human <-> Macaque 1:1 relationships
    # ---------------------------------------------------------
    df_mac = df[[h_id, mac_id, mac_conf]].dropna(subset=[h_id, mac_id]).copy()
    # Sort by confidence descending
    df_mac = df_mac.sort_values(by=mac_conf, ascending=False)
    # Greedy deduplication: enforce uniqueness for both Human and Macaque IDs, keeping the highest confidence
    df_mac = df_mac.drop_duplicates(subset=[h_id], keep='first')
    df_mac = df_mac.drop_duplicates(subset=[mac_id], keep='first')

    # ---------------------------------------------------------
    # 2. Independently resolve Human <-> Mouse 1:1 relationships
    # ---------------------------------------------------------
    df_mus = df[[h_id, mus_id, mus_conf]].dropna(subset=[h_id, mus_id]).copy()
    df_mus = df_mus.sort_values(by=mus_conf, ascending=False)
    df_mus = df_mus.drop_duplicates(subset=[h_id], keep='first')
    df_mus = df_mus.drop_duplicates(subset=[mus_id], keep='first')

    # ---------------------------------------------------------
    # 3. Intersect to build a strict 1:1:1 Triplet coordinate system
    # ---------------------------------------------------------
    # Inner join using Human ID as the anchor
    df_triplets = pd.merge(df_mac[[h_id, mac_id]], df_mus[[h_id, mus_id]], on=h_id, how='inner')

    # Build unified mapping dictionaries (Any species ID -> Unified Human Anchor ID)
    id_mapping = {}
    anchor_to_native = {} 

    for _, row in df_triplets.iterrows():
        anchor = row[h_id]
        m_id = row[mac_id]
        ms_id = row[mus_id]

        # Reverse lookup dict: map any species-specific ID back to the anchor
        id_mapping[anchor] = anchor
        id_mapping[m_id] = anchor
        id_mapping[ms_id] = anchor

        # Forward tracking dict: useful for downstream traceability analysis
        anchor_to_native[anchor] = {
            'Human': anchor,
            'Macaque': m_id,
            'Mouse': ms_id
        }

    print(f"Cleaning complete: Parsed {len(df_triplets)} strict 1-to-1-to-1 shared ortholog gene families.")
    return id_mapping, anchor_to_native


def build_cross_species_expression_dict(
    file_path, 
    output_pt_path, 
    id_mapping, 
    reference_anchor_ids=None, # Reference gene list to ensure consistent dimensions across species
    min_tpm_threshold=1.0      # Threshold to filter out unexpressed genes from statistical calculations
):
    """
    Aligns single-species count files and extracts expression vectors.
    Performs robust Z-score normalization on the WHOLE GENOME first, avoiding zero-inflation, 
    then maps to orthologs.
    """
    print(f"\nReading counts file: {file_path}")
    df = pd.read_csv(file_path, sep='\t', comment='#')
    
    bam_cols = df.columns[6:]
    rename_dict = {col: os.path.basename(col).split('.')[0] for col in bam_cols}
    df = df.rename(columns=rename_dict)
    sample_cols = list(rename_dict.values())
    
    df['Clean_Geneid'] = df['Geneid'].str.split('.').str[0]
    
    # ==========================================
    # 1. Whole-Genome TPM and Robust Z-score
    # ==========================================
    print(f"Calculating RPK, TPM, and robust Z-score (excluding TPM <= {min_tpm_threshold} from distribution stats)...")
    zscore_cols = []
    
    for col in sample_cols:
        # Calculate native RPK
        rpk = df[col] / (df['Length'] / 1000.0)
        # Calculate whole-transcriptome sequencing depth
        scaling_factor = rpk.sum() / 1e6  
        # Calculate TPM
        tpm = rpk / scaling_factor
        
        # Log2 smoothing
        log_tpm = np.log2(tpm + 1.0)
        
        # ---------------------------------------------------------
        # Robust Standardization Strategy:
        # Identify actively expressed genes to compute the background distribution.
        # This prevents the massive zero-inflation from skewing the mean/std downward.
        # ---------------------------------------------------------
        active_mask = tpm > min_tpm_threshold
        
        # Fallback to global mean/std if the sample is completely dead (rare edge case)
        if active_mask.sum() > 0:
            active_mean = log_tpm[active_mask].mean()
            active_std = log_tpm[active_mask].std()
        else:
            active_mean = log_tpm.mean()
            active_std = log_tpm.std()
        
        # Standardize ALL genes using the active distribution.
        # Genes with 0 TPM will correctly receive a negative Z-score, indicating suppression.
        z_score = (log_tpm - active_mean) / (active_std + 1e-8)
        
        z_col_name = f"{col}_Zscore"
        df[z_col_name] = z_score
        zscore_cols.append(z_col_name)

    # ==========================================
    # 2. Map to Ortholog Anchor IDs
    # ==========================================
    # Map to Anchor ID (genes without orthology will become NaN)
    df['Anchor_ID'] = df['Clean_Geneid'].map(id_mapping)
    aligned_df = df.dropna(subset=['Anchor_ID']).copy()
    
    # Merge any residual many-to-one mappings safely using mean
    grouped_zscore = aligned_df.groupby('Anchor_ID')[zscore_cols].mean()

    # ==========================================
    # 3. Align to Reference Coordinate System
    # ==========================================
    if reference_anchor_ids is not None:
        print(f"Aligning to reference gene coordinates (Dimension: {len(reference_anchor_ids)})...")
        # Reindex using the reference list. Missing annotations are filled with 0.0
        # (0.0 represents the neutral mean of the *active* transcriptome here)
        final_df = grouped_zscore.reindex(reference_anchor_ids).fillna(0.0)
        final_gene_ids = reference_anchor_ids
    else:
        print("No reference coordinates provided. Using all currently mapped orthologs as the baseline...")
        final_df = grouped_zscore.copy()
        final_gene_ids = final_df.index.tolist()
        print(f"-> Retained {len(final_gene_ids)} ortholog genes for the reference coordinate system.")
    
    # ==========================================
    # 4. Pack into Dictionary and Save
    # ==========================================
    expr_dict = {}
    for col, z_col in zip(sample_cols, zscore_cols):
        expr_dict[col] = torch.tensor(final_df[z_col].values, dtype=torch.float16)
        
    os.makedirs(os.path.dirname(output_pt_path) or '.', exist_ok=True)
    torch.save(expr_dict, output_pt_path)
    print(f"✅ Successfully saved expression dictionary to: {output_pt_path}")
    print(f"Cell type: {expr_dict.keys()}; Array shape: {expr_dict[sample_cols[0]].shape}")
    
    return expr_dict, final_gene_ids

# ==========================================
# Usage Example: Cross-species Pipeline
# ==========================================
if __name__ == "__main__":
    # Config file paths
    ortholog_csv = "/home/user/data3/rbase/genome_ref/Homolog/human_macaque_mouse_orthologs.tsv" 
    
    human_counts = "/home/user/data3/yaoc/translation_model/rna-seq/counts_gene/matched_samples_gene_counts.txt"
    macque_counts = "/home/user/data3/yaoc/translation_model/rna-seq/counts_gene/macaque_featureCounts.txt"
    mouse_counts = "/home/user/data3/yaoc/translation_model/rna-seq/counts_gene/mouse_counts.txt"
    
    out_dir = "/home/user/data3/rbase/translation_model/models/lib"

    # 1. Preprocess the ortholog table
    id_mapping, anchor_to_native = prepare_ortholog_mapping(ortholog_csv)

    # ---------------------------------------------------------
    # Phase 1: Process Human data to establish the "Global Reference Coordinates"
    # ---------------------------------------------------------
    print("\n========== Phase 1: Establishing Human Reference Coordinates ==========")
    human_pt = os.path.join(out_dir, "human_expression_dict.pt")
    _, global_anchor_ids = build_cross_species_expression_dict(
        file_path=human_counts, 
        output_pt_path=human_pt, 
        id_mapping=id_mapping,
        reference_anchor_ids=None, # Set to None to let the script define the universe via Human data
        min_tpm_threshold=0      # Adjusted via the new parameter
    )

    # Save this reference coordinate system for future lookups
    order_file = os.path.join(out_dir, "global_anchor_gene_order.txt")
    with open(order_file, 'w') as f:
        f.write("\n".join(global_anchor_ids))
    print(f"✅ Global reference coordinate system saved to: {order_file}")

    # Save species traceability dictionary
    mapping_file = os.path.join(out_dir, "global_species_id_mapping.json")
    final_mapping_dict = {gid: anchor_to_native[gid] for gid in global_anchor_ids if gid in anchor_to_native}
    with open(mapping_file, 'w') as f:
        json.dump(final_mapping_dict, f, indent=4)

    # ---------------------------------------------------------
    # Phase 2: Process other species, forcibly aligning to the reference coordinates
    # ---------------------------------------------------------
    print("\n========== Phase 2: Aligning Macaque Data ==========")
    macaque_pt = os.path.join(out_dir, "macaque_expression_dict.pt")
    build_cross_species_expression_dict(
        file_path=macque_counts, 
        output_pt_path=macaque_pt, 
        id_mapping=id_mapping,
        reference_anchor_ids=global_anchor_ids, # Forcible alignment
        min_tpm_threshold=0
    )

    # print("\n========== Phase 2: Aligning Mouse Data ==========")
    # mouse_pt = os.path.join(out_dir, "mouse_expression_dict.pt")
    # build_cross_species_expression_dict(
    #     file_path=mouse_counts, 
    #     output_pt_path=mouse_pt, 
    #     id_mapping=id_mapping,
    #     reference_anchor_ids=global_anchor_ids, # Forcible alignment
    #     min_tpm_threshold=0
    # )

    # print("\n🎉 All cross-species alignment tasks are complete! You can now merge human_pt and mouse_pt to feed the model.")