import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from eval.calculate_te import calculate_morf_mean_signal

#=======================================
# 1. 基础权重字典配置 (与之前一致)
# ==========================================

HUMAN_CODON_WEIGHTS = {
    'TTT': 17.1, 'TTC': 19.5, 'TTA': 7.8,  'TTG': 12.9,
    'CTT': 13.3, 'CTC': 19.2, 'CTA': 7.2,  'CTG': 39.4,
    'ATT': 15.8, 'ATC': 19.9, 'ATA': 7.5,  'ATG': 21.3,
    'GTT': 10.9, 'GTC': 14.0, 'GTA': 7.1,  'GTG': 27.6,
    'TCT': 15.4, 'TCC': 17.7, 'TCA': 12.5, 'TCG': 4.5,
    'CCT': 17.9, 'CCC': 20.3, 'CCA': 17.3, 'CCG': 7.3,
    'ACT': 13.3, 'ACC': 18.4, 'ACA': 15.1, 'ACG': 6.0,
    'GCT': 18.4, 'GCC': 28.1, 'GCA': 16.1, 'GCG': 7.6,
    'TAT': 12.0, 'TAC': 14.6, 'TAA': 0.5,  'TAG': 0.4, 
    'CAT': 11.1, 'CAC': 15.2, 'CAA': 12.7, 'CAG': 34.9,
    'AAT': 17.1, 'AAC': 18.6, 'AAA': 25.1, 'AAG': 31.9,
    'GAT': 22.3, 'GAC': 25.1, 'GAA': 30.4, 'GAG': 40.7,
    'TGT': 10.6, 'TGC': 12.2, 'TGA': 0.9,  'TGG': 12.1,
    'CGT': 4.5,  'CGC': 10.5, 'CGA': 6.2,  'CGG': 11.7,
    'AGT': 12.6, 'AGC': 19.9, 'AGA': 12.0, 'AGG': 11.9,
    'GGT': 10.6, 'GGC': 22.3, 'GGA': 16.5, 'GGG': 16.5
}

KOZAK_WEIGHTS = {
    '-6': {'A': 0.00, 'C': -0.03, 'G': 0.05, 'T': -0.02},
    '-5': {'A': -0.06, 'C': 0.04,  'G': -0.01, 'T': 0.03},
    '-4': {'A': 0.07, 'C': 0.08,  'G': -0.06, 'T': -0.09}, 
    '-3': {'A': 0.14, 'C': -0.08,  'G': 0.14, 'T': -0.29}, 
    '-2': {'A': 0.03, 'C': 0.06,  'G': -0.12, 'T': -0.01},
    '-1': {'A': 0.02, 'C': 0.04,  'G': 0.02, 'T': 0.02}, 
    '+4': {'A': -0.02, 'C': -0.09,  'G': 0.12, 'T': -0.02}, 
    '+5': {'A': -0.01, 'C': 0.04,  'G': 0.05, 'T': -0.07},
}

START_CODON_WEIGHTS = {
    'ATG': 1.0,
    'CTG': 0.3,
    'GTG': 0.2,
    'TTG': 0.0,
    'ACG': -0.2,
}

# ==========================================
# 2. 特征计算函数定义 (与之前一致)
# ==========================================

def calculate_codon_freq_sum(cds_seq):
    cds_seq = cds_seq.upper().replace('U', 'T')
    codons = [cds_seq[i:i+3] for i in range(0, len(cds_seq) - len(cds_seq)%3, 3)]
    if not codons: return np.nan
    score = 0.0
    for codon in codons:
        if codon in HUMAN_CODON_WEIGHTS and codon not in ['TAA', 'TAG', 'TGA']: 
            score += HUMAN_CODON_WEIGHTS[codon]
    return float(score)

def calculate_gc_content(seq):
    if not seq or len(seq) == 0: return np.nan
    seq = seq.upper()
    g_count = seq.count('G')
    c_count = seq.count('C')
    return (g_count + c_count) / len(seq)

def calculate_kozak_score(full_seq, cds_start):
    if cds_start < 6 or cds_start + 4 >= len(full_seq): return np.nan
    full_seq = full_seq.upper().replace('U', 'T')
    score = 0.0
    for i, pos_name in enumerate(['-6', '-5', '-4', '-3', '-2', '-1']):
        base = full_seq[cds_start - 6 + i]
        score += KOZAK_WEIGHTS[pos_name].get(base, 0)
    start_codon = full_seq[cds_start:cds_start+3]
    score += START_CODON_WEIGHTS.get(start_codon, -1.0)
    for i, pos_name in enumerate(['+4', '+5']):
        base = full_seq[cds_start + 3 + i]
        score += KOZAK_WEIGHTS[pos_name].get(base, 0)
    return float(score)

# ==========================================
# 3. 核心提取引擎
# ==========================================

def generate_comprehensive_baselines(seq_pkl_path, out_dir="./results/baselines", 
                                     dataset=None, tx_cds_file=None, 
                                     unlog_data=True, target_cell_types=None):
    """
    Generate baseline features.
    Priority 1: Use `dataset` to extract real mORF_Mean_Density if available.
    Priority 2: Fallback to `tx_cds_file` to only extract sequence-intrinsic features.
    If `target_cell_types` is provided, features are broadcasted to these cells (with missing TE as NaN).
    """
    if dataset is None and tx_cds_file is None:
        raise ValueError("You must provide either 'dataset' or 'tx_cds_file'.")
        
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading sequence dictionary from {seq_pkl_path}...")
    with open(seq_pkl_path, 'rb') as f:
        seqs_dict = pickle.load(f)
        
    feature_dict = {}
    cells_to_broadcast = set(target_cell_types) if target_cell_types else set()

    # ==========================================
    # PATH A: Use Dataset (Preferred)
    # ==========================================
    if dataset is not None:
        print(f"Extracting baseline features from Dataset (N={len(dataset)})...")
        for i in tqdm(range(len(dataset))):
            uuid, species, cell_type, expr_vector, meta_info, seq_emb, count_emb = dataset[i]
            uuid_str = str(uuid)
            parts = uuid_str.split('-')
            
            if len(parts) < 2: continue
            tid = parts[0]
            
            lookup_tid = tid
            if lookup_tid not in seqs_dict:
                tid_no_version = tid.split('.')[0]
                if tid_no_version in seqs_dict:
                    lookup_tid = tid_no_version
                else:
                    continue
                    
            cds_s = int(meta_info.get("cds_start_pos", -1))
            cds_e = int(meta_info.get("cds_end_pos", -1))
            if cds_s == -1 or cds_e == -1: continue
                
            m_start = max(0, cds_s - 1)
            m_end = cds_e
            full_seq = seqs_dict[lookup_tid].upper()
            
            if m_start < 0 or m_end > len(full_seq) or m_start >= m_end: continue
            
            # Sequence Segmentation
            utr5_seq = full_seq[:m_start]
            cds_seq = full_seq[m_start:m_end]
            utr3_seq = full_seq[m_end:]
            
            cds_len = len(cds_seq)
            if cds_len <= 0: continue
                
            # Sequence Features
            utr5_len = len(utr5_seq)
            utr3_len = len(utr3_seq)
            codon_freq_sum = calculate_codon_freq_sum(cds_seq)
            cds_gc = calculate_gc_content(cds_seq)
            kozak_score = calculate_kozak_score(full_seq, m_start) 
            uaug_count = utr5_seq.replace('U', 'T').count('ATG')
            
            seq_features = {
                '5UTR_Length': utr5_len,
                'Inverse_5UTR_Length': 1000 / (utr5_len + 10),
                'CDS_Length': cds_len,
                'Inverse_CDS_Length': 1000 / (cds_len + 10),
                '3UTR_Length': utr3_len,
                'Inverse_3UTR_Length': 1000 / (utr3_len + 10),
                'CAI': codon_freq_sum / cds_len,
                'CDS_GC_Content': cds_gc,
                'Inverse_CDS_GC_Content': 1 - cds_gc,
                'Kozak_Score': kozak_score,
                'uAUG_Count': uaug_count,
                'Inverse_uAUG_Count': 10 / (uaug_count + 1)
            }

            # Extract real TE indicators
            te_scale_val = meta_info.get("te_scale", np.nan)
            if te_scale_val is not None and not pd.isna(te_scale_val):
                te_scale_val = float(te_scale_val)

            if isinstance(count_emb, torch.Tensor):
                density = count_emb.detach().cpu().numpy()
            else:
                density = count_emb

            if unlog_data:
                density = np.expm1(density.astype(np.float32))
            
            if len(density.shape) > 1 and density.shape[1] > 1:
                density_profile = np.sum(density, axis=1)
            elif len(density.shape) > 1 and density.shape[1] == 1:
                density_profile = density.flatten()
            else:
                density_profile = density
                
            morf_mean_density = calculate_morf_mean_signal(density_profile, m_start, m_end)

            # Broadcasting
            cells_to_process = cells_to_broadcast.copy()
            cells_to_process.add(cell_type) 
            
            for ct in cells_to_process:
                key = (tid, ct)
                if key not in feature_dict:
                    row_data = {'Tid': tid, 'Cell_Type': ct}
                    row_data.update(seq_features)
                    row_data['te_scale'] = np.nan
                    row_data['mORF_Mean_Density'] = np.nan
                    feature_dict[key] = row_data
                
                # Fill real TE values only for the actual cell type present in the dataset
                if ct == cell_type:
                    feature_dict[key]['te_scale'] = te_scale_val
                    feature_dict[key]['mORF_Mean_Density'] = morf_mean_density

    # ==========================================
    # PATH B: Use tx_cds_file (Fallback)
    # ==========================================
    else:
        print(f"Loading transcript CDS metadata from {tx_cds_file}...")
        with open(tx_cds_file, 'rb') as f:
            cds_dict = pickle.load(f)
            
        print(f"Extracting baseline features entirely from sequence dictionary...")
        if not cells_to_broadcast:
            print("Warning: No target_cell_types provided. Using ['Generic'] as default.")
            cells_to_broadcast = {'Generic'}
            
        for tid, seq_str in tqdm(seqs_dict.items(), desc="Calculating Features"):
            clean_tid = str(tid).split('.')[0] if str(tid).startswith('ENST') else str(tid).split('|')[0]
            
            cds_info = cds_dict.get(clean_tid, cds_dict.get(tid))
            if not cds_info: continue
                
            cds_s = int(cds_info.get("cds_start_pos", -1)) if isinstance(cds_info, dict) else getattr(cds_info, "cds_start_pos", -1)
            cds_e = int(cds_info.get("cds_end_pos", -1)) if isinstance(cds_info, dict) else getattr(cds_info, "cds_end_pos", -1)
            
            if cds_s == -1 or cds_e == -1: continue
                
            m_start = max(0, cds_s - 1)
            m_end = cds_e
            full_seq = seq_str.upper()
            
            if m_start < 0 or m_end > len(full_seq) or m_start >= m_end: continue
            
            # Sequence Segmentation
            utr5_seq = full_seq[:m_start]
            cds_seq = full_seq[m_start:m_end]
            utr3_seq = full_seq[m_end:]
            
            cds_len = len(cds_seq)
            if cds_len <= 0: continue
                
            # Sequence Features
            utr5_len = len(utr5_seq)
            utr3_len = len(utr3_seq)
            codon_freq_sum = calculate_codon_freq_sum(cds_seq)
            cds_gc = calculate_gc_content(cds_seq)
            kozak_score = calculate_kozak_score(full_seq, m_start) 
            uaug_count = utr5_seq.replace('U', 'T').count('ATG')
            
            seq_features = {
                '5UTR_Length': utr5_len,
                'Inverse_5UTR_Length': 1000 / (utr5_len + 10),
                'CDS_Length': cds_len,
                'Inverse_CDS_Length': 1000 / (cds_len + 10),
                '3UTR_Length': utr3_len,
                'Inverse_3UTR_Length': 1000 / (utr3_len + 10),
                'CAI': codon_freq_sum / cds_len,
                'CDS_GC_Content': cds_gc,
                'Inverse_CDS_GC_Content': 1 - cds_gc,
                'Kozak_Score': kozak_score,
                'uAUG_Count': uaug_count,
                'Inverse_uAUG_Count': 10 / (uaug_count + 1)
            }

            # Broadcasting
            for ct in cells_to_broadcast:
                key = (clean_tid, ct)
                if key not in feature_dict:
                    row_data = {'Tid': clean_tid, 'Cell_Type': ct}
                    row_data.update(seq_features)
                    # No real data to extract, keep them NaN
                    row_data['te_scale'] = np.nan
                    row_data['mORF_Mean_Density'] = np.nan
                    feature_dict[key] = row_data

    # Convert to DataFrame
    df_all = pd.DataFrame(list(feature_dict.values()))
    
    if df_all.empty:
        print("No valid baseline data generated.")
        return
        
    # ==========================================
    # Batch Export CSVs
    # ==========================================
    metrics_to_export = [
        ('5UTR_Length', 'baseline_5utr_length.csv'),
        ('Inverse_5UTR_Length', 'baseline_inv_5utr_length.csv'),
        ('CDS_Length', 'baseline_cds_length.csv'),
        ('Inverse_CDS_Length', 'baseline_inv_cds_length.csv'),
        ('3UTR_Length', 'baseline_3utr_length.csv'),
        ('Inverse_3UTR_Length', 'baseline_inv_3utr_length.csv'),
        ('CAI', 'baseline_cai.csv'),
        ('CDS_GC_Content', 'baseline_cds_gc.csv'),
        ('Inverse_CDS_GC_Content', 'baseline_inv_cds_gc.csv'),
        ('Kozak_Score', 'baseline_kozak_score.csv'),
        ('uAUG_Count', 'baseline_uaug_count.csv'),
        ('Inverse_uAUG_Count', 'baseline_inv_uaug_count.csv')
    ]
    
    # Only export real TE metrics if Dataset was used
    if dataset is not None:
        metrics_to_export.extend([
            ('te_scale', 'baseline_te_scale.csv'),
            ('mORF_Mean_Density', 'baseline_morf_mean_density.csv') 
        ])
    
    print("\n>>> Exporting Baseline CSVs:")
    for col_name, file_name in metrics_to_export:
        df_sub = df_all[['Tid', 'Cell_Type', col_name]].copy()
        
        # Drop rows where the metric could not be calculated (or is NaN)
        df_sub = df_sub.dropna(subset=[col_name])
        
        if df_sub.empty:
            continue
            
        # Standardize metric column name to 'TE' for downstream pipeline compatibility
        df_sub.rename(columns={col_name: 'TE'}, inplace=True)
        
        save_path = os.path.join(out_dir, file_name)
        df_sub.to_csv(save_path, index=False)
        print(f"  - {file_name}")


def generate_mean_te_baselines(dataset, out_dir="./results/baselines", unlog_data=True):
    """
    遍历 Dataset，计算真实的 TE，并生成两种统计学基线：
    1. Transcript-Mean: 计算一个转录本在所有细胞类型中的平均翻译强度，并复制到该转录本的每个细胞记录中。
    2. Cell-Mean: 计算一个细胞类型内所有转录本的平均翻译强度，并复制到该细胞的每个转录本记录中。
    """
    os.makedirs(out_dir, exist_ok=True)
    results = []
    
    print(f"Extracting true TE from Dataset (N={len(dataset)})...")
    for i in tqdm(range(len(dataset))):
        # 兼容你的 dataset 返回结构
        uuid, species, cell_type, expr_vector, meta_info, seq_emb, count_emb = dataset[i]
        uuid_str = str(uuid)
        parts = uuid_str.split('-')
        
        if len(parts) < 2: continue
        tid = parts[0]
                
        cds_s = int(meta_info.get("cds_start_pos", -1))
        cds_e = int(meta_info.get("cds_end_pos", -1))
        if cds_s == -1 or cds_e == -1: 
            continue
            
        m_start = max(0, cds_s - 1)
        m_end = cds_e
        
        # 处理 count_emb (真实 Ribo-seq 密度)
        if isinstance(count_emb, torch.Tensor):
            truth = count_emb.detach().cpu().numpy()
        else:
            truth = count_emb
            
        if len(truth.shape) > 1 and truth.shape[1] > 1:
            truth = np.sum(truth, axis=1)
        elif len(truth.shape) > 1 and truth.shape[1] == 1:
            truth = truth.flatten()
            
        truth = truth.astype(np.float32)

        # 还原 log 转换的数据 (如果 dataset 里存储的是 log(x+1))
        if unlog_data:
            truth = np.expm1(truth)
            
        # 计算该样本真实的 TE
        actual_te = calculate_morf_mean_signal(truth, m_start, m_end)
        
        results.append({
            'Tid': tid,
            'Cell_Type': cell_type,
            'Actual_TE': actual_te
        })

    df_all = pd.DataFrame(results)
    
    if df_all.empty:
        print("No valid Ground Truth TE data extracted.")
        return
        
    print("\nCalculating statistical means...")
    
    # ---------------------------------------------------------
    # 基线 1: Transcript-Mean (同一个转录本跨细胞类型的平均)
    # ---------------------------------------------------------
    transcript_mean_df = df_all.groupby('Tid')['Actual_TE'].mean().reset_index()
    transcript_mean_df.rename(columns={'Actual_TE': 'Transcript_Mean_TE'}, inplace=True)
    
    # ---------------------------------------------------------
    # 基线 2: Cell-Mean (同一个细胞内跨转录本的平均)
    # ---------------------------------------------------------
    cell_mean_df = df_all.groupby('Cell_Type')['Actual_TE'].mean().reset_index()
    cell_mean_df.rename(columns={'Actual_TE': 'Cell_Mean_TE'}, inplace=True)

    # ---------------------------------------------------------
    # 数据合并：将计算好的全局均值贴回每一个具体的 (Tid, Cell_Type) 样本上
    # ---------------------------------------------------------
    df_merged = df_all.merge(transcript_mean_df, on='Tid', how='left')
    df_merged = df_merged.merge(cell_mean_df, on='Cell_Type', how='left')

    # ==========================================
    # 3. 批量导出 CSV (统一格式: Tid, Cell_Type, TE)
    # ==========================================
    print("\n>>> Exporting Baseline CSVs:")
    
    # 导出 Transcript-Mean
    df_transcript_out = df_merged[['Tid', 'Cell_Type', 'Transcript_Mean_TE']].copy()
    df_transcript_out.rename(columns={'Transcript_Mean_TE': 'TE'}, inplace=True)
    t_path = os.path.join(out_dir, 'baseline_transcript_mean.csv')
    df_transcript_out.to_csv(t_path, index=False)
    print(f"  - baseline_transcript_mean.csv saved.")
    
    # 导出 Cell-Mean
    df_cell_out = df_merged[['Tid', 'Cell_Type', 'Cell_Mean_TE']].copy()
    df_cell_out.rename(columns={'Cell_Mean_TE': 'TE'}, inplace=True)
    c_path = os.path.join(out_dir, 'baseline_cell_mean.csv')
    df_cell_out.to_csv(c_path, index=False)
    print(f"  - baseline_cell_mean.csv saved.")
    
    print("\n✅ Mean baselines successfully generated!")
