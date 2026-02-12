import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from plotnine import *
from scipy.stats import spearmanr, ranksums
from eval.calculate_te import calculate_morf_efficiency, calculate_morf_mean_density

# --- 1. Helper Functions ---

def scan_ccc_motif(sequence):
    """
    扫描序列中 CCC motif 的数量。
    规则：逐个核苷酸查找，找到 CCC 后跳过 3 个核苷酸继续查找（非重叠计数）。
    例如: CCCCC -> 1个 (前三个算，后两个不够), CCCCCC -> 2个
    """
    count = 0
    i = 0
    length = len(sequence)
    
    while i <= length - 3:
        codon = sequence[i : i+3]
        if codon == 'CCC':
            count += 1
            i += 3  # 找到了，跳过这3个
        else:
            i += 1  # 没找到，步进1
            
    return count


# --- 2. Feature Extraction Function ---

def extract_ccc_features(preds, seqs, ratio_mask=1.0, min_utr_len=150):
    """
    遍历预测结果，统计 5' UTR 上的 CCC 数量以及 Main ORF 的翻译效率。
    """
    results = []
    
    print(f"Scanning 5' UTRs for CCC motifs (ratio_mask={ratio_mask})...")
    
    for uuid, sample in tqdm(preds.items()):
        # 1. 基础校验
        tid = uuid.split("-")[0]
        if tid not in seqs: continue
        if ratio_mask not in sample['ratios']: continue
        
        # 2. Get CDS boundaries
        cds_info = sample.get('cds_info', None)
        if cds_info is None: 
            continue
        if cds_info['start'] == -1 or cds_info['end'] == -1:
            continue
        
        m_start = cds_info['start'] - 1 # Assuming 1-based in pickle, converting to 0-based
        m_end = cds_info['end']
            
        # 获取完整序列
        seq_str = seqs[tid].upper()
        
        # 3. 定义 5' UTR
        if m_start < min_utr_len: continue
        # 5' UTR 是从开头到 mORF 起始位点之前的序列
        utr5_seq = seq_str[:min_utr_len]
            
        # 4. 统计 CCC 数量
        ccc_count = scan_ccc_motif(utr5_seq)
        
        # 5. 获取预测密度并计算 TE
        try:
            # 还原 Log 空间到 Linear 空间
            pred_arr = np.expm1(sample['ratios'][ratio_mask]['pred'].reshape(-1))
            
            # 对齐长度
            seq_len = len(seq_str)
            if len(pred_arr) != seq_len:
                pred_arr = pred_arr[:seq_len] # 简单截断
            
            te = calculate_morf_mean_density(pred_arr, m_start, m_end)
            
        except Exception as e:
            continue
            
        results.append({
            'UUID': uuid,
            'Transcript_ID': tid,
            'UTR5_Length': len(utr5_seq),
            'CCC_Count': ccc_count,
            'mORF_TE': te
        })
        
    if not results:
        print("Warning: No data extracted.")
        return pd.DataFrame()
        
    return pd.DataFrame(results)

# --- 3. Plotting & Analysis ---

def plot_ccc_dose_dependent(df, out_dir, suffix=""):
    """
    绘制 CCC 数量 vs mORF TE 的 Violin + Boxplot。
    为了绘图美观，会将数量很少的 High Count 组（比如 >5）合并为 "5+"。
    """
    if df.empty: return
    
    # 设定一个阈值
    cutoff = 5

    # 1. 数据预处理：分组处理长尾数据
    plot_df = df[(df['CCC_Count'] <= 5)].copy()
    
    # 计算 Count 和 TE 的相关性 (使用原始 Count)
    r, p = spearmanr(plot_df['CCC_Count'], plot_df['mORF_TE'])
    p_text = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
    stats_label = f"Spearman R = {r:.3f}, P = {p_text}"
    print(stats_label)
    
    # 确保 X 轴顺序正确
    plot_df['CCC_Group'] = plot_df['CCC_Count'].apply(lambda x: str(x))
    categories = [str(i) for i in range(cutoff+1)]
    plot_df['CCC_Group'] = pd.Categorical(plot_df['CCC_Group'], categories=categories, ordered=True)

    # 2. 绘图
    p = (
        ggplot(plot_df, aes(x='CCC_Group', y='mORF_TE'))
        + geom_violin(aes(fill='CCC_Group'), color="gray", size=0.8, alpha=0.5, trim=True, show_legend=False)
        + geom_boxplot(width=0.15, fill="white", size=0.8, alpha=0.8, 
                       outlier_alpha=0, outlier_size=0, outlier_shape=None, 
                       show_legend=False)
        # 统计标签
        + annotate("text", x=1, y=plot_df['mORF_TE'].min() * 1, label=stats_label, ha='left', size=10)
        + theme_bw()
        + theme(
            axis_text=element_text(size=12),
            axis_title=element_text(size=14)
        )
        + labs(
            x=f"Number of CCC motifs in 5' UTR",
            y="mORF translation efficiency"
        )
        + scale_fill_brewer(type="seq", palette="Blues") # 渐变色
    )
    
    # 保存
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"ccc_motif_dose_effect.{suffix}.pdf")
    p.save(plot_path, width=5, height=5, dpi=300)
    print(f"Saved dose-dependent plot to {plot_path}")

    # 3. 保存简单的统计表
    stats_path = os.path.join(out_dir, f"ccc_motif_stats.{suffix}.csv")
    stats = plot_df.groupby('CCC_Group')['mORF_TE'].agg(['count', 'mean', 'median', 'std']).reset_index()
    stats.to_csv(stats_path, index=False)
    print(f"Saved stats summary to {stats_path}")

# --- 4. Main Execution Function ---

def evaluate_5utr_ccc_motif(pred_pkl, seq_pkl, out_dir="./results/ccc_eval", ratio_mask=1.0, suffix=""):
    """
    主入口函数：读取数据 -> 提取 CCC 特征 -> 绘图。
    """
    # 1. Load Data
    print(f"Loading predictions: {pred_pkl}")
    with open(pred_pkl, 'rb') as f:
        preds = pickle.load(f)
        
    print(f"Loading sequences: {seq_pkl}")
    with open(seq_pkl, 'rb') as f:
        seqs = pickle.load(f)
        
    # 2. Extract Features
    df_features = extract_ccc_features(preds, seqs, ratio_mask=ratio_mask)
    
    if df_features.empty:
        print("No features extracted. Exiting.")
        return

    # 保存中间结果 CSV
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"ccc_features_raw.{suffix}.csv")
    df_features.to_csv(csv_path, index=False)
    print(f"Saved raw features to {csv_path}")
    
    # 3. Plotting
    plot_ccc_dose_dependent(df_features, out_dir, suffix=suffix)