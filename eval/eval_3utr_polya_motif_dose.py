import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from plotnine import *
from scipy.stats import spearmanr, ranksums
from eval.calculate_te import *

# --- 1. Helper Functions ---

def count_motif_occurrences(sequence, motif):
    """
    计算 Motif 在序列中出现的次数 (允许重叠与否取决于需求，这里使用标准 count 非重叠)
    """
    return sequence.count(motif)


# --- 2. Feature Extraction ---

def extract_3utr_motif_abundance(preds, seqs, ratio_mask=1.0, min_3utr_len=50):
    """
    提取 3' UTR 序列并统计 CPSF, CSTF, CELF1-binding motif 的丰度。
    """
    results = []
    
    # 定义感兴趣的 Motif
    # 注意：T 代表 U
    target_motifs = {
        'CPSF (AAUAAA)': 'AATAAA',  # Core PolyA Signal
        'CSTF (UGUA)': 'TGTA',      # Upstream Sequence Element (CFIm)
        'CELF1 (GUGU)': 'GTGT'   # (Instability/DSE-like)
    }
    
    print(f"Analyzing 3' UTR Motif Abundance (ratio_mask={ratio_mask})...")
    
    for uuid, sample in tqdm(preds.items()):
        # 1. 基础校验
        tid = uuid.split("-")[0]
        if tid not in seqs: continue
        if ratio_mask not in sample['ratios']: continue
        
        # 2. 获取 CDS 信息
        cds_info = sample.get('cds_info', None)
        if cds_info is None or cds_info['end'] == -1: continue
        
        m_start = cds_info['start'] - 1 
        m_end = cds_info['end']
        
        # 3. 提取 3' UTR
        seq_str = seqs[tid].upper()
        seq_len = len(seq_str)
        
        # 3' UTR 是从 CDS 结束到转录本结束
        if m_end >= seq_len: continue # 没有 3' UTR
        
        utr3_seq = seq_str[m_end:]
        if len(utr3_seq) < min_3utr_len: continue # 过滤过短的 3' UTR
        
        # 4. 计算 TE
        try:
            pred_arr = np.expm1(sample['ratios'][ratio_mask]['pred'].reshape(-1))
            if len(pred_arr) != seq_len: pred_arr = pred_arr[:seq_len]
            te = calculate_morf_mean_efficiency(pred_arr, m_start, m_end)
            if te < 1e-6: continue
        except: continue
        
        # 5. 统计 Motif
        for motif_name, motif_seq in target_motifs.items():
            count = count_motif_occurrences(utr3_seq, motif_seq)
            
            results.append({
                'UUID': uuid,
                'Motif_Type': motif_name,
                'Count': count,
                'UTR3_Length': len(utr3_seq),
                'mORF_TE': te
            })
            
    return pd.DataFrame(results)

# --- 3. Plotting & Analysis ---

def plot_3utr_motif_dose(df, out_dir, suffix=""):
    if df.empty: return
    
    # 2. 数据分箱 (Binning)
    # Motif 数量是离散的，且长尾 (大部分是 0, 1, 2，
    cutoff = 2

    # 1. 过滤离群 TE
    upper_limit = df['mORF_TE'].quantile(0.99)
    plot_df = df[(df['Count'] <= cutoff) & (df['mORF_TE']<upper_limit)].copy()
    
    def bin_count(x):
        # if x >= cutoff: return f"{cutoff}+"
        return str(int(x))
    
    plot_df['Count_Group'] = plot_df['Count'].apply(bin_count)
    
    # 设定分类顺序
    cats = [str(i) for i in range(cutoff + 1)]
    plot_df['Count_Group'] = pd.Categorical(plot_df['Count_Group'], categories=cats, ordered=True)
    
    # 3. 计算相关性 (Annotation)
    # 使用原始 Count 计算 Spearman R
    cor_data = []
    for mtype in plot_df['Motif_Type'].unique():
        sub = plot_df[plot_df['Motif_Type'] == mtype]
        if len(sub) > 10:
            r, p = spearmanr(sub['Count'], sub['mORF_TE'])
            p_text = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
            
            # 简单的趋势描述
            trend = "Pos" if r > 0 else "Neg"
            
            cor_data.append({
                'Motif_Type': mtype,
                'Label': f"R = {r:.3f}\nP = {p_text}",
                'x': 1, 
                'y': sub['mORF_TE'].max() * 0.95
            })
    cor_df = pd.DataFrame(cor_data)
    
    # 4. 绘图
    p = (
        ggplot(plot_df, aes(x='Count_Group', y='mORF_TE', fill='Motif_Type'))
        + geom_violin(trim=True, alpha=0.6, show_legend=False)
        + geom_boxplot(width=0.15, fill='white', alpha=0.8, outlier_size=0, show_legend=False)
        
        # 分面：按 Motif 类型
        + facet_wrap('~Motif_Type', scales='free_y')
        
        # 添加统计标签
        + geom_text(data=cor_df, mapping=aes(x='x', y='y', label='Label'),
                    ha='left', size=10, inherit_aes=False)
        
        + theme_bw()
        + theme(
            figure_size=(12, 6),
            axis_text=element_text(size=12),
            axis_title=element_text(size=14),
            strip_text=element_text(size=12),
            strip_background=element_blank(),
        )
        + labs(
            x=f"Motif Count in 3' UTR (Bin >={cutoff})",
            y="mORF Translation Efficiency (TE)"
        )
        + scale_fill_brewer(type="qual", palette="Set2")
    )
    
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"3utr_motif_abundance.{suffix}.pdf")
    p.save(save_path, width=12, height=6, dpi=300)
    print(f"Saved 3' UTR motif plot to {save_path}")
    
    # 5. 保存统计表 (Mean TE per group)
    stats_path = os.path.join(out_dir, f"3utr_motif_stats.{suffix}.csv")
    stats = plot_df.groupby(['Motif_Type', 'Count_Group'])['mORF_TE'].agg(['count', 'mean', 'median']).reset_index()
    stats.to_csv(stats_path, index=False)
    print(f"Saved stats to {stats_path}")

# --- 4. Main Execution ---

def evaluate_3utr_cis_elements(pred_pkl, seq_pkl, out_dir="./results/3utr_eval", ratio_mask=1.0, suffix=""):
    """
    主入口函数
    """
    print(f"Loading data for 3' UTR Analysis...")
    with open(pred_pkl, 'rb') as f: preds = pickle.load(f)
    with open(seq_pkl, 'rb') as f: seqs = pickle.load(f)
    
    # 1. Extract
    df_motifs = extract_3utr_motif_abundance(preds, seqs, ratio_mask=ratio_mask)
    
    if df_motifs.empty:
        print("No valid 3' UTR data extracted.")
        return
    
    # 保存原始数据
    os.makedirs(out_dir, exist_ok=True)
    df_motifs.to_csv(os.path.join(out_dir, f"3utr_motif_raw.{suffix}.csv"), index=False)
    
    # 2. Plot
    plot_3utr_motif_dose(df_motifs, out_dir, suffix=suffix)

# if __name__ == "__main__":
#     evaluate_3utr_cis_elements("preds.pkl", "seqs.pkl", suffix="test_run")