import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from plotnine import *
from scipy.stats import spearmanr, pearsonr
from eval.calculate_te import *


# --- 2. Feature Extraction Function ---

def extract_cap_proximal_data(preds, seqs, ratio_mask=1.0, min_utr_len=50):
    """
    提取 Cap-proximal (1-15nt) 序列和 TE。
    """
    results = []
    print(f"Extracting Cap-proximal sequences (1-15nt)...")
    
    for uuid, sample in tqdm(preds.items()):
        tid = uuid.split("-")[0]
        if tid not in seqs: continue
        if ratio_mask not in sample['ratios']: continue
        
        # 获取 CDS 信息
        cds_info = sample.get('cds_info', None)
        if cds_info is None or cds_info['start'] == -1: 
            continue
        
        m_start = cds_info['start'] - 1 # 0-based
        m_end = cds_info['end']

        if m_start <= min_utr_len: # remove tx with short utr
            continue
        
        # 获取序列
        seq_str = seqs[tid].upper()
        
        # 提取前 15nt (Cap-proximal)
        # 如果序列长度小于 15，则跳过，避免偏差
        if len(seq_str) < 15:
            continue
        cap_seq = seq_str[:15]
        
        # 计算 TE
        try:
            pred_arr = np.expm1(sample['ratios'][ratio_mask]['pred'].reshape(-1))
            seq_len = len(seq_str)
            if len(pred_arr) != seq_len:
                pred_arr = pred_arr[:seq_len]
            
            te = calculate_morf_efficiency(pred_arr, m_start, m_end)

            re_te = np.mean(pred_arr[m_start:m_end])/(np.mean(pred_arr) + 1e-9)
            
            # 简单的质量控制，去除极低 TE (Log space 下的噪音)
            if te < 1e-6: continue

        except Exception:
            continue
            
        results.append({
            'UUID': uuid,
            'Cap_Seq': cap_seq,
            'mORF_TE': te,
            'Relative_TE': re_te
        })
        
    return pd.DataFrame(results)

# --- 3. Frequency Calculation Logic ---

def calculate_positional_freq(df, group_name):
    """
    输入包含 'Cap_Seq' 的 DataFrame，计算 1-15 位置的 ACGT 频率。
    返回格式：[Position, Base, Frequency, Group]
    """
    if df.empty: return pd.DataFrame()
    
    # 将序列列表转换为字符矩阵 (N_samples x 15)
    # 这比循环快得多
    seq_matrix = np.array([list(s) for s in df['Cap_Seq'].values])
    
    freq_data = []
    bases = ['A', 'C', 'G', 'T']
    n_samples = len(df)
    
    for i in range(15): # Position 0 to 14
        col = seq_matrix[:, i]
        # 统计当前位置各碱基数量
        unique, counts = np.unique(col, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        
        for base in bases:
            count = counts_dict.get(base, 0)
            freq = count / n_samples
            freq_data.append({
                'Position': i + 1, # 1-based index for plot
                'Base': base,
                'Frequency': freq,
                'Group': group_name
            })
            
    return pd.DataFrame(freq_data)

def process_groups_and_freqs(df):
    """
    根据 TE 分位数划分组别，并计算频率。
    """
    # 计算分位数
    q10 = df['mORF_TE'].quantile(0.10)
    q90 = df['mORF_TE'].quantile(0.90)
    
    print(f"TE Thresholds: Top 10% > {q90:.4f}, Bottom 10% < {q10:.4f}")
    
    # 划分数据集
    groups = {
        'All': df,
        'Top 10%': df[df['mORF_TE'] > q90],
        'Bottom 10%': df[df['mORF_TE'] < q10]
    }
    
    all_freqs = []
    for name, sub_df in groups.items():
        if len(sub_df) > 0:
            freq_df = calculate_positional_freq(sub_df, name)
            all_freqs.append(freq_df)
            
    return pd.concat(all_freqs, ignore_index=True)

# --- 4. Plotting ---

def plot_cap_proximal_lines(freq_df, out_dir, suffix=""):
    """
    绘制 ACGT 四个面板的折线图。
    参考提供的图片风格：
    - Top/All 为实线
    - Bottom 为虚线
    """
    # 设定 Group 的顺序，保证图例好看
    group_order = ['Top 10%', 'All', 'Bottom 10%']
    freq_df['Group'] = pd.Categorical(freq_df['Group'], categories=group_order, ordered=True)
    
    # 设定线型映射 (Line Type Map)
    # Top 和 All 用实线 ('solid')，Bottom 用虚线 ('dashed')
    linetype_map = {
        'All': 'solid',
        'Bottom 10%': 'dashed',
        'Top 10%': 'solid'
    }
    
    # 设定颜色映射 (Color Map)
    # Top 用绿色系，Bottom 用橙红系，All 用灰色
    color_map = {
        'All': '#737373',       # 灰色
        'Bottom 10%': '#fd8d3c',# 浅橙
        'Top 10%': '#e31a1c'  # 深红
    }
    
    # 设定粗细
    size_map = {
        'All': 1.0,
        'Bottom 10%': 1.2,
        'Top 10%': 1.2
    }

    p = (
        ggplot(freq_df, aes(x='Position', y='Frequency', group='Group', color='Group', linetype='Group'))
        # + geom_line(aes(size='Group'))
        + geom_smooth(aes(size='Group'), method='loess', span=1, se=False, alpha=0.8)
        + facet_wrap('~Base', scales='free_y', nrow=2) # 分面展示 ACGT
        + scale_x_continuous(breaks=range(1, 16, 2))   # X轴刻度 1, 3, 5...
        + scale_color_manual(values=color_map)
        + scale_linetype_manual(values=linetype_map)
        + scale_size_manual(values=size_map)
        + theme_bw()
        + theme(
            figure_size=(10, 8),
            axis_text=element_text(size=8, color="black"),
            axis_title=element_text(size=10),
            strip_text=element_text(size=10, face="bold"), # 分面标题 A, C, G, T
            strip_background=element_blank(),
            legend_position="top",
            legend_title=element_blank(),
            panel_grid_minor=element_blank()
        )
        + labs(
            x="Position (nt from Cap)",
            y="Nucleotide frequency"
        )
    )
    
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"cap_proximal_content.{suffix}.pdf")
    p.save(plot_path, width=6, height=5, dpi=300)
    print(f"Saved Cap-proximal plot to {plot_path}")

# --- 5. Main Execution ---

def evaluate_cap_proximal_content(pred_pkl, seq_pkl, out_dir="./results/cap_eval", ratio_mask=1.0, suffix=""):
    """
    主入口函数
    """
    # 1. Load Data
    print(f"Loading data for Cap analysis...")
    with open(pred_pkl, 'rb') as f: preds = pickle.load(f)
    with open(seq_pkl, 'rb') as f: seqs = pickle.load(f)
    
    # 2. Extract Data
    df_raw = extract_cap_proximal_data(preds, seqs, ratio_mask=ratio_mask)
    if df_raw.empty:
        print("No valid data found.")
        return

    # 3. Calculate Frequencies by Group
    print("Calculating positional frequencies...")
    df_freq = process_groups_and_freqs(df_raw)
    
    # 保存中间频率数据，方便自己检查
    csv_path = os.path.join(out_dir, f"cap_proximal_freqs.{suffix}.csv")
    os.makedirs(out_dir, exist_ok=True)
    df_freq.to_csv(csv_path, index=False)
    
    # 4. Plot
    print("Plotting...")
    plot_cap_proximal_lines(df_freq, out_dir, suffix=suffix)


# --- 1. Feature Extraction (Modified for GC Content) ---

def extract_5utr_gc_content(preds, seqs, ratio_mask=1.0, min_utr_len=50):
    """
    专门提取 5' UTR 的 GC 含量。
    Calculation: (Count(G) + Count(C)) / Length
    """
    results = []
    
    print(f"Analyzing 5' UTR GC Content (ratio_mask={ratio_mask})...")
    
    for uuid, sample in tqdm(preds.items()):
        # 1. Basic Validation
        tid = uuid.split("-")[0]
        if tid not in seqs: continue
        if ratio_mask not in sample['ratios']: continue
        
        # 2. Get CDS info
        cds_info = sample.get('cds_info', None)
        if cds_info is None or cds_info['start'] == -1: continue
        
        m_start = cds_info['start'] - 1 # 0-based
        m_end = cds_info['end']
        
        # 3. Extract 5' UTR
        seq_str = seqs[tid].upper()
        if m_start < min_utr_len: continue # Skip if UTR is too short
        
        utr_seq = seq_str[:m_start]
        utr_len = len(utr_seq)
        
        # 4. Calculate TE
        try:
            pred_arr = sample['ratios'][ratio_mask]['pred'].reshape(-1)
            if len(pred_arr) != len(seq_str): 
                pred_arr = pred_arr[:len(seq_str)]
            
            te = calculate_morf_efficiency(pred_arr, m_start, m_end)
            if te < 1e-6: continue # Skip noise
            
        except: continue

        # 5. Calculate GC Content (Modified Here)
        # 不再循环 ACGT，只计算 GC
        gc_count = utr_seq.count('G') + utr_seq.count('C')
        gc_percent = gc_count / utr_len
        
        results.append({
            'UUID': uuid,
            'Base': 'GC', # 统一标记为 GC
            'Content': gc_percent, # 0.0 to 1.0
            'mORF_TE': te
        })
            
    return pd.DataFrame(results)

# --- 2. Plotting (Modified for GC Content) ---

def plot_gc_content_effect(df, out_dir, suffix=""):
    if df.empty: return
    bins = [0.2, 0.45, 0.50, 0.55, 0.60, 0.8]
    labels = ["<45%", "45-50%", "50-55%","55-60%", ">60%"]

    # 1. 离群点过滤 (只保留 Top 99% 以下的数据)
    # upper_limit = df['mORF_TE'].quantile(0.99)
    plot_df = df[(df["Content"]>bins[0]) & (df["Content"]<bins[-1])].copy()
    
    # 2. X轴分箱 (Binning)
    # GC 含量通常集中在 30%-70%，但我们保留全范围 bin 以防万一
    plot_df['Content_Bin'] = pd.cut(plot_df['Content'], bins=bins, labels=labels, include_lowest=True)
    
    # 3. 计算相关性 (Annotation Data)
    cor_data = []
    # 这里只计算 GC 的相关性
    sub = plot_df[plot_df['Base'] == 'GC']
    if len(sub) > 10:
        # 使用 Spearman 因为生物学数据通常不是严格线性的
        r, p = pearsonr(sub['Content'], sub['mORF_TE'])
        p_text = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
        label = f"Spearman R = {r:.3f}\nP = {p_text}"
        
        cor_data.append({
            'Base': 'GC', 
            'Label': label,
            'x': 1, # Label位置
            'y': sub['mORF_TE'].min() * 1.05
        })
    cor_df = pd.DataFrame(cor_data)

    # 4. 绘图
    p = (
        ggplot(plot_df, aes(x='Content_Bin', y='mORF_TE', fill='Base'))
        + geom_violin(aes(fill='Content_Bin'), color="gray", size=0.8, alpha=0.5, trim=True, show_legend=False)
        + geom_boxplot(width=0.15, fill="white", size=0.8, alpha=0.8, 
                       outlier_alpha=0.2, outlier_size=0.5, show_legend=False)
        + geom_text(data=cor_df, mapping=aes(x='x', y='y', label='Label'), 
                    ha='left', size=10, inherit_aes=False)    
        + theme_bw()
        + theme(
            figure_size=(8, 6), # 调整尺寸，因为只有一个面，不需要太宽
            axis_text_x=element_text(rotation=45, hjust=1, size=10),
            axis_text_y=element_text(size=10),
            strip_text=element_text(size=14, face='bold'),
            axis_title=element_text(size=14)
        )
        + labs(
            x="GC Content in 5' UTR (%)",
            y="mORF translation efficiency"
        )
        + scale_fill_brewer(type="seq", palette="Blues") # 渐变色
    )
    
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"gc_content_dose_effect.{suffix}.pdf")
    p.save(save_path, width=6, height=6, dpi=300)
    print(f"Saved GC content plot to {save_path}")


# --- 3. Main Execution Module ---
def evaluate_5utr_gc_dose(pred_pkl, seq_pkl, out_dir="./results/gc_eval", ratio_mask=1.0, suffix=""):
    """
    主入口函数
    """
    print(f"Loading data for GC Content Analysis...")
    with open(pred_pkl, 'rb') as f: preds = pickle.load(f)
    with open(seq_pkl, 'rb') as f: seqs = pickle.load(f)
    
    # 1. Extract
    df_content = extract_5utr_gc_content(preds, seqs, ratio_mask=ratio_mask)
    
    if df_content.empty:
        print("No valid data extracted.")
        return
        
    # 保存一下中间数据
    os.makedirs(out_dir, exist_ok=True)
    df_content.to_csv(os.path.join(out_dir, f"gc_content_data.{suffix}.csv"), index=False)
    
    # 2. Plot
    plot_gc_content_effect(df_content, out_dir, suffix=suffix)