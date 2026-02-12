import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from plotnine import *
from eval.calculate_te import calculate_morf_efficiency, calculate_morf_mean_efficiency, calculate_morf_mean_density

# --- 1. Helper Functions ---

def find_motif_starts(sequence, motif):
    """
    返回 Motif 在序列中所有出现的起始位置索引 (0-based)
    """
    starts = []
    pos = sequence.find(motif)
    while pos != -1:
        starts.append(pos)
        pos = sequence.find(motif, pos + 1)
    return starts

# --- 2. Feature Extraction & Matrix Construction ---

def extract_positional_matrix(preds, seqs, ratio_mask=1.0, region_len=100):
    """
    提取 3' UTR 末端上游 region_len 长度内的 Motif 分布矩阵。
    """
    # 定义 Motif (T/U 兼容)
    target_motifs = {
        'CPSF (AAUAAA)': ['AATAAA', 'ATTAAA'], # CPSF 主要识别位点
        'FIP1L1 subunit (UU)': ['TT'],   # U-rich signal
        'CSTF (UGUA)': ['TGTA'],
        # 'CELF1 (GUGU)': ['TGTG', 'GTGT']   # GU-rich signal inside 3' UTR
    }
    
    # 存储中间数据
    data_list = []
    
    print(f"Scanning Motif Positions in 3' UTR (Last {region_len}nt)...")
    
    # 1. 第一遍遍历：收集 TE 和 序列信息
    for uuid, sample in tqdm(preds.items()):
        tid = uuid.split("-")[0]
        if tid not in seqs: continue
        if ratio_mask not in sample['ratios']: continue
        
        cds_info = sample.get('cds_info', None)
        if cds_info is None or cds_info['end'] == -1: continue
        
        m_start = cds_info['start'] - 1
        m_end = cds_info['end']
        
        seq_str = seqs[tid].upper()
        if m_end >= len(seq_str): continue
        
        # 提取 3' UTR
        utr3_seq = seq_str[m_end:]
        if len(utr3_seq) < region_len: continue # 太短忽略
        
        # 计算 TE
        try:
            pred_arr = np.expm1(sample['ratios'][ratio_mask]['pred'].reshape(-1))
            if len(pred_arr) != len(seq_str): pred_arr = pred_arr[:len(seq_str)]
            te = calculate_morf_mean_efficiency(pred_arr, m_start, m_end)
            if te < 1e-6: continue
        except: continue
        
        data_list.append({
            'UTR3_Seq': utr3_seq,
            'TE': te
        })
        
    df = pd.DataFrame(data_list)
    if df.empty: return pd.DataFrame()
    
    # 2. 定义分组 (Top/Bottom 10%)
    q10 = df['TE'].quantile(0.10)
    q90 = df['TE'].quantile(0.90)
    
    print(f"Grouping: Top 10% (TE > {q90:.4f}), Bottom 10% (TE < {q10:.4f})")
    
    groups = {
        'Top 10%': df[df['TE'] > q90],
        'Bottom 10%': df[df['TE'] < q10]
    }
    
    # 3. 构建位置频率矩阵
    # X轴: -100, -99, ..., -1, 0 (PolyA site)
    x_axis = np.arange(-region_len, 1) 
    plot_data = []

    for group_name, sub_df in groups.items():
        n_samples = len(sub_df)
        if n_samples == 0: continue
        
        for motif_name, patterns in target_motifs.items():
            # 初始化计数数组 (index 0 corresponds to -region_len)
            # 长度为 region_len + 1
            pos_counts = np.zeros(region_len + 1)
            
            for seq in sub_df['UTR3_Seq']:
                seq_len = len(seq)
                # 我们关心的是相对于 3' 末端的位置
                # e.g., seq="...TGTA...", len=200. "TGTA" at index 190.
                # relative_pos = 190 - 200 = -10.
                
                for pat in patterns:
                    starts = find_motif_starts(seq, pat)
                    for s in starts:
                        rel_dist = s - seq_len 
                        # rel_dist 是负数，例如 -10
                        # 映射到数组索引: rel_dist + region_len
                        # 如果 rel_dist 在 [-region_len, 0] 范围内
                        if -region_len <= rel_dist <= 0:
                            idx = rel_dist + region_len
                            pos_counts[idx] += 1
            
            # 计算频率 (Fraction of transcripts having motif at this pos)
            freqs = pos_counts / n_samples
            
            # 平滑处理 (Rolling Mean, window=4)
            freqs_smooth = np.convolve(freqs, np.ones(5)/5, mode='same')
            
            # 收集数据用于绘图
            for i, freq in enumerate(freqs_smooth):
                dist = i - region_len # 还原回 -100, -99...
                plot_data.append({
                    'Position': dist,
                    'Frequency': freq,
                    'Motif_Type': motif_name,
                    'Group': group_name
                })
                
    return pd.DataFrame(plot_data)

# --- 3. Plotting ---
def plot_motif_position_profile(df, out_dir, suffix=""):
    if df.empty: return
    
    # 设定线型和颜色映射
    # Top: 实线, 红色/暖色; Bottom: 虚线, 蓝色/冷色 (参考常见的高低表达配色)
    # 或者参考您之前的：Top 绿色，Bottom 红色
    color_map = {'Top 10%': '#E64B35', 'Bottom 10%': '#4DBBD5'} # NPG 风格配色
    linetype_map = {'Top 10%': 'solid', 'Bottom 10%': 'dashed'}
    plot_df = df.copy()
    plot_df['Motif_Type'] = pd.Categorical(plot_df['Motif_Type'], 
                                           categories=['CPSF (AAUAAA)', 'FIP1L1 subunit (UU)', 'CSTF (UGUA)'], 
                                           ordered=True)
    
    p = (
        ggplot(plot_df, aes(x='Position', y='Frequency', color='Group', linetype='Group'))
        + geom_line(size=1)
        
        # 分面：上下排列或左右排列
        + facet_wrap('~Motif_Type', scales='free_y', ncol=1)
        
        # 标注 PolyA site
        + geom_vline(xintercept=0, linetype="--", color="gray")
        + annotate("text", x=7, y=0, label="PolyA", size=12, ha='right', va='bottom', color="gray")
        
        # 标注常见 PAS 区域 (-30 到 -10)
        + annotate("rect", xmin=-30, xmax=-10, ymin=-np.inf, ymax=np.inf, alpha=0.1, fill="gray")
        + labs(
            x="Distance from 3' End (nt)",
            y="Motif Frequency (Smoothed)"
        )
        + scale_x_continuous(expand=(0.01, 0))
        + scale_color_manual(values=color_map)
        + scale_linetype_manual(values=linetype_map)
        + coord_cartesian(xlim=[-50, None])
        + theme_bw()
        + theme(
            figure_size=(8, 10),
            axis_text=element_text(size=12),
            axis_title=element_text(size=14),
            strip_text=element_text(size=14),
            strip_background=element_blank(),
            legend_position="top",
            legend_title=element_blank()
        )
    )
    
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"3utr_positional_profile.{suffix}.pdf")
    p.save(save_path, width=5, height=6, dpi=300)
    print(f"Saved positional profile to {save_path}")

# --- 4. Main Execution ---
def evaluate_3utr_positional_profile(pred_pkl, seq_pkl, out_dir="./results/3utr_pos", ratio_mask=1.0, suffix=""):
    """
    入口函数
    """
    print(f"Loading data for Positional Profile...")
    with open(pred_pkl, 'rb') as f: preds = pickle.load(f)
    with open(seq_pkl, 'rb') as f: seqs = pickle.load(f)
    
    # 1. Extract & Calculate Matrix
    df_pos = extract_positional_matrix(preds, seqs, ratio_mask=ratio_mask, region_len=100)
    
    if df_pos.empty:
        print("No valid positional data extracted.")
        return
    
    # 保存原始绘图数据
    os.makedirs(out_dir, exist_ok=True)
    df_pos.to_csv(os.path.join(out_dir, f"positional_data.{suffix}.csv"), index=False)
    
    # 2. Plot
    plot_motif_position_profile(df_pos, out_dir, suffix=suffix)