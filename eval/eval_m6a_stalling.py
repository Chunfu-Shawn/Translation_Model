import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from plotnine import *

# --- 1. Helper Functions ---

def is_rrach(seq, idx):
    """
    判断 seq[idx] 处的 'A' 是否符合 RRACH motif。
    RRACH motif definition (5-mer):
    Pos -2: R = A, G
    Pos -1: R = A, G
    Pos  0: A = A (Center)
    Pos +1: C = C
    Pos +2: H = A, C, U (not G)
    """
    # 边界检查
    if idx < 2 or idx + 2 >= len(seq): return False
    
    # 按照 RRACH 定义检查
    if seq[idx] != 'A': return False     # Center
    if seq[idx+1] != 'C': return False   # +1
    if seq[idx-1] not in ['A', 'G']: return False # -1
    if seq[idx-2] not in ['A', 'G']: return False # -2
    if seq[idx+2] == 'G': return False   # +2
    
    return True

# --- 2. Feature Extraction ---

def extract_m6a_profiles(preds, seqs, ratio_mask=1.0, window_size=30):
    """
    遍历所有转录本，收集 P-site 分布。
    CDS RRACH 特殊处理：对齐到 Codon Start (A-site alignment)。
    """
    # 初始化累加器 (window_size * 2 + 1)
    profile_len = window_size * 2 + 1
    
    data_agg = {
        'Background_A': {'sum': np.zeros(profile_len), 'count': 0},
        'UTR_RRACH':    {'sum': np.zeros(profile_len), 'count': 0},
        'CDS_RRACH':    {'sum': np.zeros(profile_len), 'count': 0}
    }

    print(f"Scanning RRACH (Align CDS sites to codon start)... Window: +/-{window_size}")
    
    for uuid, sample in tqdm(preds.items()):
        # 1. 基础校验
        tid = uuid.split("-")[0]
        if tid not in seqs: continue
        if ratio_mask not in sample['ratios']: continue
        
        cds_info = sample.get('cds_info', None)
        if cds_info is None or cds_info['end'] == -1: continue
        
        m_start = cds_info['start'] - 1 
        m_end = cds_info['end']
        
        seq_str = seqs[tid].upper()
        seq_len = len(seq_str)
        
        # 2. 获取预测并归一化
        try:
            pred_arr = np.expm1(sample['ratios'][ratio_mask]['pred'].reshape(-1))
            if len(pred_arr) != seq_len: pred_arr = pred_arr[:seq_len]
            
            # Global Mean Normalization
            global_mean = np.mean(pred_arr) + 1e-9
            norm_arr = pred_arr / global_mean
        except Exception: 
            continue

        # 3. 遍历寻找 'A'
        # 优化遍历范围
        # 对于 CDS 对齐，最大可能左移 2nt，所以左边界要多留一点空间
        safe_start = window_size + 2
        safe_end = seq_len - window_size - 1
        
        if safe_start >= safe_end: continue

        for i in range(safe_start, safe_end):
            if seq_str[i] != 'A': continue
            
            # 基础窗口 (以 A 为中心)
            base_window = norm_arr[i - window_size : i + window_size + 1]
            
            # --- Category 1: Background A ---
            data_agg['Background_A']['sum'] += base_window
            data_agg['Background_A']['count'] += 1
            
            # --- Check RRACH ---
            if is_rrach(seq_str, i):
                # 判断位置
                in_cds = (m_start <= i < m_end)
                
                if in_cds:
                    # --- CDS RRACH Special Logic: Align to Codon Start ---
                    
                    # 1. 计算 Frame (0, 1, 2)
                    # 距离 CDS Start 的距离
                    dist_from_start = i - m_start
                    frame = dist_from_start % 3
                    
                    # 2. 计算 Codon Start Index
                    # 如果 frame=0 (第1位), start = i
                    # 如果 frame=1 (第2位), start = i - 1
                    # 如果 frame=2 (第3位), start = i - 2
                    codon_start_idx = i - frame
                    
                    # 3. 提取窗口 (以 Codon Start 为中心 0 点)
                    # align to A-site
                    left_b = codon_start_idx - window_size - 3
                    right_b = codon_start_idx + window_size + 1 - 3
                    # 检查边界 (虽然上面有 safe_start，但 shift 后可能越界，需再次检查)
                    if left_b < 0 or right_b > seq_len:
                        continue
                    
                    aligned_window = norm_arr[left_b: right_b]
                    
                    data_agg['CDS_RRACH']['sum'] += aligned_window
                    data_agg['CDS_RRACH']['count'] += 1
                    
                else:
                    # UTR: 保持以 A 为中心
                    data_agg['UTR_RRACH']['sum'] += base_window
                    data_agg['UTR_RRACH']['count'] += 1
    
    # 4. 整理结果
    plot_data = []
    x_axis = np.arange(-window_size, window_size + 1)
    
    for cat, val in data_agg.items():
        if val['count'] == 0: continue
        mean_profile = val['sum'] / val['count']
        
        for pos, density in zip(x_axis, mean_profile):
            plot_data.append({
                'Distance': pos,
                'Normalized_Density': density,
                'Category': cat,
                'N_Sites': val['count']
            })
            
    print("Extraction stats:")
    for cat, val in data_agg.items():
        print(f"  {cat}: {val['count']} sites")
        
    return pd.DataFrame(plot_data)

# --- 3. Plotting ---

def plot_m6a_stalling(df, out_dir, suffix=""):
    if df.empty:
        print("No data to plot.")
        return
    
    # 生成带样本量的标签
    n_map = df[['Category', 'N_Sites']].drop_duplicates().set_index('Category')['N_Sites'].to_dict()
    
    # 注意说明 CDS 的对齐方式变化
    label_map = {
        'Background_A': f"All 'A' sites (n={n_map.get('Background_A',0)})",
        'UTR_RRACH':    f"UTR RRACH (Center A, n={n_map.get('UTR_RRACH',0)})",
        'CDS_RRACH':    f"CDS RRACH (Codon Start, n={n_map.get('CDS_RRACH',0)})"
    }
    
    df['Label'] = df['Category'].map(label_map)
    
    color_map = {
        label_map['Background_A']: '#999999', 
        label_map['UTR_RRACH']:    '#2ca02c', 
        label_map['CDS_RRACH']:    '#ff7f0e' 
    }
    size_map = {
        label_map['Background_A']: 0.8,
        label_map['UTR_RRACH']:    1.2,
        label_map['CDS_RRACH']:    1.2
    }

    p = (
        ggplot(df, aes(x='Distance', y='Normalized_Density', color='Label', size='Label'))
        + geom_line()
        + geom_vline(xintercept=0, linetype="dashed", color="gray", alpha=0.5)
        
        # 这里的 0 点含义不同了，对于 CDS 是 Codon Start
        + annotate("text", x=0, y=df['Normalized_Density'].min(), label="A site / Codon Start", size=8, color="gray", va="bottom")
        
        + theme_bw()
        + theme(
            figure_size=(6, 6),
            axis_text=element_text(size=12),
            axis_title=element_text(size=13),
            legend_position="top",
            legend_title=element_blank(),
            legend_direction='vertical' # 竖直排列图例防止太宽
        )
        + labs(
            x="Distance from Center (nt)",
            y="Normalized Ribosome Density"
        )
        + scale_color_manual(values=color_map)
        + scale_size_manual(values=size_map)
    )
    
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"m6a_stalling_profile.{suffix}.pdf")
    p.save(save_path, width=6, height=6, dpi=300)
    print(f"Saved plot to {save_path}")

# --- 4. Main Execution ---

def evaluate_m6a_effect(pred_pkl, seq_pkl, out_dir="./results/m6a_eval", ratio_mask=1.0, suffix=""):
    print(f"Loading data for m6A Analysis...")
    with open(pred_pkl, 'rb') as f: preds = pickle.load(f)
    with open(seq_pkl, 'rb') as f: seqs = pickle.load(f)
    
    df_data = extract_m6a_profiles(preds, seqs, ratio_mask=ratio_mask, window_size=30)
    
    if df_data.empty: return

    os.makedirs(out_dir, exist_ok=True)
    df_data.to_csv(os.path.join(out_dir, f"m6a_profile_data.{suffix}.csv"), index=False)
    
    plot_m6a_stalling(df_data, out_dir, suffix=suffix)