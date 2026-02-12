import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from plotnine import *
from scipy.stats import ranksums
from scipy.stats import pearsonr, spearmanr
import itertools

# --- 1. Global Constants & Helper Functions (Kozak Logic) ---

KOZAK_WEIGHTS = {
    '-6': {'A': 0, 'C': -0.03, 'G': 0.05, 'T': -0.02},
    '-5': {'A': -0.06, 'C': 0.04,  'G': -0.01, 'T': 0.03},
    '-4': {'A': 0.07, 'C': 0.08,  'G': -0.06, 'T': -0.09}, 
    '-3': {'A': 0.14, 'C': -0.08,  'G': 0.14, 'T': -0.29},  # Critical
    '-2': {'A': 0.03, 'C': 0.06,  'G': -0.12, 'T': -0.01},
    '-1': {'A': 0.02, 'C': 0.04,  'G': 0.02, 'T': 0.02}, 
    '+4': {'A': -0.02, 'C': -0.09,  'G': 0.12, 'T': -0.02},  # Critical
    '+5': {'A': -0.01, 'C': 0.04,  'G': 0.05, 'T': -0.07},
}
# KOZAK_WEIGHTS = {
#     '-6': {'A': 0.1, 'C': 0.05, 'G': 0.3, 'T': 0.05},
#     '-5': {'A': 0.1, 'C': 0.2,  'G': 0.1, 'T': 0.1},
#     '-4': {'A': 0.1, 'C': 0.4,  'G': 0.1, 'T': 0.1}, 
#     '-3': {'A': 1.0, 'C': 0.0,  'G': 0.5, 'T': 0.0},  # Critical
#     '-2': {'A': 0.1, 'C': 0.3,  'G': 0.1, 'T': 0.1},
#     '-1': {'A': 0.1, 'C': 0.4,  'G': 0.1, 'T': 0.1}, 
#     '+4': {'A': 0.2, 'C': 0.1,  'G': 1.0, 'T': 0.1},  # Critical
#     '+5': {'A': 0.1, 'C': 0.9,  'G': 0.1, 'T': 0.7},
# }

def calculate_kozak_score(sequence, start_idx):
    """
    量化打分 Kozak 序列强度。
    """
    if start_idx < 6 or start_idx + 5 >= len(sequence):
        return None
    
    score = 0
    # Calculate -6 to -1
    for i, pos_name in enumerate(['-6', '-5', '-4', '-3', '-2', '-1']):
        base = sequence[start_idx - 6 + i]
        score += KOZAK_WEIGHTS[pos_name].get(base, 0)
    
    # Calculate +4 (sequence index start_idx + 3 because start_idx is +1)
    # Correct logic based on original code: sequence[start_idx + 3] corresponds to +4 position relative to ATG start
    for i, pos_name in enumerate(['+4', '+5']):
        base = sequence[start_idx + 3 + i]
        score += KOZAK_WEIGHTS[pos_name].get(base, 0)
    
    return score

def classify_kozak_context(sequence, start_idx):
    """
    基于规则分类 Kozak 强度：
    Rule: +4 Must be G, -3 Best is A or G.
    """
    if start_idx < 3 or start_idx + 4 >= len(sequence):
        return None
        
    # 获取关键位置碱基
    pos_minus_3 = sequence[start_idx - 3]
    pos_plus_4  = sequence[start_idx + 3]
    
    is_plus4_G = (pos_plus_4 == 'G')
    is_minus3_R = (pos_minus_3 in ['A', 'G'])
    
    if is_plus4_G and is_minus3_R:
        return "Strong (+4G, -3R)"
    elif is_minus3_R:
        return "Moderate (-3R)"     # 有最关键的+4G，但-3不理想 (注：原注释似乎反了，但逻辑保留原样)
    elif is_plus4_G:
        return "Moderate (+4G)"   # 有次关键的-3R，但+4不理想
    else:
        return "Weak"             # 两个关键位点都不符合

# --- 2. Feature Extraction Function ---

def extract_initiation_features(preds, seqs, ratio_mask=1.0):
    """
    遍历预测结果，提取起始密码子附近的特征。
    
    Args:
        preds (dict): 预测结果字典，key为uuid。
        seqs (dict): 序列字典，key为transcript_id。
        ratio_mask (float/int): 指定使用的 ratio mask (原代码中为 1.0)。
    
    Returns:
        pd.DataFrame: 包含所有提取出的 Motif 信息的 DataFrame。
    """
    results = []
    # 定义感兴趣的起始密码子
    target_codons = {'ATG', 'CTG', 'TTG', 'GTG'}
    
    print(f"Scanning full transcripts using ratio_mask={ratio_mask}...")
    
    for uuid, sample in tqdm(preds.items()):
        # 1. Get Sequence
        tid = uuid.split("-")[0]
        if tid not in seqs: continue
        
        seq_str = seqs[tid].upper()
        seq_len = len(seq_str)
        
        # 2. Check Prediction Data Availability
        # 使用传入的 ratio_mask 替代硬编码的 1.0
        if ratio_mask not in sample['ratios']: 
            continue
        
        # 还原 P-site 信号 (Linear Space)
        # 注意：这里使用了 ratio_mask 来获取对应的 pred
        pred_arr = np.expm1(sample['ratios'][ratio_mask]['pred'].reshape(-1))
        
        # 简单的长度对齐检查
        if len(pred_arr) != seq_len:
            limit = min(len(pred_arr), seq_len)
        else:
            limit = seq_len
            
        # 计算该转录本的全局平均密度 (用于归一化)
        global_mean = np.mean(pred_arr[:limit]) + 1e-6
        
        # 3. 遍历序列扫描 Motif
        # 从 6 开始到 limit-5，预留 Kozak 上下文空间
        for i in range(6, limit - 5):
            codon = seq_str[i : i+3]
            
            if codon in target_codons:
                # 获取 Kozak 分类
                k_class = classify_kozak_context(seq_str, i)
                if k_class is None: continue
                
                # 获取 Kozak 分数 (数值)
                k_score = calculate_kozak_score(seq_str, i)
                
                # 提取 P-site 强度 (At position 0)
                p_site_intensity = pred_arr[i] / (np.sum(pred_arr[i-3:i+3]) + global_mean)
                
                # 记录信息
                results.append({
                    'UUID': uuid,
                    'Start codon': codon,
                    'Kozak class': k_class,
                    'Kozak score': k_score,
                    'Normalized_Density': p_site_intensity
                })

    if not results:
        print("Warning: No motifs extracted. Check your data or ratio_mask.")
        return pd.DataFrame()

    meta_df = pd.DataFrame(results)
    
    # 设定分类的顺序 (供绘图使用)
    meta_df['Kozak class'] = pd.Categorical(
        meta_df['Kozak class'], 
        categories=["Weak", "Moderate (+4G)", "Moderate (-3R)", "Strong (+4G, -3R)"], 
        ordered=True
    )
    # 设定 Codon 顺序
    meta_df['Start codon'] = pd.Categorical(
        meta_df['Start codon'],
        categories=['ATG', 'CTG', 'GTG', 'TTG'],
        ordered=True
    )
    
    print(f"Extracted {len(meta_df)} motifs from transcripts.")
    return meta_df

def plot_kozak_correlation_scatter(meta_df, out_dir, suffix=""):
    """
    Scatter plot with Density: Kozak Score (X) vs Log P-site Density (Y).
    Faceted by Start Codon.
    Features:
      - geom_bin2d: To show point density.
      - Pearson correlation label: In the top-left corner.
      - Linear regression line.
    """
    if meta_df.empty: 
        return

    # --- 1. 数据采样 (仅用于绘图渲染加速) ---
    # 统计计算建议使用全量数据，绘图可以使用采样数据
    if len(meta_df) > 10000:
        print("Sampling 10,000 points for scatter plot rendering...")
        plot_df = meta_df.sample(10000, random_state=42)
    else:
        plot_df = meta_df

    # plot_df = plot_df[plot_df["Normalized_Density"]!=0]

    # --- 2. 计算 Pearson 相关性 (按 Start Codon 分组) ---
    cor_data = []
    # 使用全量数据 plot_df 计算相关性，确保统计准确
    for codon, group in plot_df.groupby('Start codon'):
        # 去除 NaN
        sub = group.dropna(subset=['Kozak score', 'Normalized_Density'])
        if len(sub) < 2: continue
        
        r, p = spearmanr(sub['Kozak score'], sub['Normalized_Density'])
        
        # 格式化 P 值
        p_text = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
        label_text = f"R = {r:.3f}\nP = {p_text}"
        
        # 确定标签位置 (左上角)
        # X轴位置: 最小值
        # Y轴位置: 最大值
        x_pos = sub['Kozak score'].min()
        y_pos = sub['Normalized_Density'].max()
        
        cor_data.append({
            'Start codon': codon,
            'Label': label_text,
            'x': x_pos,
            'y': y_pos
        })
    
    cor_df = pd.DataFrame(cor_data)
    # 确保 cor_df 的分面列类型与主数据一致
    if 'Start codon' in plot_df.columns and isinstance(plot_df['Start codon'].dtype, pd.CategoricalDtype):
         cor_df['Start codon'] = pd.Categorical(cor_df['Start codon'], categories=plot_df['Start codon'].cat.categories)

    # --- 3. 绘图 ---
    p = (
        ggplot(plot_df, aes(x='Kozak score', y='Normalized_Density'))
        # 散点：透明度调低，防止重叠
        + geom_point(alpha=0.1, size=1.5, color='#2c3e50', shape='.') 
        # 趋势线：线性回归 (lm)，红色
        + geom_smooth(method='lm', color='#e74c3c', size=1, linetype='dashed', fill='#e74c3c', alpha=0.2)
        
        # Layer 3: 相关性文字标签
        # 注意: inherit_aes=False 很重要，避免 aes 冲突
        + geom_text(
            data=cor_df,
            mapping=aes(x='x', y='y', label='Label'),
            ha='left', va='top', # 左对齐，顶对齐
            size=10, 
            color='black',
            nudge_x=0.05, # 向右微调一点，不要紧贴坐标轴
            nudge_y=-0.05, # 向下微调一点
            inherit_aes=False
        )

        # Layer 4: 分面与主题
        + facet_wrap('Start codon', scales='fixed', nrow=1)
        + theme_bw()
        + theme(
            figure_size=(14, 4), # 调整图片长宽比
            strip_background=element_blank(),
            strip_text=element_text(size=12),
            axis_title=element_text(size=12)
        )
        + labs(
            x="Quantified Kozak Score",
            y="Initiation strength by predicted P-site density",
            fill="Count"
        )
    )
    
    # --- 4. 保存 ---
    plot_path = os.path.join(out_dir, f"kozak_score_scatter.{suffix}.pdf")
    # 使用较高的 dpi 保证文字清晰
    p.save(plot_path, width=14, height=4, dpi=300)
    print(f"Saved correlation scatter plot to {plot_path}")


def analyze_and_plot_initiation(meta_df, out_dir, suffix=""):
    """
    对提取的特征进行绘图和统计检验。
    """
    if meta_df.empty:
        print("DataFrame is empty, skipping plot and stats.")
        return

    os.makedirs(out_dir, exist_ok=True)
    
    df_plot = meta_df.copy()
    # Log 变换，加一个小值避免 log(0)
    df_plot['Log_Density'] = np.log2(df_plot['Normalized_Density'] + 1e-3)
    
    # --- A. 绘图 (Plotting) ---
    p = (
        ggplot(df_plot, aes(x='Start codon', y='Normalized_Density', color='Kozak class'))
        + geom_boxplot(
            fill='white',
            width=0.6, 
            outlier_alpha=0, 
            outlier_size=0,
            outlier_shape=None,
            size=1,
            position=position_dodge(width=0.7)
            )
        + theme_bw()
        + theme(
            axis_text_x=element_text(size=12),
            legend_position="top"
            )
        + labs(
            y="Initiation strength by predicted P-site density"
        )
        + scale_color_manual(values=["darkgray","#4292C6", "#2171B5", "#08306B"]) # 颜色区分 Kozak 强度
    )
    
    plot_filename = f"start_codon_kozak_boxplot.{suffix}.pdf" if suffix else "start_codon_kozak_boxplot.pdf"
    plot_path = os.path.join(out_dir, plot_filename)
    p.save(plot_path, width=6, height=5)
    print(f"Saved boxplot to {plot_path}")

    # --- B. 统计检验 (Statistics) ---
    stats_results = []
    
    # 1. Start Codon 大组间比较 (Overall comparison between Codons)
    codons = ['ATG', 'CTG', 'GTG', 'TTG']
    for c1, c2 in itertools.combinations(codons, 2):
        group1 = df_plot[df_plot['Start codon'] == c1]['Log_Density']
        group2 = df_plot[df_plot['Start codon'] == c2]['Log_Density']
        
        if len(group1) > 0 and len(group2) > 0:
            stat, pval = ranksums(group1, group2)
            stats_results.append({
                'Type': 'Between_Codons',
                'Group1': c1,
                'Group2': c2,
                'Statistic': stat,
                'P_Value': pval
            })

    # 2. 每种 Start Codon 内部，不同 Kozak Class 两两比较
    kozak_levels = ["Strong (+4G, -3R)", "Moderate (-3R)", "Moderate (+4G)", "Weak"]
    
    for codon in codons:
        subset = df_plot[df_plot['Start codon'] == codon]
        
        for k1, k2 in itertools.combinations(kozak_levels, 2):
            group1 = subset[subset['Kozak class'] == k1]['Log_Density']
            group2 = subset[subset['Kozak class'] == k2]['Log_Density']
            
            if len(group1) > 0 and len(group2) > 0:
                stat, pval = ranksums(group1, group2)
                stats_results.append({
                    'Type': f'Within_{codon}',
                    'Group1': k1,
                    'Group2': k2,
                    'Statistic': stat,
                    'P_Value': pval
                })

    # 保存统计结果
    stats_df = pd.DataFrame(stats_results)
    stats_filename = f"wilcox_stats_results.{suffix}.csv" if suffix else "wilcox_stats_results.csv"
    stats_path = os.path.join(out_dir, stats_filename)
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved statistics to {stats_path}")

# --- 4. Main Execution Function ---

def evaluate_start_codon_kozak_motif(pred_pkl, seq_pkl, out_dir="./results/initiation_eval", ratio_mask=1.0, suffix=""):
    """
    主入口函数。
    """
    print(f"Loading predictions: {pred_pkl}")
    with open(pred_pkl, 'rb') as f:
        preds = pickle.load(f)
        
    print(f"Loading sequences: {seq_pkl}")
    with open(seq_pkl, 'rb') as f:
        seqs = pickle.load(f)
        
    # Step 1: 提取特征
    df_features = extract_initiation_features(preds, seqs, ratio_mask=ratio_mask)
    
    if not df_features.empty:
        # Step 2: 保存原始特征表
        os.makedirs(out_dir, exist_ok=True)
        csv_filename = f"initiation_features_fullscan.{suffix}.csv" if suffix else "initiation_features_fullscan.csv"
        csv_path = os.path.join(out_dir, csv_filename)
        df_features.to_csv(csv_path, index=False)
        print(f"Features saved to {csv_path}")
        
        # Step 3: 分析和绘图
        analyze_and_plot_initiation(df_features, out_dir, suffix=suffix)
        plot_kozak_correlation_scatter(df_features, out_dir, suffix=suffix)