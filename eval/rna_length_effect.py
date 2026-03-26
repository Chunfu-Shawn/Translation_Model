import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from plotnine import *
from scipy.stats import spearmanr, pearsonr
from eval.calculate_te import *


# --- Feature Extraction ---
def extract_length_vs_te_data(preds, seqs, ratio_mask=1.0):
    """
    提取各部分序列长度 (5'UTR, CDS, 3'UTR, Total) 和 TE。
    """
    results = []
    print(f"Extracting Sequence Lengths vs TE (ratio_mask={ratio_mask})...")
    
    for uuid, sample in tqdm(preds.items()):
        # 1. 基础校验
        tid = uuid.split("-")[0]
        if tid not in seqs: continue
        if ratio_mask not in sample['ratios']: continue
        
        cds_info = sample.get('cds_info', None)
        if cds_info is None or cds_info['end'] == -1: continue
        
        m_start = cds_info['start'] - 1 
        m_end = cds_info['end']
        
        seq_str = seqs[tid] # 不需要 upper，只算长度
        seq_len = len(seq_str)
        
        # 2. 计算各部分长度
        len_5utr = m_start
        len_cds = m_end - m_start
        len_3utr = seq_len - m_end
        
        # 简单的过滤：如果长度异常 (如负数)，跳过
        if len_5utr < 0 or len_cds < 0 or len_3utr < 0: continue

        # 3. 计算 TE
        try:
            pred_arr = np.expm1(sample['ratios'][ratio_mask]['pred'].reshape(-1))
            if len(pred_arr) != seq_len: pred_arr = pred_arr[:seq_len]
            te = calculate_morf_mean_density(pred_arr, m_start, m_end)
            if te < 1e-6: continue
        except: continue
        
        results.append({
            'UUID': uuid,
            'Len_5UTR': len_5utr,
            'Len_CDS': len_cds,
            'Len_3UTR': len_3utr,
            'Len_Total': seq_len,
            'TE': te
        })
        
    return pd.DataFrame(results)

# --- 3. Plotting ---

def plot_length_correlation_faceted(df, out_dir, suffix=""):
    """
    绘制长度 vs TE 的散点图 (4个分面：5'UTR, CDS, 3'UTR, Total)。
    不需要输入 target_region，一次性画出所有。
    """
    if df.empty:
        print("No data to plot.")
        return
        
    # --- 1. 数据预处理 ---
    
    # 1.1 全局过滤 TE 极值 (Top 1%) 防止压缩 Y 轴
    upper_te = df['TE'].quantile(0.99)
    clean_df = df[df['TE'] <= upper_te].copy()
    
    # 1.2 数据重塑 (Wide to Long)
    # 我们需要的列：UUID, TE, 以及四个长度列
    id_vars = ['UUID', 'TE']
    value_vars = ['Len_5UTR', 'Len_CDS', 'Len_3UTR', 'Len_Total']
    
    # 转换
    plot_df = clean_df.melt(id_vars=id_vars, value_vars=value_vars, 
                            var_name='Region_Raw', value_name='Length')
    
    # 1.3 过滤掉长度 <= 0 的点 (防止 Log 轴报错)
    plot_df = plot_df[plot_df['Length'] > 0].copy()
    
    # 1.4 映射可读的标签并设定顺序
    label_map = {
        'Len_5UTR': "5' UTR length",
        'Len_CDS': "CDS length",
        'Len_3UTR': "3' UTR length",
        'Len_Total': "Transcript length"
    }
    plot_df['Region'] = plot_df['Region_Raw'].map(label_map)
    
    # 设定 Region 的 Categorical 顺序，保证画图顺序符合逻辑 (5'->CDS->3'->Total)
    region_order = ["5' UTR length", "CDS length", "3' UTR length", "Transcript length"]
    plot_df['Region'] = pd.Categorical(plot_df['Region'], categories=region_order, ordered=True)
    
    # --- 2. 计算每个分面的相关性 ---
    
    cor_stats = []
    
    # 遍历每个区域单独计算
    for region in region_order:
        sub_df = plot_df[plot_df['Region'] == region]
        if len(sub_df) < 10: continue # 样本太少不计算
        
        # 计算相关性
        r_spearman, p_s = spearmanr(sub_df['Length'], sub_df['TE'])
        r_pearson, p_p = pearsonr(sub_df['Length'], sub_df['TE'])
        
        # 格式化文本
        p_s_text = f"{p_s:.1e}" if p_s < 0.001 else f"{p_s:.3f}"
        p_p_text = f"{p_p:.1e}" if p_p < 0.001 else f"{p_p:.3f}"
        
        label_text = (f"Spearman R = {r_spearman:.3f} (P={p_s_text})\n"
                      f"Pearson R = {r_pearson:.3f} (P={p_p_text})")
        
        # 确定标签位置 (左上角)
        # 注意：由于是 log 坐标，取最小值时要小心不要取到 0
        x_pos = sub_df['Length'].min()
        y_pos = sub_df['TE'].max() * 0.95
        
        cor_stats.append({
            'Region': region,
            'Label': label_text,
            'x': x_pos,
            'y': y_pos
        })
        
    cor_df = pd.DataFrame(cor_stats)
    # 确保 cor_df 的 Region 也是 Categorical，否则分面可能会乱
    cor_df['Region'] = pd.Categorical(cor_df['Region'], categories=region_order, ordered=True)

    # --- 3. 绘图 ---
    p = (
        ggplot(plot_df, aes(x='Length', y='TE'))
        # 散点：透明度低一点，去描边
        + geom_point(alpha=0.1, size=2, stroke=0, color="#2E86C1")
        # 趋势线
        + geom_smooth(method='lm', color="#E74C3C", fill="#E74C3C", alpha=0.2)
        + facet_wrap('~Region', scales='free_x', ncol=2)
        
        # 统计标注 (使用独立的数据框 cor_df)
        + geom_text(data=cor_df, mapping=aes(x='x', y='y', label='Label'),
                    ha='left', va='top', size=10, inherit_aes=False)
        
        # 坐标轴设置
        + scale_x_log10()
        + theme_bw()
        + theme(
            figure_size=(10, 8), # 调整画布大小以容纳4个图
            axis_text=element_text(size=10),
            axis_title=element_text(size=12),
            strip_text=element_text(size=12),
            strip_background=element_blank(),
            panel_grid_minor=element_blank()
        )
        + labs(
            x="Length (nt, log10 scale)",
            y="CDS translation efficiency"
        )
    )
    
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"corr_length_faceted.{suffix}.pdf")
    p.save(save_path, width=7, height=7, dpi=300)
    print(f"Saved faceted plot to {save_path}")

def plot_length_correlation(df, out_dir, x_col='Len_5UTR', suffix=""):
    """
    绘制长度 vs TE 的散点图。
    x_col: 可以是 'Len_5UTR', 'Len_3UTR', 'Len_CDS', 'Len_Total'
    """
    if df.empty:
        print("No data to plot.")
        return
        
    # 1. 数据清洗
    # 过滤 TE 极值 (Top 1%) 防止压缩 Y 轴
    upper_te = df['TE'].quantile(0.99)
    # 过滤长度为 0 的点 (防止 Log 轴报错)
    plot_df = df[(df['TE'] <= upper_te) & (df[x_col] > 0)].copy()
    
    # 2. 计算相关性
    # 长度分布通常跨度很大，且与 TE 的关系往往是非线性的 (e.g. log-linear)
    # Spearman (秩相关) 对此更稳健
    r_spearman, p_s = spearmanr(plot_df[x_col], plot_df['TE'])
    r_pearson, p_p = pearsonr(plot_df[x_col], plot_df['TE'])
    
    p_s_text = f"{p_s:.2e}" if p_s < 0.001 else f"{p_s:.3f}"
    p_p_text = f"{p_s:.2e}" if p_s < 0.001 else f"{p_s:.3f}"
    stats_label = (f"Spearman R = {r_spearman:.3f} (P={p_s_text})\n"
                   f"Pearson R = {r_pearson:.3f} (P={p_p_text})")
    
    # 设置显示标签
    label_map = {
        'Len_5UTR': "5' UTR length",
        'Len_3UTR': "3' UTR length",
        'Len_CDS': "CDS length",
        'Len_Total': "Transcript length"
    }
    x_label = label_map.get(x_col, x_col)
    
    # 3. 绘图
    p = (
        ggplot(plot_df, aes(x=x_col, y='TE'))
        # 散点：透明度低一点以显示密度
        + geom_point(alpha=0.2, size=2, stroke=0, color="#2E86C1")
        # 趋势线
        + geom_smooth(method='lm', color="#E74C3C", fill="#E74C3C", alpha=0.2)
        # 统计标注
        + annotate("text", x=plot_df[x_col].min(), y=plot_df['TE'].max()*0.95, 
                   label=stats_label, ha='left', va='top', size=10)
        + scale_x_log10()
        + theme_bw()
        + theme(
            figure_size=(8, 6),
            axis_text=element_text(size=12),
            axis_title=element_text(size=12)
        )
        + labs(
            x=f"{x_label} (nt, log10 scale)",
            y="CDS translation efficiency"
        )
    )
    
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"corr_{x_col}_vs_te.{suffix}.pdf")
    p.save(save_path, width=5, height=5, dpi=300)
    print(f"Saved plot to {save_path}")

# --- 4. Main Execution ---

def evaluate_length_correlation(pred_pkl, seq_pkl, out_dir="./results/len_eval", ratio_mask=1.0, suffix=""):
    """
    主入口函数。
    Args:
        target_region: 选择要分析的区域，可选:
                       'Len_5UTR', 'Len_3UTR', 'Len_CDS', 'Len_Total'
    """
    print(f"Loading data for Length Analysis...")
    with open(pred_pkl, 'rb') as f: preds = pickle.load(f)
    with open(seq_pkl, 'rb') as f: seqs = pickle.load(f)
    
    # 1. Extract
    df_data = extract_length_vs_te_data(preds, seqs, ratio_mask=ratio_mask)
    
    if df_data.empty:
        print("No valid data extracted.")
        return

    # 保存数据
    os.makedirs(out_dir, exist_ok=True)
    df_data.to_csv(os.path.join(out_dir, f"length_data.{suffix}.csv"), index=False)
    
    # 2. Plot
    plot_length_correlation_faceted(df_data, out_dir, suffix=suffix)