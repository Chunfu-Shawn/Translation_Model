import pickle
import numpy as np
import pandas as pd
import os
import RNA  # ViennaRNA Python interface
from tqdm import tqdm
from multiprocessing import Pool
from plotnine import *
from scipy.stats import spearmanr, pearsonr
from eval.calculate_te import calculate_morf_efficiency, calculate_morf_mean_efficiency, calculate_morf_mean_density

# 1. 定义单个计算函数 (必须是顶层函数)
def process_single_seq(seq_str):
    """
    输入序列字符串，返回 (Length, MFE)
    """
    if not seq_str or 'N' in seq_str:
        return len(seq_str), None
    
    # RNA.fold 返回 (structure, mfe)
    # 我们只需要 mfe
    try:
        _, mfe = RNA.fold(seq_str)
        return len(seq_str), mfe
    except:
        return len(seq_str), None

# 2. 修改主逻辑
def extract_mfe_te_parallel(preds, seqs, out_dir, suffix, ratio_mask=1.0, region='5utr', num_workers=50):
    """
    并行计算 MFE，支持断点续传/缓存读取。
    TE 每次都会重新计算。
    """
    # --- Step A: 尝试加载缓存 ---
    cache_file = os.path.join(out_dir, f"mfe_data_{region}.{suffix}.csv")
    mfe_cache = {}
    
    if os.path.exists(cache_file):
        print(f"Found existing MFE cache: {cache_file}")
        try:
            df_cache = pd.read_csv(cache_file)
            # 检查必要列是否存在
            if all(col in df_cache.columns for col in ['UUID', 'MFE', 'Length']):
                print(f"Loading {len(df_cache)} entries from cache...")
                # 将缓存转换为字典以便快速查找: UUID -> row dict
                # orient='index' 会以索引为key，所以我们手动构建
                for _, row in df_cache.iterrows():
                    mfe_cache[row['UUID']] = {
                        'Length': row['Length'],
                        'MFE': row['MFE'],
                        # 兼容旧文件可能没有 Density 的情况
                        'MFE_Density': row.get('MFE_Density', row['MFE']/row['Length'] if row['Length']>0 else 0)
                    }
            else:
                print("Cache file format incorrect, ignoring.")
        except Exception as e:
            print(f"Error reading cache: {e}, will recalculate MFE.")

    # --- Step B: 准备任务 ---
    tasks = []       # 需要新计算 MFE 的序列
    new_meta = []    # 新任务的元数据
    
    final_results = [] # 存储最终结果 (Cached + New)

    print(f"Preparing data for {region}...")
    
    for uuid, sample in tqdm(preds.items(), desc="Matching & TE Calc"):
        # ... 基础校验 ...
        tid = uuid.split("-")[0]
        if tid not in seqs: continue
        if ratio_mask not in sample['ratios']: continue
        
        cds_info = sample.get('cds_info', None)
        if cds_info is None or cds_info['end'] == -1: continue
        
        m_start = cds_info['start'] - 1 
        m_end = cds_info['end']
        
        # 1. 无论是否有缓存，先计算 TE (因为它很快，且可能受 ratio_mask 影响)
        try:
            pred_arr = np.expm1(sample['ratios'][ratio_mask]['pred'].reshape(-1))
            # 简单的长度对齐
            seq_len_total = len(seqs[tid])
            if len(pred_arr) != seq_len_total: pred_arr = pred_arr[:seq_len_total]
            
            te = calculate_morf_mean_density(pred_arr, m_start, m_end)
            if te < 1e-6: continue
        except: continue

        # 2. 检查 MFE 缓存
        if uuid in mfe_cache:
            # [Hit] 命中缓存：直接使用 MFE 数据 + 新算的 TE
            cached_item = mfe_cache[uuid]
            final_results.append({
                'UUID': uuid,
                'Length': cached_item['Length'],
                'MFE': cached_item['MFE'],
                'MFE_Density': cached_item['MFE_Density'],
                'TE': te # 使用本次运行计算的 TE
            })
        else:
            # [Miss] 未命中：需要提取序列并加入计算队列
            seq_str = seqs[tid].upper()
            if region == '5utr':
                if m_start < 10: continue
                target_seq = seq_str[:m_start]
            else: # full
                target_seq = seq_str
            
            # 加入任务列表
            tasks.append(target_seq)
            new_meta.append({
                'UUID': uuid,
                'TE': te
            })

    # --- Step C: 对缺失数据进行并行计算 ---
    if tasks:
        print(f"Calculating MFE for {len(tasks)} new/missing sequences using {num_workers} cores...")
        with Pool(num_workers) as p:
            fold_results = list(tqdm(p.imap(process_single_seq, tasks), total=len(tasks)))
        
        # 合并新结果
        for meta, (length, mfe) in zip(new_meta, fold_results):
            if mfe is not None:
                final_results.append({
                    'UUID': meta['UUID'],
                    'Length': length,
                    'MFE': mfe,
                    'MFE_Density': mfe / length if length > 0 else 0,
                    'TE': meta['TE']
                })
    else:
        print("All sequences found in cache! Skipping MFE calculation.")

    return pd.DataFrame(final_results)

# --- 3. Plotting (保持不变) ---
def plot_mfe_vs_te(df, out_dir, suffix="", x_axis='MFE_Density'):
    if df.empty:
        print("No data to plot.")
        return
        
    upper_te = df['TE'].quantile(0.99)
    plot_df = df[df['TE'] <= upper_te].copy()
    
    r_spearman, p_s = spearmanr(plot_df[x_axis], plot_df['TE'])
    r_pearson, p_p = pearsonr(plot_df[x_axis], plot_df['TE'])
    
    stats_label = (f"Spearman R = {r_spearman:.3f} (P={p_s:.2e})\n"
                   f"Pearson R = {r_pearson:.3f} (P={p_p:.2e})")
    
    if x_axis == 'MFE':
        x_lab = "Minimum Free Energy (kcal/mol)"
        title_lab = "Absolute MFE"
    else:
        x_lab = "MFE Density (kcal/mol per nt)"
        title_lab = "Normalized MFE (Stability Density)"
        
    p = (
        ggplot(plot_df, aes(x=x_axis, y='TE'))
        + geom_point(alpha=0.2, size=2, stroke=0, color="#2C3E50") # stroke=0 去除描边
        + geom_smooth(method='lm', color="#E74C3C", fill="#E74C3C", alpha=0.2)
        + annotate("text", x=plot_df[x_axis].min(), y=plot_df['TE'].max() * 0.95, 
                   label=stats_label, ha='left', va='top', size=10)
        + theme_bw()
        + theme(
            axis_text=element_text(size=12),
            axis_title=element_text(size=13)
        )
        + labs(
            x=x_lab,
            y="CDS translation efficiency"
        )
    )
    
    # 注意：这里根据你的描述，绘图保存的文件名和数据保存的文件名是分开的
    # 数据文件: mfe_data_{region}.{suffix}.csv
    # 图片文件: te_{x_axis}.{suffix}.pdf
    plot_save_path = os.path.join(out_dir, f"te_{x_axis}_cor.{suffix}.pdf")
    p.save(plot_save_path, width=5, height=5, dpi=300)
    print(f"Saved plot to {plot_save_path}")

# --- 4. Main Execution ---
def evaluate_rna_structure_correlation(
        pred_pkl, seq_pkl, out_dir="./results/structure_eval", ratio_mask=1.0, suffix="", region='5utr'):
    
    print(f"Loading data for Structure Analysis...")
    with open(pred_pkl, 'rb') as f: preds = pickle.load(f)
    with open(seq_pkl, 'rb') as f: seqs = pickle.load(f)
    
    os.makedirs(out_dir, exist_ok=True)

    # 1. Extract (传入 out_dir 和 suffix 用于查找缓存)
    df_data = extract_mfe_te_parallel(preds, seqs, out_dir, suffix, ratio_mask=ratio_mask, region=region)
    
    if df_data.empty:
        print("No valid data extracted.")
        return

    # 保存/更新缓存数据
    # 下次运行时，这里保存的文件就会被上面的 Step A 读取
    csv_path = os.path.join(out_dir, f"mfe_data_{region}.{suffix}.csv")
    df_data.to_csv(csv_path, index=False)
    print(f"Updated MFE cache saved to {csv_path}")
    
    # 2. Plot
    print("Plotting MFE Density vs TE...")
    plot_mfe_vs_te(df_data, out_dir, suffix=f"{suffix}_{region}", x_axis='MFE_Density')
    
    print("Plotting Absolute MFE vs TE...")
    plot_mfe_vs_te(df_data, out_dir, suffix=f"{suffix}_{region}", x_axis='MFE')