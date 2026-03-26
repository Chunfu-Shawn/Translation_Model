import pickle
import numpy as np
import pandas as pd
import os
import RNA  # ViennaRNA Python interface
from tqdm import tqdm
from multiprocessing import Pool
from plotnine import *
from scipy.stats import spearmanr, pearsonr
from typing import Union, Dict

# 假设这里有你之前定义的 calculate_te 相关模块
from eval.calculate_te import *

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
def extract_mfe_te_parallel(
        truth_dataset, all_predictions, seqs, out_dir, suffix, region='5utr', num_workers=50):
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
                for _, row in df_cache.iterrows():
                    mfe_cache[row['UUID']] = {
                        'Length': row['Length'],
                        'MFE': row['MFE'],
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
    
    for i in tqdm(range(len(truth_dataset)), desc="Matching & TE Calc"):
        # 1. 解析 Dataset 获取 Meta 信息
        uuid, cell_type_idx, meta_info, seq_emb, count_emb = truth_dataset[i]
        uuid_str = str(uuid)
        
        parts = uuid_str.split('-')
        if len(parts) < 2: continue
            
        tid = parts[0]
        cell_type = parts[1]
        
        # 2. 检查序列和预测是否存在
        if tid not in seqs: continue
        if cell_type not in all_predictions: continue
        
        predictions = all_predictions[cell_type]
        lookup_tid = tid
        if lookup_tid not in predictions:
            tid_no_version = tid.split('.')[0]
            if tid_no_version in predictions:
                lookup_tid = tid_no_version
            else:
                continue
                
        # 获取预测数据并做反对数转换 (如果你的预测是线性值请去掉 np.expm1)
        pred_signal = predictions[lookup_tid]
        pred_arr = np.expm1(pred_signal.reshape(-1).astype(np.float32))

        # 3. 解析 CDS 边界
        cds_s = int(meta_info.get("cds_start_pos", -1))
        cds_e = int(meta_info.get("cds_end_pos", -1))
        
        if cds_s == -1 or cds_e == -1: 
            continue
            
        m_start = max(0, cds_s - 1) # 依据原逻辑转为 0-based
        m_end = cds_e
        
        # 简单的长度对齐
        seq_len_total = len(seqs[tid])
        if len(pred_arr) != seq_len_total: 
            pred_arr = pred_arr[:seq_len_total]
            
        # 计算 TE
        try:
            te = calculate_morf_mean_signal(pred_arr, m_start, m_end)
            if te < 1e-6: continue
        except: 
            continue

        # 4. 检查 MFE 缓存
        if uuid_str in mfe_cache:
            # [Hit] 命中缓存：直接使用 MFE 数据 + 新算的 TE
            cached_item = mfe_cache[uuid_str]
            final_results.append({
                'UUID': uuid_str,
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
                'UUID': uuid_str,
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
        print("All valid sequences found in cache! Skipping MFE calculation.")

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
    
    plot_save_path = os.path.join(out_dir, f"te_{x_axis}_cor.{suffix}.pdf")
    p.save(plot_save_path, width=5, height=5, dpi=300, verbose=False)
    print(f"Saved plot to {plot_save_path}")

# --- 4. Main Execution ---
def evaluate_rna_structure_correlation(
        truth_dataset, 
        pkl_input: Union[Dict[str, str], str], 
        seq_pkl: str, 
        out_dir="./results/structure_eval", 
        suffix="", 
        region='5utr',
        num_workers=50):
    
    # 1. 灵活加载预测文件
    print(">>> Loading prediction files...")
    all_predictions = {}
    if isinstance(pkl_input, str):
        with open(pkl_input, 'rb') as f:
            loaded_data = pickle.load(f)
            if isinstance(loaded_data, dict):
                all_predictions = loaded_data
            else:
                raise ValueError("Single pickle file does not contain a dictionary.")
    elif isinstance(pkl_input, dict):
        for cell_type, pkl_path in pkl_input.items():
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                all_predictions[cell_type] = data.get(cell_type, data)
    
    # 2. 加载序列字典
    print(f">>> Loading sequences from {seq_pkl}...")
    with open(seq_pkl, 'rb') as f: 
        seqs = pickle.load(f)
    
    os.makedirs(out_dir, exist_ok=True)

    # 3. 提取特征
    df_data = extract_mfe_te_parallel(
        truth_dataset, all_predictions, seqs, 
        out_dir=out_dir, suffix=suffix, 
        region=region, num_workers=num_workers
    )
    
    if df_data.empty:
        print("No valid data extracted.")
        return

    # 保存/更新缓存数据
    csv_path = os.path.join(out_dir, f"mfe_data_{region}.{suffix}.csv")
    df_data.to_csv(csv_path, index=False)
    print(f"Updated MFE cache saved to {csv_path}")
    
    # 4. 绘图
    print("Plotting MFE Density vs TE...")
    plot_mfe_vs_te(df_data, out_dir, suffix=f"{suffix}_{region}", x_axis='MFE_Density')
    
    print("Plotting Absolute MFE vs TE...")
    plot_mfe_vs_te(df_data, out_dir, suffix=f"{suffix}_{region}", x_axis='MFE')