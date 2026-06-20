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
from eval.calculate_te import *

# 1. 定义单个计算函数 (必须是顶层函数)
def process_single_seq(seq_str):
    """
    输入序列字符串，返回 (Length, MFE)
    """
    if not seq_str or 'N' in seq_str:
        return len(seq_str), None
    
    # RNA.fold 返回 (structure, mfe)
    try:
        _, mfe = RNA.fold(seq_str)
        return len(seq_str), mfe
    except:
        return len(seq_str), None

# 2. 修改主逻辑
# =================================================================
# [MODIFIED] Added target_cell_type parameter
# =================================================================
def extract_mfe_te_parallel(
        tx_cds_dict, all_predictions, seqs, out_dir, suffix, region='5utr', num_workers=50, target_cell_type=None):
    """
    并行计算 MFE，支持断点续传/缓存读取。
    不依赖 Dataset，直接遍历 predictions，利用 tx_cds_dict 划定边界。
    """
    # --- Step A: 尝试加载缓存 ---
    cache_file = os.path.join(out_dir, f"mfe_data_{region}.{suffix}.csv")
    mfe_cache = {}
    
    if os.path.exists(cache_file):
        print(f"Found existing MFE cache: {cache_file}")
        try:
            df_cache = pd.read_csv(cache_file)
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
    tasks = []       
    new_meta = []    
    final_results = [] 

    print(f"Preparing data for {region}...")
    if target_cell_type:
        print(f"Filtering for target cell type: {target_cell_type}")
    
    # =================================================================
    # 核心遍历逻辑：双层遍历 all_predictions
    # =================================================================
    for cell_type, tid_dict in all_predictions.items():
        # =================================================================
        # [NEW] Skip irrelevant cell types
        # =================================================================
        if target_cell_type is not None and cell_type != target_cell_type:
            continue
            
        print(f"  -> Processing cell type: {cell_type}")
        
        for tid, pred_signal in tqdm(tid_dict.items(), leave=False):
            # 1. Safely clean Tid
            clean_tid = str(tid).split('.')[0] if str(tid).startswith('ENST') else str(tid).split('|')[0]
            
            # UUID 定义为 Tid-Cell_type
            uuid_str = f"{tid}-{cell_type}"
            
            seq_key = tid if tid in seqs else clean_tid
            if seq_key not in seqs: 
                continue
                
            # 2. 从传入的 CDS 字典获取边界
            meta = tx_cds_dict.get(clean_tid, tx_cds_dict.get(tid))
            if not meta: 
                continue
                
            cds_s = int(meta.get("cds_start_pos", -1)) if isinstance(meta, dict) else getattr(meta, "cds_start_pos", -1)
            cds_e = int(meta.get("cds_end_pos", -1)) if isinstance(meta, dict) else getattr(meta, "cds_end_pos", -1)
            
            if cds_s == -1 or cds_e == -1: 
                continue
                
            m_start = max(0, cds_s - 1) 
            m_end = cds_e
            
            seq_str = seqs[seq_key].upper()
            seq_len_total = len(seq_str)
            
            # 3. 解析预测信号
            if hasattr(pred_signal, 'cpu'):
                pred_signal = pred_signal.cpu().numpy()
            pred_arr = np.expm1(np.array(pred_signal).reshape(-1).astype(np.float32))
            
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
                cached_item = mfe_cache[uuid_str]
                final_results.append({
                    'UUID': uuid_str,
                    'Length': cached_item['Length'],
                    'MFE': cached_item['MFE'],
                    'MFE_Density': cached_item['MFE_Density'],
                    'TE': te 
                })
            else:
                if region == '5utr':
                    if m_start < 10: continue
                    target_seq = seq_str[:m_start]
                else: 
                    target_seq = seq_str
                
                tasks.append(target_seq)
                new_meta.append({
                    'UUID': uuid_str,
                    'TE': te
                })

    # --- Step C: 并行计算 ---
    if tasks:
        print(f"Calculating MFE for {len(tasks)} new/missing sequences using {num_workers} cores...")
        with Pool(num_workers) as p:
            fold_results = list(tqdm(p.imap(process_single_seq, tasks), total=len(tasks)))
        
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

# --- 3. Plotting (Scatter Plot 保持不变) ---
def plot_mfe_vs_te(df, out_dir, suffix="", x_axis='MFE_Density'):
    if df.empty: return
        
    upper_te = df['TE'].quantile(0.99)
    plot_df = df[df['TE'] <= upper_te].copy()
    
    r_spearman, p_s = spearmanr(plot_df[x_axis], plot_df['TE'])
    r_pearson, p_p = pearsonr(plot_df[x_axis], plot_df['TE'])
    
    stats_label = (f"Spearman R = {r_spearman:.3f} (P={p_s:.2e})\n"
                   f"Pearson R = {r_pearson:.3f} (P={p_p:.2e})")
    
    if x_axis == 'MFE':
        x_lab = "Minimum Free Energy (kcal/mol)"
    else:
        x_lab = "MFE Density (kcal/mol per nt)"
        
    p = (
        ggplot(plot_df, aes(x=x_axis, y='TE'))
        + geom_point(alpha=0.2, size=2, stroke=0, color="#2C3E50") 
        + geom_smooth(method='lm', color="#E74C3C", fill="#E74C3C", alpha=0.2)
        + annotate("text", x=plot_df[x_axis].min(), y=plot_df['TE'].max() * 0.95, 
                   label=stats_label, ha='left', va='top', size=10)
        + theme_classic()
        + theme(
            axis_text=element_text(size=12),
            axis_title=element_text(size=13)
        )
        + labs(
            x=x_lab,
            y="Mean CDS translation signal"
        )
    )
    
    plot_save_path = os.path.join(out_dir, f"te_{x_axis}_cor.{suffix}.pdf")
    p.save(plot_save_path, width=5, height=5, dpi=300, verbose=False)


def plot_mfe_binned_boxplot(df, out_dir, suffix="", x_axis='MFE_Density', bins=10):
    
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    if df.empty: return
    
    plot_df = df.copy()
    
    # 防止极端离群点拉伸 Y 轴
    upper_te = plot_df['TE'].quantile(0.99)
    plot_df = plot_df[plot_df['TE'] <= upper_te].copy()
    
    # 1. 计算全局 Spearman 相关性 (用于左上角标注)
    r_spearman, p_s = spearmanr(plot_df[x_axis], plot_df['TE'])
    p_text = f"{p_s:.2e}" if p_s < 0.001 else f"{p_s:.3f}"
    stats_label = f"Spearman R = {r_spearman:.3f}\nP = {p_text}"
    
    # 2. 动态优雅分箱处理
    if isinstance(bins, int):
        # 通过 retbins=True 获取实际生成的箱数
        _, bin_edges = pd.qcut(plot_df[x_axis], q=bins, duplicates='drop', retbins=True)
        actual_bins = len(bin_edges) - 1
        step = 100 / actual_bins
        
        # 动态生成百分比标签：100%-90%, 90%-80%, ..., 10%-0%
        percentile_labels = [f"{int(100 - i*step)}%-{int(100 - (i+1)*step)}%" for i in range(actual_bins)]
        
        # 使用 qcut 并直接打上百分比标签
        plot_df['Bin_Label'] = pd.qcut(plot_df[x_axis], q=bins, labels=percentile_labels, duplicates='drop')
        label_order = percentile_labels
    else:
        plot_df['MFE_Bin'] = pd.cut(plot_df[x_axis], bins=bins)
        plot_df = plot_df.dropna(subset=['MFE_Bin'])
        
        def format_bin(b):
            if x_axis == 'MFE':
                return f"{int(b.left)} to {int(b.right)}"
            else:
                return f"{b.left:.2f} to {b.right:.2f}"
                
        plot_df['Bin_Label'] = plot_df['MFE_Bin'].apply(format_bin)
        bin_order = sorted(plot_df['MFE_Bin'].unique())
        label_order = [format_bin(b) for b in bin_order]
        
    plot_df = plot_df.dropna(subset=['Bin_Label'])
    
    # 获取有序的类别顺序
    plot_df['Bin_Label'] = pd.Categorical(plot_df['Bin_Label'], categories=label_order, ordered=True)
    
    # 色带逻辑完美契合
    cmap = cm.get_cmap('Blues')
    color_vals = np.linspace(0.3, 0.95, len(label_order))
    hex_colors = [mcolors.to_hex(cmap(c)) for c in color_vals]
    
    if x_axis == 'MFE':
        x_lab = "Minimum Free Energy percentiles" if isinstance(bins, int) else "Minimum Free Energy (kcal/mol)"
    else:
        x_lab = "MFE Density percentiles" if isinstance(bins, int) else "MFE Density (kcal/mol per nt)"
        
    # 构造标注锚点 (左上角)
    anno_df = pd.DataFrame({'x': [1.2], 'y': [plot_df['TE'].max() * 0.95], 'label': [stats_label]})

    # 3. 绘图
    p = (
        ggplot(plot_df, aes(x='Bin_Label', y='TE'))
        + geom_boxplot(
            aes(color='Bin_Label'), 
            fill='white',               
            alpha=1.0,
            width=0.7,
            size=0.8,                   
            outlier_shape=None,         
            outlier_alpha=0,
            outlier_size=0,
            position=position_dodge(width=0.6)
        )
        
        + geom_smooth(
            aes(group=1), 
            method='lm', 
            color='#FF4500', 
            linetype='dashed', 
            size=1.5, 
            se=False,
            alpha=0.8
        )
        
        + geom_text(
            data=anno_df,
            mapping=aes(x='x', y='y', label='label'),
            inherit_aes=False,
            ha='left', va='top', 
            size=11, 
            color='black'
        )
        + theme_classic()
        + theme(
            axis_text_x=element_text(rotation=30, hjust=1, size=11, color='black'),
            axis_text_y=element_text(size=11, color='black'),
            axis_title=element_text(size=12, color='black'),
            legend_position="none" 
        )
        + labs(
            x=x_lab,
            y="Mean CDS translation signal"
        )
        + scale_color_manual(values=hex_colors)
        + coord_cartesian(ylim=(0, plot_df['TE'].max()))
    )
    
    plot_save_path = os.path.join(out_dir, f"te_{x_axis}_binned_boxplot.{suffix}.pdf")
    p.save(plot_save_path, width=4, height=5, dpi=300, verbose=False)
    print(f"Saved elegant binned boxplot to {plot_save_path}")


# --- 4. Main Execution ---
# =================================================================
# [MODIFIED] Added target_cell_type parameter to main execution function
# =================================================================
def evaluate_rna_structure_correlation(
        pkl_input: Union[Dict[str, str], str], 
        seq_pkl: str, 
        tx_cds_file: str, 
        out_dir="./results/structure_eval", 
        suffix="", 
        region='5utr',
        num_workers=50,
        mfe_bins: Union[int, list] = 10,
        target_cell_type: str = None):
    
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
                
    # 从 pkl 中加载 tx_cds_dict (即 meta_dict)
    print(f">>> Loading transcript metadata from {tx_cds_file}...")
    with open(tx_cds_file, 'rb') as f:
        tx_cds_dict = pickle.load(f)
    
    print(f">>> Loading sequences from {seq_pkl}...")
    with open(seq_pkl, 'rb') as f: 
        seqs = pickle.load(f)
    
    os.makedirs(out_dir, exist_ok=True)

    # =================================================================
    # [MODIFIED] Pass target_cell_type to extraction function
    # =================================================================
    df_data = extract_mfe_te_parallel(
        tx_cds_dict, all_predictions, seqs, 
        out_dir=out_dir, suffix=suffix, 
        region=region, num_workers=num_workers,
        target_cell_type=target_cell_type
    )
    
    if df_data.empty:
        print("No valid data extracted.")
        return

    csv_path = os.path.join(out_dir, f"mfe_data_{region}.{suffix}.csv")
    df_data.to_csv(csv_path, index=False)
    print(f"Updated MFE cache saved to {csv_path}")
    
    print("\n>>> Plotting MFE Density vs TE...")
    plot_mfe_vs_te(df_data, out_dir, suffix=f"{suffix}_{region}", x_axis='MFE_Density')
    plot_mfe_binned_boxplot(df_data, out_dir, suffix=f"{suffix}_{region}", x_axis='MFE_Density', bins=mfe_bins)
    
    print("\n>>> Plotting Absolute MFE vs TE...")
    plot_mfe_vs_te(df_data, out_dir, suffix=f"{suffix}_{region}", x_axis='MFE')
    plot_mfe_binned_boxplot(df_data, out_dir, suffix=f"{suffix}_{region}", x_axis='MFE', bins=mfe_bins)