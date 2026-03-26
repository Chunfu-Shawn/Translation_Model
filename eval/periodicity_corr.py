import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from plotnine import *
from scipy.stats import spearmanr, pearsonr

def calculate_periodicity(signal_array, cds_start, cds_end):
    """
    计算给定区间的 0-frame 周期性比例: Frame_0_Sum / Total_Sum
    """
    if cds_start == -1 or cds_end == -1 or cds_end <= cds_start:
        return np.nan
        
    region_data = signal_array[cds_start:cds_end]
    total_sum = np.sum(region_data)
    
    # 过滤掉总信号极低或长度不足的序列
    if total_sum < 1e-6 or len(region_data) < 3:
        return np.nan
        
    # 计算当前区间的相对 frame 
    frames = np.arange(len(region_data)) % 3
    f0_sum = np.sum(region_data[frames == 0])
    
    return f0_sum / total_sum

def plot_periodicity_scatter(df, out_dir="./results/periodicity", suffix=""):
    """
    绘制预测 vs 真实的三碱基周期性散点图，管家基因用红色标注
    """
    if df.empty: return
    os.makedirs(out_dir, exist_ok=True)
    
    # 保证画图顺序，让红点 (管家基因) 画在最上层
    df = df.sort_values(by="Is_Housekeeping")
    
    # 获取各个模型的相关性指标，用于显示在图上
    stats_data = []
    r_val, p_val = spearmanr(df['Pred_Periodicity'], df['GT_Periodicity'])
    p_text = f"{p_val:.1e}" if p_val < 0.001 else f"{p_val:.3f}"
    label = f"Spearman R = {r_val:.3f}, P = {p_text}"
    stats_data.append({'Label': label})
    df_stats = pd.DataFrame(stats_data)
    
    color_map = {
        "No": "#B0B0B0",    # 浅灰色代表普通基因
        "Yes": "#E74C3C"    # 亮红色代表管家基因
    }
    
    p = (
        ggplot(df, aes(x='Pred_Periodicity', y='GT_Periodicity', color='Is_Housekeeping'))
        + geom_point(alpha=0.6, size=2, stroke=0)
        # + geom_abline(intercept=0, slope=1, color="black", linetype="dashed", size=1)
        # + geom_smooth(method="lm", color="#2C6B9A", se=False, size=1)
        + geom_text(
            data=df_stats, 
            mapping=aes(x=0.1, y=0.9, label='Label'), 
            inherit_aes=False, size=11, ha='left', color="black"
        )
        + scale_color_manual(values=color_map)
        + labs(
            x="Predicted tri-nucleotide periodicity",
            y="Ground truth periodicity",
            color="Housekeeping gene"
        )
        + theme_bw()
        + theme(
            legend_position='top',
            strip_text=element_text(fontweight='bold', size=11)
        )
        + coord_cartesian(xlim=[0, 1], ylim=[0, 1])
    )
    
    plot_path = os.path.join(out_dir, f"periodicity_scatter_{suffix}.pdf" if suffix else "periodicity_scatter.pdf")

    p.save(plot_path, width=4, height=4)
    print(f"Saved scatter plot to {plot_path}")

def evaluate_periodicity_correlation(
        dataset, 
        pkl_path: str,
        hk_genes_path: str ="/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/other_gene_list/housekeeping_genes.tsv",
        out_dir="./results/periodicity",
        suffix=""):
    """
    评估单模型预测的三碱基周期性与真实值的相关性，并标注管家基因。
    
    Args:
        dataset: TranslationDataset 实例 (含 GT 信号和 CDS 边界)。
        pkl_path: 模型的预测结果 pkl 文件路径。
        hk_genes_path: 管家基因参考表路径 (需包含 ensg_id)。
    """
    os.makedirs(out_dir, exist_ok=True)
    print("--- Evaluating Tri-nucleotide Periodicity ---")

    # 1. 加载管家基因集合
    print(f"Loading housekeeping genes from {hk_genes_path}...")
    hk_df = pd.read_csv(hk_genes_path, sep='\t' if hk_genes_path.endswith('.tsv') else ',')
    # 假设列名为 'ensg_id'，提取并去除可能的版本号
    hk_ensg_set = set(hk_df['Transcript ID'].dropna().astype(str).str.split('.').str[0])
    print(f"  -> Found {len(hk_ensg_set)} unique housekeeping genes.")

    all_results = []

    # 3. 遍历 Dataset 提取 GT 周期性
    print(f"Extracting Ground Truth periodicity from Dataset...")
    gt_records = {}
    for i in tqdm(range(len(dataset))):
        uuid, cell_type_idx, meta_info, seq_emb, count_emb = dataset[i]
        uuid_str = str(uuid)
        parts = uuid_str.split('-')
        if len(parts) < 2: continue
        
        tid = parts[0]
        clean_tid = tid.split('.')[0]
        cell_type = parts[1]
        
        cds_s = int(meta_info.get("cds_start_pos", -1))
        cds_e = int(meta_info.get("cds_end_pos", -1))
        
        if cds_s == -1 or cds_e == -1: continue
            
        m_start = max(0, cds_s - 1)
        m_end = cds_e
        
        gt_signal = np.expm1(count_emb.numpy().flatten().astype(np.float32))
        gt_periodicity = calculate_periodicity(gt_signal, m_start, m_end)

        # if gt_periodicity < 0.3:
        #     print(seq_emb[cds_s:cds_s+3,:], gt_periodicity)
        
        if not np.isnan(gt_periodicity):
            # tid = tid2gene.get(clean_tid, "Unknown")
            is_hk = "Yes" if clean_tid in hk_ensg_set else "No"
            gt_records[uuid_str] = {
                'Tid': tid,
                'Tid_clean': clean_tid,
                'Cell_Type': cell_type,
                'GT_Periodicity': gt_periodicity,
                'Is_Housekeeping': is_hk,
                'CDS_Start': m_start,
                'CDS_End': m_end
            }

    # 4. 提取单模型预测周期性
    print(f"\nProcessing {pkl_path}")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        preds_dict = pickle.load(f)

    # 根据 preds_dict 的格式 {cell_type: {tid: pred}} 解析
    for uuid_str, record in tqdm(gt_records.items(), desc="Matching Predictions"):
        cell_type = record['Cell_Type']
        tid = record['Tid']
        clean_tid = record['Tid_clean']
        
        if cell_type not in preds_dict: continue
        
        cell_preds = preds_dict[cell_type]
        # 回退匹配逻辑
        lookup_tid = clean_tid
        if lookup_tid not in cell_preds:
            if tid in cell_preds: # 防止 clean_tid 找不到，尝试用原始 tid
                lookup_tid = tid
            else:
                continue
            
        pred_raw = cell_preds[lookup_tid]
        pred_signal = np.expm1(pred_raw.reshape(-1).astype(np.float32))
        
        # 对齐长度并计算周期性
        if len(pred_signal) < record['CDS_End']:
            continue
            
        pred_periodicity = calculate_periodicity(pred_signal, record['CDS_Start'], record['CDS_End'])
        
        if not np.isnan(pred_periodicity):
            all_results.append({
                'Tid': tid,
                'Cell_Type': cell_type,
                'GT_Periodicity': record['GT_Periodicity'],
                'Pred_Periodicity': pred_periodicity,
                'Is_Housekeeping': record['Is_Housekeeping']
            })

    df_final = pd.DataFrame(all_results)
    save_path = os.path.join(out_dir, f"periodicity_eval_results_{suffix}.csv" if suffix else "periodicity_eval_results.csv")
    df_final.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")

    # plot
    plot_periodicity_scatter(df_final, out_dir, suffix)
    return df_final