import os
import re
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def calculate_periodicity(signal_array, cds_start, cds_end):
    """
    计算给定区间的 0-frame 周期性比例: Frame_0_Sum / Total_Sum。
    如果无 CDS 边界信息 (cds_start == -1)，则计算全长序列上 3 个 frame 的周期性比例，并返回最大值。
    """
    if cds_start != -1 and cds_end != -1 and cds_end > cds_start:
        region_data = signal_array[cds_start:cds_end]
        total_sum = np.sum(region_data)
        
        if total_sum < 1e-6 or len(region_data) < 3:
            return np.nan
            
        frames = np.arange(len(region_data)) % 3
        f0_sum = np.sum(region_data[frames == 0])
        return f0_sum / total_sum
        
    else:
        region_data = signal_array
        total_sum = np.sum(region_data)
        
        if total_sum < 1e-6 or len(region_data) < 3:
            return np.nan
            
        frames = np.arange(len(region_data)) % 3
        f0_sum = np.sum(region_data[frames == 0])
        f1_sum = np.sum(region_data[frames == 1])
        f2_sum = np.sum(region_data[frames == 2])
        
        max_frame_sum = max(f0_sum, f1_sum, f2_sum)
        return max_frame_sum / total_sum


def plot_periodicity_scatter(df, out_dir="./results/periodicity", suffix=""):
    """
    绘制预测 vs 真实的三碱基周期性散点图。
    使用 Seaborn JointGrid 在上方和右侧附加不同 transcript_type 的密度分布图。
    清理了外围密度图的所有标度线和边框，并将散点图刻度间隔强制设为 0.2。
    """
    if df.empty: return
    os.makedirs(out_dir, exist_ok=True)
    
    unique_types = df['Gene_Type'].unique().tolist()
    
    # 提取所有具体的 ncRNA type (排除 Other 和 Housekeeping)
    nc_types = sorted([t for t in unique_types if t not in ["Other", "Housekeeping"]])
    
    # 强制设定画图图层顺序：Other (底) -> 各种 ncRNA (中) -> Housekeeping (顶)
    category_order = ["Other"] + nc_types + ["Housekeeping"]
    df['Gene_Type'] = pd.Categorical(df['Gene_Type'], categories=category_order, ordered=True)
    df = df.sort_values(by="Gene_Type")
    
    # 颜色分配引擎
    color_map = {
        "Other": "#B0B0B0",        # 浅灰色 (底层噪音)
        "Housekeeping": "#E74C3C"  # 亮红色 (顶层重点)
    }
    
    distinct_colors = ["#3498DB", "#2ECC71", "#9B59B6", "#F1C40F", "#E67E22", "#1ABC9C", "#34495E"]
    for i, nc_type in enumerate(nc_types):
        color_map[nc_type] = distinct_colors[i % len(distinct_colors)]
        
    print("Applied Color Mapping for scatter plot:")
    for k, v in color_map.items():
        print(f"  {k}: {v}")

    r_val, p_val = spearmanr(df['GT_Periodicity'], df['Pred_Periodicity'])
    p_text = f"{p_val:.1e}" if p_val < 0.001 else f"{p_val:.3f}"
    label_text = f"Spearman $R$ = {r_val:.3f}\n$P$ = {p_text}"
    
    print("\n>>> Generating Scatter Plot with Marginal Densities...")
    
    # 初始化 JointGrid
    g = sns.JointGrid(
        data=df,
        x='GT_Periodicity',     
        y='Pred_Periodicity',   
        hue='Gene_Type',        
        palette=color_map,
        height=6,               
        ratio=6,                
        space=0                 
    )
    
    # 中间主图：散点图
    g.plot_joint(sns.scatterplot, alpha=0.4, s=15, edgecolor="none")
    
    # 上下边缘图：核密度分布图
    g.plot_marginals(sns.kdeplot, fill=True, alpha=0.3, linewidth=1.2, common_norm=False)
    
    # ---------------------------------------------------------
    # 主图美化 (Main Plot Adjustments)
    # ---------------------------------------------------------
    
    # 主图背景网格线
    g.ax_joint.grid(True, color='lightgray', linestyle='-', linewidth=1.0, alpha=0.8)
    g.ax_joint.set_axisbelow(True) # 强制将网格线推至所有图形的最底层
    
    # 添加 zorder=10，强制 0.5 基准线悬浮于散点图最顶层
    g.ax_joint.axvline(x=0.5, color='#6b6b6b', linestyle='--', linewidth=1.5, alpha=1.0, zorder=10)
    g.ax_joint.axhline(y=0.5, color='#6b6b6b', linestyle='--', linewidth=1.5, alpha=1.0, zorder=10)
    
    # 设定坐标轴范围
    g.ax_joint.set_xlim(0.3, None)
    g.ax_joint.set_ylim(0.3, None)
    
    # =================================================================
    # [NEW] 强制设定主刻度间隔为 0.2，让网格线和 Ticks 变稀疏
    # =================================================================
    g.ax_joint.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    g.ax_joint.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    
    # 设置坐标轴标签
    g.ax_joint.set_xlabel("Observed periodicity", fontsize=14, color="black")
    g.ax_joint.set_ylabel("Predicted periodicity", fontsize=14, color="black")
    
    # 添加相关性文字标注
    g.ax_joint.text(
        0.95, 0.05, label_text, 
        transform=g.ax_joint.transAxes,
        fontsize=12, ha='right', va='bottom', color='black'
    )
    
    # 图例设置
    g.ax_joint.legend(
        title="Transcript Type", title_fontsize=14, fontsize=12,
        loc='upper left', bbox_to_anchor=(1.2, 1), frameon=False
    )
    
    # 主图向外刻度和粗黑边框
    g.ax_joint.tick_params(axis='both', which='major', labelsize=12, direction='out', length=6, width=1.5, colors='black')
    for spine in g.ax_joint.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    
    # 关闭顶部密度图 (marg_x) 的所有刻度和边框
    g.ax_marg_x.tick_params(axis='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    g.ax_marg_x.grid(False)
    for spine in g.ax_marg_x.spines.values():
        spine.set_visible(False)

    # 关闭右侧密度图 (marg_y) 的所有刻度和边框
    g.ax_marg_y.tick_params(axis='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    g.ax_marg_y.grid(False)
    for spine in g.ax_marg_y.spines.values():
        spine.set_visible(False)

    # 保存图表
    plot_path = os.path.join(out_dir, f"periodicity_scatter_density_{suffix}.pdf" if suffix else "periodicity_scatter_density.pdf")
    g.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved scatter plot with completely clean marginal densities to {plot_path}")


def evaluate_periodicity_correlation(
        dataset, 
        pkl_path: str,
        hk_genes_path: str ="/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/other_gene_list/housekeeping_genes.tsv",
        gtf_path: str = None, 
        out_dir="./results/periodicity",
        suffix=""):
    """
    评估单模型预测的三碱基周期性与真实值的相关性，并精确标注管家基因及各种 transcript_type 负对照。
    """
    os.makedirs(out_dir, exist_ok=True)
    print("--- Evaluating Tri-nucleotide Periodicity ---")

    # 1. 加载管家基因集合
    print(f"Loading housekeeping genes from {hk_genes_path}...")
    hk_df = pd.read_csv(hk_genes_path, sep='\t' if hk_genes_path.endswith('.tsv') else ',')
    hk_ensg_set = set(hk_df['Transcript ID'].dropna().astype(str).str.split('.').str[0])
    print(f"  -> Found {len(hk_ensg_set)} unique housekeeping genes.")

    # =================================================================
    # 2. 解析 GTF 建立 {Transcript_ID : Transcript_Type} 映射字典
    # =================================================================
    nc_tid_to_type = {}
    if gtf_path and os.path.exists(gtf_path):
        print(f"Parsing GTF file for specific transcript types from {gtf_path}...")
        # 目标负对照 biotype 集合
        # target_nc_biotypes = {'lncRNA', 'snoRNA', 'snRNA', 'tRNA', 'misc_RNA', 'rRNA', 'processed_transcript'}
        
        tid_re = re.compile(r'transcript_id "([^"]+)"')
        btype_re = re.compile(r'transcript_(?:bio)?type "([^"]+)"')
        
        with open(gtf_path, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                cols = line.split('\t')
                if len(cols) > 8 and cols[2] == 'transcript':
                    attr = cols[8]
                    btype_match = btype_re.search(attr)
                    if btype_match:
                        btype = btype_match.group(1)
                        if btype == 'lncRNA': #in target_nc_biotypes:
                            tid_match = tid_re.search(attr)
                            if tid_match:
                                clean_tid = tid_match.group(1).split('.')[0]
                                nc_tid_to_type[clean_tid] = btype 
                                
        print(f"  -> Extracted {len(nc_tid_to_type)} negative control transcripts across {len(set(nc_tid_to_type.values()))} specific biotypes.")
    else:
        print("  -> GTF path not provided or not found. Skipping detailed negative control labeling.")

    all_results = []

    # 3. 遍历 Dataset 提取 GT 周期性
    print(f"Extracting Ground Truth periodicity from Dataset...")
    gt_records = {}
    for i in tqdm(range(len(dataset))):
        uuid, species, cell_type, expr_vector, meta_info, seq_emb, count_emb = dataset[i]
        uuid_str = str(uuid)
        parts = uuid_str.split('-')
        if len(parts) < 2: continue
        
        tid = parts[0]
        clean_tid = tid.split('.')[0]
        cell_type = parts[1]
        
        cds_s = int(meta_info.get("cds_start_pos", -1))
        cds_e = int(meta_info.get("cds_end_pos", -1))
        
        if cds_s != -1 and cds_e != -1 and cds_e > cds_s:
            m_start = max(0, cds_s - 1)
            m_end = cds_e
        else:
            m_start = -1
            m_end = -1
        
        gt_signal = np.expm1(count_emb.numpy().flatten().astype(np.float32))
        gt_periodicity = calculate_periodicity(gt_signal, m_start, m_end)
        
        if not np.isnan(gt_periodicity):
            if clean_tid in hk_ensg_set:
                gene_type = "Housekeeping"
            elif clean_tid in nc_tid_to_type:
                gene_type = nc_tid_to_type[clean_tid] 
            else:
                gene_type = "Other"
                
            gt_records[uuid_str] = {
                'Tid': tid,
                'Tid_clean': clean_tid,
                'Cell_Type': cell_type,
                'GT_Periodicity': gt_periodicity,
                'Gene_Type': gene_type, 
                'CDS_Start': m_start,
                'CDS_End': m_end
            }

    # 4. 提取单模型预测周期性
    print(f"\nProcessing {pkl_path}")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        preds_dict = pickle.load(f)

    for uuid_str, record in tqdm(gt_records.items(), desc="Matching Predictions"):
        cell_type = record['Cell_Type']
        tid = record['Tid']
        clean_tid = record['Tid_clean']
        
        if cell_type not in preds_dict: continue
        
        cell_preds = preds_dict[cell_type]
        lookup_tid = clean_tid
        if lookup_tid not in cell_preds:
            if tid in cell_preds: 
                lookup_tid = tid
            else:
                continue
            
        pred_raw = cell_preds[lookup_tid]
        pred_signal = np.expm1(pred_raw.reshape(-1).astype(np.float32))
        
        if record['CDS_End'] != -1 and len(pred_signal) < record['CDS_End']:
            continue
            
        pred_periodicity = calculate_periodicity(pred_signal, record['CDS_Start'], record['CDS_End'])
        
        if not np.isnan(pred_periodicity):
            all_results.append({
                'Tid': tid,
                'Cell_Type': cell_type,
                'GT_Periodicity': record['GT_Periodicity'],
                'Pred_Periodicity': pred_periodicity,
                'Gene_Type': record['Gene_Type'] 
            })

    df_final = pd.DataFrame(all_results)
    save_path = os.path.join(out_dir, f"periodicity_eval_results_{suffix}.csv" if suffix else "periodicity_eval_results.csv")
    df_final.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")

    # 调用 [MODIFIED] 的画图函数
    plot_periodicity_scatter(df_final, out_dir, suffix)
    return df_final