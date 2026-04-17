import os
import re
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from eval.calculate_te import *
from scipy.stats import gaussian_kde, spearmanr, pearsonr
from plotnine import *
import seaborn as sns
import matplotlib.pyplot as plt


# =================================================================
# 核心坐标转换工具
# =================================================================
def map_genomic_to_transcript(genomic_pos, strand, sorted_exons):
    """
    将基因组坐标映射为转录本序列上的 0-based 坐标。
    sorted_exons: 已经按照转录方向排好序的 exon 列表 (start, end)
    """
    t_len = 0
    for start, end in sorted_exons:
        if start <= genomic_pos <= end:
            if strand == '+':
                return t_len + (genomic_pos - start)
            else:
                return t_len + (end - genomic_pos)
        t_len += (end - start + 1)
    return None

def parse_gtf_for_transcript_cds(gtf_path):
    """
    解析 GTF 文件，提取外显子和 CDS 信息，
    计算并返回每个转录本的 CDS 在完整 mRNA 序列上的 0-based start 和 end。
    """
    print(f"Parsing GTF file to map genomic coordinates: {gtf_path}...")
    transcript_info = {}
    
    with open(gtf_path, 'r') as f:
        for line in f:
            if line.startswith('#'): 
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9: 
                continue
            
            feature = parts[2]
            if feature not in ['exon', 'CDS']: 
                continue
            
            strand = parts[6]
            start, end = int(parts[3]), int(parts[4])
            
            # 提取 transcript_id
            attr_str = parts[8]
            tid_match = re.search(r'transcript_id "([^"]+)"', attr_str)
            if not tid_match: 
                continue
            
            # 去除版本号以保证最广泛的兼容性
            tid = tid_match.group(1).split('.')[0]
            
            if tid not in transcript_info:
                transcript_info[tid] = {'strand': strand, 'exons': [], 'cds_genomic': []}
            
            if feature == 'exon':
                transcript_info[tid]['exons'].append((start, end))
            elif feature == 'CDS':
                transcript_info[tid]['cds_genomic'].append((start, end))

    cds_mapping = {}
    print("Mapping genomic CDS bounds to transcript space...")
    for tid, info in transcript_info.items():
        if not info['cds_genomic']: 
            continue  # 跳过没有 CDS 的非编码转录本
        
        strand = info['strand']
        exons = info['exons']
        
        # 按照转录方向对外显子进行排序
        if strand == '+':
            exons.sort(key=lambda x: x[0])
        else:
            exons.sort(key=lambda x: x[1], reverse=True)
            
        cds_starts = [x[0] for x in info['cds_genomic']]
        cds_ends = [x[1] for x in info['cds_genomic']]
        
        genomic_min = min(cds_starts)
        genomic_max = max(cds_ends)
        
        # 确定转录本上最先翻译和最后翻译的基因组绝对坐标
        if strand == '+':
            first_base = genomic_min
            last_base = genomic_max
        else:
            first_base = genomic_max
            last_base = genomic_min
            
        t_start = map_genomic_to_transcript(first_base, strand, exons)
        t_end_inclusive = map_genomic_to_transcript(last_base, strand, exons)
        
        if t_start is not None and t_end_inclusive is not None:
            # 返回标准的 Python 切片边界 [start:end] (end 为 exclusive)
            cds_mapping[tid] = (t_start, t_end_inclusive + 1)
            
    print(f"Successfully mapped CDS coordinates for {len(cds_mapping)} transcripts.")
    return cds_mapping


# =================================================================
# TE 指标评估引擎
# =================================================================
class TranslationEfficiencyAnalyzerGTF:
    def __init__(self, gtf_path, preds_pkl_path, unlog_data=True):
        """
        Args:
            gtf_path: 小鼠/人类的标准 GTF 文件路径。
            preds_pkl_path: 预测结果的 pkl 文件路径 ({cell_type: {tid: prediction}})。
            unlog_data: 是否对数据进行 np.expm1 还原。
        """
        self.unlog_data = unlog_data
        
        # 1. 预解析 GTF 文件
        if not os.path.exists(gtf_path):
            raise FileNotFoundError(f"GTF file not found: {gtf_path}")
        self.cds_mapping = parse_gtf_for_transcript_cds(gtf_path)
        
        # 2. 读取预测结果
        if not os.path.exists(preds_pkl_path):
            raise FileNotFoundError(f"PKL file not found: {preds_pkl_path}")
            
        print(f"Loading predictions from {preds_pkl_path}...")
        with open(preds_pkl_path, 'rb') as f:
            self.preds_dict = pickle.load(f)
            
        # 展平预测任务以支持进度条
        self.tasks = []
        for cell_type, preds in self.preds_dict.items():
            for tid, count_emb in preds.items():
                self.tasks.append((cell_type, tid, count_emb))
        print(f"Loaded {len(self.tasks)} total prediction records from PKL.")

    def run(self, out_dir="./results", suffix=""):
        results = []
        os.makedirs(out_dir, exist_ok=True)
        
        matched_count = 0
        print(f"Start processing and calculating TE metrics...")
        
        for cell_type, transcript_id, count_emb in tqdm(self.tasks):
            try:
                # 去除版本号进行匹配
                tid_clean = str(transcript_id).split('.')[0]
                
                # 如果转录本在 GTF 中找不到对应的 CDS 映射，则跳过
                if tid_clean not in self.cds_mapping:
                    continue
                    
                cds_start, cds_end = self.cds_mapping[tid_clean]
                uuid = f"{tid_clean}-{cell_type}-Prediction"
                
                # 由于没有实验数据的 dataset，真实 TE 设定为 NaN 占位
                te_val = np.nan 
                
                # 执行核心计算
                self._process_and_append(results, uuid, tid_clean, cell_type, count_emb, cds_start, cds_end, te_val)
                matched_count += 1
                
            except Exception as e:
                print(f"Error processing TID {transcript_id}: {e}")
                continue

        # ==========================================
        # 保存与返回
        # ==========================================
        df = pd.DataFrame(results)
        
        if df.empty:
            print("Warning: No valid records were matched between PKL and GTF. Returning empty DataFrame.")
            return df
            
        print(f"\nSuccessfully matched and processed {matched_count} transcripts.")
        print("Head of Result DataFrame:")
        print(df.head())
        
        file_suffix = f".{suffix}" if suffix else ""
        save_path = os.path.join(out_dir, f"translation_efficiency_metrics{file_suffix}.csv")
        df.to_csv(save_path, index=False)
        print(f"Metrics saved to: {save_path}")
        
        return df

    def _process_and_append(self, results_list, uuid, transcript_id, cell_type, count_emb, cds_start, cds_end, te_val):
        """
        内部核心逻辑：处理 Density 数组并计算翻译指标
        """
        # 转为 Numpy
        if isinstance(count_emb, torch.Tensor):
            density = count_emb.detach().cpu().numpy()
        else:
            density = count_emb

        # 还原 Log
        if self.unlog_data:
            density = np.expm1(density.astype(np.float32))
        
        # 维度压缩兼容性处理 (L, 10) -> (L,) 或 (L, 1) -> (L,)
        if len(density.shape) > 1 and density.shape[1] > 1:
            density_profile = np.sum(density, axis=1)
        elif len(density.shape) > 1 and density.shape[1] == 1:
            density_profile = density.flatten()
        else:
            density_profile = density

        # 防护：如果序列长度小于计算出的 CDS_end，说明 FASTA 序列可能不完整，进行截断
        cds_end = min(cds_end, len(density_profile))
        
        if cds_end <= cds_start or np.sum(density_profile[cds_start: cds_end]) == 0:
            return None

        # --- 计算指标 ---
        te_morf_ratio = calculate_morf_signal_ratio(density_profile, cds_start, cds_end)
        te_morf_mean_ratio = calculate_morf_mean_signal_ratio(density_profile, cds_start, cds_end)
        te_morf_mean_signal = calculate_morf_mean_signal(density_profile, cds_start, cds_end)
        te_total_sum = calculate_sum_signal(density_profile, cds_start, cds_end)
        te_total_mean = calculate_mean_signal(density_profile)

        # --- 记录结果 ---
        results_list.append({
            'UUID': uuid,
            'Tid': transcript_id,
            'Cell_Type': cell_type,
            'TE': te_val,
            'mORF_Sum_Ratio': te_morf_ratio,
            'mORF_Mean_Ratio': te_morf_mean_ratio,
            'mORF_Mean_Density': te_morf_mean_signal,
            'mORF_Ribo_Load': te_total_sum,
            'Global_Mean_Density': te_total_mean,
            'Transcript_Length': len(density_profile)
        })


# =================================================================
# 1. 核心评估函数 (动态适配任意 Group)
# =================================================================
def evaluate_silac_processed_data(pred_csv_path: str, silac_csv_path: str, out_dir: str = "./results"):
    """
    计算模型预测指标与经过预处理的长表 SILAC translation rate 之间的相关性。
    动态识别 SILAC 表中的 Group 列，完美兼容 Dataset 1 和 Dataset 2。
    """
    print(f"Loading predictions from: {pred_csv_path}")
    pred_df = pd.read_csv(pred_csv_path)
    
    print(f"Loading processed SILAC data from: {silac_csv_path}")
    silac_df = pd.read_csv(silac_csv_path)
    
    # 确保 SILAC 列名存在
    target_s_metric = "Translation rate"
    if target_s_metric not in silac_df.columns or 'Group' not in silac_df.columns:
        raise ValueError(f"SILAC data must contain '{target_s_metric}' and 'Group' columns.")
        
    pred_metrics = [
        "mORF_Sum_Ratio", "mORF_Mean_Ratio", 
        "mORF_Mean_Density", "mORF_Ribo_Load", "Global_Mean_Density"
    ]
    
    # 1. 数据合并 (Inner Join)
    print("Merging predictions with SILAC ground truth...")
    pred_df['Tid_clean'] = pred_df['Tid'].astype(str).str.split('.').str[0]
    silac_df['ENSEMBL ID'] = silac_df['ENSEMBL ID'].astype(str).str.strip()
    
    # 如果 SILAC 表里同一个 ENSEMBL ID 有多行相同的 Group (比如同一个基因映射到多个转录本)，先取均值
    silac_grouped = silac_df.groupby(['ENSEMBL ID', 'Group'], as_index=False)[target_s_metric].mean()
    
    merged_df = pd.merge(pred_df, silac_grouped, left_on='Tid_clean', right_on='ENSEMBL ID', how='inner')
    
    if merged_df.empty:
        raise ValueError("No matching Tids found between predictions and SILAC data!")
    print(f"Successfully matched transcripts.")

    # 2. 动态遍历 Group 计算相关性
    results = []
    groups = merged_df['Group'].unique()
    
    for p_metric in pred_metrics:
        if p_metric not in merged_df.columns:
            continue
            
        for grp in groups:
            # 提取当前 Group 的数据
            group_df = merged_df[merged_df['Group'] == grp].copy()
            valid_df = group_df.dropna(subset=[p_metric, target_s_metric])
            
            # 过滤极端值 (上下 1%)
            if len(valid_df) > 10:
                p01_p = valid_df[p_metric].quantile(0.01)
                p99_p = valid_df[p_metric].quantile(0.99)
                p01_s = valid_df[target_s_metric].quantile(0.01)
                p99_s = valid_df[target_s_metric].quantile(0.99)
                
                valid_df = valid_df[
                    (valid_df[p_metric] >= p01_p) & (valid_df[p_metric] <= p99_p) &
                    (valid_df[target_s_metric] >= p01_s) & (valid_df[target_s_metric] <= p99_s)
                ]
            
            if len(valid_df) < 5:
                continue
                
            x = valid_df[p_metric].values
            y = valid_df[target_s_metric].values
            
            sp_r, sp_pval = spearmanr(x, y)
            pe_r, pe_pval = pearsonr(x, y)
            
            results.append({
                'Predicted_Metric': p_metric,
                'SILAC_Group': str(grp).capitalize(), 
                'Spearman_R': sp_r,
                'Spearman_P': sp_pval,
                'Pearson_R': pe_r,
                'Pearson_P': pe_pval,
                'Matched_N': len(valid_df)
            })
            
    corr_df = pd.DataFrame(results)
    
    os.makedirs(out_dir, exist_ok=True)
    dataset_name = os.path.basename(silac_csv_path).replace('.csv', '')
    csv_path = os.path.join(out_dir, f"correlation_{dataset_name}.csv")
    corr_df.to_csv(csv_path, index=False)
    
    print(f"\nCorrelation results saved to {csv_path}")
    print(corr_df[['Predicted_Metric', 'SILAC_Group', 'Spearman_R', 'Pearson_R', 'Matched_N']])
    
    return corr_df, merged_df


# =================================================================
# 2. 热图绘制函数 
# =================================================================
def plot_cross_species_heatmap(corr_df, out_dir="./results", method="Spearman", prefix="dataset"):
    """绘制热图，将不同 Group 作为列展示（自动适配动态和多类别样本）"""
    if corr_df.empty:
        print("Warning: Input DataFrame is empty. Skipping heatmap generation.")
        return
        
    val_col = f"{method}_R"
    # 透视表：行是评估指标，列是各个不同的样本 Group
    pivot_df = corr_df.pivot(index="Predicted_Metric", columns="SILAC_Group", values=val_col)
    
    # ---------------------------------------------------------
    # 智能排序逻辑：常规样本按字母排序，汇总类样本放最右边
    # ---------------------------------------------------------
    all_cols = list(pivot_df.columns)
    summary_keywords = ['Mean', 'Average', 'Median', 'All']
    
    # 拆分并分别排序
    normal_cols = sorted([c for c in all_cols if c not in summary_keywords])
    summary_cols = sorted([c for c in all_cols if c in summary_keywords])
    ordered_cols = normal_cols + summary_cols
    
    pivot_df = pivot_df[ordered_cols]
    
    # ---------------------------------------------------------
    # 动态画布缩放与画图
    # ---------------------------------------------------------
    # 宽度：基础宽度6，每多一列增加 1.2 英寸
    # 高度：基础高度4，每多一行增加 0.8 英寸 (防止行也变多时挤压)
    fig_width = max(6, len(ordered_cols) * 1.2)
    fig_height = max(4, len(pivot_df.index) * 0.8)
    
    plt.figure(figsize=(fig_width, fig_height))
    cmap = sns.color_palette("vlag", as_cmap=True)
    
    sns.heatmap(
        pivot_df, 
        annot=True, 
        fmt=".3f", 
        cmap=cmap, 
        center=0,
        cbar_kws={'label': f'{method} Correlation (R)'},
        linewidths=1, 
        linecolor='white'
    )
    
    plt.title(f"SILAC Translation Rate Correlation\n({prefix})", pad=15, fontsize=14)
    plt.ylabel("Predicted TE Metrics", fontsize=12)
    plt.xlabel("SILAC Data Group", fontsize=12)
    
    # 标签旋转45度，且强制右对齐(ha='right')，防止多列时文字互相重叠
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    # ---------------------------------------------------------
    # 保存输出
    # ---------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"heatmap_{prefix}_{method}.pdf")
    # bbox_inches='tight' 确保旋转的超长 X 轴标签不会被裁掉
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap successfully saved to {save_path}")

# =================================================================
# 3. 密度散点图函数
# =================================================================

def plot_density_scatter(df, x_col, y_col, out_dir="./results", suffix="", log_x=True, log_y=False, corr_method="Spearman"):
    """
    使用 Plotnine 绘制密度散点图 (已修复坐标轴压缩、KDE报错和极值过滤逻辑)
    
    :param corr_method: 字符串, "Spearman" 或 "Pearson", 决定计算哪种相关性系数。
    """
    print(f"\nGenerating density scatter plot for {x_col} vs {y_col} (Method: {corr_method})...")
    df_clean = df.dropna(subset=[x_col, y_col]).copy()
    print(f" -> Initial points after dropna: {len(df_clean)}")
    
    # 0. 对数模式下的安全过滤
    if log_x: 
        df_clean = df_clean[df_clean[x_col] > 0]
        print(f" -> Points after filtering {x_col} > 0 (for log scale): {len(df_clean)}")
    if log_y: 
        df_clean = df_clean[df_clean[y_col] > 0]
        print(f" -> Points after filtering {y_col} > 0 (for log scale): {len(df_clean)}")
        
    if len(df_clean) < 10: 
        print(" -> Warning: Not enough points to plot. Aborting.")
        return

    # 1. 极值过滤！防止单个离群值把整个坐标系撑爆
    p01_x, p99_x = df_clean[x_col].quantile(0.01), df_clean[x_col].quantile(0.99)
    p01_y, p99_y = df_clean[y_col].quantile(0.01), df_clean[y_col].quantile(0.99)
    
    # 安全判定：防止某些数据高度集中导致 p01 == p99 把数据删空
    if p01_x < p99_x:
        df_clean = df_clean[(df_clean[x_col] >= p01_x) & (df_clean[x_col] <= p99_x)]
    if p01_y < p99_y:
        df_clean = df_clean[(df_clean[y_col] >= p01_y) & (df_clean[y_col] <= p99_y)]
        
    print(f" -> Points after 1%-99% outlier trimming: {len(df_clean)}")

    x, y = df_clean[x_col].values, df_clean[y_col].values
    
    # =================================================================
    # [MODIFIED] 2. 动态计算相关性系数 (Spearman vs Pearson)
    # =================================================================
    if corr_method.lower() == "pearson":
        r, p = pearsonr(x, y)
        r_label = "Pearson R"
    elif corr_method.lower() == "spearman":
        r, p = spearmanr(x, y)
        r_label = "Spearman R"
    else:
        raise ValueError("corr_method must be either 'Spearman' or 'Pearson'")

    p_str = "P-value < 1e-100" if p < 1e-100 else f"P-value = {p:.1e}"
    # 动态生成图表上的文字标签
    textstr = f"{r_label} = {r:.2f}\n{p_str}\nN = {len(x)}"

    # 3. 在剔除极值之后再计算 KDE 密度，确保对比度和颜色正确
    x_kde = np.log10(x) if log_x else x
    y_kde = np.log10(y) if log_y else y
    xy = np.vstack([x_kde, y_kde])
    
    try:
        z = gaussian_kde(xy)(xy)
    except Exception as e:
        print(f" -> KDE calculation failed ({e}), falling back to uniform density.")
        z = np.ones_like(x) # 防止点太少或完全共线导致奇异矩阵报错
        
    df_clean['Density'] = z
    df_clean = df_clean.sort_values(by='Density', ascending=True)

    # 动态计算文本框位置 (防爆轴)
    text_x = 10**(np.log10(x.min()) + 0.05 * (np.log10(x.max()) - np.log10(x.min()))) if log_x else x.min() + 0.05 * (x.max() - x.min())
    text_y = 10**(np.log10(y.min()) + 0.95 * (np.log10(y.max()) - np.log10(y.min()))) if log_y else y.min() + 0.95 * (y.max() - y.min())

    plot = (
        ggplot(df_clean, aes(x=x_col, y=y_col, color='Density'))
        + geom_point(size=2, alpha=0.7, stroke=0)
        + scale_color_cmap(cmap_name="magma")
        + annotate("text", x=text_x, y=text_y, label=textstr, ha='left', va='top', size=13, color='black')
        + labs(x=f"{x_col} (Log Scale)" if log_x else x_col, y=f"{y_col} (Log Scale)" if log_y else y_col)
        + theme_classic()
        + theme(legend_position='none')
    )
    if log_x: plot += scale_x_log10()
    if log_y: plot += scale_y_log10()

    os.makedirs(out_dir, exist_ok=True)
    # [MODIFIED] 将计算方法加入文件名中以防覆盖
    save_path = os.path.join(out_dir, f"scatter_{corr_method}_{suffix}.pdf")
    plot.save(save_path, width=6, height=6, dpi=300, verbose=False)
    print(f" -> Scatter plot successfully saved to: {save_path}\n")