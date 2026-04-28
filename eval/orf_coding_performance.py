import os
import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from plotnine import *
import seaborn as sns
import matplotlib.pyplot as plt

# =====================================================================
# Module 1: Data Loading and Preprocessing
# =====================================================================
def load_and_filter_data(
        pred_csv_path: str, 
        gt_csv_path: str, 
        target_transcript_ids: Optional[List[str]] = None,
        # =================================================================
        # [MODIFIED] 引入上下界参数，替代原本单一的 ORF_length_limit
        # =================================================================
        min_orf_len: Optional[int] = None,
        max_orf_len: Optional[int] = None):
    """Read, clean, and filter data based on target transcripts and exact length ranges."""
    print(f"Loading Ground Truth from: {gt_csv_path}")
    try:
        gt_df = pd.read_csv(gt_csv_path, sep='\t')
        if 'PacBio_ID' not in gt_df.columns:
            gt_df = pd.read_csv(gt_csv_path, sep=',')
    except Exception as e:
        raise ValueError(f"Error reading GT: {e}")
        
    gt_df['Tid_clean'] = gt_df['PacBio_ID'].astype(str)
    gt_df['start_gt'] = gt_df['CDS_Start_0based']
    gt_df['stop_gt'] = gt_df['CDS_End_0based']
    gt_df['length'] = gt_df['stop_gt'] - gt_df['start_gt']
    
    print(f"Loading Predictions from: {pred_csv_path}")
    pred_df = pd.read_csv(pred_csv_path)
    
    if 'score' not in pred_df.columns:
        raise ValueError("Validation Error: The predictions file MUST contain a 'score' column for evaluation.")

    pred_df['Tid_clean'] = pred_df['Tid'].astype(str)
    if 'length' not in pred_df.columns:
        pred_df['length'] = pred_df['stop'] - pred_df['start']

    # 1. Filter by transcript IDs
    if target_transcript_ids is not None:
        print(f"Filtering datasets to {len(target_transcript_ids)} target transcripts...")
        target_set = set(str(t) for t in target_transcript_ids)
        gt_df = gt_df[gt_df['Tid_clean'].isin(target_set)].copy()
        pred_df = pred_df[pred_df['Tid_clean'].isin(target_set)].copy()
        
    # =================================================================
    # [NEW] 鲁棒且优雅的双向长度过滤逻辑 (适用于小肽评估)
    # =================================================================
    if min_orf_len is not None or max_orf_len is not None:
        # 参数校验
        if min_orf_len is not None and max_orf_len is not None and min_orf_len > max_orf_len:
            raise ValueError(f"Invalid length range: min_orf_len ({min_orf_len}) cannot be greater than max_orf_len ({max_orf_len}).")
            
        lower_bound = min_orf_len if min_orf_len is not None else 0
        upper_bound = max_orf_len if max_orf_len is not None else float('inf')
        
        print(f"Filtering ORFs by length range: {lower_bound} <= Length <= {upper_bound} nt...")
        
        gt_orig_len = len(gt_df)
        pred_orig_len = len(pred_df)
        
        gt_df = gt_df[(gt_df['length'] >= lower_bound) & (gt_df['length'] <= upper_bound)].copy()
        pred_df = pred_df[(pred_df['length'] >= lower_bound) & (pred_df['length'] <= upper_bound)].copy()
        
        print(f"  -> Ground Truth ORFs retained : {len(gt_df)} / {gt_orig_len}")
        print(f"  -> Predicted ORFs retained    : {len(pred_df)} / {pred_orig_len}")

    if len(gt_df) == 0:
        raise ValueError("No Ground Truth data left after filtering! Please check your Transcript IDs or Length conditions.")

    gt_df = gt_df.reset_index(drop=True)
    gt_df['gt_idx'] = gt_df.index
    
    pred_df = pred_df.sort_values('score', ascending=False).reset_index(drop=True)
    pred_df['pred_idx'] = pred_df.index
    
    return pred_df, gt_df

# =====================================================================
# Module 2: NMS Matching and Unified Evaluation Table Construction
# =====================================================================
def match_and_build_eval_df(pred_df: pd.DataFrame, gt_df: pd.DataFrame, eval_metrics: List[str], overlap_threshold: float) -> pd.DataFrame:
    """Execute ultra-fast NMS matching and build a unified evaluation dataframe containing y_true, length, and scores."""
    print(f"\nMemory-Safe Matching (Frame Consistent & Overlap > {overlap_threshold*100}%)...")
    
    # Build Ground Truth dictionary for O(1) lookups
    gt_dict = {}
    for row in gt_df.itertuples(index=False):
        if row.Tid_clean not in gt_dict:
            gt_dict[row.Tid_clean] = []
        gt_dict[row.Tid_clean].append((row.gt_idx, row.start_gt, row.stop_gt))
        
    pred_to_gt = {} # Map successful predictions to their GT indices
    matched_gt_indices = set()
    
    # Matching logic
    for row in pred_df.itertuples(index=False):
        tid = row.Tid_clean
        if tid not in gt_dict: continue
            
        p_start, p_stop, p_idx, p_len = row.start, row.stop, row.pred_idx, row.length
        
        for g_idx, g_start, g_stop in gt_dict[tid]:
            if g_idx in matched_gt_indices: continue 
            if p_start % 3 != g_start % 3: continue
                
            overlap_s = max(p_start, g_start)
            overlap_e = min(p_stop, g_stop)
            overlap_l = max(0, overlap_e - overlap_s)
            
            if overlap_l > 0:
                g_len = g_stop - g_start
                if (overlap_l / (p_len + g_len - overlap_l)) >= overlap_threshold:
                    pred_to_gt[p_idx] = g_idx
                    matched_gt_indices.add(g_idx)
                    break 

    print("Assembling Unified Evaluation DataFrame...")
    eval_records = []
    gt_lengths = dict(zip(gt_df['gt_idx'], gt_df['length']))
    
    # 1. Assemble Predictions (TP and FP)
    for row in pred_df.itertuples(index=False):
        is_tp = row.pred_idx in pred_to_gt
        # Core logic: If TP, use actual GT length; if FP, use predicted length
        eval_len = gt_lengths[pred_to_gt[row.pred_idx]] if is_tp else row.length
        
        record = {'y_true': 1 if is_tp else 0, 'length': eval_len}
        for m in eval_metrics:
            val = getattr(row, m, 0.0) if hasattr(row, m) else 0.0 
            record[m] = float(val)
        eval_records.append(record)
        
    # 2. Assemble False Negatives (FN)
    for row in gt_df.itertuples(index=False):
        if row.gt_idx not in matched_gt_indices:
            record = {'y_true': 1, 'length': row.length}
            for m in eval_metrics: record[m] = -1.0 # Missed GTs receive a score of -1.0
            eval_records.append(record)
            
    eval_df = pd.DataFrame(eval_records)
    
    print("-" * 40)
    print(f"Total Evaluated MS Ground Truth : {len(gt_df)}")
    print(f"Successfully Matched (TP)       : {len(matched_gt_indices)}")
    print(f"Missed Ground Truths (FN)       : {len(gt_df) - len(matched_gt_indices)}")
    print(f"False Positives (FP)            : {len(pred_df) - len(matched_gt_indices)}")
    print("-" * 40)
    
    return eval_df

# =====================================================================
# Module 3: Global Evaluation Plotting
# =====================================================================
def evaluate_and_plot_global(eval_df: pd.DataFrame, eval_metrics: List[str], display_names: dict, out_dir: str):
    """Calculate global metrics, plot ROC/PR and Heatmap."""
    print("\nCalculating overall metrics and generating global plots...")
    roc_dfs, pr_dfs, auc_records = [], [], []
    y_true = eval_df['y_true'].values
    baseline = np.sum(y_true) / len(y_true)

    def subsample_curve(x_array, y_array, max_points=2000):
        if len(x_array) <= max_points: return x_array, y_array
        indices = np.linspace(0, len(x_array) - 1, max_points).astype(int)
        return x_array[indices], y_array[indices]

    for metric in eval_metrics:
        scores = eval_df[metric].values
        d_name = display_names.get(metric, metric)
        
        # Calculate ROC
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        fpr_plot, tpr_plot = subsample_curve(fpr, tpr)
        roc_dfs.append(pd.DataFrame({'FPR': fpr_plot, 'TPR': tpr_plot, 'Metric': d_name, 'AUC': roc_auc}))
        
        # Calculate PR
        prec, rec, _ = precision_recall_curve(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
        rec_plot, prec_plot = subsample_curve(rec, prec)
        pr_dfs.append(pd.DataFrame({'Recall': rec_plot, 'Precision': prec_plot, 'Metric': d_name, 'AUC': pr_auc}))
        
        auc_records.append({'Feature': d_name, 'ROC-AUC': roc_auc, 'PR-AUC': pr_auc})

    all_roc_df = pd.concat(roc_dfs, ignore_index=True)
    all_pr_df = pd.concat(pr_dfs, ignore_index=True)
    metrics_df = pd.DataFrame(auc_records)

    all_roc_df['Legend'] = all_roc_df.apply(lambda row: f"{row['Metric']} (AUC={row['AUC']:.3f})", axis=1)
    all_pr_df['Legend'] = all_pr_df.apply(lambda row: f"{row['Metric']} (AUC={row['AUC']:.3f})", axis=1)

    color_palette = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f1c40f", "#34495e"]
    
    p_roc = (
        ggplot(all_roc_df, aes(x='FPR', y='TPR', color='Legend'))
        + geom_line(size=1.2, alpha=0.8) + geom_abline(intercept=0, slope=1, linetype='dashed', color='gray')
        + scale_color_manual(values=color_palette) + theme_bw()
        + labs(title="Overall ROC Curves", x="False Positive Rate", y="True Positive Rate")
        + theme(figure_size=(7, 6), panel_border=element_rect(color="black", size=1), legend_position="bottom", legend_title=element_blank())
    )
    p_roc.save(os.path.join(out_dir, "Overall_ROC_Curves.pdf"), verbose=False)

    p_pr = (
        ggplot(all_pr_df, aes(x='Recall', y='Precision', color='Legend'))
        + geom_line(size=1.2, alpha=0.8) + geom_hline(yintercept=baseline, linetype='dashed', color='gray')
        + scale_color_manual(values=color_palette) + theme_bw()
        + labs(title="Overall PR Curves", x="Recall", y="Precision")
        + theme(figure_size=(7, 6), panel_border=element_rect(color="black", size=1), legend_position="bottom", legend_title=element_blank())
    )
    p_pr.save(os.path.join(out_dir, "Overall_PR_Curves.pdf"), verbose=False)

    heatmap_data = metrics_df.set_index('Feature')[['ROC-AUC', 'PR-AUC']].sort_values(by='ROC-AUC', ascending=False)
    plt.figure(figsize=(6, 5))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=1, linecolor='white')
    plt.title("Overall AUC Metrics", pad=15, fontsize=14)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Overall_AUC_Heatmap.pdf"), dpi=300)
    plt.close()
    
    metrics_df.to_csv(os.path.join(out_dir, "overall_metrics.csv"), index=False)


# =====================================================================
# Module 4: Binned Evaluation Plotting and Stacked Bar Charts
# =====================================================================
def evaluate_and_plot_binned(eval_df: pd.DataFrame, eval_metrics: List[str], display_names: dict, out_dir: str):
    """Bin by length and execute raw TP/FP/FN classification based on coordinate NMS matching."""
    print("\nCalculating Binned AUCs and Raw Classifications across ORF Lengths...")
    
    # 动态分箱：如果过滤后样本极度集中，可能会导致 qcut 报错，需要加上错误处理
    try:
        q_bins = pd.qcut(eval_df['length'], q=8, duplicates='drop')
        unique_intervals = sorted(q_bins.cat.categories, key=lambda x: x.left)
        ordered_bin_labels = [f"[{int(i.left)}, {int(i.right)})" for i in unique_intervals]
        interval_to_str = {inter: f"[{int(inter.left)}, {int(inter.right)})" for inter in unique_intervals}
        eval_df['Length_Bin'] = [interval_to_str[inter] for inter in q_bins]
    except ValueError:
        print("  -> Insufficient variation in lengths for 8 bins. Skipping binned plotting.")
        return
    
    binned_records = []
    binned_counts = []
    
    for bin_label in ordered_bin_labels:
        bin_df = eval_df[eval_df['Length_Bin'] == bin_label]
        y_true_bin = bin_df['y_true'].values
        
        valid_auc = (y_true_bin.sum() > 0) and ((1 - y_true_bin).sum() > 0)
        
        for metric in eval_metrics:
            scores_bin = bin_df[metric].values
            roc_auc = auc(*roc_curve(y_true_bin, scores_bin)[:2]) if valid_auc else np.nan
            pr_auc = average_precision_score(y_true_bin, scores_bin) if valid_auc else np.nan
            binned_records.append({
                'Length_Bin': bin_label, 'Metric': display_names.get(metric, metric),
                'ROC-AUC': roc_auc, 'PR-AUC': pr_auc
            })
            
            if metric == 'score':
                tp = ((y_true_bin == 1) & (scores_bin >= 0)).sum()
                fn = ((y_true_bin == 1) & (scores_bin < 0)).sum()
                fp = (y_true_bin == 0).sum()
                
                binned_counts.extend([
                    {'Length_Bin': bin_label, 'Type': 'True Positive (TP)', 'Count': tp},
                    {'Length_Bin': bin_label, 'Type': 'False Negative (FN)', 'Count': fn},
                    {'Length_Bin': bin_label, 'Type': 'False Positive (FP)', 'Count': fp}
                ])

    binned_df = pd.DataFrame(binned_records)
    binned_df['Length_Bin'] = pd.Categorical(binned_df['Length_Bin'], categories=ordered_bin_labels, ordered=True)
    counts_df = pd.DataFrame(binned_counts)
    counts_df['Length_Bin'] = pd.Categorical(counts_df['Length_Bin'], categories=ordered_bin_labels, ordered=True)

    print("Generating Binned Line Plots and Stacked Bar Chart...")
    color_palette = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f1c40f", "#34495e"]
    
    if not binned_df['ROC-AUC'].isna().all():
        p_roc_binned = (
            ggplot(binned_df.dropna(subset=['ROC-AUC']), aes(x='Length_Bin', y='ROC-AUC', color='Metric', group='Metric'))
            + geom_point(size=3) + geom_line(size=1.2) + scale_color_manual(values=color_palette) + theme_bw() 
            + labs(title="ROC-AUC across ORF Lengths", x="ORF Length (nt)", y="ROC-AUC")
            + theme(axis_text_x=element_text(rotation=30, hjust=1), figure_size=(7, 5), panel_border=element_rect(color="black", size=1), legend_title=element_blank())
        )
        p_roc_binned.save(os.path.join(out_dir, "Binned_Length_ROC_AUC.pdf"), verbose=False)

    if not binned_df['PR-AUC'].isna().all():
        p_pr_binned = (
            ggplot(binned_df.dropna(subset=['PR-AUC']), aes(x='Length_Bin', y='PR-AUC', color='Metric', group='Metric'))
            + geom_point(size=3) + geom_line(size=1.2) + scale_color_manual(values=color_palette) + theme_bw() 
            + labs(title="PR-AUC across ORF Lengths", x="ORF Length (nt)", y="PR-AUC")
            + theme(axis_text_x=element_text(rotation=30, hjust=1), figure_size=(7, 5), panel_border=element_rect(color="black", size=1), legend_title=element_blank())
        )
        p_pr_binned.save(os.path.join(out_dir, "Binned_Length_PR_AUC.pdf"), verbose=False)

    p_counts_bar = (
        ggplot(counts_df, aes(x='Length_Bin', y='Count', fill='Type'))
        + geom_bar(stat='identity', position='stack', alpha=0.85)
        + scale_fill_manual(values={'True Positive (TP)': '#2ecc71', 'False Negative (FN)': '#95a5a6', 'False Positive (FP)': '#e74c3c'})
        + theme_bw() 
        + labs(
            title="Raw Prediction Outcomes (NMS Matched) across Lengths", 
            x="ORF Length (nt)", y="Count"
        )
        + theme(axis_text_x=element_text(rotation=30, hjust=1), figure_size=(7, 5), panel_border=element_rect(color="black", size=1), legend_title=element_blank(), legend_position="bottom")
    )
    p_counts_bar.save(os.path.join(out_dir, "Binned_Length_Distribution.pdf"), verbose=False)

# =====================================================================
# Main Orchestrator
# =====================================================================
def evaluate_orf_level_predictions(
        pred_csv_path: str, 
        gt_csv_path: str, 
        target_transcript_ids: Optional[List[str]] = None,
        # =================================================================
        # [MODIFIED] 修改主函数的签名为 min/max 长度限定
        # =================================================================
        min_orf_len: Optional[int] = None,
        max_orf_len: Optional[int] = None,
        out_dir: str = "./results/eval",
        overlap_threshold: float = 0.70):
    """
    Main Orchestrator: Highly decoupled ORF evaluation pipeline.
    Automatically identifies and plots available metrics, dropping missing ones.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Filter and Load
    pred_df, gt_df = load_and_filter_data(
        pred_csv_path, gt_csv_path, target_transcript_ids, min_orf_len, max_orf_len)
    
    all_possible_metrics = {
        'score': 'Final Score', 
        'mean_intensity': 'Mean Intensity', 
        'tri_nucleotide_periodicity': 'Periodicity',
        'uniformity_of_signal': 'Uniformity', 
        'step_up_contrast': 'Step-up Contrast', 
        'drop_off': 'Drop-off'
    }
    
    eval_metrics = [m for m in all_possible_metrics.keys() if m in pred_df.columns]
    print(f"\nDynamically selected metrics for evaluation: {eval_metrics}")
    
    display_names = {k: all_possible_metrics[k] for k in eval_metrics}

    # 2. Match and build unified evaluation dataframe
    eval_df = match_and_build_eval_df(pred_df, gt_df, eval_metrics, overlap_threshold)
    
    eval_df.to_csv(os.path.join(out_dir, "unified_evaluation_table.csv"), index=False)
    
    print("\nCalculating Overall Prediction Summary and Best F1-Score...")
    
    tp_count = ((eval_df['y_true'] == 1) & (eval_df['score'] >= 0)).sum()
    fp_count = (eval_df['y_true'] == 0).sum()
    total_predictions = tp_count + fp_count
    overall_precision = tp_count / total_predictions if total_predictions > 0 else 0.0

    y_true_all = eval_df['y_true'].values
    scores_all = eval_df['score'].values
    
    prec, rec, pr_threshs = precision_recall_curve(y_true_all, scores_all)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-9)
    
    opt_idx = np.argmax(f1_scores)
    opt_thresh_val = pr_threshs[opt_idx] if opt_idx < len(pr_threshs) else pr_threshs[-1]
    best_f1 = f1_scores[opt_idx]
    
    best_tp_count = ((eval_df['y_true'] == 1) & (eval_df['score'] >= opt_thresh_val) & (eval_df['score'] >= 0)).sum()
    best_fp_count = ((eval_df['y_true'] == 0) & (eval_df['score'] >= opt_thresh_val) & (eval_df['score'] >= 0)).sum()

    summary_df = pd.DataFrame({
        'Total_Predictions': [total_predictions],
        'True_Positives_TP': [tp_count],
        'False_Positives_FP': [fp_count],
        'Overall_Precision': [overall_precision],
        'Best_F1_Score': [best_f1],
        'Best_Threshold': [opt_thresh_val],
        'TP_at_Best_Threshold': [best_tp_count],
        'FP_at_Best_Threshold': [best_fp_count]
    })
    
    summary_save_path = os.path.join(out_dir, "overall_prediction_summary.csv")
    summary_df.to_csv(summary_save_path, index=False)
    print(f"  -> Saved Overall TP ({tp_count}) and Precision ({overall_precision:.4f})")
    print(f"  -> Found Best F1 ({best_f1:.4f}) at Threshold {opt_thresh_val:.4f} (TP={best_tp_count}, FP={best_fp_count})")
    print(f"  -> Summary saved to: {summary_save_path}")

    # 3. Global plotting
    evaluate_and_plot_global(eval_df, eval_metrics, display_names, out_dir)
    
    # 4. Binned plotting
    evaluate_and_plot_binned(eval_df, eval_metrics, display_names, out_dir)
    
    print(f"\n✅ All Evaluation processes successfully finished! Output directory: {out_dir}")


# =====================================================================
# Module 1: Precision@K Calculation Engine
# =====================================================================
def calculate_top_k_precision(
        pred_csv_path: str, 
        gt_csv_path: str, 
        min_orf_len: Optional[int] = None,
        max_orf_len: Optional[int] = None,
        overlap_threshold: float = 0.70,) -> pd.DataFrame:
    """
    Calculate the Precision@K for predicted ORFs against the Ground Truth.
    Matches are based on Frame consistency and spatial overlap.
    Includes robust length filtering for specific ORF ranges (e.g., sORFs).
    """
    print(f"\nLoading and preparing data for Precision@K evaluation...")
    
    # 1. Load Ground Truth
    gt_df = pd.read_csv(gt_csv_path, sep='\t' if '\t' in open(gt_csv_path).readline() else ',')
    gt_df['Tid_clean'] = gt_df['PacBio_ID'].astype(str)
    gt_df['start_gt'] = gt_df['CDS_Start_0based']
    gt_df['stop_gt'] = gt_df['CDS_End_0based']
    gt_df['length'] = gt_df['stop_gt'] - gt_df['start_gt'] # [NEW] 确保计算了GT长度
    
    # 2. Load Predictions 
    pred_df = pd.read_csv(pred_csv_path)
    pred_df['Tid_clean'] = pred_df['Tid'].astype(str)
    if 'length' not in pred_df.columns:
        pred_df['length'] = pred_df['stop'] - pred_df['start']
        
    # =================================================================
    # [NEW] 双向长度过滤逻辑
    # =================================================================
    if min_orf_len is not None or max_orf_len is not None:
        if min_orf_len is not None and max_orf_len is not None and min_orf_len > max_orf_len:
            raise ValueError(f"Invalid length range: min_orf_len ({min_orf_len}) cannot be greater than max_orf_len ({max_orf_len}).")
            
        lower_bound = min_orf_len if min_orf_len is not None else 0
        upper_bound = max_orf_len if max_orf_len is not None else float('inf')
        
        print(f"Filtering ORFs by length range: {lower_bound} <= Length <= {upper_bound} nt...")
        
        gt_orig_len = len(gt_df)
        pred_orig_len = len(pred_df)
        
        gt_df = gt_df[(gt_df['length'] >= lower_bound) & (gt_df['length'] <= upper_bound)].copy()
        pred_df = pred_df[(pred_df['length'] >= lower_bound) & (pred_df['length'] <= upper_bound)].copy()
        
        print(f"  -> Ground Truth ORFs retained : {len(gt_df)} / {gt_orig_len}")
        print(f"  -> Predicted ORFs retained    : {len(pred_df)} / {pred_orig_len}")
        
    if len(gt_df) == 0 or len(pred_df) == 0:
        print("Warning: No Ground Truth or Predicted ORFs left after filtering. Returning empty dataframe.")
        return pd.DataFrame(columns=['K', 'TP_Count', 'Precision'])
    # =================================================================

    # 排序预测结果 (Ensure it is sorted by score in descending order!)
    pred_df = pred_df.sort_values(by='score', ascending=False).reset_index(drop=True)
    pred_df['pred_idx'] = pred_df.index
    
    # 3. Fast Coordinate Matching (Frame & Overlap)
    print(f"Executing ultra-fast coordinate matching (Overlap > {overlap_threshold*100}%)...")
    gt_dict = {}
    for row in gt_df.itertuples(index=False):
        if row.Tid_clean not in gt_dict:
            gt_dict[row.Tid_clean] = []
        gt_dict[row.Tid_clean].append((row.start_gt, row.stop_gt))
        
    is_tp_list = []
    
    for row in pred_df.itertuples(index=False):
        tid = row.Tid_clean
        p_start, p_stop, p_len = row.start, row.stop, row.length
        
        matched = False
        if tid in gt_dict:
            for g_start, g_stop in gt_dict[tid]:
                if p_start % 3 != g_start % 3: 
                    continue
                    
                overlap_s = max(p_start, g_start)
                overlap_e = min(p_stop, g_stop)
                overlap_l = max(0, overlap_e - overlap_s)
                
                if overlap_l > 0:
                    g_len = g_stop - g_start
                    if (overlap_l / (p_len + g_len - overlap_l)) >= overlap_threshold:
                        matched = True
                        break
        
        is_tp_list.append(1 if matched else 0)

    # 4. Vectorized Precision@K Calculation
    print("Calculating Cumulative Precision@K...")
    is_tp_array = np.array(is_tp_list)
    
    tp_cumsum = np.cumsum(is_tp_array)
    k_array = np.arange(1, len(is_tp_array) + 1)
    
    precision_at_k = tp_cumsum / k_array
    
    pk_df = pd.DataFrame({
        'K': k_array,
        'TP_Count': tp_cumsum,
        'Precision': precision_at_k
    })
    
    print(f"Done! Evaluated Top {len(pk_df)} predictions.")
    return pk_df

# =====================================================================
# Module 2: Precision@K Plotting Function (保持不变)
# =====================================================================
def plot_top_k_precision(pk_df: pd.DataFrame, out_dir: str = "./results/eval", max_k: Optional[int] = None):
    """
    Plot the Precision@K line chart. 
    Optionally restrict the x-axis to a maximum K value to zoom in on the top predictions.
    """
    if pk_df.empty:
        print("Dataframe is empty, skipping plot generation.")
        return

    print("\nGenerating Precision@K line chart...")
    os.makedirs(out_dir, exist_ok=True)
    
    plot_df = pk_df.copy()
    if max_k is not None:
        plot_df = plot_df[plot_df['K'] <= max_k]
        
    if len(plot_df) > 5000:
        indices = np.linspace(0, len(plot_df) - 1, 5000).astype(int)
        plot_df = plot_df.iloc[indices]
        
    baseline_precision = pk_df['Precision'].iloc[-1]
    
    p = (
        ggplot(plot_df, aes(x='K', y='Precision'))
        + geom_line(color="#2980b9", size=1.5, alpha=0.9)
        + geom_hline(yintercept=baseline_precision, linetype="dashed", color="#e74c3c", size=1)
        + theme_classic()
        + labs(
            title="Precision@K: Top Predicted ORFs vs Ground Truth",
            x="Top K Predicted ORFs (Ranked by Final Score)",
            y="Precision (Proportion of True Positives)"
        )
        + annotate("text", x=plot_df['K'].max() * 0.2, y=baseline_precision - 0.05, 
                   label=f"Overall Baseline: {baseline_precision:.3f}", color="#e74c3c", size=10)
        + scale_y_continuous(limits=(0, 1.05))
        + scale_x_log10()
        + theme(
            figure_size=(6, 5),
            axis_title=element_text(size=12),
            axis_text=element_text(size=10)
        )
    )
    
    filename = f"TopK_Precision_Curve_{'All' if max_k is None else max_k}.pdf"
    save_path = os.path.join(out_dir, filename)
    p.save(save_path, dpi=300, verbose=False)
    
    print(f"Chart successfully saved to: {save_path}")