import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from plotnine import *

def evaluate_model_performance(df_pred, gt_df, out_dir="./results/eval", target_tx=None):
    """
    使用 ROC 和 PR 曲线评估模型预测 ORF 的表现，包含综合 coding_score。
    """
    
    # 1. 准备 Ground Truth 集合 (建立一个唯一标识符用于快速匹配)
    # 确保 tid 格式一致 (去掉版本号)
    gt_df['tid_clean'] = gt_df['tid'].apply(lambda x: str(x).split('.')[0])

    # 过滤目标转录本
    if target_tx:
        target_clean_tx = [str(tid).split('.')[0] for tid in target_tx]
        gt_df = gt_df[gt_df['tid_clean'].isin(target_clean_tx)]

    gt_set = set(zip(gt_df['tid_clean'], gt_df['Start_pos'], gt_df['End_pos']))

    # [CHANGE] 2. 计算综合 coding_score
    # 权重为 PIF:Uniformity:Drop_off = 3:1:1
    # 公式：coding_score = (3*PIF + 1*Uni + 1*Drop) / 5
    # 这样分值越高代表与最优值 (1, 1, 1) 的加权距离越小
    df_pred['coding_score'] = (2 * df_pred['PIF'] + 1 * df_pred['Uniformity'] + 1 * df_pred['Drop_off']) / 4

    # 3. 标记样本 (Labeling)
    # 如果预测的 (tid, start_pos, end_pos) 在 GT 集合中，则为 1，否则为 0
    df_pred['label'] = df_pred.apply(
        lambda row: 1 if (str(row['tid']).split('.')[0], row['start_pos'], row['end_pos']) in gt_set else 0, 
        axis=1
    )

    print(f"Total candidates: {len(df_pred)}")
    print(f"Total ground truth: {len(gt_set)}")
    print(f"True Positive candidates: {df_pred['label'].sum()}")

    # [CHANGE] 4. 计算不同指标下的 AUC (增加了 coding_score)
    metrics_to_eval = ['PIF', 'Uniformity', 'Drop_off', 'coding_score']
    plot_data = []

    for metric in metrics_to_eval:
        y_true = df_pred['label']
        y_score = df_pred[metric]

        # ROC 计算
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # PR 计算
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
        
        print(f"Metric: {metric:15} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

        # 收集绘图数据 (为了使用 plotnine)
        roc_df = pd.DataFrame({'x': fpr, 'y': tpr, 'Metric': f"{metric} (AUC={roc_auc:.3f})", 'Type': 'ROC'})
        pr_df = pd.DataFrame({'x': recall, 'y': precision, 'Metric': f"{metric} (AUC={pr_auc:.3f})", 'Type': 'PR'})
        plot_data.append(roc_df)
        plot_data.append(pr_df)

    # 5. 可视化
    df_plot = pd.concat(plot_data)
    
    # 柔和配色方案
    soft_colors = ["#F8766D", "#7CAE00", "#00BFC4", "#C77CFF"] # 增加了一个紫色给 coding_score

    os.makedirs(out_dir, exist_ok=True)
    
    # 绘制 ROC 曲线
    p_roc = (
        ggplot(df_plot[df_plot['Type'] == 'ROC'], aes(x='x', y='y', color='Metric'))
        + geom_line(size=1)
        + geom_abline(linetype='dashed', color='gray')
        + scale_color_manual(values=soft_colors)
        + theme_bw()
        + labs(title="ROC Curve for ORF Prediction", x="False Positive Rate", y="True Positive Rate")
        + theme(figure_size=(6, 5), panel_border=element_rect(color="black", size=1))
    )
    p_roc.save(os.path.join(out_dir, "orf_roc_curve.pdf"))

    # 绘制 PR 曲线
    p_pr = (
        ggplot(df_plot[df_plot['Type'] == 'PR'], aes(x='x', y='y', color='Metric'))
        + geom_line(size=1)
        + scale_color_manual(values=soft_colors)
        + theme_bw()
        + labs(title="Precision-Recall Curve for ORF Prediction", x="Recall", y="Precision")
        + theme(figure_size=(6, 5), panel_border=element_rect(color="black", size=1))
    )
    p_pr.save(os.path.join(out_dir, "orf_pr_curve.pdf"))

    return df_pred[df_pred['label']==1]