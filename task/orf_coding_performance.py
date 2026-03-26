import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from plotnine import *

def evaluate_coding_predictions(
        gt_file_path: str, 
        pkl_file_path: str, 
        cell_type = "brain_fetal", 
        out_dir: str = "./results/eval"):
    """
    读取真实 ORF 表格和模型的预测 pkl 文件，计算并绘制单碱基水平的 ROC 和 PR 曲线。
    只评估在 pkl 文件中存在的转录本。
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading Ground Truth from: {gt_file_path}")
    
    # 1. 读入 Ground Truth
    # 自动兼容制表符 \t 或逗号 , 分隔的表格
    try:
        gt_df = pd.read_csv(gt_file_path, sep='\t')
        if 'tid' not in gt_df.columns:
            gt_df = pd.read_csv(gt_file_path, sep=',')
    except Exception as e:
        raise ValueError(f"Error reading Ground Truth file: {e}")
        
    # 去除 tid 的版本号以方便匹配
    gt_df['tid_clean'] = gt_df['tid'].apply(lambda x: str(x).split('.')[0])
    
    # 构建快速查询字典 (同一转录本可能对应多个蛋白产物)
    gt_dict = {}
    for _, row in gt_df.iterrows():
        tid = row['tid_clean']
        if tid not in gt_dict:
            gt_dict[tid] = []
        gt_dict[tid].append({
            'start': int(row['Start_pos']) - 1,   # 转换为 0-based 坐标
            'end': int(row['End_pos']) - 3        # 指向终止密码子的第一个碱基
        })

    # 2. 读入模型预测结果
    print(f"Loading Predictions from: {pkl_file_path}")
    with open(pkl_file_path, 'rb') as f:
        predictions = pickle.load(f)

    if cell_type not in predictions:
        print(f"❌ Error: Cell type '{cell_type}' not found in the prediction file.")
        return
    
    pred = predictions[cell_type]
    y_true_tis, y_pred_tis = [], []
    y_true_tts, y_pred_tts = [], []
    
    print("Flattening sequence arrays for evaluation...")
    for uuid, pred_array in pred.items():
        # 提取 tid (解析类似 ENST00000350669-Liver-0 的 uuid)
        tid_clean = str(uuid).split('-')[0].split('.')[0]
        
        seq_len = pred_array.shape[0] # shape: (L, 2)
        
        # 构建严格的 0/1 硬标签
        gt_tis = np.zeros(seq_len, dtype=int)
        gt_tts = np.zeros(seq_len, dtype=int)
        
        # 如果这个转录本在质谱 GT 表里，就在对应坐标打上 1
        if tid_clean in gt_dict:
            for orf in gt_dict[tid_clean]:
                s = orf['start']
                e = orf['end']
                if 0 <= s < seq_len:
                    gt_tis[s] = 1
                if 0 <= e < seq_len:
                    gt_tts[e] = 1
                    
        # 将每个位点的真实标签和预测概率拼接到一维长列表
        y_true_tis.extend(gt_tis)
        y_pred_tis.extend(pred_array[:, 0])
        
        y_true_tts.extend(gt_tts)
        y_pred_tts.extend(pred_array[:, 1])

    # 转为 numpy array 以便 sklearn 处理
    y_true_tis = np.array(y_true_tis)
    y_pred_tis = np.array(y_pred_tis)
    y_true_tts = np.array(y_true_tts)
    y_pred_tts = np.array(y_pred_tts)
    
    print(f"Total evaluated nucleotides: {len(y_true_tis)}")
    print(f"Positive TIS count: {np.sum(y_true_tis)} | Positive TTS count: {np.sum(y_true_tts)}")

    # =====================================================================
    # [CHANGE] 新增安全检查与 NaN 处理拦截机制
    # =====================================================================
    if np.isnan(y_pred_tis).any() or np.isnan(y_pred_tts).any():
        nan_tis_count = np.isnan(y_pred_tis).sum()
        nan_tts_count = np.isnan(y_pred_tts).sum()
        print(f"⚠️ WARNING: 预测结果中检测到 NaN!")
        print(f"  -> TIS NaN 数量: {nan_tis_count} / {len(y_pred_tis)}")
        print(f"  -> TTS NaN 数量: {nan_tts_count} / {len(y_pred_tts)}")
        
        if nan_tis_count == len(y_pred_tis):
            print("🚨 灾难性错误: 全部预测值都是 NaN！你的模型在训练时已经崩溃，请检查训练 Loss 或调低学习率。")
            return None
            
        print("正在将 NaN 值安全替换为 0.0 以继续评估...")
        y_pred_tis = np.nan_to_num(y_pred_tis, nan=0.0)
        y_pred_tts = np.nan_to_num(y_pred_tts, nan=0.0)
    # =====================================================================
        
    # 3. 计算评价指标
    print("Calculating ROC and PR metrics...")
    
    # TIS (Start)
    fpr_tis, tpr_tis, _ = roc_curve(y_true_tis, y_pred_tis)
    roc_auc_tis = auc(fpr_tis, tpr_tis)
    prec_tis, rec_tis, _ = precision_recall_curve(y_true_tis, y_pred_tis)
    pr_auc_tis = average_precision_score(y_true_tis, y_pred_tis)

    # TTS (Stop)
    fpr_tts, tpr_tts, _ = roc_curve(y_true_tts, y_pred_tts)
    roc_auc_tts = auc(fpr_tts, tpr_tts)
    prec_tts, rec_tts, _ = precision_recall_curve(y_true_tts, y_pred_tts)
    pr_auc_tts = average_precision_score(y_true_tts, y_pred_tts)
    
    # 打印结果
    print("-" * 40)
    print(f"TIS (Start) -> ROC-AUC: {roc_auc_tis:.4f} | PR-AUC: {pr_auc_tis:.4f}")
    print(f"TTS (Stop)  -> ROC-AUC: {roc_auc_tts:.4f} | PR-AUC: {pr_auc_tts:.4f}")
    print("-" * 40)

    # 4. 可视化 (Plotnine)
    print("Generating plots...")
    
    # 为 ROC 曲线构建 DataFrame
    df_roc_tis = pd.DataFrame({'FPR': fpr_tis, 'TPR': tpr_tis, 'Type': f'TIS (AUC={roc_auc_tis:.3f})'})
    df_roc_tts = pd.DataFrame({'FPR': fpr_tts, 'TPR': tpr_tts, 'Type': f'TTS (AUC={roc_auc_tts:.3f})'})
    df_roc = pd.concat([df_roc_tis, df_roc_tts], ignore_index=True)

    # 为 PR 曲线构建 DataFrame
    df_pr_tis = pd.DataFrame({'Recall': rec_tis, 'Precision': prec_tis, 'Type': f'TIS (AUC={pr_auc_tis:.3f})'})
    df_pr_tts = pd.DataFrame({'Recall': rec_tts, 'Precision': prec_tts, 'Type': f'TTS (AUC={pr_auc_tts:.3f})'})
    df_pr = pd.concat([df_pr_tis, df_pr_tts], ignore_index=True)

    # 配色方案
    soft_colors = ["#00BFC4", "#F8766D"]

    # --- 绘制 ROC 曲线 ---
    p_roc = (
        ggplot(df_roc, aes(x='FPR', y='TPR', color='Type'))
        + geom_line(size=1)
        + geom_abline(intercept=0, slope=1, linetype='dashed', color='gray')
        + scale_color_manual(values=soft_colors)
        + theme_bw()
        + labs(title="ROC Curve for Protein Output Prediction", x="False Positive Rate", y="True Positive Rate")
        + theme(figure_size=(6, 5), panel_border=element_rect(color="black", size=1))
    )
    roc_path = os.path.join(out_dir, "Coding_ROC_Curve.pdf")
    p_roc.save(roc_path, verbose=False)

    # --- 绘制 PR 曲线 ---
    p_pr = (
        ggplot(df_pr, aes(x='Recall', y='Precision', color='Type'))
        + geom_line(size=1)
        + scale_color_manual(values=soft_colors)
        + theme_bw()
        + labs(title="Precision-Recall Curve for Protein Output Prediction", x="Recall", y="Precision")
        + theme(figure_size=(6, 5), panel_border=element_rect(color="black", size=1))
    )
    pr_path = os.path.join(out_dir, "Coding_PR_Curve.pdf")
    p_pr.save(pr_path, verbose=False)

    # 保存指标至 CSV 便于后续查阅
    metrics_df = pd.DataFrame({
        'Metric': ['ROC-AUC', 'PR-AUC'],
        'TIS': [roc_auc_tis, pr_auc_tis],
        'TTS': [roc_auc_tts, pr_auc_tts]
    })
    metrics_df.to_csv(os.path.join(out_dir, "coding_evaluation_metrics.csv"), index=False)
    
    print(f"Plots and metrics saved to: {out_dir}")
    return metrics_df