import os
import pandas as pd
import numpy as np
from plotnine import *
from typing import Optional


def plot_model_benchmark(
        manifest: list, 
        out_dir: str = "./results/benchmark",
        depth_levels: list = ['1M', '2M', '5M', '10M', 'Total']
):
    """
    一次性读取多个模型的评估结果 CSV，绘制 Ribo-seq 深度与 AUC 的趋势对比图。
    支持母目录自动遍历，极大简化输入 manifest 配置。
    """
    os.makedirs(out_dir, exist_ok=True)
    print("Loading and aggregating AUC benchmark data...")
    
    records = []
    
    # =========================================================
    # 内部辅助函数：统一定义如何从 CSV 提取 ROC-AUC 和 PR-AUC
    # =========================================================
    def extract_auc_metrics(df, target_feature):
        if target_feature and 'Feature' in df.columns:
            sub_df = df[df['Feature'] == target_feature]
            if sub_df.empty: return None, None
            row = sub_df.iloc[0]
        else:
            row = df.sort_values(by='PR-AUC', ascending=False).iloc[0]
        return row['ROC-AUC'], row['PR-AUC']
        
    # =========================================================
    # [MODIFIED] 极简的智能文件读取引擎 (与 TP-Precision 保持完全一致)
    # =========================================================
    for cfg in manifest:
        model_name = cfg['model']
        model_type = cfg['type']  # 'w/ Ribo-seq' 或 'w/o Ribo-seq'
        target_feature = cfg.get('feature', None)
        
        if model_type == 'w/o Ribo-seq':
            # 深度无关模型 (单点文件)：读取后将分数广播到所有 depth_levels
            csv_path = cfg['path']
            if not os.path.exists(csv_path):
                print(f"  [Warning] File not found: {csv_path}. Skipping...")
                continue
                
            df = pd.read_csv(csv_path)
            roc_auc, pr_auc = extract_auc_metrics(df, target_feature)
            
            if roc_auc is not None and pr_auc is not None:
                for d in depth_levels:
                    records.append({
                        'Model': model_name, 'Type': model_type, 'Depth': d,
                        'ROC-AUC': roc_auc, 'PR-AUC': pr_auc
                    })
            else:
                print(f"  [Warning] Feature '{target_feature}' not found in {csv_path}.")
                
        elif model_type == 'w/ Ribo-seq':
            # 深度依赖模型 (动态轨迹)：自动按 depth 向量遍历母目录
            base_dir = cfg['base_dir']
            file_name = cfg.get('file_name', 'overall_metrics.csv')
            target_depths = cfg.get('depths', depth_levels)
            
            for d in target_depths:
                # 智能路径拼接：base_dir / 深度(例如1M) / file_name
                csv_path = os.path.join(base_dir, d, file_name)
                if not os.path.exists(csv_path):
                    print(f"  [Warning] File not found: {csv_path}. Skipping...")
                    continue
                    
                df = pd.read_csv(csv_path)
                roc_auc, pr_auc = extract_auc_metrics(df, target_feature)
                
                if roc_auc is not None and pr_auc is not None:
                    records.append({
                        'Model': model_name, 'Type': model_type, 'Depth': d,
                        'ROC-AUC': roc_auc, 'PR-AUC': pr_auc
                    })
                else:
                    print(f"  [Warning] Feature '{target_feature}' not found in {csv_path}.")
    # =========================================================
            
    if not records:
        raise ValueError("No valid records extracted. Please check your manifest and file paths.")
        
    # 汇总为总表
    plot_df = pd.DataFrame(records)
    plot_df.to_csv(os.path.join(out_dir, "aggregated_benchmark_metrics.csv"), index=False)
    
    # 将 Depth 设定为有序的 Categorical 变量
    plot_df['Depth'] = pd.Categorical(plot_df['Depth'], categories=depth_levels, ordered=True)

    print("Generating Benchmark Trend Plots...")
    
    model_order = [
        "TRACE", "Convolution", 
        "TranslationAI", "RiboTISH", "RibORF",
        "ORF-structure"
    ]
    all_models_in_data = plot_df['Model'].unique().tolist()
    for m in all_models_in_data:
        if m not in model_order:
            model_order.append(m)
            
    plot_df['Model'] = pd.Categorical(plot_df['Model'], categories=model_order, ordered=True)

    # 颜色分配引擎
    color_mapping = {}
    gray_idx, brown_idx = 0, 0
    grays = ["#555555", "#777777", "#999999", "#BBBBBB", "#DDDDDD", "#E5E5E5"]
    browns = ["#AF804F", "#B98C57", "#C3975F", "#CDA367", "#D7AF6F", "#E1BA77", "#EBC67F"]
    
    unique_models = plot_df['Model'].unique().tolist()
    for m_name in unique_models:
        if "TRACE" in m_name:
            color_mapping[m_name] = "#2C6B9A"
        elif "ORF-structure" in m_name or "Baseline" in m_name:
            color_mapping[m_name] = browns[brown_idx % len(browns)]
            brown_idx += 1
        else:
            color_mapping[m_name] = grays[gray_idx % len(grays)]
            gray_idx += 1
            
    print("Applied Color Mapping:")
    for k, v in color_mapping.items():
        print(f"  {k}: {v}")
    
    # 内部画图函数
    def build_trend_plot(metric_name: str, y_label: str):
        p = (
            ggplot(plot_df, aes(x='Depth', y=metric_name, color='Model', group='Model'))
            + geom_line(aes(linetype='Type'), size=1.5, alpha=0.8)
            + geom_point(data=plot_df[plot_df['Type'] == 'w/ Ribo-seq'], size=3.5, alpha=0.9)
            
            + scale_color_manual(values=color_mapping)
            + scale_linetype_manual(values={'w/ Ribo-seq': 'dashed', 'w/o Ribo-seq': 'solid'})
            + scale_x_discrete(expand=[0, 0])
            + theme_bw()
            + labs(
                x="Ribo-seq Data Depth",
                y=y_label
            )
            + theme(
                panel_border=element_rect(color="black", size=1),
                axis_title=element_text(size=12),
                axis_text_x=element_text(rotation=0, ha='center', size=10),
                axis_text_y=element_text(size=10),
                legend_position="right",
                legend_text=element_text(size=10),
                legend_title=element_blank()
            )
        )
        return p

    # 绘制并保存 ROC-AUC
    p_roc = build_trend_plot('ROC-AUC', 'ROC-AUC')
    p_roc.save(os.path.join(out_dir, "Benchmark_ROC_AUC_Trend.pdf"), dpi=300, verbose=False)
    
    # 绘制并保存 PR-AUC
    p_pr = build_trend_plot('PR-AUC', 'PR-AUC')
    p_pr.save(os.path.join(out_dir, "Benchmark_PR_AUC_Trend.pdf"), dpi=300, verbose=False)
    
    print(f"✅ Benchmark Complete! Plots saved to: {out_dir}")


import os
import pandas as pd
import numpy as np
from plotnine import *

# =================================================================
# [MODIFIED] 重命名函数，增加 x_col, y_col 及 label 动态参数
# =================================================================
def plot_tradeoff_benchmark(
        manifest: list, 
        out_dir: str = "./results/benchmark",
        depth_levels: list = ['1M', '5M', '10M', '50M', '100M', 'Total'],
        x_col: str = 'TP_at_Best_Threshold',
        y_col: str = 'Best_F1_Score',
        x_label: str = 'True Positives at Best F1 (Log Scale)',
        y_label: str = 'Best F1-Score',
        title: str = 'Quantity-Quality Trade-off at Best Threshold'
):
    """
    读取 overall_prediction_summary.csv 绘制任意两个指标的气泡轨迹图。
    默认设置为 x=TP_at_Best_Threshold, y=Best_F1_Score。
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading and aggregating benchmark data ({x_col} vs {y_col})...")
    
    records = []
    
    for cfg in manifest:
        model_name = cfg['model']
        model_type = cfg['type']  # 'w/ Ribo-seq' 或 'w/o Ribo-seq'
        
        if model_type == 'w/o Ribo-seq':
            csv_path = cfg['path']
            if not os.path.exists(csv_path):
                print(f"  [Warning] File not found: {csv_path}. Skipping...")
                continue
                
            df = pd.read_csv(csv_path)
            # =========================================================
            # [MODIFIED] 动态提取用户指定的 x_col 和 y_col
            # =========================================================
            records.append({
                'Model': model_name,
                'Type': model_type,
                'Depth': 'Constant',
                x_col: df.iloc[0][x_col],
                y_col: df.iloc[0][y_col]
            })
            
        elif model_type == 'w/ Ribo-seq':
            base_dir = cfg['base_dir']
            file_name = cfg.get('file_name', 'overall_prediction_summary.csv')
            target_depths = cfg.get('depths', depth_levels)
            
            for d in target_depths:
                csv_path = os.path.join(base_dir, d, file_name)
                if not os.path.exists(csv_path):
                    print(f"  [Warning] File not found: {csv_path}. Skipping...")
                    continue
                    
                df = pd.read_csv(csv_path)
                # =========================================================
                # [MODIFIED] 动态提取用户指定的 x_col 和 y_col
                # =========================================================
                records.append({
                    'Model': model_name,
                    'Type': model_type,
                    'Depth': d,
                    x_col: df.iloc[0][x_col],
                    y_col: df.iloc[0][y_col]
                })
            
    if not records:
        raise ValueError("No valid records extracted. Please check your manifest and file paths.")
        
    plot_df = pd.DataFrame(records)
    # [MODIFIED] 保存的聚合表名称也随之动态变化
    csv_out = os.path.join(out_dir, f"aggregated_{x_col}_{y_col}_metrics.csv")
    plot_df.to_csv(csv_out, index=False)

    print(f"Generating {x_col} vs {y_col} Benchmark Plot...")
    
    model_order = [
        "TRACE", "Convolution", 
        "TranslationAI", "RiboTISH", "RibORF",
        "ORF-structure"
    ]
    all_models_in_data = plot_df['Model'].unique().tolist()
    for m in all_models_in_data:
        if m not in model_order:
            model_order.append(m)
    plot_df['Model'] = pd.Categorical(plot_df['Model'], categories=model_order, ordered=True)

    # 颜色分配引擎
    color_mapping = {}
    gray_idx, brown_idx = 0, 0
    grays = ["#555555", "#777777", "#999999", "#BBBBBB", "#DDDDDD", "#E5E5E5"]
    browns = ["#AF804F", "#B98C57", "#C3975F", "#CDA367", "#D7AF6F", "#E1BA77", "#EBC67F"]
    
    unique_models = plot_df['Model'].unique().tolist()
    for m_name in unique_models:
        if "TRACE" in m_name:
            color_mapping[m_name] = "#2C6B9A"
        elif "ORF-structure" in m_name or "Baseline" in m_name:
            color_mapping[m_name] = browns[brown_idx % len(browns)]
            brown_idx += 1
        else:
            color_mapping[m_name] = grays[gray_idx % len(grays)]
            gray_idx += 1
    
    # =================================================================
    # [MODIFIED] 动态映射 x 和 y
    # =================================================================
    p = (
        ggplot(plot_df, aes(x=x_col, y=y_col, color='Model'))
        + geom_line(
            data=plot_df[plot_df['Type'] == 'w/ Ribo-seq'], 
            mapping=aes(group='Model'), 
            linetype='dashed', size=1.2, alpha=0.7
        )
        + geom_point(size=4.5, alpha=0.9, stroke=0.5)
        + scale_x_log10()
        + scale_color_manual(values=color_mapping)
        + theme_bw()
        + labs(
            title=title,
            x=x_label,
            y=y_label
        )
        + theme(
            panel_border=element_rect(color="black", size=1),
            axis_title=element_text(size=12),
            axis_text=element_text(size=10),
            legend_position="right",
            legend_text=element_text(size=10),
            legend_title=element_blank()
        )
    )
    
    # [MODIFIED] 根据指定的列名动态生成输出文件名
    save_path = os.path.join(out_dir, f"Benchmark_Tradeoff_{x_col}_vs_{y_col}.pdf")
    p.save(save_path, dpi=300, verbose=False)
    print(f"✅ Benchmark Complete! Trade-off plot saved to: {save_path}")



def plot_multi_model_top_k_precision(
        manifest: list, 
        out_dir: str = "./results/benchmark", 
        max_k: Optional[int] = None
):
    """
    绘制多模型 Top-K Precision 对比折线图。
    支持输入已计算好的 Precision@K 表，或统一评估表。
    """
    print("\nCalculating and aggregating Precision@K for multiple models...")
    os.makedirs(out_dir, exist_ok=True)
    
    all_pk_data = []
    
    for cfg in manifest:
        model_name = cfg['model']
        csv_path = cfg['path']
        score_col = cfg.get('score_col', 'score') 
        
        if not os.path.exists(csv_path):
            print(f"  [Warning] File not found: {csv_path}. Skipping...")
            continue
            
        df = pd.read_csv(csv_path)
        
        if 'Precision' in df.columns and 'K' in df.columns:
            pk_df = df[['K', 'Precision']].copy()
            
        elif 'y_true' in df.columns and score_col in df.columns:
            df_sorted = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
            df_sorted = df_sorted[df_sorted[score_col] >= 0].copy()
            
            if df_sorted.empty:
                print(f"  [Warning] {model_name} has no valid positive predictions. Skipping...")
                continue
                
            k_array = np.arange(1, len(df_sorted) + 1)
            tp_cumsum = df_sorted['y_true'].cumsum()
            
            pk_df = pd.DataFrame({
                'K': k_array,
                'Precision': tp_cumsum / k_array
            })
        else:
            print(f"  [Warning] {csv_path} lacks required columns.")
            continue
            
        pk_df['Model'] = model_name
        all_pk_data.append(pk_df)
        
    if not all_pk_data:
        raise ValueError("No valid Top-K data processed. Please check your manifest.")
        
    # 合并所有模型数据
    plot_df = pd.concat(all_pk_data, ignore_index=True)
    
    # 限制 X 轴范围 (Zoom-in)
    if max_k is not None:
        plot_df = plot_df[plot_df['K'] <= max_k]
        
    # =================================================================
    # [NEW] 智能按模型平滑 (Smart Smoothing per Model)
    # =================================================================
    smoothing_window = 50  # 你可以根据需要调大或调小这个窗口
    
    def apply_smoothing(group):
        # 使用 min_periods=1 保证最开头的几个点不会变成 NaN
        group['Precision_Smooth'] = group['Precision'].rolling(window=smoothing_window, min_periods=1).mean()
        return group
        
    print(f"Applying rolling average smoothing (window={smoothing_window})...")
    plot_df = plot_df.groupby('Model', group_keys=False).apply(apply_smoothing)

    # =================================================================
    # 智能按模型降采样 (Smart Downsampling per Model)
    # =================================================================
    def downsample(group, max_pts=3000):
        if len(group) > max_pts:
            indices = np.linspace(0, len(group) - 1, max_pts).astype(int)
            return group.iloc[indices]
        return group
        
    plot_df = plot_df.groupby('Model', group_keys=False).apply(downsample)

    model_order = [
        "TRACE", "Convolution", 
        "TranslationAI", "RiboTISH", "RibORF",
        "ORF-structure"
    ]
    all_models_in_data = plot_df['Model'].unique().tolist()
    for m in all_models_in_data:
        if m not in model_order:
            model_order.append(m)
    plot_df['Model'] = pd.Categorical(plot_df['Model'], categories=model_order, ordered=True)

    # =================================================================
    # [MODIFIED] 颜色与线型分配引擎 (Color & Linetype Mapping)
    # =================================================================
    color_mapping = {}
    linetype_mapping = {}  # [NEW] 存储每个模型的线型
    
    gray_idx, brown_idx = 0, 0
    grays = ["#555555", "#777777", "#999999", "#BBBBBB", "#DDDDDD", "#E5E5E5"]
    browns = ["#AF804F", "#B98C57", "#C3975F", "#CDA367", "#D7AF6F", "#E1BA77", "#EBC67F"]
    
    for m_name in plot_df['Model'].cat.categories:
        # 1. 我们的模型
        if "TRACE" in m_name:
            color_mapping[m_name] = "#2C6B9A"
            linetype_mapping[m_name] = "solid"    # [NEW] 独享实线
            
        # 2. Baseline 系列
        elif "ORF-structure" in m_name or "Baseline" in m_name:
            color_mapping[m_name] = browns[brown_idx % len(browns)]
            linetype_mapping[m_name] = "dashed"   # [NEW] 降级虚线
            brown_idx += 1
            
        # 3. 其他竞争模型
        else:
            color_mapping[m_name] = grays[gray_idx % len(grays)]
            linetype_mapping[m_name] = "dashed"   # [NEW] 降级虚线
            gray_idx += 1
            
    print("Applied Color & Linetype Mapping for Precision@K:")
    for k in color_mapping.keys():
        print(f"  {k}: {color_mapping[k]} | {linetype_mapping[k]}")
    # =================================================================
    
    print("Generating Multi-Model Precision@K line chart...")
    
    # =================================================================
    # 绘图逻辑
    # =================================================================
    p = (
        ggplot(plot_df, aes(x='K', y='Precision_Smooth', color='Model'))
        # [MODIFIED] 将线型映射加入到 geom_line 中
        + geom_line(aes(linetype='Model'), size=1.5, alpha=0.85)
        
        + scale_color_manual(values=color_mapping)
        # [NEW] 传入线型字典，并隐藏线型的图例（防止图例臃肿）
        + scale_linetype_manual(values=linetype_mapping, guide=None)
        
        + scale_y_continuous(limits=(0, 1.05))
        + scale_x_log10() # 使用对数坐标看头部趋势更清晰
        + theme_classic()
        + labs(
            title=f"Precision@K Benchmark {'(Top ' + str(max_k) + ')' if max_k else '(All Predictions)'}",
            x="Top K Predicted ORFs (Log Scale, Ranked by Conf. Score)",
            y="Precision (Proportion of True Positives)"
        )
        + theme(
            figure_size=(8, 5),
            # panel_border=element_rect(color="black", size=1.5),
            axis_title=element_text(size=12, face="bold"),
            axis_text=element_text(size=10),
            legend_position="right",
            legend_text=element_text(size=10),
            legend_title=element_blank()
        )
    )
    
    filename = f"Benchmark_TopK_Precision_Curve_{'All' if max_k is None else max_k}.pdf"
    save_path = os.path.join(out_dir, filename)
    p.save(save_path, dpi=300, verbose=False)
    
    print(f"✅ Multi-Model Precision@K Chart successfully saved to: {save_path}")