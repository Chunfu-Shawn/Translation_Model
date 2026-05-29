import os
import numpy as np
import pandas as pd
from typing import Optional
from plotnine import *


GLOBAL_MODEL_COLORS = {
    "TRACE": "#2C6B9A",
    "Convolution": "#637D96",
    "TranslationAI": "#555555",
    "RiboTIE": "#777777",
    "RibORF": "#BBBBBB",
    "RiboTISH": "#999999",
    "ORF-structure": "#AF804F",
    "Transcription-level": "#EBC67F"
}


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
    
    def extract_auc_metrics(df, target_feature):
        if target_feature and 'Feature' in df.columns:
            sub_df = df[df['Feature'] == target_feature]
            if sub_df.empty: return None, None
            row = sub_df.iloc[0]
        else:
            row = df.sort_values(by='PR-AUC', ascending=False).iloc[0]
        return row['ROC-AUC'], row['PR-AUC']
        
    for cfg in manifest:
        model_name = cfg['model']
        model_type = cfg['type']  
        target_feature = cfg.get('feature', None)
        
        if model_type == 'w/o Ribo-seq':
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
            base_dir = cfg['base_dir']
            file_name = cfg.get('file_name', 'overall_metrics.csv')
            target_depths = cfg.get('depths', depth_levels)
            
            for d in target_depths:
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
            
    if not records:
        raise ValueError("No valid records extracted. Please check your manifest and file paths.")
        
    plot_df = pd.DataFrame(records)
    plot_df.to_csv(os.path.join(out_dir, "aggregated_benchmark_metrics.csv"), index=False)
    
    plot_df['Depth'] = pd.Categorical(plot_df['Depth'], categories=depth_levels, ordered=True)

    print("Generating Benchmark Trend Plots...")
    
    model_order = [
        "TRACE", "Convolution", 
        "TranslationAI", "RiboTIE", "RiboTISH", "RibORF",
        "ORF-structure", "Transcription-level"
    ]
    all_models_in_data = plot_df['Model'].unique().tolist()
    for m in all_models_in_data:
        if m not in model_order:
            model_order.append(m)
            
    plot_df['Model'] = pd.Categorical(plot_df['Model'], categories=model_order, ordered=True)

    # =================================================================
    # [MODIFIED] 使用全局统一颜色字典
    # =================================================================
    color_mapping = {}
    for m_name in plot_df['Model'].unique():
        color_mapping[m_name] = GLOBAL_MODEL_COLORS.get(m_name, "#C0C0C0") # 默认灰色兜底
            
    print("Applied Color Mapping:")
    for k, v in color_mapping.items():
        print(f"  {k}: {v}")
    
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

    p_roc = build_trend_plot('ROC-AUC', 'ROC-AUC')
    p_roc.save(os.path.join(out_dir, "Benchmark_ROC_AUC_Trend.pdf"), dpi=300, verbose=False)
    
    p_pr = build_trend_plot('PR-AUC', 'PR-AUC')
    p_pr.save(os.path.join(out_dir, "Benchmark_PR_AUC_Trend.pdf"), dpi=300, verbose=False)
    
    print(f"✅ Benchmark Complete! Plots saved to: {out_dir}")


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
        model_type = cfg['type']  
        
        if model_type == 'w/o Ribo-seq':
            csv_path = cfg['path']
            if not os.path.exists(csv_path):
                print(f"  [Warning] File not found: {csv_path}. Skipping...")
                continue
                
            df = pd.read_csv(csv_path)
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
    
    all_depth_categories = depth_levels + ['Constant']
    plot_df['Depth'] = pd.Categorical(plot_df['Depth'], categories=all_depth_categories, ordered=True)

    csv_out = os.path.join(out_dir, f"aggregated_{x_col}_{y_col}_metrics.csv")
    plot_df.to_csv(csv_out, index=False)

    print(f"Generating {x_col} vs {y_col} Benchmark Plot...")
    
    model_order = [
        "TRACE", "Convolution",
        "TranslationAI", "RiboTIE", "RiboTISH", "RibORF",
        "ORF-structure", "Transcription-level"
    ]
    all_models_in_data = plot_df['Model'].unique().tolist()
    for m in all_models_in_data:
        if m not in model_order:
            model_order.append(m)
    plot_df['Model'] = pd.Categorical(plot_df['Model'], categories=model_order, ordered=True)

    # =================================================================
    # [MODIFIED] 使用全局统一颜色字典
    # =================================================================
    color_mapping = {}
    for m_name in plot_df['Model'].unique():
        color_mapping[m_name] = GLOBAL_MODEL_COLORS.get(m_name, "#C0C0C0")
            
    # 点大小分配引擎 
    min_size = 2  
    max_size = 5  
    depth_sizes = np.linspace(min_size, max_size, len(depth_levels))
    size_mapping = {d: s for d, s in zip(depth_levels, depth_sizes)}
    size_mapping['Constant'] = 5 

    p = (
        ggplot(plot_df, aes(x=x_col, y=y_col, color='Model'))
        + geom_line(
            data=plot_df[plot_df['Type'] == 'w/ Ribo-seq'], 
            mapping=aes(group='Model'), 
            linetype='dashed', size=1.2, alpha=0.7
        )
        + geom_point(mapping=aes(size='Depth'), alpha=0.9, stroke=0.5)
        + scale_x_log10()
        + scale_color_manual(values=color_mapping)
        + scale_size_manual(values=size_mapping, breaks=depth_levels, name="Ribo-seq Depth")
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
            legend_title=element_text(size=10, face="bold") 
        )
    )
    
    save_path = os.path.join(out_dir, f"Benchmark_Tradeoff_{x_col}_vs_{y_col}.pdf")
    p.save(save_path, dpi=300, verbose=False)
    print(f"✅ Benchmark Complete! Trade-off plot saved to: {save_path}")


def plot_multi_model_top_k_precision(
        manifest: list, 
        out_dir: str = "./results/benchmark", 
        min_k: Optional[int] = None,
        max_k: Optional[int] = None, 
        suffix: str = ""
):
    """
    绘制多模型 Top-K Precision 对比折线图。
    支持输入已计算好的 Precision@K 表，或统一评估表。
    支持自定义 X 轴展示范围 (min_k, max_k)。
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
        
    plot_df = pd.concat(all_pk_data, ignore_index=True)
        
    smoothing_window = 50  
    
    def apply_smoothing(group):
        group['Precision_Smooth'] = group['Precision'].rolling(window=smoothing_window, min_periods=1).mean()
        return group
        
    print(f"Applying rolling average smoothing (window={smoothing_window})...")
    plot_df = plot_df.groupby('Model', group_keys=False).apply(apply_smoothing)

    if min_k is not None:
        plot_df = plot_df[plot_df['K'] >= min_k]
    if max_k is not None:
        plot_df = plot_df[plot_df['K'] <= max_k]

    def downsample(group, max_pts=3000):
        if len(group) > max_pts:
            indices = np.linspace(0, len(group) - 1, max_pts).astype(int)
            return group.iloc[indices]
        return group
        
    plot_df = plot_df.groupby('Model', group_keys=False).apply(downsample)

    model_order = [
        "TRACE", "Convolution",
        "TranslationAI", "RiboTIE", "RiboTISH", "RibORF",
        "ORF-structure", "Transcription-level"
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
    linetype_mapping = {} 
    
    for m_name in plot_df['Model'].cat.categories:
        # 使用全局统一颜色
        color_mapping[m_name] = GLOBAL_MODEL_COLORS.get(m_name, "#C0C0C0")
        
        # 线型保留你原有的逻辑：只有 TRACE 是实线，其余 Baseline 都是虚线
        if "TRACE" in m_name:
            linetype_mapping[m_name] = "solid"
        else:
            linetype_mapping[m_name] = "dashed"  
            
    print("Generating Multi-Model Precision@K line chart...")
    
    if min_k is not None and max_k is not None:
        title_suffix = f"(K: {min_k} to {max_k})"
        file_suffix = f"{suffix}_{min_k}_to_{max_k}"
    elif min_k is not None:
        title_suffix = f"(K >= {min_k})"
        file_suffix = f"{suffix}_{min_k}_to_All"
    elif max_k is not None:
        title_suffix = f"(Top {max_k})"
        file_suffix = f"{suffix}_1_to_{max_k}"
    else:
        title_suffix = "(All Predictions)"
        file_suffix = f"{suffix}_All"

    p = (
        ggplot(plot_df, aes(x='K', y='Precision_Smooth', color='Model'))
        + geom_line(aes(linetype='Model'), size=1.5, alpha=0.85)
        + scale_color_manual(values=color_mapping)
        + scale_linetype_manual(values=linetype_mapping, guide=None)
        + scale_y_continuous(limits=(0, 1.05))
        + scale_x_log10() 
        + theme_classic()
        + labs(
            title=f"Precision@K Benchmark {title_suffix}",
            x="Top K Predicted ORFs (Log Scale, Ranked by Conf. Score)",
            y="Precision (Proportion of True Positives)"
        )
        + theme(
            figure_size=(7, 5),
            axis_title=element_text(size=12, face="bold"),
            axis_text=element_text(size=10),
            legend_position="right",
            legend_text=element_text(size=10),
            legend_title=element_blank()
        )
    )
    
    filename = f"Benchmark_TopK_Precision_Curve_{file_suffix}.pdf"
    save_path = os.path.join(out_dir, filename)
    p.save(save_path, dpi=300, verbose=False)
    
    print(f"✅ Multi-Model Precision@K Chart successfully saved to: {save_path}")


def plot_top_k_precision_bar(
        manifest: list, 
        target_k: int,
        out_dir: str = "./results/benchmark", 
        suffix: str = ""
):
    """
    绘制指定 Top-K 的 Precision Bar + Jitter 图。
    Bar 代表模型的平均 Precision，误差棒代表 SEM。
    点代表每一次具体的评估，形状映射为 Dataset，颜色映射为 Cell_type。
    """
    print(f"\nCalculating Precision@{target_k} for multiple models...")
    os.makedirs(out_dir, exist_ok=True)
    
    records = []
    
    for cfg in manifest:
        model_name = cfg['model']
        csv_path = cfg['path']
        dataset_name = cfg.get('dataset', 'Unknown_Dataset')
        cell_type = cfg.get('cell_type', 'Unknown_Cell')
        score_col = cfg.get('score_col', 'score') 
        
        if not os.path.exists(csv_path):
            print(f"  [Warning] File not found: {csv_path}. Skipping...")
            continue
            
        df = pd.read_csv(csv_path)
        prec_val = np.nan
        
        if 'Precision' in df.columns and 'K' in df.columns:
            if target_k in df['K'].values:
                prec_val = df.loc[df['K'] == target_k, 'Precision'].values[0]
            else:
                max_k_avail = df['K'].max()
                if 'TP_Count' in df.columns:
                    max_tp = df.loc[df['K'] == max_k_avail, 'TP_Count'].values[0]
                    prec_val = max_tp / target_k
                else:
                    prec_val = (df['Precision'].iloc[-1] * max_k_avail) / target_k
                    
        elif 'y_true' in df.columns and score_col in df.columns:
            df_sorted = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
            df_sorted = df_sorted[df_sorted[score_col] >= 0].copy()
            
            if df_sorted.empty:
                prec_val = 0.0
            else:
                if len(df_sorted) >= target_k:
                    prec_val = df_sorted['y_true'].iloc[:target_k].sum() / target_k
                else:
                    prec_val = df_sorted['y_true'].sum() / target_k
        else:
            print(f"  [Warning] {csv_path} lacks required columns.")
            continue
            
        records.append({
            'Model': model_name,
            'Dataset': dataset_name,
            'Cell_type': cell_type,
            'Precision': prec_val
        })
        
    if not records:
        raise ValueError("No valid Top-K data processed. Please check your manifest.")
        
    plot_df = pd.DataFrame(records)

    summary_df = plot_df.groupby('Model', observed=False).agg(
        Overall_Mean=('Precision', 'mean'),
        SEM=('Precision', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0)
    ).reset_index()
    
    summary_df['ymin'] = summary_df['Overall_Mean'] - summary_df['SEM']
    summary_df['ymax'] = summary_df['Overall_Mean'] + summary_df['SEM']

    model_order = [
        "TRACE", "Convolution", 
        "TranslationAI", "RiboTIE", "RibORF", "RiboTISH",
        "ORF-structure", "Transcription-level"
    ]
    valid_models = [m for m in model_order if m in plot_df['Model'].unique()]
    for m in plot_df['Model'].unique():
        if m not in valid_models:
            valid_models.append(m)
            
    plot_df['Model'] = pd.Categorical(plot_df['Model'], categories=valid_models, ordered=True)
    summary_df['Model'] = pd.Categorical(summary_df['Model'], categories=valid_models, ordered=True)

    # =================================================================
    # [MODIFIED] 获取全局字典的颜色，确保一致性
    # =================================================================
    model_colors = {}
    for m in valid_models:
        model_colors[m] = GLOBAL_MODEL_COLORS.get(m, "#C0C0C0")
            
    unique_cells = plot_df['Cell_type'].unique().tolist()
    unseen_cells = [c for c in unique_cells if 'unseen' in str(c).lower()]
    seen_cells = [c for c in unique_cells if c not in unseen_cells]
    ordered_cells = seen_cells + unseen_cells
    
    plot_df['Cell_type'] = pd.Categorical(plot_df['Cell_type'], categories=ordered_cells, ordered=True)
        
    cell_colors = {}
    for ct in seen_cells:
        cell_colors[ct] = "#202020"
    for ct in unseen_cells:
        cell_colors[ct] = "#D6715E"

    unique_datasets = plot_df['Dataset'].unique().tolist()
    plot_df['Dataset'] = pd.Categorical(plot_df['Dataset'], categories=unique_datasets, ordered=True)
    
    shapes_pool = ['o', '^', 's', 'D', 'v', 'p', 'h', '8']
    dataset_shapes = {}
    for i, ds in enumerate(unique_datasets):
        dataset_shapes[ds] = shapes_pool[i % len(shapes_pool)]

    print(f"Generating Precision@{target_k} Bar Chart...")
    
    p = (
        ggplot()
        + geom_col(
            data=summary_df, 
            mapping=aes(x='Model', y='Overall_Mean', fill='Model'), 
            width=0.7
        )
        + geom_errorbar(
            data=summary_df, 
            mapping=aes(x='Model', ymin='ymin', ymax='ymax'), 
            width=0.2, 
            size=0.8, 
            color="black"
        )
        + geom_jitter(
            data=plot_df, 
            mapping=aes(x='Model', y='Precision', shape='Dataset', color='Cell_type'), 
            width=0.15, 
            size=3.0, 
            stroke=0.8,
            alpha=0.85
        )
        + scale_fill_manual(values=model_colors, guide=None) 
        + scale_shape_manual(values=dataset_shapes, name="Dataset") 
        + scale_color_manual(values=cell_colors, name="Cell type")
        + theme_bw()
        + labs(
            x="",
            y=f"Precision @ K={target_k}"
        )
        + theme(
            axis_text_x=element_text(angle=45, hjust=1, size=12, color="black"),
            axis_text_y=element_text(size=12, color="black"),
            axis_title_y=element_text(size=14, margin={'r': 10}),
            panel_grid_major_x=element_blank(), 
            legend_position="right",
            legend_title=element_text(size=13, fontweight='bold'),
            legend_text=element_text(size=11)
        )
    )

    file_suffix = f".{suffix}" if suffix else ""
    save_path = os.path.join(out_dir, f"precision_at_{target_k}_bar{file_suffix}.pdf")
    p.save(save_path, width=7, height=5, dpi=300, verbose=False)
    print(f"✅ Bar chart saved to: {save_path}")

    return summary_df, plot_df