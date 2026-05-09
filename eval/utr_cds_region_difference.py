import pickle
import numpy as np
import pandas as pd
import os
from typing import Union, Dict
from tqdm import tqdm
from plotnine import *

def calculate_region_metrics(signal_array, global_start_idx, region_start, region_end, total_transcript_sum, threshold=0.01):
    """
    Calculate Proportion, Coverage, and Periodicity for a specific region.
    """
    # 1. Slice region
    if region_end <= region_start:
        return None 
        
    region_data = signal_array[region_start:region_end]
    L = len(region_data)
    
    if L < 3: return None

    # 2. [Modified] Proportion (Sum of region / Sum of total transcript)
    # Avoid division by zero
    if total_transcript_sum < 1e-6:
        proportion = 0.0
    else:
        proportion = np.sum(region_data) / total_transcript_sum
    
    # 3. Coverage
    coverage = np.sum(region_data > threshold) / L
    
    # 4. Periodicity
    global_indices = np.arange(region_start, region_end)
    frames = (global_indices - global_start_idx) % 3
    
    f0_sum = np.sum(region_data[frames == 0])
    region_sum = np.sum(region_data)
    
    if region_sum < 1e-6:
        periodicity = np.nan
    else:
        periodicity = f0_sum / region_sum
        
    return {
        'Proportion': proportion, # Renamed from Density
        'Coverage': coverage,
        'Periodicity': periodicity
    }


def evaluate_region_specificity(
        truth_dataset, 
        pkl_input: Union[Dict[str, str], str], 
        out_dir: str = "./results/plots", 
        suffix: str = ""):
    """
    Compare P-site Proportion, Coverage, and Periodicity across 5'UTR, CDS, and 3'UTR.
    Handles nested dictionary pkl formats: {cell_type: {tid: prediction}}
    """
    
    # 1. 灵活加载预测文件 (直接沿用你的参考逻辑)
    print(">>> Loading prediction files...")
    all_predictions = {}
    
    if isinstance(pkl_input, str):
        print(f"  - Loading combined predictions from: {pkl_input}")
        with open(pkl_input, 'rb') as f:
            loaded_data = pickle.load(f)
            if isinstance(loaded_data, dict):
                all_predictions = loaded_data
            else:
                raise ValueError("The provided single pickle file does not contain a dictionary.")
    elif isinstance(pkl_input, dict):
        for cell_type, pkl_path in pkl_input.items():
            print(f"  - Loading predictions for {cell_type}: {pkl_path}")
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                if cell_type in data and isinstance(data[cell_type], dict):
                    all_predictions[cell_type] = data[cell_type]
                else:
                    all_predictions[cell_type] = data
    else:
        raise TypeError("pkl_input must be either a file path (str) or a dictionary mapping (dict).")
        
    os.makedirs(out_dir, exist_ok=True)
    metrics_data = []
    
    # 2. 遍历 Dataset 提取真值与预测值进行对比
    print(f"\n>>> Analyzing region specificity...")
    
    for i in tqdm(range(len(truth_dataset))):
        uuid, species, cell_type, expr_array, meta_info, seq_emb, count_emb = truth_dataset[i]
        uuid_str = str(uuid)
        
        parts = uuid_str.split('-')
        if len(parts) < 2:
            continue
            
        tid = parts[0]
        cell_type = parts[1]
        
        if cell_type not in all_predictions:
            continue
            
        predictions = all_predictions[cell_type]
        
        # 处理带有版本号的 TID 回退
        lookup_tid = tid
        if lookup_tid not in predictions:
            tid_no_version = tid.split('.')[0]
            if tid_no_version in predictions:
                lookup_tid = tid_no_version
            else:
                continue
                
        # 3. 提取信号
        pred_signal = predictions[lookup_tid]
        gt_signal = count_emb.numpy().flatten()
        
        # 假定此处需要进行反 log1p 转换（如果你预测的是线性数据，请去掉 np.expm1）
        pred_linear = np.expm1(pred_signal.astype(np.float32))
        truth_linear = np.expm1(gt_signal.astype(np.float32))
        
        pred_len = len(pred_linear)
        gt_len = len(truth_linear)
        
        # 4. 获取 CDS 坐标信息
        cds_s = int(meta_info.get("cds_start_pos", -1))
        cds_e = int(meta_info.get("cds_end_pos", -1))
        has_cds = (cds_s != -1 and cds_e != -1)
        
        if not has_cds:
            continue
            
        # 按照参考代码逻辑划定 CDS 的准确边界
        start_idx = max(0, cds_s - 1)
        end_idx = cds_e + 3
        cds_len = end_idx - start_idx
        
        # 5. 校验：只有 "全长 (Full-length)" 预测才能评估 UTR 分布
        is_pred_full = abs(pred_len - gt_len) < abs(pred_len - cds_len)
        if not is_pred_full:
            continue
            
        # 对齐长度（防止极端情况下的越界）
        min_len = min(pred_len, gt_len)
        if min_len < 2 or end_idx >= min_len: 
            continue
            
        pred_aligned = pred_linear[:min_len]
        truth_aligned = truth_linear[:min_len]
        
        total_sum_gt = np.sum(truth_aligned)
        total_sum_pred = np.sum(pred_aligned)
        
        # 6. 定义三大区域 (Regions)
        regions = {
            '5\'UTR': (0, start_idx),
            'CDS': (start_idx, end_idx),
            '3\'UTR': (end_idx, min_len)
        }
        
        # 7. 计算评价指标
        for region_name, (r_start, r_end) in regions.items():
            # Ground Truth
            m_gt = calculate_region_metrics(truth_aligned, start_idx, r_start, r_end, total_sum_gt)
            if m_gt:
                m_gt['Condition'] = 'Ground Truth'
                m_gt['Region'] = region_name
                m_gt['UUID'] = uuid_str
                metrics_data.append(m_gt)
            
            # Prediction
            m_pred = calculate_region_metrics(pred_aligned, start_idx, r_start, r_end, total_sum_pred)
            if m_pred:
                m_pred['Condition'] = 'Prediction'
                m_pred['Region'] = region_name
                m_pred['UUID'] = uuid_str
                metrics_data.append(m_pred)

    # 8. 整理与保存
    df = pd.DataFrame(metrics_data)
    if df.empty:
        print("Warning: No valid transcripts found for region evaluation.")
        return df
        
    csv_path = os.path.join(out_dir, f"region_specificity_stats.{suffix}.csv" if suffix else "region_specificity_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n>>> Stats saved to {csv_path}")
    
    # 调用原有的画图函数
    plot_region_comparison(df, out_dir, suffix)
    
    return df

def plot_region_comparison(df, out_dir, suffix):
    """
    使用 plotnine 绘制分面箱线图。
    特点：白色填充、彩色边框、紧凑布局。
    """
    # 1. 数据转换：宽格式 -> 长格式 (Melt)
    # 将 Proportion, Coverage, Periodicity 合并到一列 'Value'，用 'Metric' 区分
    plot_df = df.melt(
        id_vars=['UUID', 'Condition', 'Region'], 
        value_vars=['Proportion', 'Coverage', 'Periodicity'],
        var_name='Metric', 
        value_name='Value'
    )
    
    # 2. 设定 Region 的顺序 (Categorical)
    plot_df['Region'] = pd.Categorical(
        plot_df['Region'], 
        categories=['5\'UTR', 'CDS', '3\'UTR'], 
        ordered=True
    )
    plot_df['Metric'] = pd.Categorical(
        plot_df['Metric'], 
        categories=['Proportion', 'Coverage', 'Periodicity'], 
        ordered=True
    )
    
    # 3. 颜色定义
    colors = {'Ground Truth': '#555555', 'Prediction': '#3498db'}
    
    # 4. 绘图
    p = (
        ggplot(plot_df, aes(x='Region', y='Value', color='Condition'))
        # 箱线图
        + geom_boxplot(
            fill='white',        # 内部白色填充
            size=0.8,            # 边框稍粗 (默认是0.5)
            outlier_shape=None,  # 不显示离群点
            outlier_alpha=0,     # 彻底隐藏离群点
            width=0.6,           # 箱体宽度
            position=position_dodge(width=0.8) # 调整并排间距
        )
        + facet_wrap('~Metric', scales='free_y', nrow=1)
        + scale_color_manual(values=colors)
        + theme_bw()
        + theme(
            legend_position='top',      # 图例放上面节省横向空间
            legend_title=element_blank(), # 去掉图例标题
            axis_title_x=element_blank(), # 去掉 X 轴标题 "Region"
            axis_title_y=element_blank(), # 去掉 Y 轴标题 "Value"
            strip_background=element_blank(),
            strip_text=element_text(weight='bold', size=10) # 分面标题文字
        )
    )
    
    # 保存
    save_path = os.path.join(out_dir, f"region_specificity_comparison.{suffix}.pdf")
    p.save(filename=save_path, width=6, height=4, verbose=False)
    print(f"Plotnine comparison plot saved to {save_path}")