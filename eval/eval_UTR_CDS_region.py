import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from plotnine import *

def calculate_region_metrics(signal_array, global_start_idx, region_start, region_end, total_transcript_sum, threshold=0.05):
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
        pkl_path, out_dir="./results/plots", mask_ratio=1, suffix=""):
    """
    Compare P-site Proportion, Coverage, and Periodicity across 5'UTR, CDS, and 3'UTR.
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")
        
    print(f"Loading data from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    os.makedirs(out_dir, exist_ok=True)
    
    metrics_data = []
    
    print(f"Analyzing regions ...")
    
    for uuid, sample in tqdm(data.items(), desc="Processing transcripts"):
        # Get Data
        truth_linear = np.expm1(sample['truth'].reshape(-1).astype(np.float32))
        
        if mask_ratio not in sample['ratios']: # Ensure assuming mask ratio 1.0 for reconstruction
            continue
            
        pred_linear = np.expm1(sample['ratios'][mask_ratio]['pred'].reshape(-1).astype(np.float32))
        
        # Calculate Total Sums for Proportion
        total_sum_gt = np.sum(truth_linear)
        total_sum_pred = np.sum(pred_linear)

        # Get CDS boundaries
        cds_info = sample.get('cds_info', None)
        if cds_info is None: 
            continue
        if cds_info['start'] == -1 or cds_info['end'] == -1:
            continue
        
        cds_start = cds_info['start'] - 1 # Assuming 1-based in pickle, converting to 0-based
        cds_end = cds_info['end']
        seq_len = len(truth_linear)
        
        regions = {
            '5\'UTR': (0, cds_start),
            'CDS': (cds_start, cds_end),
            '3\'UTR': (cds_end, seq_len)
        }
        
        for region_name, (r_start, r_end) in regions.items():
            # Ground Truth
            m_gt = calculate_region_metrics(truth_linear, cds_start, r_start, r_end, total_sum_gt)
            if m_gt:
                m_gt['Condition'] = 'Ground Truth'
                m_gt['Region'] = region_name
                m_gt['UUID'] = uuid
                metrics_data.append(m_gt)
            
            # Prediction
            m_pred = calculate_region_metrics(pred_linear, cds_start, r_start, r_end, total_sum_pred)
            if m_pred:
                m_pred['Condition'] = 'Prediction'
                m_pred['Region'] = region_name
                m_pred['UUID'] = uuid
                metrics_data.append(m_pred)

    df = pd.DataFrame(metrics_data)
    
    csv_path = os.path.join(out_dir, f"region_specificity_stats.{suffix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Stats saved to {csv_path}")
    
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
            figure_size=(8, 4),        # 紧凑尺寸 (宽, 高)
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
    p.save(filename=save_path, dpi=300, verbose=False)
    print(f"Plotnine comparison plot saved to {save_path}")