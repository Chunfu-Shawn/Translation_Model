import pickle
import numpy as np
import os
import pandas as pd
import torch
from plotnine import *
from tqdm import tqdm

def evaluate_metagene_TIS_TTS_profile(
        dataset, pkl_path, 
        window_size=12, out_dir="./results/plots", 
        suffix="", unlog_data=True):
    """
    评估模型在 TIS (起始) 和 TTS (终止) 附近的聚合分布 (Meta-gene Profile)。
    
    Args:
        dataset: 包含真实数据和元信息的 TranslationDataset 实例
        pkl_path: 预测结果 pickle 文件路径, 结构为 {cell_type: {tid: signal}}
        window_size: 观察窗口大小 (例如 12nt 表示观察 -12 到 +11)
        out_dir: 图片保存路径
        unlog_data: 是否使用 expm1 将 log 数据还原为线性计数 (强烈建议还原后再求和)
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")
    
    print(f"Loading predictions from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        preds_data = pickle.load(f)
        
    os.makedirs(out_dir, exist_ok=True)
    
    # 存储所有样本的原始切片，用于最后【汇总】
    tis_slices_gt = []
    tis_slices_pred = []
    tts_slices_gt = []
    tts_slices_pred = []
    
    valid_count = 0
    print(f"Scanning {len(dataset)} transcripts in dataset...")
    
    # ==========================================
    # 1. 遍历 Dataset 并匹配提取数据
    # ==========================================
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        uuid = str(sample[0])
        
        # 解析 UUID (tid-cell_type-order)
        parts = uuid.split("-")
        tid = parts[0]
        cell_type = parts[1] if len(parts) > 1 else "Unknown"
        
        # 从 pkl 中获取预测信号
        if cell_type not in preds_data or tid not in preds_data[cell_type]:
            continue
            
        prediction = preds_data[cell_type][tid].reshape(-1).astype(np.float32)
        
        # 从 dataset 获取真实信号和 meta
        meta = sample[4]
        count_emb = sample[6]
        
        if isinstance(count_emb, torch.Tensor):
            truth = count_emb.detach().cpu().numpy()
        else:
            truth = count_emb
            
        # 压缩维度
        if len(truth.shape) > 1 and truth.shape[1] > 1:
            truth = np.sum(truth, axis=1)
        elif len(truth.shape) > 1 and truth.shape[1] == 1:
            truth = truth.flatten()
        truth = truth.astype(np.float32)
        
        # 还原 Log 空间 (计算 metagene 时在线性空间相加才符合物理意义)
        if unlog_data:
            truth = np.expm1(truth)
            prediction = np.expm1(prediction)
        
        # 获取 CDS 信息
        start_idx = int(meta.get('cds_start_pos', -1))
        end_idx = int(meta.get('cds_end_pos', -1))
        
        if start_idx == -1 or end_idx == -1:
            continue
        start_idx = start_idx - 1 # 根据你的原始逻辑修正为 0-based
        
        # 检查长度防止越界
        if start_idx < window_size or (end_idx + window_size) > len(truth):
            continue
        if (end_idx - start_idx) < window_size: 
            continue

        # --- TIS 切片 ---
        t_slice_tis = truth[start_idx - window_size : start_idx + window_size]
        p_slice_tis = prediction[start_idx - window_size : start_idx + window_size]
        
        # --- TTS 切片 ---
        stop_codon_start = end_idx - 3
        t_slice_tts = truth[stop_codon_start - window_size : stop_codon_start + window_size]
        p_slice_tts = prediction[stop_codon_start - window_size : stop_codon_start + window_size]

        # 【修改点 1】：不再逐个转录本 Normalization，直接 append 原始信号
        tis_slices_gt.append(t_slice_tis)
        tis_slices_pred.append(p_slice_tis)
        tts_slices_gt.append(t_slice_tts)
        tts_slices_pred.append(p_slice_tts)
        
        valid_count += 1

    print(f"Used {valid_count} valid transcripts for meta-gene analysis.")
    if valid_count == 0:
        print("No valid transcripts found.")
        return

    # ==========================================
    # 2. 聚合后再归一化 (压制低表达噪音)
    # ==========================================
    eps = 1e-6
    
    # TIS
    sum_tis_gt = np.sum(np.array(tis_slices_gt), axis=0)
    sum_tis_pred = np.sum(np.array(tis_slices_pred), axis=0)
    
    norm_tis_gt = sum_tis_gt / (np.sum(sum_tis_gt) + eps)
    norm_tis_pred = sum_tis_pred / (np.sum(sum_tis_pred) + eps)
    
    # TTS
    sum_tts_gt = np.sum(np.array(tts_slices_gt), axis=0)
    sum_tts_pred = np.sum(np.array(tts_slices_pred), axis=0)
    
    norm_tts_gt = sum_tts_gt / (np.sum(sum_tts_gt) + eps)
    norm_tts_pred = sum_tts_pred / (np.sum(sum_tts_pred) + eps)

    # ==========================================
    # 3. 使用 Plotnine 绘图
    # ==========================================
    x = np.arange(-window_size, window_size)
    
    # 构建 DataFrame
    df_tis_gt = pd.DataFrame({'Position': x, 'Density': norm_tis_gt, 'Source': 'Ground Truth', 'Region': 'TIS (Start)'})
    df_tis_pred = pd.DataFrame({'Position': x, 'Density': norm_tis_pred, 'Source': 'Prediction', 'Region': 'TIS (Start)'})
    
    df_tts_gt = pd.DataFrame({'Position': x, 'Density': norm_tts_gt, 'Source': 'Ground Truth', 'Region': 'TTS (Stop)'})
    df_tts_pred = pd.DataFrame({'Position': x, 'Density': norm_tts_pred, 'Source': 'Prediction', 'Region': 'TTS (Stop)'})
    
    df = pd.concat([df_tis_gt, df_tis_pred, df_tts_gt, df_tts_pred])
    
    # 指定 Source 的顺序，让图例固定
    df['Source'] = pd.Categorical(df['Source'], categories=['Ground Truth', 'Prediction'])
    
    # 准备背景高亮的 DataFrame
    # 1. CDS 区域 (TIS后, TTS前)
    annot_cds = pd.DataFrame([
        {'Region': 'TIS (Start)', 'xmin': 0, 'xmax': window_size},
        {'Region': 'TTS (Stop)', 'xmin': -window_size, 'xmax': 0}
    ])
    # 2. 起始/终止密码子区域 (0-3)
    annot_codon = pd.DataFrame([
        {'Region': 'TIS (Start)', 'xmin': 0, 'xmax': 3, 'type': 'Start Codon', 'color': 'limegreen'},
        {'Region': 'TTS (Stop)', 'xmin': 0, 'xmax': 3, 'type': 'Stop Codon', 'color': '#e74c3c'}
    ])

    file_suffix = f".{suffix}" if suffix else ""
    # plot_title = f"Metagene Profile (n={valid_count})"
    
    # 绘图逻辑
    p = (
        ggplot(df, aes(x='Position', y='Density'))
        
        # 添加 CDS 背景 (透明灰)
        + geom_rect(
            data=annot_cds, 
            mapping=aes(xmin='xmin', xmax='xmax', ymin=-np.inf, ymax=np.inf),
            fill='gray', alpha=0.15, inherit_aes=False
        )
        
        # 添加密码子背景 (透明绿/红)
        + geom_rect(
            data=annot_codon, 
            mapping=aes(xmin='xmin', xmax='xmax', ymin=-np.inf, ymax=np.inf, fill='color'),
            alpha=0.2, inherit_aes=False, show_legend=False
        )
        + scale_fill_identity() # 使 DataFrame 里的 color 列名称直接生效
        
        # 绘制主曲线
        + geom_line(aes(color='Source', linetype='Source'), size=1)
        + scale_color_manual(values={'Ground Truth': 'darkgray', 'Prediction': '#005b96'})
        + scale_linetype_manual(values={'Ground Truth': 'solid', 'Prediction': 'dashed'})
        
        # 分面
        + facet_wrap('~Region', ncol=2, scales='free_y')
        
        # 样式与主题
        + theme_bw()
        + theme(
            strip_background=element_blank(),
            strip_text=element_text(size=11),
            legend_title=element_blank(),
            legend_text=element_text(size=11),
            legend_position='top'
        )
        + labs(
            # title=plot_title,
            x="Distance from Start/Stop Codon (nt)", 
            y="Normalized P-site Density"
        )
    )

    save_file = os.path.join(out_dir, f"metagene_tis_tts_profile{file_suffix}.pdf")
    p.save(save_file, width=8, height=4, verbose=False)
    print(f"Metagene plot saved to: {save_file}")