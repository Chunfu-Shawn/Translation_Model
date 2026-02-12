import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

def evaluate_metagene_TIS_TTS_profile(
        pkl_path, window_size=12, out_dir="./results/plots", mask_ratio=1, suffix=""):
    """
    评估模型在 TIS (起始) 和 TTS (终止) 附近的聚合分布 (Meta-gene Profile)。
    
    Args:
        pkl_path: 预测结果 pickle 文件路径
        window_size: 观察窗口大小 (例如 12nt 表示观察 -12 到 +12)
        out_dir: 图片保存路径
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")
    
    print(f"Loading data from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    os.makedirs(out_dir, exist_ok=True)
    
    # 初始化累加器
    # 长度 = window_size (左) + 1 (中心) + window_size (右)
    span = window_size * 2
    
    # 存储所有样本的切片，最后取平均
    tis_slices_gt = []
    tis_slices_pred = []
    tts_slices_gt = []
    tts_slices_pred = []
    
    valid_count = 0
    
    print(f"Aggregating profiles for {len(data)} transcripts...")
    
    for uuid, sample in data.items():
        # 获取基础信息
        truth = sample['truth'].reshape(-1).astype(np.float32)
        cds_info = sample.get('cds_info', None)
        
        # 如果没有 CDS 信息或者 CDS 太短，跳过
        if cds_info is None:
            continue

        start_idx = cds_info['start']
        end_idx = cds_info['end']
        if start_idx == -1 or end_idx == -1:
            continue
        else:
            start_idx = start_idx - 1
        
        # 简单的长度检查，防止切片越界
        if start_idx < window_size or (end_idx + window_size) > len(truth):
            continue
        if (end_idx - start_idx) < window_size: # CDS 太短也不要
            continue

        prediction = sample['ratios'][mask_ratio]['pred'].reshape(-1).astype(np.float32)

        # --- TIS 切片 (Around Start) ---
        t_slice_tis = truth[start_idx - window_size : start_idx + window_size]
        p_slice_tis = prediction[start_idx - window_size : start_idx + window_size]
        
        # --- TTS 切片 (Around Stop) ---
        # 我们希望 0 点对齐到 Stop Codon 的第一个碱基
        # cds_info['end'] 通常是 CDS 结束位置（不包含）。Stop codon 是 [end-3 : end]
        # 所以 Stop Codon Start 是 end_idx - 3
        stop_codon_start = end_idx - 3
        t_slice_tts = truth[stop_codon_start - window_size : stop_codon_start + window_size]
        p_slice_tts = prediction[stop_codon_start - window_size : stop_codon_start + window_size]

        # --- 归一化 (关键步骤) ---
        # 为了让不同表达量的基因可以平均，我们除以该窗口内的总信号
        # 加一个极小值 eps 防止除零
        eps = 1e-6
        
        # Normalize TIS
        norm_t_tis = t_slice_tis / (np.sum(t_slice_tis) + eps)
        norm_p_tis = p_slice_tis / (np.sum(p_slice_tis) + eps)
        
        # Normalize TTS
        norm_t_tts = t_slice_tts / (np.sum(t_slice_tts) + eps)
        norm_p_tts = p_slice_tts / (np.sum(p_slice_tts) + eps)

        tis_slices_gt.append(norm_t_tis)
        tis_slices_pred.append(norm_p_tis)
        tts_slices_gt.append(norm_t_tts)
        tts_slices_pred.append(norm_p_tts)
        
        valid_count += 1

    print(f"Used {valid_count} valid transcripts for meta-gene analysis.")
    
    if valid_count == 0:
        print("No valid transcripts found (check CDS info or length).")
        return

    # --- 计算平均 Profile ---
    avg_tis_gt = np.mean(np.array(tis_slices_gt), axis=0)
    avg_tis_pred = np.mean(np.array(tis_slices_pred), axis=0)
    
    avg_tts_gt = np.mean(np.array(tts_slices_gt), axis=0)
    avg_tts_pred = np.mean(np.array(tts_slices_pred), axis=0)
    
    # --- 绘图 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # X 轴坐标: -12 到 +12
    x = np.arange(-window_size, window_size)
    
    # 设置颜色
    color_gt = 'darkgray'
    color_pred = '#3498db' # Blue
    
    # === Panel 1: TIS (Start) ===
    # 绘制 GT 和 Pred
    ax1.plot(x, avg_tis_gt, color=color_gt, label='Ground Truth', linewidth=2, alpha=0.7)
    ax1.plot(x, avg_tis_pred, color=color_pred, label='Prediction', linewidth=2, linestyle='--')
    
    # 填充背景标示 CDS 区域
    # TIS 之后是 CDS (0 到 window_size)
    ax1.axvspan(0, window_size, color='lightgray', alpha=0.3, label='CDS Region')
    # 标记 Stop Codon 位置 (0-3)
    ax1.axvspan(0, 3, color='limegreen', alpha=0.2, label='Start Codon')
    
    ax1.set_title(f"Metagene Profile around TIS (Start)\n(n={valid_count})")
    ax1.set_xlabel("Distance from Start Codon (nt)")
    ax1.set_ylabel("Normalized P-site Density")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)

    # === Panel 2: TTS (Stop) ===
    # 绘制 GT 和 Pred
    ax2.plot(x, avg_tts_gt, color=color_gt, label='Ground Truth', linewidth=2, alpha=0.7)
    ax2.plot(x, avg_tts_pred, color=color_pred, label='Prediction', linewidth=2, linestyle='--')
    
    # 填充背景标示 CDS 区域
    # Stop Codon Start 之前是 CDS (-window_size 到 0)
    # 0 到 3 是 Stop Codon (通常也算在 CDS 结构内，但在翻译上是终止点)
    # 3 之后是 UTR
    ax2.axvspan(-window_size, 0, color='lightgray', alpha=0.3, label='CDS Region')
    # 标记 Stop Codon 位置 (0-3)
    ax2.axvspan(0, 3, color='#e74c3c', alpha=0.2, label='Stop Codon')
    
    ax2.set_title(f"Metagene Profile around TTS (Stop)\n(n={valid_count})")
    ax2.set_xlabel("Distance from Stop Codon Start (nt)")
    ax2.set_ylabel("Normalized P-site Density")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    save_file = os.path.join(out_dir, f"metagene_tis_tts_profile.{suffix}.pdf")
    plt.savefig(save_file)
    plt.close()
    print(f"Metagene plot saved to: {save_file}")