import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_coding_prediction(
    pkl_file_path: str, 
    target_tid: str, 
    gt_file_path: str = None, 
    cell_type: str =  "brain_fetal",
    out_dir: str = "./results/plots",
    width = 10,
    height = 4
):
    """
    可视化特定转录本的 TIS 和 TTS 预测分数。
    如果有 Ground Truth，则在主图下方添加一个独立的轨道标注真实的 CDS 位置。
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. 加载预测数据
    print(f"Loading predictions from {pkl_file_path}...")
    with open(pkl_file_path, 'rb') as f:
        predictions = pickle.load(f)
    
    if cell_type not in predictions:
        print(f"❌ Error: Cell type '{cell_type}' not found in the prediction file.")
        return
    cell_preds = predictions[cell_type]

    # 在 pkl 的 key 中精确寻找指定的转录本 (处理 ENST00000350669-Liver-0 这种 UUID)
    target_uuid = None
    target_tid_clean = target_tid.split('.')[0]
    
    for uuid in cell_preds.keys():
        if str(uuid).split('-')[0].split('.')[0] == target_tid_clean:
            target_uuid = uuid
            break
            
    if not target_uuid:
        print(f"❌ Error: Transcript {target_tid} not found in the prediction file.")
        return
        
    print(f"Found target: {target_uuid}")
    pred_array = cell_preds[target_uuid] # shape: (L, 2)
    pred_array = np.nan_to_num(pred_array, nan=0.0) # 替换潜在的 NaN
    
    seq_len = pred_array.shape[0]
    x_positions = np.arange(seq_len)
    tis_scores = pred_array[:, 0]
    tts_scores = pred_array[:, 1]

    # 2. 加载并匹配 Ground Truth (如果提供)
    gt_regions = []
    if gt_file_path and os.path.exists(gt_file_path):
        try:
            gt_df = pd.read_csv(gt_file_path, sep='\t' if '\t' in open(gt_file_path).read(1024) else ',')
            gt_df['tid_clean'] = gt_df['tid'].apply(lambda x: str(x).split('.')[0])
            matched_gt = gt_df[gt_df['tid_clean'] == target_tid_clean]
            
            for _, row in matched_gt.iterrows():
                # 转换为 0-based 坐标用于绘图
                gt_regions.append({
                    'start': int(row['Start_pos']) - 1,
                    'end': int(row['End_pos']) - 1
                })
        except Exception as e:
            print(f"⚠️ Warning: Could not process Ground Truth file: {e}")

    # 3. 开始绘图布局设计
    plt.style.use('default')
    # 如果有 GT，将画布分为上下两部分 (高度比 5:1)
    if gt_regions:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, height), gridspec_kw={'height_ratios': [5, 1]}, sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(width, height))
        ax2 = None

    # ==========================================
    # Ax1: 预测概率主图 (Barplots)
    # ==========================================
    # 使用 width=1.0 使得柱子紧密排列，alpha=0.7 允许重叠区域透色
    ax1.bar(x_positions, tis_scores, width=1.0, color='#00BFC4', alpha=0.7, label='Start (TIS) Prob')
    ax1.bar(x_positions, tts_scores, width=1.0, color='#F8766D', alpha=0.7, label='Stop (TTS) Prob')
    
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(0, seq_len)
    ax1.set_ylabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax1.set_title(f'Translation Coding Prediction: {target_uuid}', fontsize=14, pad=15)
    ax1.legend(loc='upper right', frameon=True, edgecolor='black')
    
    # 美化边框和网格
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    # ax1.spines['top'].set_visible(False)
    # ax1.spines['right'].set_visible(False)

    # ==========================================
    # Ax2: 真实的 CDS 轨道 (如果有)
    # ==========================================
    if ax2:
        # 画一条代表转录本全长的灰色主轴
        ax2.plot([0, seq_len], [0, 0], color='lightgray', linewidth=3, zorder=1) 
        
        # 绘制真实的 CDS 方块
        for i, region in enumerate(gt_regions):
            s, e = region['start'], region['end']
            # 添加金色的矩形块代表真实的蛋白编码区
            rect = patches.Rectangle(
                (s, -0.3), e - s, 0.6, 
                linewidth=1.5, edgecolor='black', facecolor='#FFD700', zorder=2
            )
            ax2.add_patch(rect)
            
            # 在方块中央添加文字
            ax2.text((s + e) / 2, 0.45, f'True CDS {i+1}' if len(gt_regions)>1 else 'True CDS', 
                     ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
                     
        # 隐藏 y 轴坐标，只保留轨道视觉
        ax2.set_ylim(-1, 1)
        ax2.set_yticks([])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        
        # 设置 x 轴
        ax2.set_xlabel('Transcript Position (nt)', fontsize=12, fontweight='bold')
    else:
        ax1.set_xlabel('Transcript Position (nt)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    
    # 4. 保存输出
    save_path = os.path.join(out_dir, f"{target_uuid}_predictions.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✅ Plot successfully saved to: {save_path}")