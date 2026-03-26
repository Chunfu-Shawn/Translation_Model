import numpy as np

def compute_pif(cds_signal_real, eps=0.1):
    """计算 P-sites in frame 1 (PIF)"""
    if len(cds_signal_real) == 0:
        return 0
    f1_sum = np.sum(cds_signal_real[0::3])
    total_sum = np.sum(cds_signal_real) + eps
    return f1_sum / total_sum if total_sum > 0 else 0

def compute_uniformity(f1_signal_real):
    """计算信号分布均匀度 (Uniformity)"""
    if len(f1_signal_real) == 0:
        return 0
    num_nonzero = np.count_nonzero(f1_signal_real > 0)
    return num_nonzero / len(f1_signal_real)

def compute_dropoff(full_pred, cds_start, cds_end, window=15, eps=0.1):
    """
    计算 Ribosome Release Score (Drop-off)。
    注意：这里遵循你代码中的逻辑，Drop-off 使用原始 pred 信号（通常是 log 空间）进行求和。
    """
    win_s = max(0, cds_end - window)
    win_e = min(len(full_pred), cds_end + window)
    
    # 建立索引序列
    win_idx = np.arange(win_s, win_e)
    win_before_idx = np.arange(win_s, cds_end)
    
    # 提取窗口内属于 Frame 1 的位置 (相对于 cds_start 的偏移)
    f1_win_idx = win_idx[(win_idx - cds_start) % 3 == 0]
    f1_win_sum = np.sum(full_pred[f1_win_idx]) + eps
    
    f1_win_before_idx = win_before_idx[(win_before_idx - cds_start) % 3 == 0]
    f1_win_before = np.sum(full_pred[f1_win_before_idx]) + eps
    
    return f1_win_before / f1_win_sum if f1_win_sum > eps else 0