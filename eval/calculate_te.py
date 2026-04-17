import numpy as np

def calculate_morf_signal_ratio(density_array, m_start, m_end, eps=1e-6):
    """
    计算翻译效率 (TE)
    """
    if m_start >= len(density_array): 
        return 0.0
    
    valid_end = min(len(density_array), m_end)
    if m_start >= valid_end: 
        return 0.0

    morf_sum = np.sum(density_array[m_start:valid_end]) + eps
    total_sum = np.sum(density_array) + eps

    return morf_sum / total_sum

def calculate_morf_mean_signal_ratio(density_array, m_start, m_end, eps=1e-6):
    """
    计算翻译效率 (TE)
    """
    if m_start >= len(density_array): 
        return 0.0
    
    valid_end = min(len(density_array), m_end)
    if m_start >= valid_end: 
        return 0.0

    morf_sum = np.mean(density_array[m_start:valid_end]) + eps
    total_sum = np.mean(density_array) + eps

    return morf_sum / total_sum

def calculate_morf_mean_signal(density_array, m_start, m_end, eps=1e-6):
    """
    计算翻译效率 (TE)
    """
    if m_start >= len(density_array): 
        return 0.0
    
    valid_end = min(len(density_array), m_end)
    if m_start >= valid_end: 
        return 0.0

    morf_mean = np.mean(density_array[m_start:valid_end]) + eps

    return morf_mean

def calculate_morf_median_signal(density_array, m_start, m_end, eps=1e-6):
    """
    计算翻译效率 (TE)
    """
    if m_start >= len(density_array): 
        return 0.0
    
    valid_end = min(len(density_array), m_end)
    if m_start >= valid_end: 
        return 0.0
    nonzero_arrary = density_array[density_array > 0]
    morf_median = np.median(nonzero_arrary[m_start:valid_end]) + eps

    return morf_median

def calculate_sum_signal(density_array, m_start, m_end, eps=1e-6):
    """
    计算总翻译负载
    """
    if m_start >= len(density_array): 
        return 0.0
    valid_end = min(len(density_array), m_end)
    if m_start >= valid_end: 
        return 0.0
    
    return np.sum(density_array[m_start:valid_end]) + eps

def calculate_mean_signal(density_array, eps=1e-6):
    """
    计算平均翻译效率
    """

    return np.mean(density_array) + eps
