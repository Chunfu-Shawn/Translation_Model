import numpy as np

def calculate_morf_efficiency(density_array, m_start, m_end, eps=1e-6):
    """
    计算翻译效率 (TE)
    """
    if m_start >= len(density_array): 
        return 0.0
    
    valid_end = min(len(density_array), m_end)
    if m_start >= valid_end: 
        return 0.0

    morf_sum = np.sum(density_array[m_start:valid_end])
    total_sum = np.sum(density_array) + eps

    return morf_sum / total_sum

def calculate_morf_mean_efficiency(density_array, m_start, m_end, eps=1e-6):
    """
    计算翻译效率 (TE)
    """
    if m_start >= len(density_array): 
        return 0.0
    
    valid_end = min(len(density_array), m_end)
    if m_start >= valid_end: 
        return 0.0

    morf_sum = np.mean(density_array[m_start:valid_end])
    total_sum = np.mean(density_array) + eps

    return morf_sum / total_sum

def calculate_morf_mean_density(density_array, m_start, m_end, eps=1e-6):
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


def calculate_mean_efficiency(density_array, eps=1e-6):
    """
    计算平均翻译效率
    """

    return np.mean(density_array) + eps
