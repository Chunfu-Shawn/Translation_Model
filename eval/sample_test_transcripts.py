import numpy as np
import torch
from tqdm import tqdm


def get_samples_with_clean_utr(
        dataset, top_n=50, utr_len_range=[50, 800], 
        forbidden_motifs=['ATG', 'TGA', 'TAA', 'TAG'], check_region=[50, 800]):
    """
    筛选指定 5'UTR 长度范围且天然不含特定 Motif 的样本 (不进行突变)。

    Args:
        dataset: 你的数据集对象
        top_n: 返回的最大样本数 (按 UTR 长度降序排列后取前 N 个)
        utr_len_range: [min_len, max_len], 筛选 5'UTR 长度在此区间的样本
        forbidden_motifs: List[str], 如果 5'UTR 中包含这些 Motif，则丢弃该样本。

    Returns:
        List of dicts: 筛选出的样本列表 (原始数据，未修改)
    """
    candidates = []
    min_len, max_len = utr_len_range
    print(f"Scanning dataset for natural clean 5'UTRs (Length: {min_len}-{max_len})...")
    print(f"Rejecting 5'UTRs containing: {forbidden_motifs}")
    
    # 1. 预处理 Motif 匹配规则
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, 'N': 4}
    
    # 将字符串 motif 转换为索引元组
    target_patterns = set()
    for m in forbidden_motifs:
        m = m.upper()
        if all(b in base_map for b in m):
            pattern = tuple(base_map[b] for b in m)
            target_patterns.add(pattern)

    for i in tqdm(range(len(dataset))):
        try:
            item = dataset[i]
            # 根据实际 dataset 结构解包
            cell_type = item[0]
            cds_pos = item[1]
            seq_emb = item[3] 
            uuid = dataset.uuids[i]
            
            # 解析 CDS Start
            start = int(cds_pos[0]) - 1 # 0-based index of ATG start
            end = int(cds_pos[1])
            
            # 1. 长度筛选
            if not (min_len <= start <= max_len):
                continue
            
            # 2. 获取 Numpy Embedding (Copy)
            if isinstance(seq_emb, torch.Tensor):
                seq_emb_np = seq_emb.cpu().numpy()
            else:
                seq_emb_np = seq_emb.copy()
            
            # 维度调整
            if seq_emb_np.shape[0] == 4 and seq_emb_np.shape[1] > 4:
                seq_emb_np = seq_emb_np.T 

            # 3. 提取 5'UTR 区域索引用于检查
            utr_region_indices = np.argmax(seq_emb_np[check_region[0] : check_region[1]], axis=1)
            
            # 4. 检查是否包含 Forbidden Motif
            has_forbidden = False
            
            if target_patterns:
                # 遍历每一个黑名单 Motif
                for motif_tuple in target_patterns:
                    m_len = len(motif_tuple)
                    # 简单滑动窗口检查
                    # 对于纯筛选，我们可以利用 Python 字符串查找 (更快) 或者转为 list/tuple 查找
                    # 这里为了保持跟 indices 一致性，使用 tuple 转换
                    
                    # 将整个 UTR 转为 tuple (hashable)
                    utr_tuple = tuple(utr_region_indices)
                    
                    # 简单的子序列检查: 遍历 UTR
                    # 这种方式对于极长的 UTR 可能稍慢，但逻辑最准确
                    for k in range(len(utr_tuple) - m_len + 1):
                        if utr_tuple[k : k+m_len] == motif_tuple:
                            has_forbidden = True
                            break
                    
                    if has_forbidden:
                        break
            
            if has_forbidden:
                continue

            # 5. 通过筛选，保存样本
            candidates.append({
                'index': i,
                'cell_type': cell_type,
                'utr5_len': start,
                'morf_start': start,
                'morf_end': end,
                'uuid': uuid,
                'seq_emb': seq_emb_np # Original
            })
            
        except Exception as e:
            continue
            
    # 6. 排序与截断
    candidates.sort(key=lambda x: x['utr5_len'], reverse=True)
    selected = candidates[:top_n]
    
    if len(selected) > 0:
        print(f"Selected {len(selected)} naturally clean transcripts.")
        print(f"Length Range: {selected[-1]['utr5_len']} - {selected[0]['utr5_len']} nt")
    else:
        print("Warning: No transcripts met the criteria.")
        
    return selected


def get_samples_5utr_no_start_stop(
        dataset, top_n=50, utr_len_range=[300, 2000], check_region_len=300):
    """
    筛选指定 5'UTR 长度范围的 top_n 个样本，并进行背景清洗。
    
    清洗规则：
    在 CDS Start 上游 check_region_len 范围内，如果发现 ATG, CTG, TGA, TAA，
    将其第3个碱基突变为 C (index 1)。
    
    Args:
        dataset: 数据集对象
        top_n: 返回样本数量
        utr_len_range: [min_len, max_len], 筛选条件
        check_region_len: 在 Start Codon 上游多少 nt 范围内进行清洗
    """
    candidates = []
    min_len, max_len = utr_len_range
    print(f"Scanning dataset (UTR len: {min_len}-{max_len}) and cleaning 5'UTR backgrounds (Range: -{check_region_len}nt)...")
    
    # 定义需要清洗的密码子集合 (基于索引: A=0, C=1, G=2, T=3)
    # 替换规则：将 codon[2] (第3位) 变为 1 (C)
    targets = {
        (0, 3, 2), # ATG -> ATC
        (1, 3, 2), # CTG -> CTC
        (3, 2, 0), # TGA -> TGC
        (3, 0, 0)  # TAA -> TAC
    }
    
    # 定义替换碱基 C 的 One-hot 向量
    C_VECTOR = np.array([0, 1, 0, 0], dtype=np.float32)

    for i in tqdm(range(len(dataset))):
        try:
            item = dataset[i]
            # 根据实际 dataset 结构解包
            cell_type = item[0]
            cds_pos = item[1]
            seq_emb = item[3] # Tensor or Numpy
            uuid = dataset.uuids[i]
            
            # 解析 CDS Start
            start = int(cds_pos[0]) - 1 # 0-based index of ATG start
            end = int(cds_pos[1])
            
            # 1. 长度筛选 (使用参数范围)
            if not (min_len <= start <= max_len):
                continue
            
            # 2. 获取 Numpy Embedding
            if isinstance(seq_emb, torch.Tensor):
                seq_emb_np = seq_emb.cpu().numpy()
            else:
                seq_emb_np = seq_emb.copy()
                
            # 维度调整：确保 shape 为 (Length, 4)
            if seq_emb_np.shape[0] == 4 and seq_emb_np.shape[1] > 4:
                seq_emb_np = seq_emb_np.T 

            # 3. 定义扫描区域
            # 这里的 region_end 就是 CDS start (不包含 ATG 本身，只处理上游)
            region_end = start
            # region_start 是上游 check_region_len 处，但不能小于 0
            region_start = max(0, start - check_region_len)
            
            # 提取该区域的索引序列 (0,1,2,3) 用于快速查找
            # 注意：这里我们修改的是 seq_emb_np 本身
            upstream_indices = np.argmax(seq_emb_np[region_start : region_end], axis=1)
            
            # 4. 滑动窗口扫描与替换 (从 5' -> 3')
            mutation_count = 0
            
            # 只需要遍历到 len-3 即可
            for k in range(len(upstream_indices) - 2):
                # 提取当前 3-mer
                codon = tuple(upstream_indices[k : k+3])
                
                if codon in targets:
                    # 发现目标！执行替换
                    
                    # 1. 修改 Embedding (全局位置 = region_start + k + 2)
                    # 将第 3 个碱基 (index k+2) 变为 C
                    mutation_pos = region_start + k + 2
                    seq_emb_np[mutation_pos] = C_VECTOR
                    
                    # 2. 同步更新 indices 数组，防止逻辑错乱 (虽然这里其实不需要再次匹配它，但保持一致是个好习惯)
                    upstream_indices[k+2] = 1 # 1 is C
                    mutation_count += 1

            # 5. 保存样本
            candidates.append({
                'index': i,
                'cell_type': cell_type,
                'utr5_len': start,
                'morf_start': start,
                'morf_end': end,
                'uuid': uuid,
                'seq_emb': seq_emb_np, # 这是已经清洗过的 clean embedding
                'mutations_made': mutation_count
            })
            
        except Exception as e:
            continue
            
    # 排序并取 Top N
    candidates.sort(key=lambda x: x['utr5_len'], reverse=True)
    selected = candidates[:top_n]
    
    if len(selected) > 0:
        print(f"Selected {len(selected)} transcripts.")
        print(f"5'UTR Length Range: {selected[-1]['utr5_len']} - {selected[0]['utr5_len']} nt")
        total_muts = sum(s['mutations_made'] for s in selected)
        print(f"Total background mutations performed in check regions: {total_muts}")
    else:
        print("Warning: No transcripts met the criteria.")
    
    return selected