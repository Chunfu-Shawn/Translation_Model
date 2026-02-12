import torch
import random
from typing import List, Optional, Union, Dict

class BatchMaskingAdapter:
    def __init__(self, 
                 mask_value: float = 0,
                 mask_token_vec: Optional[torch.Tensor] = None):
        self.mask_value = mask_value
        self.mask_token_vec = mask_token_vec  # 1-D tensor of size D or None
        
    def _mask_amount(self, seq_len: int, mask_perc: float):
        return max(1, int(round(seq_len * mask_perc)))

    def _apply_mask_values(self, 
                           masked_emb: torch.Tensor, 
                           emb_mask: torch.Tensor, 
                           batch_idx: int, 
                           positions: torch.Tensor, 
                           generator: torch.Generator):
        """
        通用函数：对选定的位置(positions)应用BERT风格的mask策略 (80% mask, 10% random, 10% original)
        positions: 1-D tensor of indices to mask
        """
        if positions.numel() == 0:
            return

        device = masked_emb.device
        D = masked_emb.shape[-1]
        
        # 决定替换类型
        r = torch.rand(positions.numel(), generator=generator, device=device)
        
        # 80% 替换为 mask_value / mask_vec
        mask_indices = positions[r < 0.8]
        if mask_indices.numel() > 0:
            if self.mask_token_vec is not None:
                masked_emb[batch_idx, mask_indices, :] = self.mask_token_vec.to(device)
            else:
                masked_emb[batch_idx, mask_indices, :] = self.mask_value

        # 10% 替换为随机噪声
        rand_indices = positions[(r >= 0.8) & (r < 0.9)]
        if rand_indices.numel() > 0:
            masked_emb[batch_idx, rand_indices, :] = torch.randn((rand_indices.numel(), D), device=device, generator=generator)
            
        # 剩下 10% 保持原样 (pass)，但仍需标记在 emb_mask 中
        emb_mask[batch_idx, positions] = True

    def mask_random_single_base_batch(self,
                                      embeddings: torch.Tensor,
                                      mask_perc: float = 0.15,
                                      pad_mask: Optional[torch.Tensor] = None,
                                      generator: Optional[torch.Generator] = None):
        B, L, _ = embeddings.shape
        device = embeddings.device
        g = generator or torch.Generator(device=device)
        masked = embeddings.clone()
        emb_mask = torch.zeros((B, L), dtype=torch.bool, device=device)

        for i in range(B):
            # get candidate positions
            if pad_mask is not None:
                cand = torch.nonzero(pad_mask[i], as_tuple=False).view(-1)
            else:
                cand = torch.arange(L, device=device)
            
            if cand.numel() == 0: continue
            
            k = self._mask_amount(cand.numel(), mask_perc)
            # random sampling
            perm = torch.randperm(cand.numel(), generator=g, device=device)[:k]
            pos = cand[perm]
            
            # apply mask value
            self._apply_mask_values(masked, emb_mask, i, pos, g)

        return masked, emb_mask

    def mask_random_trinucleotide_batch(self,
                                        embeddings: torch.Tensor,
                                        cds_starts: List[int],
                                        mask_perc: float = 0.15,
                                        pad_mask: Optional[torch.Tensor] = None,
                                        generator: Optional[torch.Generator] = None):
        B, L, _ = embeddings.shape
        device = embeddings.device
        g = generator or torch.Generator(device=device)
        masked = embeddings.clone()
        emb_mask = torch.zeros((B, L), dtype=torch.bool, device=device)

        for i in range(B):
            valid_len = L
            if pad_mask is not None:
                valid_indices = torch.nonzero(pad_mask[i], as_tuple=False).view(-1)
                if valid_indices.numel() == 0: continue
                valid_len = valid_indices.numel()

            cds = cds_starts[i]
            if cds == -1: continue 

            start = int((cds - 1) % 3)
            tri_len = int((valid_len - start) // 3)
            if tri_len <= 0: continue

            mask_tri_amount = max(1, int((valid_len * mask_perc) // 3))
            
            # 随机选择密码子索引
            p_indices = torch.randperm(tri_len, generator=g, device=device)[:mask_tri_amount]
            
            # 将密码子索引展开为所有涉及的base索引
            # shape: (num_masked_codons, 3) -> flatten
            base_indices = (start + p_indices.unsqueeze(1) * 3 + torch.arange(3, device=device)).view(-1)
            
            # 过滤掉可能的越界（如果是padding引起的）
            if pad_mask is not None:
                # 简单处理：仅保留在pad_mask范围内的
                # 实际上如果valid_len计算正确，一般不会越界，除非pad在中间
                base_indices = base_indices[base_indices < L]
                base_indices = base_indices[pad_mask[i][base_indices]]

            self._apply_mask_values(masked, emb_mask, i, base_indices, g)

        return masked, emb_mask

    def mask_random_motif_batch(self,
                                occs_list: List[List[tuple]],
                                embeddings: torch.Tensor,
                                mask_perc: float = 0.15,
                                generator: Optional[torch.Generator] = None):
        B, L, _ = embeddings.shape
        device = embeddings.device
        g = generator or torch.Generator(device=device)
        masked = embeddings.clone()
        emb_mask = torch.zeros((B, L), dtype=torch.bool, device=device)

        if not occs_list:
            return masked, emb_mask

        for i in range(B):
            occs = occs_list[i]
            if not occs: continue
            
            # Shuffle and greedy select
            occs_shuf = list(occs)
            if len(occs_shuf) > 1:
                perm = torch.randperm(len(occs_shuf), generator=g).tolist()
                occs_shuf = [occs_shuf[j] for j in perm]
            
            max_mask_tokens = int(L * mask_perc)
            current_mask_indices = []
            total = 0
            
            # 简单的贪心无重叠选择
            selected_intervals = []
            for (s, e) in occs_shuf:
                if total + (e - s) > max_mask_tokens: continue
                # 检查重叠
                if any(not (e <= existing_s or s >= existing_e) for existing_s, existing_e in selected_intervals):
                    continue
                
                selected_intervals.append((s, e))
                total += (e - s)
                # 收集所有索引
                current_mask_indices.extend(range(s, e))
                if total >= max_mask_tokens: break
            
            if current_mask_indices:
                pos = torch.tensor(current_mask_indices, dtype=torch.long, device=device)
                self._apply_mask_values(masked, emb_mask, i, pos, g)

        return masked, emb_mask

    def get_random_masked_batch(self,
                                embeddings: torch.Tensor,
                                cds_starts: Optional[Union[List[int], torch.Tensor]] = None,
                                occs: Optional[List[List[tuple]]] = None,
                                pad_mask: Optional[torch.Tensor] = None,
                                mask_perc: float = 0.15,
                                generator: Optional[torch.Generator] = None,
                                strategy_probs: Optional[Dict[str, float]] = None):
        """
        核心调度函数：
        - 如果 cds_starts 存在且有效 (!=-1)，可以使用 'tri' 策略。
        - 如果 cds_starts 无效，'tri' 策略概率归零，并在 'single' 和 'motif' 间重分配。
        """
        B, L, D = embeddings.shape
        device = embeddings.device
        g = generator or torch.Generator(device=device)

        # 1. 预处理 cds_starts
        if cds_starts is None:
            cds_starts_list = [-1] * B
        elif isinstance(cds_starts, torch.Tensor):
            cds_starts_list = cds_starts.cpu().tolist()
        else:
            cds_starts_list = cds_starts

        # 2. 准备基础概率
        if strategy_probs is None:
            # 默认概率: single 0.4, tri 0.3, motif 0.3
            base_probs = {"single": 0.4, "tri": 0.3, "motif": 0.3}
        else:
            base_probs = strategy_probs.copy()

        strategies = ["single", "tri", "motif"]
        
        # 计算两种概率分布：
        # Plan A: 标准分布 (有 CDS)
        sum_full = sum(base_probs.values())
        probs_full = [base_probs[s]/sum_full for s in strategies]
        
        # Plan B: 无 CDS 分布 (去除 tri，重归一化)
        probs_no_tri = [base_probs["single"], 0.0, base_probs["motif"]]
        sum_no_tri = sum(probs_no_tri)
        if sum_no_tri > 0:
            probs_no_tri = [p/sum_no_tri for p in probs_no_tri]
        else:
            # 极端情况 fallback
            probs_no_tri = [1.0, 0.0, 0.0]

        # 3. 为每个样本分配策略
        # 我们在这里手动进行策略采样，以便处理逐样本的概率差异
        # 生成 [0, 1) 随机数
        r_strategy = torch.rand(B, generator=g, device=device).cpu().tolist()
        
        batch_strategies = []
        for i, r_val in enumerate(r_strategy):
            has_cds = (cds_starts_list[i] != -1)
            p_dist = probs_full if has_cds else probs_no_tri
            
            # 简单的累积概率选择
            cum = 0.0
            chosen = strategies[-1] # default
            for s_idx, s_name in enumerate(strategies):
                cum += p_dist[s_idx]
                if r_val < cum:
                    chosen = s_name
                    break
            batch_strategies.append(chosen)

        # 4. 分组处理 (Group indices by strategy)
        idxs_by_strategy = {s: [] for s in strategies}
        for i, s in enumerate(batch_strategies):
            idxs_by_strategy[s].append(i)

        # 准备输出容器
        masked_emb = embeddings.clone()
        emb_mask = torch.zeros((B, L), dtype=torch.bool, device=device)

        # 定义辅助函数：处理子 batch 并写回原 tensor
        def _process_subbatch(strategy_name, indices):
            if not indices: return
            
            # 提取子数据 (view)
            sub_emb = embeddings[indices] # clone slicing
            sub_pad = pad_mask[indices] if pad_mask is not None else None
            
            if strategy_name == 'single':
                sub_out, sub_mask = self.mask_random_single_base_batch(
                    sub_emb, mask_perc=mask_perc, pad_mask=sub_pad, generator=g)
            
            elif strategy_name == 'tri':
                sub_cds = [cds_starts_list[i] for i in indices]
                sub_out, sub_mask = self.mask_random_trinucleotide_batch(
                    sub_emb, cds_starts=sub_cds, mask_perc=mask_perc, pad_mask=sub_pad, generator=g)
                
            elif strategy_name == 'motif':
                sub_occs = [occs[i] for i in indices] if occs else []
                sub_out, sub_mask = self.mask_random_motif_batch(
                    sub_occs, sub_emb, mask_perc=mask_perc, generator=g)
            
            # Write back (Scatter)
            # 注意：indices 是 list，直接用作 tensor index 可能会慢，转为 tensor
            idx_tensor = torch.tensor(indices, device=device)
            masked_emb[idx_tensor] = sub_out
            emb_mask[idx_tensor] = sub_mask

        # 5. 执行
        for s in strategies:
            _process_subbatch(s, idxs_by_strategy[s])

        return masked_emb, emb_mask