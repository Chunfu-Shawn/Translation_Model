import torch
import random
from typing import List, Optional, Union, Dict, Tuple

def get_dynamic_mask_ratio(current_step, total_steps, 
                            start_ratio=0.1, end_ratio=1.0):
    """
    Calcualte the range of mask ratio with "Linear Curriculum"
    """
    
    progress = min(1.0, current_step / total_steps)
    
    # linear interpolation
    current_ratio = start_ratio + (end_ratio - start_ratio) * progress
    
    # boundary limitation 
    current_ratio = max(0.1, min(current_ratio, 1.1))
    
    # return a range of mask ratio, keeping dynamics
    lower_bound = max(0.05, current_ratio - 0.1)
    upper_bound = min(1.0, current_ratio + 0.1)
    
    return (lower_bound, upper_bound)

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
        # ... (保持不变) ...
        if positions.numel() == 0:
            return

        device = masked_emb.device
        D = masked_emb.shape[-1]
        
        r = torch.rand(positions.numel(), generator=generator, device=device)
        
        mask_indices = positions[r < 0.8]
        if mask_indices.numel() > 0:
            if self.mask_token_vec is not None:
                masked_emb[batch_idx, mask_indices, :] = self.mask_token_vec.to(device)
            else:
                masked_emb[batch_idx, mask_indices, :] = self.mask_value

        rand_indices = positions[(r >= 0.8) & (r < 0.9)]
        if rand_indices.numel() > 0:
            masked_emb[batch_idx, rand_indices, :] = torch.randn((rand_indices.numel(), D), device=device, generator=generator)
            
        emb_mask[batch_idx, positions] = True

    def _fill_remaining_budget_randomly(self,
                                        masked_emb: torch.Tensor,
                                        emb_mask: torch.Tensor,
                                        batch_idx: int,
                                        valid_indices: torch.Tensor,
                                        target_count: int,
                                        generator: torch.Generator):
        current_masked_count = torch.sum(emb_mask[batch_idx]).item()
        needed = target_count - current_masked_count

        if needed <= 0:
            return

        candidate_mask = torch.zeros_like(emb_mask[batch_idx], dtype=torch.bool)
        candidate_mask[valid_indices] = True
        candidate_mask = candidate_mask & (~emb_mask[batch_idx])
        
        candidates = torch.nonzero(candidate_mask, as_tuple=False).view(-1)
        
        if candidates.numel() == 0:
            return

        k = min(needed, candidates.numel())
        perm = torch.randperm(candidates.numel(), generator=generator, device=masked_emb.device)[:k]
        pos = candidates[perm]

        self._apply_mask_values(masked_emb, emb_mask, batch_idx, pos, generator)

    def mask_random_single_base_batch(self,
                                      embeddings: torch.Tensor,
                                      mask_perc: Union[float, torch.Tensor] = 0.15,
                                      pad_mask: Optional[torch.Tensor] = None,
                                      generator: Optional[torch.Generator] = None):
        B, L, _ = embeddings.shape
        device = embeddings.device
        g = generator or torch.Generator(device=device)
        masked = embeddings.clone()
        emb_mask = torch.zeros((B, L), dtype=torch.bool, device=device)

        for i in range(B):
            if pad_mask is not None:
                cand = torch.nonzero(pad_mask[i], as_tuple=False).view(-1)
            else:
                cand = torch.arange(L, device=device)
            
            if cand.numel() == 0: continue
            
            ### MODIFICATION: Get specific ratio for this sample ###
            current_perc = mask_perc if isinstance(mask_perc, float) else mask_perc[i].item()
            k = self._mask_amount(cand.numel(), current_perc)
            
            perm = torch.randperm(cand.numel(), generator=g, device=device)[:k]
            pos = cand[perm]
            
            self._apply_mask_values(masked, emb_mask, i, pos, g)

        return masked, emb_mask

    def mask_random_trinucleotide_batch(self,
                                        embeddings: torch.Tensor,
                                        cds_starts: List[int],
                                        mask_perc: Union[float, torch.Tensor] = 0.15,
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
            else:
                valid_indices = torch.arange(L, device=device)

            ### MODIFICATION: Get specific ratio for this sample ###
            current_perc = mask_perc if isinstance(mask_perc, float) else mask_perc[i].item()
            target_mask_tokens = self._mask_amount(valid_len, current_perc)

            cds = cds_starts[i]
            if cds != -1: 
                start = int((cds - 1) % 3)
                tri_len = int((valid_len - start) // 3)
                
                if tri_len > 0:
                    mask_tri_amount = max(1, int((valid_len * current_perc) // 3))
                    p_indices = torch.randperm(tri_len, generator=g, device=device)[:mask_tri_amount]
                    base_indices = (start + p_indices.unsqueeze(1) * 3 + torch.arange(3, device=device)).view(-1)
                    
                    if pad_mask is not None:
                        base_indices = base_indices[base_indices < L]
                        base_indices = base_indices[pad_mask[i][base_indices]]

                    self._apply_mask_values(masked, emb_mask, i, base_indices, g)

            self._fill_remaining_budget_randomly(
                masked_emb=masked,
                emb_mask=emb_mask,
                batch_idx=i,
                valid_indices=valid_indices,
                target_count=target_mask_tokens,
                generator=g
            )

        return masked, emb_mask

    def mask_random_motif_batch(self,
                                occs_list: List[List[tuple]],
                                embeddings: torch.Tensor,
                                mask_perc: Union[float, torch.Tensor] = 0.15,
                                pad_mask: Optional[torch.Tensor] = None, 
                                generator: Optional[torch.Generator] = None):
        B, L, _ = embeddings.shape
        device = embeddings.device
        g = generator or torch.Generator(device=device)
        masked = embeddings.clone()
        emb_mask = torch.zeros((B, L), dtype=torch.bool, device=device)

        if not occs_list:
            occs_list = [[] for _ in range(B)]

        for i in range(B):
            if pad_mask is not None:
                valid_indices = torch.nonzero(pad_mask[i], as_tuple=False).view(-1)
                valid_len = valid_indices.numel()
            else:
                valid_indices = torch.arange(L, device=device)
                valid_len = L

            if valid_len == 0: continue

            ### MODIFICATION: Get specific ratio for this sample ###
            current_perc = mask_perc if isinstance(mask_perc, float) else mask_perc[i].item()
            target_mask_tokens = self._mask_amount(valid_len, current_perc)
            
            occs = occs_list[i]
            if occs:
                occs_shuf = list(occs)
                if len(occs_shuf) > 1:
                    perm = torch.randperm(len(occs_shuf), generator=g).tolist()
                    occs_shuf = [occs_shuf[j] for j in perm]
                
                current_mask_indices = []
                total_motif_masked = 0
                
                selected_intervals = []
                for (s, e) in occs_shuf:
                    if total_motif_masked + (e - s) > target_mask_tokens: continue
                    if any(not (e <= existing_s or s >= existing_e) for existing_s, existing_e in selected_intervals):
                        continue
                    
                    selected_intervals.append((s, e))
                    total_motif_masked += (e - s)
                    current_mask_indices.extend(range(s, e))
                    if total_motif_masked >= target_mask_tokens: break
                
                if current_mask_indices:
                    pos = torch.tensor(current_mask_indices, dtype=torch.long, device=device)
                    if pad_mask is not None:
                        pos = pos[pos < L]
                        pos = pos[pad_mask[i][pos]]
                    self._apply_mask_values(masked, emb_mask, i, pos, g)

            self._fill_remaining_budget_randomly(
                masked_emb=masked,
                emb_mask=emb_mask,
                batch_idx=i,
                valid_indices=valid_indices,
                target_count=target_mask_tokens,
                generator=g
            )

        return masked, emb_mask


    def get_random_masked_batch(self,
                                embeddings: torch.Tensor,
                                cds_starts: Optional[Union[List[int], torch.Tensor]] = None,
                                occs: Optional[List[List[tuple]]] = None,
                                pad_mask: Optional[torch.Tensor] = None,
                                mask_perc_range: Union[float, Tuple[float, float]] = (0.1, 0.9),
                                full_mask_perc: float = 0.0,
                                generator: Optional[torch.Generator] = None,
                                strategy_probs: Optional[Dict[str, float]] = None):
        """
        Core scheduling function:
        - Samples a random masking percentage from mask_perc_range PER SAMPLE.
        - Supports full masking based on full_mask_perc.
        """
        B, L, D = embeddings.shape
        device = embeddings.device
        g = generator or torch.Generator(device=device)

        if cds_starts is None:
            cds_starts_list = [-1] * B
        elif isinstance(cds_starts, torch.Tensor):
            cds_starts_list = cds_starts.cpu().tolist()
        else:
            cds_starts_list = cds_starts

        if strategy_probs is None:
            base_probs = {"single": 0.4, "tri": 0.3, "motif": 0.3}
        else:
            base_probs = strategy_probs.copy()

        strategies = ["single", "tri", "motif"]
        sum_full = sum(base_probs.values())
        probs_full = [base_probs[s]/sum_full for s in strategies]
        
        probs_no_tri = [base_probs["single"], 0.0, base_probs["motif"]]
        sum_no_tri = sum(probs_no_tri)
        if sum_no_tri > 0:
            probs_no_tri = [p/sum_no_tri for p in probs_no_tri]
        else:
            probs_no_tri = [1.0, 0.0, 0.0]

        r_strategy = torch.rand(B, generator=g, device=device).cpu().tolist()
        
        batch_strategies = []
        for i, r_val in enumerate(r_strategy):
            has_cds = (cds_starts_list[i] != -1)
            p_dist = probs_full if has_cds else probs_no_tri
            
            cum = 0.0
            chosen = strategies[-1] 
            for s_idx, s_name in enumerate(strategies):
                cum += p_dist[s_idx]
                if r_val < cum:
                    chosen = s_name
                    break
            batch_strategies.append(chosen)

        ### Generate Per-Sample Ratios ###
        # 1. Determine which samples are fully masked (1.0)
        is_full_mask = torch.rand(B, generator=g, device=device) < full_mask_perc
        
        # 2. Generate dynamic ratios for everyone (from range)
        if isinstance(mask_perc_range, (tuple, list)) and len(mask_perc_range) == 2:
            min_p, max_p = mask_perc_range
            # Shape (B,) - uniform sampling per sample
            dynamic_ratios = torch.rand(B, generator=g, device=device) * (max_p - min_p) + min_p
        else:
            dynamic_ratios = torch.full((B,), float(mask_perc_range), device=device)
            
        # 3. Combine: Override dynamic ratio with 1.0 where is_full_mask is True
        batch_ratios = torch.where(is_full_mask, torch.tensor(1.0, device=device), dynamic_ratios)

        idxs_by_strategy = {s: [] for s in strategies}
        for i, s in enumerate(batch_strategies):
            idxs_by_strategy[s].append(i)

        masked_emb = embeddings.clone()
        emb_mask = torch.zeros((B, L), dtype=torch.bool, device=device)

        def _process_subbatch(strategy_name, indices):
            if not indices: return
            
            sub_emb = embeddings[indices] 
            sub_pad = pad_mask[indices] if pad_mask is not None else None
            
            ### Slice ratios for this subgroup ###
            sub_ratios = batch_ratios[indices] 
            
            if strategy_name == 'single':
                sub_out, sub_mask = self.mask_random_single_base_batch(
                    sub_emb, mask_perc=sub_ratios, pad_mask=sub_pad, generator=g)
            
            elif strategy_name == 'tri':
                sub_cds = [cds_starts_list[i] for i in indices]
                sub_out, sub_mask = self.mask_random_trinucleotide_batch(
                    sub_emb, cds_starts=sub_cds, mask_perc=sub_ratios, pad_mask=sub_pad, generator=g)
                
            elif strategy_name == 'motif':
                sub_occs = [occs[i] for i in indices] if occs else []
                sub_out, sub_mask = self.mask_random_motif_batch(
                    sub_occs, sub_emb, mask_perc=sub_ratios, pad_mask=sub_pad, generator=g)
            
            idx_tensor = torch.tensor(indices, device=device)
            masked_emb[idx_tensor] = sub_out
            emb_mask[idx_tensor] = sub_mask

        for s in strategies:
            _process_subbatch(s, idxs_by_strategy[s])

        return masked_emb, emb_mask