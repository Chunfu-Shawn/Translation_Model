import torch
import random
from typing import List, Optional, Union, Dict, Tuple

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
        General function: Apply BERT-style masking strategy to selected positions 
        (80% mask token, 10% random noise, 10% original).
        positions: 1-D tensor of indices to mask
        """
        if positions.numel() == 0:
            return

        device = masked_emb.device
        D = masked_emb.shape[-1]
        
        # Determine replacement type
        r = torch.rand(positions.numel(), generator=generator, device=device)
        
        # 80% replace with mask_value / mask_vec
        mask_indices = positions[r < 0.8]
        if mask_indices.numel() > 0:
            if self.mask_token_vec is not None:
                masked_emb[batch_idx, mask_indices, :] = self.mask_token_vec.to(device)
            else:
                masked_emb[batch_idx, mask_indices, :] = self.mask_value

        # 10% replace with random noise
        rand_indices = positions[(r >= 0.8) & (r < 0.9)]
        if rand_indices.numel() > 0:
            masked_emb[batch_idx, rand_indices, :] = torch.randn((rand_indices.numel(), D), device=device, generator=generator)
            
        # Remaining 10% keep original (pass), but still mark in emb_mask
        emb_mask[batch_idx, positions] = True

    def _fill_remaining_budget_randomly(self,
                                        masked_emb: torch.Tensor,
                                        emb_mask: torch.Tensor,
                                        batch_idx: int,
                                        valid_indices: torch.Tensor,
                                        target_count: int,
                                        generator: torch.Generator):
        """
        Helper function to fill the remaining masking budget with random single bases 
        if structural masking (motif/tri) didn't reach the target ratio.
        """
        current_masked_count = torch.sum(emb_mask[batch_idx]).item()
        needed = target_count - current_masked_count

        if needed <= 0:
            return

        # Get all valid indices that are NOT yet masked
        # We assume valid_indices contains all non-padding positions
        # We need to filter out those already set to True in emb_mask[batch_idx]
        
        # 1. Create a boolean mask of valid positions
        # shape: (L,)
        candidate_mask = torch.zeros_like(emb_mask[batch_idx], dtype=torch.bool)
        candidate_mask[valid_indices] = True
        
        # 2. Exclude already masked positions
        candidate_mask = candidate_mask & (~emb_mask[batch_idx])
        
        # 3. Get indices
        candidates = torch.nonzero(candidate_mask, as_tuple=False).view(-1)
        
        if candidates.numel() == 0:
            return

        # 4. Sample 'needed' amount
        k = min(needed, candidates.numel())
        perm = torch.randperm(candidates.numel(), generator=generator, device=masked_emb.device)[:k]
        pos = candidates[perm]

        # 5. Apply mask
        self._apply_mask_values(masked_emb, emb_mask, batch_idx, pos, generator)

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
            # Get candidate positions
            if pad_mask is not None:
                cand = torch.nonzero(pad_mask[i], as_tuple=False).view(-1)
            else:
                cand = torch.arange(L, device=device)
            
            if cand.numel() == 0: continue
            
            k = self._mask_amount(cand.numel(), mask_perc)
            # Random sampling
            perm = torch.randperm(cand.numel(), generator=g, device=device)[:k]
            pos = cand[perm]
            
            # Apply mask value
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
            # Determine valid indices (non-padding)
            if pad_mask is not None:
                valid_indices = torch.nonzero(pad_mask[i], as_tuple=False).view(-1)
                if valid_indices.numel() == 0: continue
                valid_len = valid_indices.numel()
            else:
                valid_indices = torch.arange(L, device=device)

            ### Calculate total target tokens to mask based on ratio ###
            target_mask_tokens = self._mask_amount(valid_len, mask_perc)

            cds = cds_starts[i]
            # If CDS is invalid, we will skip the structured part and go straight to filling
            # (Logic handled by falling through to _fill_remaining_budget_randomly)
            
            if cds != -1: 
                start = int((cds - 1) % 3)
                tri_len = int((valid_len - start) // 3)
                
                if tri_len > 0:
                    # Calculate how many codons we *ideally* want to mask
                    # (This is an approximation, usually ratio / 3)
                    mask_tri_amount = max(1, int((valid_len * mask_perc) // 3))
                    
                    # Randomly select codon indices
                    p_indices = torch.randperm(tri_len, generator=g, device=device)[:mask_tri_amount]
                    
                    # Expand codon indices to all involved base indices
                    # shape: (num_masked_codons, 3) -> flatten
                    base_indices = (start + p_indices.unsqueeze(1) * 3 + torch.arange(3, device=device)).view(-1)
                    
                    # Filter out possible out-of-bounds (if caused by padding)
                    if pad_mask is not None:
                        # Simple handling: keep only those within L and valid in pad_mask
                        base_indices = base_indices[base_indices < L]
                        base_indices = base_indices[pad_mask[i][base_indices]]

                    self._apply_mask_values(masked, emb_mask, i, base_indices, g)

            ### Check if target ratio is met, if not, fill with random single bases ###
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
                                mask_perc: float = 0.15,
                                pad_mask: Optional[torch.Tensor] = None, 
                                generator: Optional[torch.Generator] = None):
        B, L, _ = embeddings.shape
        device = embeddings.device
        g = generator or torch.Generator(device=device)
        masked = embeddings.clone()
        emb_mask = torch.zeros((B, L), dtype=torch.bool, device=device)

        # Handle empty list case gracefully
        if not occs_list:
            occs_list = [[] for _ in range(B)]

        for i in range(B):
            # Determine valid indices for fallback calculation
            if pad_mask is not None:
                valid_indices = torch.nonzero(pad_mask[i], as_tuple=False).view(-1)
                valid_len = valid_indices.numel()
            else:
                valid_indices = torch.arange(L, device=device)
                valid_len = L

            if valid_len == 0: continue

            ### Calculate total target tokens to mask ###
            target_mask_tokens = self._mask_amount(valid_len, mask_perc)
            
            occs = occs_list[i]
            if occs:
                # Shuffle and greedy select
                occs_shuf = list(occs)
                if len(occs_shuf) > 1:
                    perm = torch.randperm(len(occs_shuf), generator=g).tolist()
                    occs_shuf = [occs_shuf[j] for j in perm]
                
                # NOTE: In original code, max_mask_tokens limited the motif selection loop.
                # We keep this logic to avoid over-masking with motifs alone.
                current_mask_indices = []
                total_motif_masked = 0
                
                # Greedy non-overlapping selection
                selected_intervals = []
                for (s, e) in occs_shuf:
                    # Check if adding this motif exceeds target
                    if total_motif_masked + (e - s) > target_mask_tokens: continue
                    
                    # Check overlap
                    if any(not (e <= existing_s or s >= existing_e) for existing_s, existing_e in selected_intervals):
                        continue
                    
                    selected_intervals.append((s, e))
                    total_motif_masked += (e - s)
                    # Collect all indices
                    current_mask_indices.extend(range(s, e))
                    
                    if total_motif_masked >= target_mask_tokens: break
                
                if current_mask_indices:
                    pos = torch.tensor(current_mask_indices, dtype=torch.long, device=device)
                    # Ensure indices are within bounds and valid (in case motif data is noisy)
                    if pad_mask is not None:
                        pos = pos[pos < L]
                        pos = pos[pad_mask[i][pos]]
                        
                    self._apply_mask_values(masked, emb_mask, i, pos, g)

            ### Check if target ratio is met, if not, fill with random single bases ###
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
                                mask_perc_range: Union[float, Tuple[float, float]] = (0.1, 1.0),
                                generator: Optional[torch.Generator] = None,
                                strategy_probs: Optional[Dict[str, float]] = None):
        """
        Core scheduling function:
        - Samples a random masking percentage from mask_perc_range.
        - Dispatches 'single', 'tri', or 'motif' strategy.
        - Fills gaps with random masking if structural strategies fall short.
        """
        B, L, D = embeddings.shape
        device = embeddings.device
        g = generator or torch.Generator(device=device)

        # 1. Preprocess cds_starts
        if cds_starts is None:
            cds_starts_list = [-1] * B
        elif isinstance(cds_starts, torch.Tensor):
            cds_starts_list = cds_starts.cpu().tolist()
        else:
            cds_starts_list = cds_starts

        # 2. Prepare base probabilities
        if strategy_probs is None:
            # Default probs: single 0.4, tri 0.3, motif 0.3
            base_probs = {"single": 0.4, "tri": 0.3, "motif": 0.3}
        else:
            base_probs = strategy_probs.copy()

        strategies = ["single", "tri", "motif"]
        
        # Calculate probability distributions:
        # Plan A: Standard distribution (Has CDS)
        sum_full = sum(base_probs.values())
        probs_full = [base_probs[s]/sum_full for s in strategies]
        
        # Plan B: No CDS distribution (Remove tri, re-normalize)
        probs_no_tri = [base_probs["single"], 0.0, base_probs["motif"]]
        sum_no_tri = sum(probs_no_tri)
        if sum_no_tri > 0:
            probs_no_tri = [p/sum_no_tri for p in probs_no_tri]
        else:
            # Extreme fallback
            probs_no_tri = [1.0, 0.0, 0.0]

        # 3. Assign strategy for each sample
        # Generate [0, 1) random numbers
        r_strategy = torch.rand(B, generator=g, device=device).cpu().tolist()
        
        batch_strategies = []
        for i, r_val in enumerate(r_strategy):
            has_cds = (cds_starts_list[i] != -1)
            p_dist = probs_full if has_cds else probs_no_tri
            
            # Simple cumulative probability selection
            cum = 0.0
            chosen = strategies[-1] # default
            for s_idx, s_name in enumerate(strategies):
                cum += p_dist[s_idx]
                if r_val < cum:
                    chosen = s_name
                    break
            batch_strategies.append(chosen)

        ### Randomly sample mask percentage from range ###
        if isinstance(mask_perc_range, (tuple, list)) and len(mask_perc_range) == 2:
            min_p, max_p = mask_perc_range
            # Sample a float between min and max
            # Note: We sample ONE ratio for the entire batch for efficiency/stability, 
            # but you could move this inside the loop to have per-sample ratios.
            rand_ratio = torch.rand(1, generator=g, device=device).item()
            target_mask_perc = min_p + rand_ratio * (max_p - min_p)
        else:
            # Handle case where a single float is passed
            target_mask_perc = float(mask_perc_range)

        # 4. Group processing (Group indices by strategy)
        idxs_by_strategy = {s: [] for s in strategies}
        for i, s in enumerate(batch_strategies):
            idxs_by_strategy[s].append(i)

        # Prepare output containers
        masked_emb = embeddings.clone()
        emb_mask = torch.zeros((B, L), dtype=torch.bool, device=device)

        # Helper function: Process sub-batch and write back to original tensor
        def _process_subbatch(strategy_name, indices):
            if not indices: return
            
            # Extract sub-data (view/clone)
            sub_emb = embeddings[indices] 
            sub_pad = pad_mask[indices] if pad_mask is not None else None
            
            if strategy_name == 'single':
                sub_out, sub_mask = self.mask_random_single_base_batch(
                    sub_emb, mask_perc=target_mask_perc, pad_mask=sub_pad, generator=g)
            
            elif strategy_name == 'tri':
                sub_cds = [cds_starts_list[i] for i in indices]
                # Note: We pass pad_mask here now, as updated method uses it for filling
                sub_out, sub_mask = self.mask_random_trinucleotide_batch(
                    sub_emb, cds_starts=sub_cds, mask_perc=target_mask_perc, pad_mask=sub_pad, generator=g)
                
            elif strategy_name == 'motif':
                sub_occs = [occs[i] for i in indices] if occs else []
                # Note: We pass pad_mask here now, as updated method uses it for filling
                sub_out, sub_mask = self.mask_random_motif_batch(
                    sub_occs, sub_emb, mask_perc=target_mask_perc, pad_mask=sub_pad, generator=g)
            
            # Write back (Scatter)
            # Convert indices list to tensor
            idx_tensor = torch.tensor(indices, device=device)
            masked_emb[idx_tensor] = sub_out
            emb_mask[idx_tensor] = sub_mask

        # 5. Execute
        for s in strategies:
            _process_subbatch(s, idxs_by_strategy[s])

        return masked_emb, emb_mask