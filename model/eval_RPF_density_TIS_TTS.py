import os
import numpy as np
import pandas as pd

def build_length_position_matrix(counts_dict, tx_cds, result_dir,
                                 region="start",
                                 left=40, right=60,
                                 min_len=21, max_len=40,
                                 prefix="read_count"):
    """
    返回：
      lengths_sorted: list of lengths present (sorted)
      mat: np.array shape (n_lengths, W) where W = left+right+1,
           each entry is total counts summed across transcripts for that length and relative position.
      total_counts_per_length: np.array shape (n_lengths,) - total counts for each length across window
      rel_positions: np.arange(-left, right+1)
    counts_dict: dict {tid: {read_len: {pos:count}}} or {tid: {pos:count}}
    tx_cds: dict[tid] with cds_start_pos and cds_end_pos (1-based)
    """
    rel_positions = np.arange(-left, right+1)
    W = len(rel_positions)
    # collect counts per length
    length_to_vector = {rl: np.ones(W, dtype=np.float64) for rl in range(min_len, max_len + 1)}

    for tid, raw_counts in counts_dict.items():
        cds = tx_cds.get(tid)
        if cds is None:
            continue
        if region == "start":
            anchor = cds.get("cds_start_pos", None)
        else:
            anchor = cds.get("cds_end_pos", None) + 1
        if anchor is None or anchor <= 0:
            continue

        # collapse readlen nested dict -> pos:count if needed
        # but we need per-length info; so if raw_counts is nested dict {len: {pos:count}}, iterate lens
        any_val = next(iter(raw_counts.values()))
        if isinstance(any_val, dict):
            # nested by length
            for pos, d in raw_counts.items():
                for rl, c in d.items():
                    if rl < min_len or rl > max_len:
                        continue
                    rel = int(pos) - int(anchor)
                    # only keep within window
                    if rel >= -left and rel <= right:
                        idx = rel + left
                        length_to_vector[rl][idx] += c
        else:
            # raw_counts already collapsed (no length info) -> skip (can't attribute length)
            continue

    if len(length_to_vector) == 0:
        raise ValueError("No data for requested lengths / anchors. Check inputs and windows.")

    lengths_sorted = sorted(length_to_vector.keys())
    mat = np.vstack([length_to_vector[l] for l in lengths_sorted])
    
    # save
    M_df = pd.DataFrame(mat, index=lengths_sorted, columns=rel_positions)
    M_df.to_csv(os.path.join(result_dir, prefix + ".normalized_counts_around_"+ region + ".csv"), index=True, header=True)

    return lengths_sorted, mat, rel_positions

def build_length_frame_matrix(counts_dict, tx_cds, 
                              result_dir,
                              min_len=21, max_len=40, prefix="read",
                              offset_dict: dict = {}):
    """
    返回：
      lengths_sorted: list of lengths present (sorted)
      mat: np.array shape (n_lengths, W) where W = left+right+1,
           each entry is total counts summed across transcripts for that length and relative position.
      total_counts_per_length: np.array shape (n_lengths,) - total counts for each length across window
      rel_positions: np.arange(-left, right+1)
    counts_dict: dict {tid: {read_len: {pos:count}}} or {tid: {pos:count}}
    tx_cds: dict[tid] with cds_start_pos and cds_end_pos (1-based)
    """

    # collect counts per length
    length_to_vector = {rl: np.zeros(3, dtype=np.float64) for rl in range(min_len, max_len + 1)}
    length_totals = {}

    for tid, raw_counts in counts_dict.items():
        cds = tx_cds.get(tid)
        if cds is None:
            continue
        anchor = cds.get("cds_start_pos", None)

        if anchor is None or anchor <= 0:
            continue

        any_val = next(iter(raw_counts.values()))
        if isinstance(any_val, dict):
            # nested by length
            for pos, d in raw_counts.items():
                for rl, c in d.items():
                    if rl < min_len or rl > max_len:
                        continue
                    frame = (int(pos) - int(anchor) + offset_dict.get(rl, 0)) % 3
                    # only keep within window
                    length_to_vector[rl][frame] += c
                    length_totals[rl] = length_totals.get(rl, 0) + c
        else:
            # raw_counts already collapsed (no length info) -> skip (can't attribute length)
            continue

    if len(length_to_vector) == 0:
        raise ValueError("No data for requested lengths / anchors. Check inputs and windows.")

    lengths_sorted = sorted(length_to_vector.keys())
    mat = np.vstack([length_to_vector[l] for l in lengths_sorted])
    totals = np.array([length_totals.get(l, 0) for l in lengths_sorted], dtype=np.float64)

    # save
    M_df = pd.DataFrame(mat, index=lengths_sorted, columns=range(3))
    M_df.to_csv(os.path.join(result_dir, prefix + "_perc.coding_frame.csv"), index=True, header=True)

    return lengths_sorted, mat, totals

def _smooth_row(row, sigma=1.0):
    """
    Smooth a 1-D array using a Gaussian filter if available.
    If sigma is None or <=0, return the raw array.
    Fallback to a simple moving average if scipy is not installed.
    """
    if sigma is None or sigma <= 0:
        return np.asarray(row, dtype=float)
    try:
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(np.asarray(row, dtype=float), sigma=sigma, mode='reflect')
    except Exception:
        # fallback: simple moving average kernel
        k = max(1, int(2 * sigma + 1))
        kernel = np.ones(k) / k
        return np.convolve(np.asarray(row, dtype=float), kernel, mode='same')


def find_peaks_in_range_all_lengths(matrix, rel_positions, read_lens,
                                    totals,
                                    output_dir,
                                    search_range=(-17, -8),
                                    smoothing_sigma=1.0,
                                    min_fraction=0.01):
    """
    For each read length (each row of `matrix`), find the column within
    `search_range` that has the maximum value (optionally smoothed).
    If a read length's total read count < min_fraction * total_reads,
    replace its best_pos with the best_pos of the most abundant read length
    that has a valid candidate (fallback).

    Parameters
    ----------
    matrix : 2D array-like, shape (n_lengths, W)
        Each row corresponds to a read length's density or counts across positions.
    rel_positions : 1D array-like, length W
        Integer relative positions (e.g., np.arange(-left, right+1)), where 0 is annotated TTS.
    read_lens : list-like, length n_lengths
        Read lengths corresponding to matrix rows.
    totals : 1D array-like, length n_lengths
        Total read counts for each read length (used for sparsity thresholding).
    search_range : tuple (low, high), inclusive
        Interval of relative positions to search for the peak (integers).
    smoothing_sigma : float or None
        If >0, apply Gaussian smoothing with this sigma. If None or <=0, use raw values.
    min_fraction : float
        Fraction threshold (default 0.01). If totals[i] < min_fraction * sum(totals),
        that read length is considered sparse and will be replaced by fallback.

    Returns
    -------
    best_pos_dict : dict
        Mapping {read_len: best_pos_or_None}.
    df : pandas.DataFrame
        DataFrame with columns ['read_len','best_pos','best_value','col_index','total','replaced'].
    """
    M = np.asarray(matrix)
    pos = np.asarray(rel_positions)
    totals = np.asarray(totals)
    low, high = search_range
    if low > high:
        low, high = high, low

    # candidate column indices where rel_positions are within search_range
    cand_mask = (pos >= low) & (pos <= high)
    cand_idx = np.where(cand_mask)[0]
    if cand_idx.size == 0:
        raise ValueError(f"No columns in rel_positions within search_range {search_range}.")

    n_rows = M.shape[0]
    if len(read_lens) != n_rows:
        raise ValueError("length of read_lens must match number of rows in matrix")
    if len(totals) != n_rows:
        raise ValueError("length of totals must match number of rows in matrix")

    rows = []
    initial_best = {}  # store initial best_pos per read length
    initial_info = {}  # store (best_val, col_index) for fallback use

    # find per-read-length peak within the candidate indices
    for i, rl in enumerate(read_lens):
        row = M[i, :]
        s = _smooth_row(row, smoothing_sigma)
        sub = s[cand_idx]
        if np.all(np.isnan(sub)) or np.nanmax(sub) == 0:
            best_pos = None
            best_val = float(np.nan)
            best_j = None
        else:
            arg = int(np.nanargmax(sub))
            j = cand_idx[arg]
            best_pos = int(pos[j])
            best_val = float(s[j])
            best_j = int(j)
        initial_best[rl] = best_pos
        initial_info[rl] = (best_val, best_j)
        rows.append({
            'read_len': rl,
            'best_pos': best_pos,
            'best_value': best_val,
            'col_index': best_j,
            'total': float(totals[i]),
            'replaced': False
        })

    df = pd.DataFrame(rows)

    # if no reads at all, return the initial result
    total_reads = float(np.sum(totals))
    if total_reads <= 0:
        best_pos_dict = {rl: initial_best[rl] for rl in read_lens}
        return best_pos_dict, df

    threshold = min_fraction * total_reads

    # choose fallback: the most abundant read length (by totals) that has a valid best_pos
    sort_idx = np.argsort(-totals)  # indices sorted by totals descending
    fallback_pos = None
    fallback_val = None
    fallback_col = None
    fallback_rl = None
    for idx in sort_idx:
        rl_candidate = read_lens[idx]
        bp = initial_best[rl_candidate]
        if bp is not None:
            fallback_pos = bp
            fallback_val, fallback_col = initial_info[rl_candidate]
            fallback_rl = rl_candidate
            break
    # if no valid candidate found for any length, fallback_pos remains None

    # replace sparse read lengths with the fallback candidate (if available)
    inf_read_lens = [] # informative reads
    for i, rl in enumerate(read_lens):
        if totals[i] < threshold:
            if fallback_pos is not None:
                df.loc[df['read_len'] == rl, ['best_pos', 'best_value', 'col_index', 'replaced']] = [
                    fallback_pos, fallback_val, fallback_col, True
                ]
            else:
                # no fallback available; keep original (likely None)
                df.loc[df['read_len'] == rl, 'replaced'] = False
        else:
            inf_read_lens.append(rl)

    # build the return dictionary from the (possibly updated) DataFrame
    best_pos_dict = dict(zip(df['read_len'].tolist(), df['best_pos'].tolist()))
    print(f"Read length where number of read exceeds {min_fraction} of total reads: {inf_read_lens}")
    # save
    df.to_csv(os.path.join(output_dir, "best_pos_for_train.csv"))

    return best_pos_dict, sorted(inf_read_lens)