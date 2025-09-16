import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import patches

def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _find_runs(mask):
    """Find contiguous True runs in 1D boolean array.
       Returns list of (start_idx, length)."""
    runs = []
    if mask.size == 0:
        return runs
    in_run = False
    run_start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run = True
            run_start = i
        elif (not v) and in_run:
            runs.append((run_start, i - run_start))
            in_run = False
    if in_run:
        runs.append((run_start, len(mask) - run_start))
    return runs

def plot_transcript_coding(prediction,
                           targets,
                           seq_id: str = None,
                           model_name: str = None,
                           savepath: str = None,
                           figsize=(20, 4),
                           dpi=200,
                           show: bool = True):
    """
    Plot per-base scores (start/stop/in_orf) and targets for one transcript.

    Args:
      prediction: either
         - dict with keys 'start','stop','in_orf' each shape (L,) logits or probs
         - or ndarray shape (L,3) where columns order is [start, stop, in_orf] (logits or probs)
      targets: same layout as prediction but values 0/1 (either dict or ndarray (L,3))
      seq_id: optional title or filename tag
      savepath: if provided, save the image to this path
      figsize, dpi: figure size and resolution
      show: whether to plt.show()
    Returns:
      fig, ax
    """
    # normalize inputs to numpy arrays
    # predictions
    if isinstance(prediction, dict):
        start_pred = _to_numpy(prediction['start'])
        stop_pred = _to_numpy(prediction['stop'])
        inorf_pred = _to_numpy(prediction['in_orf'])
    else:
        arr = _to_numpy(prediction)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("prediction ndarray must have shape (L,3) with columns [start,stop,in_orf]")
        start_pred, stop_pred, inorf_pred = arr[:,0], arr[:,1], arr[:,2]

    # targets
    if isinstance(targets, dict):
        start_t = _to_numpy(targets['start']).astype(bool)
        stop_t = _to_numpy(targets['stop']).astype(bool)
        inorf_t = _to_numpy(targets['in_orf']).astype(bool)
    else:
        targ_arr = _to_numpy(targets)
        if targ_arr.ndim != 2 or targ_arr.shape[1] != 3:
            raise ValueError("targets ndarray must have shape (L,3) with columns [start,stop,in_orf]")
        start_t, stop_t, inorf_t = targ_arr[:,0].astype(bool), targ_arr[:,1].astype(bool), targ_arr[:,2].astype(bool)

    # ensure same length
    L = max(len(start_pred), len(stop_pred), len(inorf_pred),
            len(start_t), len(stop_t), len(inorf_t))
    def _pad(a):
        a = _to_numpy(a)
        if a.shape[0] == L:
            return a
        return np.pad(a, (0, L - a.shape[0]), constant_values=0)
    start_pred = _pad(start_pred)
    stop_pred  = _pad(stop_pred)
    inorf_pred = _pad(inorf_pred)
    start_t = _pad(start_t).astype(bool)
    stop_t  = _pad(stop_t).astype(bool)
    inorf_t = _pad(inorf_t).astype(bool)

    # if predictions look like logits (not between 0..1), apply sigmoid
    def _maybe_sigmoid(x):
        x = np.asarray(x, dtype=float)
        if x.min() < 0.0 - 1e-6 or x.max() > 1.0 + 1e-6:
            return _sigmoid(x)
        return x

    start_score = _maybe_sigmoid(start_pred)
    stop_score  = _maybe_sigmoid(stop_pred)
    inorf_score = _maybe_sigmoid(inorf_pred)

    x = np.arange(L)

    # colors: chosen per request (light/darker for targets)
    color_inorf = "#dedede"   # light gray
    color_inorf_t = "#7f7f7f" # darker gray for target blocks
    color_start = "#67be6e"   # light green for score
    color_start_t = "#2ca02c" # darker green for target blocks
    color_stop  = "#ff6666"   # light red for score
    color_stop_t  = "#d62728" # darker red for target blocks

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot order: in_orf (bottom), then start, then stop so they are not obscured by in_orf.
    ax.bar(x, inorf_score, width=1.0, color=color_inorf, alpha=0.6, align='edge', label='in_orf score')
    ax.bar(x, start_score, width=1.0, color=color_start, alpha=0.6, align='edge', label='start score')
    ax.bar(x, stop_score,  width=1.0, color=color_stop,  alpha=0.6, align='edge', label='stop score')

    # Targets: draw small rectangles below x-axis in separate rows to avoid overlap
    # define rows and positions
    row_height = 0.04
    spacing = 0.01
    # baseline for first target row below y=0
    y_base = - (row_height + spacing) * 3  # leave room for three rows

    # In_orf targets: lowest row
    for (s, ln) in _find_runs(inorf_t):
        rect = patches.Rectangle((s, y_base), ln, row_height, linewidth=0, facecolor=color_inorf_t, alpha=1.0)
        ax.add_patch(rect)
    # Stop targets: middle row
    for (s, ln) in _find_runs(stop_t):
        rect = patches.Rectangle((s, y_base + (row_height + spacing)), ln, row_height, linewidth=0, facecolor=color_stop_t, alpha=1.0)
        ax.add_patch(rect)
    # Start targets: top row (closest to x-axis)
    for (s, ln) in _find_runs(start_t):
        rect = patches.Rectangle((s, y_base + 2*(row_height + spacing)), ln, row_height, linewidth=0, facecolor=color_start_t, alpha=1.0)
        ax.add_patch(rect)

    # aesthetics
    ax.set_xlim(0, L)
    ax.set_ylim(y_base - 0.01, 1.05)
    ax.set_xlabel("position (base index)")
    ax.set_ylabel("score / probability")
    title = f"Transcript {seq_id}" if seq_id is not None else "Transcript coding prediction"
    ax.set_title(title)
    # legend: show only score labels (targets are obvious from colored bars below)
    ax.legend(loc='upper right')

    # tighten and grid
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    if savepath:
        path = os.path.join(savepath, f"{model_name}.{seq_id}_coding_pred.pdf")
        fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax
