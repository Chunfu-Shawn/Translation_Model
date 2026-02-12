import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import seaborn as sns

sns.set(style="white")

def plot_region_heatmap(M, col_positions, title="Start codon", read_lens=None, 
                        cmap="RdYlBu_r", vmin=None, vmax=None, ax=None, y_label="Read length"):
    """
    M: 2D array (n_rows, W) OR 1D array (W,) -> will be treated as single-row heatmap
    read_lens: list of integer read lengths corresponding to rows of M (if M has multiple rows)
    If M is single-row, read_lens may be None.
    """
    # ensure numpy array
    M = np.asarray(M)
    if M.ndim == 1:
        M = M[np.newaxis, :]

    n_rows, W = M.shape

    # If we have multiple rows and read_lens provided, set extent so that integer ticks align with centers
    if n_rows > 1 and read_lens is not None and len(read_lens) == n_rows:
        y0 = read_lens[0] - 0.5
        y1 = read_lens[-1] + 0.5
        y_ticks = read_lens
    else:
        # single row: make a compact vertical extent from 0 to 1 and set single tick label
        y0 = 0.0
        y1 = 1.0
        y_ticks = [0.5]

    im = ax.imshow(M, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[col_positions[0]-0.5, col_positions[-1]+0.5, y0, y1])

    ax.set_xlabel(title)
    ax.axvline(0, color='k', linestyle='--', linewidth=1)

    # y ticks & labels
    if n_rows > 1 and read_lens is not None and len(read_lens) == n_rows:
        ax.yaxis.set_major_locator(FixedLocator(y_ticks))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.set_ylabel(y_label)
        ax.tick_params(axis='y', which='major', labelsize=8)
    else:
        # single-row: place a single tick (middle) labelled as merged
        ax.yaxis.set_major_locator(FixedLocator(y_ticks))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d' if y_ticks[0] == int(y_ticks[0]) else '%.1f'))
        # label the single-row heatmap more semantically
        ax.set_yticklabels(["All\nlengths"])
        ax.set_ylabel("Merged")
        ax.tick_params(axis='y', which='major', labelsize=8)

    return im, ax

# def plot_region_heatmap(M, col_positions, title="Start codon", read_lens=[], 
#                         cmap="RdYlBu_r", vmin=None, vmax=None, ax=None):
#     """
#     M: (n_lengths, W)
#     read_lens: list of integers (sorted), must correspond to rows of M
#     """
    
#     # imshow: keep extent so each integer read_len is centered on row
#     im = ax.imshow(
#         M, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
#         extent=[col_positions[0]-0.5, col_positions[-1]+0.5,
#                 read_lens[0]-0.5, read_lens[-1]+0.5]
#     )
#     ax.set_ylabel("Read length")
#     ax.set_xlabel(title)
#     # vertical line at 0
#     ax.axvline(0, color='k', linestyle='--', linewidth=1)

#     # --- 设置 y ticks 为整数 read lengths ---
#     if len(read_lens) > 0:
#         # 用 FixedLocator 把主刻度固定到每个整数 read_len
#         ax.yaxis.set_major_locator(FixedLocator(read_lens))
#         # 标签用整数字符串（或自定义格式）
#         ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
#         # 若标签太密，可以只显示子集，例如每 2 nt：
#         # visible_labels = [rl if (i % 2 == 0) else '' for i, rl in enumerate(read_lens)]
#         # ax.set_yticklabels(visible_labels)
#         ax.tick_params(axis='y', which='major', labelsize=8)

#     return im, ax

def plot_start_stop_compact(M_start, M_stop, cols_s, cols_t, read_lens, output_path,
                            cmap="RdYlBu_r", vmin=0, vmax=0.2,
                            figsize=(7,4), mean_height_ratio=0.25, hspace=0.02,
                            normalize = "sum"):
    """
    mean_height_ratio: top row height as fraction of total figure height (0..1)
    hspace: vertical spacing between mean row and heatmap row (in fraction of subplot)
    """

    # normalization options
    if normalize == "sum":
        row_sum = M_start.sum() if M_start.sum() else 1
        M_start = M_start.sum(axis=0, keepdims=True) / row_sum
        row_sum = M_stop.sum() if M_stop.sum() else 1
        M_stop = M_stop.sum(axis=0, keepdims=True) / row_sum
    elif normalize == "max":
        row_max = M_start.sum(axis=1).max()
        row_max[row_max == 0] = 1.0
        M_start = M_start.sum(axis=0, keepdims=True) / row_max
        row_max = M_stop.sum(axis=1).max()
        row_max[row_max == 0] = 1.0
        M_stop = M_stop.sum(axis=0, keepdims=True) / row_max
    elif normalize is None:
        pass
    else:
        raise ValueError("unknown normalize option")
    
    print(M_start.shape, M_stop.shape)
    
    # gridspec: 2 rows x 2 cols; row0 for means, row1 for heatmaps
    # height_ratios choose mean row small and heatmap row large
    height_ratios = [mean_height_ratio, 1.0]
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows=2, ncols=2, height_ratios=height_ratios, hspace=hspace, wspace=0.08)

    # mean axes (row 0)
    ax_mean_l = fig.add_subplot(gs[0, 0])
    ax_mean_r = fig.add_subplot(gs[0, 1])

    # heatmap axes (row 1)
    ax_start = fig.add_subplot(gs[1, 0], sharex=ax_mean_l)
    ax_stop  = fig.add_subplot(gs[1, 1], sharex=ax_mean_r)

    # --- plot heatmaps (no mean inside) ---
    im1, _ = plot_region_heatmap(M_start, cols_s, title="Distance to start codon (nt)",
                                read_lens=read_lens, cmap=cmap,
                                ax=ax_start, vmin=vmin, vmax=vmax)

    im2, _ = plot_region_heatmap(M_stop, cols_t, title="Distance to stop codon (nt)",
                                read_lens=read_lens, cmap=cmap,
                                ax=ax_stop, vmin=vmin, vmax=vmax)

    # --- compute and plot mean profiles on the compact mean axes ---
    mean_start = M_start/M_start.sum()
    mean_stop  = M_stop/M_stop.sum()

    ax_mean_l.plot(cols_s, mean_start, lw=0.9)
    ax_mean_r.plot(cols_t, mean_stop, lw=0.9)

    ax_mean_l.axvline(0, color='k', linestyle='--', lw=0.6)
    ax_mean_r.axvline(0, color='k', linestyle='--', lw=0.6)

    # tidy mean axes: remove spines and ticks to be compact
    for axm in (ax_mean_l, ax_mean_r):
        axm.set_ylabel("Mean", fontsize=10)
        axm.tick_params(axis='y', which='both', left=True, length=2)
        axm.tick_params(labelbottom=False)
        axm.set_xlim(ax_start.get_xlim())
    
    ax_start.tick_params(axis='x', which='both', bottom=True, length=2)
    ax_start.tick_params(axis='y', which='both', left=True, length=2)
    ax_stop.tick_params(axis='x', which='both', bottom=True, length=2)
    ax_stop.tick_params(axis='y', which='both', left=True, length=2)
    ax_start.set_ylabel("Read length (nt)")
    ax_stop.set_ylabel("")

    # Shared colorbar below the two heatmaps (use an axes occupying the bottom center)
    # create a small axes at bottom spanning both columns via GridSpec
    cbar_ax = fig.add_axes([0.5, 0.05, 0.1, 0.03])   # [left, bottom, width, height] in figure coords
    cbar = fig.colorbar(im2, cax=cbar_ax, orientation='horizontal', fraction=0.1, pad=0.1, shrink=0.2, aspect=5)
    cbar.set_label("Normalized density of RPFs", fontsize=9)
    cbar.set_ticks([0.0, 0.1, 0.2])
    cbar.ax.tick_params(labelsize=9)

    # Tight layout and save-friendly settings
    plt.subplots_adjust(left=0.1, right=0.99, top=0.98, bottom=0.15, wspace=0.08, hspace=hspace)
    fig.savefig(output_path)

    return fig


def plot_start_stop_read_length(M_start, M_stop, cols_s, cols_t, read_lens, output_path,
                            cmap="RdYlBu_r", vmin=0, vmax=0.2,
                            figsize=(7,4), mean_height_ratio=0.25, hspace=0.02,
                            normalize = "row_sum"):
    """
    mean_height_ratio: top row height as fraction of total figure height (0..1)
    hspace: vertical spacing between mean row and heatmap row (in fraction of subplot)
    """

    # normalization options
    if normalize == "row_sum":
        row_sums = M_start.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        M_start = M_start / row_sums
        row_sums = M_stop.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        M_stop = M_stop / row_sums
    elif normalize == "row_max":
        row_max = M_start.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        M_start = M_start / row_max
        row_max = M_stop.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        M_stop = M_stop / row_max
    elif normalize is None:
        pass
    else:
        raise ValueError("unknown normalize option")
    
    # gridspec: 2 rows x 2 cols; row0 for means, row1 for heatmaps
    # height_ratios choose mean row small and heatmap row large
    height_ratios = [mean_height_ratio, 1.0]
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows=2, ncols=2, height_ratios=height_ratios, hspace=hspace, wspace=0.08)

    # mean axes (row 0)
    ax_mean_l = fig.add_subplot(gs[0, 0])
    ax_mean_r = fig.add_subplot(gs[0, 1])

    # heatmap axes (row 1)
    ax_start = fig.add_subplot(gs[1, 0], sharex=ax_mean_l)
    ax_stop  = fig.add_subplot(gs[1, 1], sharex=ax_mean_r)

    # --- plot heatmaps (no mean inside) ---
    im1, _ = plot_region_heatmap(M_start, cols_s, title="Distance to start codon (nt)",
                                read_lens=read_lens, cmap=cmap,
                                ax=ax_start, vmin=vmin, vmax=vmax)

    im2, _ = plot_region_heatmap(M_stop, cols_t, title="Distance to stop codon (nt)",
                                read_lens=read_lens, cmap=cmap,
                                ax=ax_stop, vmin=vmin, vmax=vmax)

    # --- compute and plot mean profiles on the compact mean axes ---
    mean_start = M_start.sum(axis=0)/M_start.sum()
    mean_stop  = M_stop.sum(axis=0)/M_stop.sum()

    ax_mean_l.plot(cols_s, mean_start, lw=0.9)
    ax_mean_r.plot(cols_t, mean_stop, lw=0.9)

    ax_mean_l.axvline(0, color='k', linestyle='--', lw=0.6)
    ax_mean_r.axvline(0, color='k', linestyle='--', lw=0.6)

    # tidy mean axes: remove spines and ticks to be compact
    for axm in (ax_mean_l, ax_mean_r):
        # axm.spines['top'].set_visible(False)
        # axm.spines['right'].set_visible(False)
        # axm.spines['left'].set_visible(False)
        # axm.set_yticks([])
        axm.set_ylabel("Mean", fontsize=10)
        axm.tick_params(axis='y', which='both', left=True, length=2)
        axm.tick_params(labelbottom=False)
        axm.set_xlim(ax_start.get_xlim())
    
    ax_start.tick_params(axis='x', which='both', bottom=True, length=2)
    ax_start.tick_params(axis='y', which='both', left=True, length=2)
    ax_stop.tick_params(axis='x', which='both', bottom=True, length=2)
    ax_stop.tick_params(axis='y', which='both', left=True, length=2)
    ax_start.set_ylabel("Read length (nt)")
    ax_stop.set_ylabel("")

    # Shared colorbar below the two heatmaps (use an axes occupying the bottom center)
    # create a small axes at bottom spanning both columns via GridSpec
    cbar_ax = fig.add_axes([0.5, 0.05, 0.1, 0.03])   # [left, bottom, width, height] in figure coords
    cbar = fig.colorbar(im2, cax=cbar_ax, orientation='horizontal', fraction=0.1, pad=0.1, shrink=0.2, aspect=5)
    cbar.set_label("Normalized density of RPFs", fontsize=9)
    cbar.set_ticks([0.0, 0.1, 0.2])
    cbar.ax.tick_params(labelsize=9)

    # Tight layout and save-friendly settings
    plt.subplots_adjust(left=0.1, right=0.99, top=0.98, bottom=0.15, wspace=0.08, hspace=hspace)
    fig.savefig(output_path, dpi=400)
    return fig