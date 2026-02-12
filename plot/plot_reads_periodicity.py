import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white")

def plot_length_periodicity(lengths, mat, totals,
                            output_path,
                            cmap="Greys",
                            figsize=(6,8),
                            heatmap_width=0.60,
                            gap=0.04,
                            bar_width=0.20,
                            left_margin=0.12,
                            right_margin=0.92,
                            top_margin=0.85,
                            bottom_margin=0.05,
                            ytick_max=25):
    """
    Improved layout:
      - heatmap left, bar right, colorbar at bottom full-width (inside margins)
      - right bar shows only bottom x-axis ticks (percent), no y ticks
      - more relaxed margins so nothing is clipped

    参数说明:
      heatmap_width: 热图占 figure 宽度的比例(0..1)
      gap: heatmap 与 bar 间隔 (figure 宽度的比例）
      bar_width: 右边柱状图宽度比例（剩下由 heatmap_width 决定）
      left_margin/right_margin/top_margin/bottom_margin: figure 边距 (0..1)
      ytick_max: 若行数 > ytick_max, 则只显示采样后的 y ticks (防止重叠)
    """
    # M: mat normalized rows (n_len x 3)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    M = mat / row_sums
    n_len = M.shape[0]

    # percent reads per length
    total_all = totals.sum() if totals.sum() > 0 else 1.0
    pct = totals / total_all * 100.0

    # compute axes positions in figure coords
    fig = plt.figure(figsize=figsize)
    # compute usable width between left_margin and right_margin
    usable_w = right_margin - left_margin
    hm_w = heatmap_width * usable_w
    bar_w = bar_width * usable_w
    # place heatmap at left_margin ... left_margin + hm_w
    hm_left = left_margin
    hm_bottom = bottom_margin
    hm_height = top_margin - bottom_margin
    heatmap_rect = [hm_left, hm_bottom, hm_w, hm_height]

    # bar placed at heatmap_right + gap
    bar_left = hm_left + hm_w + gap
    bar_rect = [bar_left, hm_bottom, bar_w, hm_height]

    # colorbar full width under heatmap and bar but inside margins
    cbar_left = left_margin
    cbar_width = right_margin - left_margin
    cbar_rect = [cbar_left, 0.95, cbar_width * 0.6, 0.03]  # height small

    # create axes
    ax_heat = fig.add_axes(heatmap_rect)
    ax_bar  = fig.add_axes(bar_rect)

    # plot heatmap: rows -> lengths, we want shortest at bottom so reverse ordering when plotting
    im = ax_heat.imshow(M[::-1, :], aspect='auto', origin='lower',
                       cmap=cmap, interpolation='nearest', vmin=0.0, vmax=0.7,
                       extent=[0-0.5, 2+0.5, 0, n_len])

    # y ticks: sample ticks to avoid overlap
    if n_len <= ytick_max:
        yticks_idx = np.arange(0, n_len)
        yticks_labels = [str(lengths[i]) for i in (n_len-1 - yticks_idx)]
        ax_heat.set_yticks(yticks_idx + 0.5)
        ax_heat.set_yticklabels(yticks_labels, fontsize=8)
    else:
        # show roughly 20 ticks
        n_ticks = min(20, ytick_max)
        idxs = np.linspace(0, n_len-1, n_ticks, dtype=int)
        yticks_idx = idxs
        yticks_labels = [str(lengths[n_len-1 - i]) for i in idxs]
        ax_heat.set_yticks(yticks_idx + 0.5)
        ax_heat.set_yticklabels(yticks_labels, fontsize=7)

    ax_heat.set_xlabel("Coding frame", fontsize=9)
    ax_heat.set_ylabel("Read length (nt)", fontsize=9)
    ax_heat.set_xticks([0,1,2])
    ax_heat.set_xticklabels([ "0", "1", "2" ], fontsize=9)
    ax_heat.tick_params(axis='both', which='both', length=3)

    # vertical divider (optional): small separation visually
    ax_heat.spines['right'].set_visible(False)
    ax_heat.spines['top'].set_visible(False)

    # right bar: horizontal bars aligned with heatmap rows; reversed order for alignment
    y = np.arange(n_len) + 0.5
    pct_rev = pct[::-1]
    ax_bar.barh(y, pct_rev, height=0.8, align='center', color='gray')
    ax_bar.set_ylim(0, n_len)
    # hide y axis entirely on bar (reads labels on heatmap)
    ax_bar.yaxis.set_visible(False)

    # Show only bottom x-axis ticks on the bar (default behavior) — ensure top ticks off
    ax_bar.xaxis.set_ticks_position('bottom')
    ax_bar.xaxis.set_label_position('bottom')
    ax_bar.set_xlabel("% of reads", fontsize=9)
    ax_bar.tick_params(axis='x', labelsize=8)

    # tighten the heatmap/bar spacing visually (we already left 'gap')
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_visible(False)
    ax_bar.spines['bottom'].set_visible(True)

    # add colorbar at bottom using our cax
    cax = fig.add_axes(cbar_rect)
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label("fraction of reads per length", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6])

    # final layout adjustments: larger margins so figure feels "宽松"
    plt.subplots_adjust(left=left_margin*0.8, right=right_margin*1.01, top=top_margin*0.99, bottom=bottom_margin*0.6)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    
    return fig, (ax_heat, ax_bar, cbar)