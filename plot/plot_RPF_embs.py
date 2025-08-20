import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import numpy as np

__author__ = "Chunfu Xiao"
__contributor__="..."
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__="1.0.0"
__maintainer__ = "Chunfu Xiao"
__email__ = "chunfushawn@gmail.com"

def plot_bar_RPF_along_tx(data, block_start, block_end, output_file,
                          log_y=False, block_color='grey', block_alpha=0.15, block_zorder=3):
    """
    在每个子图上从 block_start 到 block_end 添加阴影块并绘制 bar 图。
    参数:
      data: torch.Tensor 或类似 (tx_len, num_read_len)，会用 data.cpu().numpy()
      block_start, block_end: 阴影区间（基于位置的索引，int 或 float）
      output_file: 保存的文件路径
      log_y: 是否使用对数 y 轴
      block_color: 阴影颜色
      block_alpha: 阴影透明度
      block_zorder: 阴影 z-order（> bars 则覆盖在柱子之上）
    """
    # 转 numpy
    try:
        data_np = data.cpu().numpy()
    except AttributeError:
        data_np = np.asarray(data)
    tx_len, num_read_len = data_np.shape

    # 裁剪阴影区间到有效范围
    bs = max(0, float(block_start))
    be = min(float(block_end), float(tx_len))
    if be <= bs:
        # 如果无效区间则设为 None（不画）
        draw_block = False
    else:
        draw_block = True

    fig, axs = plt.subplots(num_read_len + 1, 1, figsize=(12, 8), sharex=True)

    # set for subplot
    for ax in axs[:-1]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        if log_y:
            ax.set_yscale('log')

    # set for bottom plot
    axs[-1].spines["top"].set_visible(False)
    axs[-1].spines["right"].set_visible(False)
    axs[-1].spines["left"].set_visible(False)
    axs[-1].tick_params(axis='y', which='both', left=False, labelleft=False)

    if log_y:
        plt.yscale("log")
    plt.subplots_adjust(left=0.1, bottom=0.1, hspace=0.5)
    plt.xlabel('Position of the transcript', fontsize=9)
    plt.ylabel('log2 RPF embeddings' if log_y else 'RPF embeddings', fontsize=8)

    x_positions = range(0, tx_len)

    # different read length
    for i in range(0, num_read_len):
        ax = axs[i]
        # 先绘制阴影（放在上层或下层由 block_zorder 决定）
        if draw_block:
            ax.axvspan(bs, be, ymin=0, ymax=1, 
                       facecolor=block_color, alpha=block_alpha, 
                       zorder=block_zorder, edgecolor="white", linewidth=0, antialiased=False)
        ax.bar(x_positions, data_np[:, i], width=1, color='#08306B', zorder=2)
        ax.set_title(str(25 + i) + " nt", fontsize=10, pad=0.5)

    # total read length
    total = np.sum(data_np, axis=1)
    axs[-1].bar(x_positions, total, width=1, color='#08306B', zorder=2)
    if draw_block:
        axs[-1].axvspan(bs, be, ymin=0, ymax=1, color=block_color, alpha=block_alpha, zorder=block_zorder)
    axs[-1].set_title("All RPF length", fontsize=10, pad=1)

    # save pdf
    plt.savefig(output_file)
    plt.close()
