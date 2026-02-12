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

def plot_bar_sum_psite_along_tx(data, block_start, block_end, output_file,
                          figsize=(12,2),
                          log_y=False, 
                          block_color='grey', 
                          block_alpha=0.15, 
                          block_zorder=3):
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
    data_np = data_np.sum(axis=1)
    tx_len = len(data_np)
    print("Length of the transcript: ", tx_len)

    # 裁剪阴影区间到有效范围
    bs = max(0, float(block_start))
    be = min(float(block_end), float(tx_len))
    draw_block = (be > bs)

    # 新建 figure，并手动放置 axes（figure coordinates: 0..1）
    fig = plt.figure(figsize=figsize)
    # 调整这两个参数控制 axes 在画布中的占比（宽、高）
    ax_w = 0.94   # 宽度占比（0..1）
    ax_h = 0.72   # 高度占比（0..1）
    left = (1.0 - ax_w) / 2.0
    bottom = (1.0 - ax_h) / 2.0
    ax = fig.add_axes([left, bottom, ax_w, ax_h])

    # 去掉上和右边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # y 轴对数刻度（如果需要）
    if log_y:
        ax.set_yscale("log")

    # x 数据与绘图：确保条覆盖到 0..tx_len 的范围
    x_positions = np.arange(tx_len)  # 0,1,...,tx_len-1
    ax.bar(x_positions, data_np, width=1.0, align='edge', zorder=2, color=["#08306B","#BABABA","#DEA193"])

    # 阴影块
    if draw_block:
        # axvspan 的 xmin/xmax 以 x 轴值计，ymin/ymax 以轴高度占比计（0..1）
        ax.axvspan(bs, be, ymin=0, ymax=1, color=block_color, alpha=block_alpha, zorder=block_zorder)

    # 让条形图贴紧左右边界（去除自动 margin）
    ax.margins(x=0)

    # 坐标和标题
    ax.set_xlim(0, tx_len)   # 精确到最后的位置
    ax.set_xlabel('Position of the transcript', fontsize=10)
    ax.set_ylabel('log2 P-site density' if log_y else 'P-site density', fontsize=10)

    # 美观的刻度：根据需要打开/关
    ax.tick_params(axis='both', which='major', labelsize=9)

    # 保存：使用 tight 裁剪并留一点微边距以防 label 被裁
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close(fig)


def plot_bar_psite_along_tx(data, block_start, block_end, output_file,
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

    if log_y:
        plt.yscale("log")
    plt.subplots_adjust(left=0.1, bottom=0.1, hspace=0.5)
    plt.xlabel('Position of the transcript', fontsize=9)
    plt.ylabel('log2 P-site density' if log_y else 'P-site density', fontsize=9)

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
    axs[-1].set_title("All P-site length", fontsize=10, pad=1)

    # save pdf
    plt.savefig(output_file)
    plt.close()
