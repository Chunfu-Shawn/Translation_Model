import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

def plot_onehot_windows(mat, block_start, block_end,
                        flank=20,
                        output_file=None,
                        figsize=(10,2),
                        one_color='#ADD8E6',   # 浅蓝
                        bg_color='white',
                        highlight_color='grey',
                        show_xticks=True):
    """
    只绘制以 block_start 和 block_end 为中心，各向前后扩长 `flank` nt 的窗口（默认 flank=20）。
    输入:
      mat: numpy array，形状 (N, L) 或 (N, L, C)（若为 one-hot 的多通道，自动折叠为二值）
      block_start, block_end: 整数位置（0-based）
      flank: 在 start/end 两侧扩展的长度（默认 20）
      output_file: 若为字符串则保存图片并关闭；否则返回 fig, ax, highlighted_rows
      one_color: one-hot = 1 的颜色（浅蓝默认）
      bg_color: 0 的背景色（白色）
      highlight_color: 高亮行边框颜色
    返回:
      (fig, ax, highlighted_rows) 或 (None, None, highlighted_rows) 若保存后关闭图
    """

    mat2d = np.asarray(mat)
    if mat.ndim == 2:
        mat = np.asarray(mat)
        L, D = mat.shape
        mat2d = np.zeros_like(mat, dtype=int)
        idx = np.argmax(mat, axis=1)            # 每行第一个最大值的列索引
        mat2d[np.arange(L), idx] = 1
    else:
        raise ValueError("mat must be 2D numpy array")

    n_cols, n_rows = mat2d.shape
    

    # 计算窗口 [s-flank, e+flank] 并裁剪到合法范围
    s = max(0, int(block_start) - int(flank))
    e = min(n_cols, int(block_end) + int(flank) + 1)

    # 提取子矩阵并记录原始列索引
    sub_mats = mat2d[s:e, :]
    sub_ncols = sub_mats.shape[0]


    # 构造只含 0/1 的二色 colormap：0->bg_color, 1->one_color
    cmap = ListedColormap([bg_color, one_color])

    # 绘图
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(sub_mats.T, aspect='auto', interpolation='nearest', cmap=cmap, vmin=0, vmax=1)


    ax.set_xticks(np.arange(0, sub_ncols))
    ax.set_xticklabels([i for i in range(s, e)], fontsize=6, rotation=90)
    ax.set_yticks(np.arange(0, n_rows))
    ax.set_yticklabels([str("ACGT")[i] for i in range(n_rows)], fontsize=6)

    ax.axvspan(block_start-0.5, block_end-1.5, ymin=0, ymax=1, 
                facecolor=highlight_color, alpha=0.15, 
                zorder=3, edgecolor="white", linewidth=0, antialiased=False)

    ax.set_xlabel("Position (original coordinates)")
    ax.set_ylabel("Sequence index (row)")
    ax.set_title(f"Windows around {block_start} and {block_end} (flank={flank})")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
