from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="white")

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="white")


def plot_offset_confusion_matrix_heatmap(mat, 
                                         output_path,
                                         cmap="RdYlBu_r",
                                         figsize=(4, 4), 
                                         vmin=None, vmax=None):
    """
    Plot heatmap with a short vertical colorbar placed at the side
    """
    # create figure and axis (do NOT use tight_layout later)
    fig, ax = plt.subplots(figsize=figsize)

    vmin = mat.min() if vmin is None else vmin
    vmax = mat.max() if vmax is None else vmax

    # draw heatmap with seaborn but don't create a colorbar there
    mesh = sns.heatmap(
        mat, ax=ax, cmap=cmap, square=True, linewidths=0.6,
        linecolor='white', cbar=False, annot=False, fmt='d',
        vmin=vmin, vmax=vmax
    ).collections[0]

    # add a black frame around the heatmap using axes-relative coords
    border = Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                      edgecolor='black', linewidth=1.5, clip_on=False, zorder=20)
    ax.add_patch(border)

    # size can be a fraction string (e.g. "5%") or a numeric value; pad is space
    cax = inset_axes(ax, width="120%", height="100%", loc='upper right',
                 bbox_to_anchor=(1.02, 0.5, 0.05, 0.5), bbox_transform=ax.transAxes, borderpad=-0.5)
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Number of reads", rotation=90, labelpad=10, fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    # set ticks (few ticks, integer labels)
    ticks = np.linspace(vmin, vmax, num=5)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([str(int(t)) for t in ticks])

    # beautify labels
    ax.set_xlabel('P-site offset prediction', fontsize=12)
    ax.set_ylabel('P-site offset truth', fontsize=12)
    ax.invert_yaxis()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # plot and save
    plt.show()
    fig.subplots_adjust(left=0.2, right=0.7, top = 1.0, bottom= 0)
    fig.savefig(output_path)

    return fig