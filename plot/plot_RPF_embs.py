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

def plot_bar_RPF_along_tx(data, output_file, log_y=False):
    data_np = data.cpu().numpy()
    tx_len, num_read_len = data.size()

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

    # different read length
    for i in range(0, num_read_len):
        axs[i].bar(range(0, tx_len), data_np[:, i], width=1, color='#08306B') #["#08306B","#BABABA","#E0E0E0"])
        axs[i].set_title(str(25 + i)+" nt", fontsize=10, pad=0.5)

    # total read length
    sum = np.sum(data_np, axis=1)
    print(sum)
    axs[-1].bar(range(0, tx_len), sum, width=1, color='#08306B') #["#08306B","#BABABA","#E0E0E0"])
    axs[-1].set_title("All RPF length", fontsize=10, pad=1)


    # save pdf
    plt.savefig(output_file)
    plt.close()