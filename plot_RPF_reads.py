import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from genome_tx_exon_index import *
from data_generate_RPF_count import *

__author__ = "Chunfu Xiao"
__contributor__="..."
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__="1.0.0"
__maintainer__ = "Chunfu Xiao"
__email__ = "chunfushawn@gmail.com"

def plot_bar_RPF_along_tx(RPF_count, tx_info, output_file, log_y=False):
    read_len = sorted(list(RPF_count.keys()))
    num_read_len = len(read_len)
    if num_read_len == 0:
        print("No RPF with this transcript")
        return
    fig, axs = plt.subplots(num_read_len + 2, 1, figsize=(12, 8), sharex=True)

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
    plt.ylabel('log2 RPF count' if log_y else 'RPF count', fontsize=8)

    # different read length
    tx_len = tx_info["tx_ends"][-1]
    sum = np.zeros(tx_len)
    for i in range(0,num_read_len):
        l = read_len[i]
        y = np.zeros(tx_len)
        for pos in RPF_count[l]:
            y[pos-1] = RPF_count[l][pos]
            sum[pos-1] += RPF_count[l][pos]
        axs[i].bar(range(0, tx_len), y, width=1, color='#08306B') #["#08306B","#BABABA","#E0E0E0"])
        axs[i].set_title(str(l)+" nt", fontsize=10, pad=0.5)
    # total read length
    axs[-2].bar(range(0, tx_len), sum, width=1, color='#08306B') #["#08306B","#BABABA","#E0E0E0"])
    axs[-2].set_title("All RPF length", fontsize=10, pad=1)

    # coding region
    axs[-1].axvspan(tx_info["cds_start"], tx_info["cds_end"], ymin=0.05, ymax=0.45, color='darkorange', label='Coding Region')
    axs[-1].legend(loc='upper right')

    # save pdf
    plt.savefig(output_file)
    plt.close()

def plot_heatmap_RPF_along_tx(RPF_count, tx_info, output_file, log_y=False):
    # data prepare
    read_len = sorted(list(RPF_count.keys()))
    if len(read_len) == 0:
        print("No RPF with this transcript")
        return
    counts = pd.DataFrame(data=None, columns=[str(x) + " nt" for x in read_len])
    tx_len = tx_info["tx_ends"][-1]
    sum = np.zeros(tx_len)
    for l in RPF_count.keys():
        y = np.zeros(tx_len)
        for pos in RPF_count[l]:
            y[pos-1] = RPF_count[l][pos]
            sum[pos-1] += RPF_count[l][pos]
        counts[str(l) + " nt"] = np.log2(y + 1) if log_y else y
    # total
    counts["All RPF length"] = np.log2(sum + 1) if log_y else sum
    print(counts.T)

    # plot
    plt.figure(figsize=(10,4))
    plt.subplots_adjust(left=0.15, bottom=0.2)
    plot = sns.heatmap(counts.T, cmap="YlGnBu", cbar_kws={'label': 'log2 RPF count' if log_y else 'RPF count'})
    plt.xlabel('Position of the transcript', fontsize=9)
    plt.ylabel('RPF length', fontsize=11)

    # save pdf
    plt.savefig(output_file)
    plt.close()


if __name__=="__main__":
    RPF_count_file = '/home/user/data3/rbase/translation_pred/models/test/SRR15513158.v48.read_count.pkl'
    tx_arrays_file = '/home/user/data3/rbase/translation_pred/models/lib/transcript_arrays.pkl'
    output_dir = '/home/user/data3/rbase/translation_pred/models/test/figures'
    tx_ids = ['ENST00000382361.8']
    # load results
    with open(RPF_count_file, 'rb') as f_RPF:
        final_counts = pickle.load(f_RPF)
    with open(tx_arrays_file, 'rb') as f_tx:
        tx_arrays = pickle.load(f_tx)
    
    # plot
    for tx_id in tx_ids:
        target_counts = final_counts[tx_id]
        tx_info = tx_arrays[tx_id]
        print("--- plot the read distribution of " + tx_id + " ---")
        # plot 
        plot_bar_RPF_along_tx(
            target_counts, tx_info, output_dir + "/Barplot RPF count along the transcript of "+ tx_id + ".pdf", True
            )
        plot_heatmap_RPF_along_tx(
            target_counts, tx_info, output_dir + "/Heatmap RPF count along the transcript of "+ tx_id + ".pdf", True
            )
    # plot_three_nucleotide_periodicity(
    #     target_counts, 
    #     output_dir + "/three_nt_periodicity_within_ORF."+ tx_id + ".pdf")