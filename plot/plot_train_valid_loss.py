import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def pretrain_loss_plot(file_path, output_path, filename, tasks=["seq", "count", "cell"]):
    with open(file_path, 'r') as f_epoch:
        epoch_data = json.load(f_epoch)
    epoch_train_loss = [x['train_loss'] for x in epoch_data]
    epoch_val_loss = [x['valid_loss'] for x in epoch_data]
    # sequence loss
    if "seq" in tasks:
        k = tasks.index("seq")
        seq_loss_df = pd.DataFrame(
            data={
                "epoch": [i for i, data in enumerate(epoch_train_loss)],
                "train": [i[k] for i in epoch_train_loss],
                "valid": [i[k] for i in epoch_val_loss]}
        ).melt(
            id_vars='epoch', value_vars=['train', 'valid'],
            var_name='mode', value_name='loss'
        )
        min_seq_val_loss = min(seq_loss_df[seq_loss_df["mode"]=="valid"].loss)
        
        # plot
        fig, ax = plt.subplots(figsize=(6,4))
        sns.lineplot(seq_loss_df, x="epoch", y="loss", hue="mode")
        plt.axhline(min_seq_val_loss, color="gray", linestyle=":")
        ax.set_title("Pre-training for sequence")
        fig.savefig(output_path + filename + "_pretrain.seq_loss.pdf")
    
    # count loss
    if "count" in tasks:
        k = tasks.index("count")
        count_loss_df = pd.DataFrame(
            data={
                "epoch": [i for i, data in enumerate(epoch_train_loss)],
                "train":[i[k] for i in epoch_train_loss], 
                "valid": [i[k] for i in epoch_val_loss]}
        ).melt(
            id_vars='epoch', value_vars=['train', 'valid'],
            var_name='mode', value_name='loss'
        )
        losses = count_loss_df[
            (count_loss_df["mode"]=="valid") & (count_loss_df["loss"] != 0)].loss
        if losses.empty:
            min_count_val_loss = 1
        else:
            min_count_val_loss = min(losses)
    
        fig, ax = plt.subplots(figsize=(6,4))
        sns.lineplot(count_loss_df, x="epoch", y="loss", hue="mode")
        plt.axhline(min_count_val_loss, color="gray", linestyle=":")
        ax.set_title("Pre-training for RPF")
        fig.savefig(output_path + filename + "_pretrain.count_loss.pdf")
    
    # count loss
    if "cell" in tasks:
        k = tasks.index("cell")
        cell_loss_df = pd.DataFrame(
            data={
                "epoch": [i for i, data in enumerate(epoch_train_loss)],
                "train":[i[k] for i in epoch_train_loss], 
                "valid": [i[k] for i in epoch_val_loss]}
        ).melt(
            id_vars='epoch', value_vars=['train', 'valid'],
            var_name='mode', value_name='loss'
        )
        losses = cell_loss_df[
            (cell_loss_df["mode"]=="valid") & (cell_loss_df["loss"] != 0)].loss

        if losses.empty:
            min_cell_val_loss = 1
        else:
            min_cell_val_loss = min(losses)
    
        fig, ax = plt.subplots(figsize=(6,4))
        sns.lineplot(cell_loss_df, x="epoch", y="loss", hue="mode")
        plt.axhline(min_cell_val_loss, color="gray", linestyle=":")
        ax.set_title("Pre-training for cell")
        fig.savefig(output_path + filename + "_pretrain.cell_loss.pdf")
    #plt.close(fig)