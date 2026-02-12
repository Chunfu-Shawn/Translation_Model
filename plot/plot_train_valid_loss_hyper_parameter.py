import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def pretrain_valid_loss_plot(loss_df_file, file_path, figsize=(6,4)):
    df = pd.read_csv(loss_df_file)
    df_valid_loss = df[df["mode"]=="valid"]
    # plot
    fig, ax = plt.subplots(figsize= figsize)
    sns.lineplot(df_valid_loss, x="epoch", y="loss", hue="model_size", style="learning_rate")
    # plt.axhline(min_seq_val_loss, color="gray", linestyle=":")
    ax.set_title("Pre-training validation loss")
    fig.savefig(file_path)