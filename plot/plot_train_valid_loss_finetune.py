import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_finetune_loss(file_path, output_path, suffix="."):
    with open(file_path, 'r') as f_epoch:
        epoch_data = json.load(f_epoch)

    epoch_train_loss = [x['train_loss'] for x in epoch_data]
    epoch_val_loss = [x['valid_loss'] for x in epoch_data]
    loss_df = pd.DataFrame(
        data={
            "epoch": [i for i, data in enumerate(epoch_train_loss)],
            "train": [i for i in epoch_train_loss],
            "valid": [i for i in epoch_val_loss]}
    ).melt(
        id_vars='epoch', value_vars=['train', 'valid'],
        var_name='mode', value_name='loss'
    )
    min_val_loss = min(loss_df[loss_df["mode"]=="valid"].loss)
    
    # plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.lineplot(loss_df, x="epoch", y="loss", hue="mode")
    plt.axhline(min_val_loss, color="gray", linestyle=":")
    ax.set_title("Pre-training for TIS prediction")


    fig.savefig(output_path + f"finetune_loss.{suffix}.pdf")
    plt.show()


def plot_finetune_batch_loss(file_path, output_path, train_window=50, suffix=".", log=False):
    # 1. Load data
    with open(file_path, 'r') as f:
        log_data = json.load(f)

    train_records = []
    valid_records = []

    # 2. Data parsing and fractional epoch mapping
    for epoch_idx, epoch_data in enumerate(log_data):
        epoch_num = epoch_data.get("epoch", epoch_idx + 1)
        
        # --- Process Train Data ---
        t_batches = epoch_data.get("train_loss", [])
        n_t = len(t_batches)
        for i, batch in enumerate(t_batches):
            # Map batch to continuous epoch float (e.g., middle of first round is 0.5)
            epoch_float = (epoch_num - 1) + (i + 1) / n_t
            train_records.append({
                "epoch": epoch_float, 
                "TIS_loss": batch[0], 
                "TTS_loss": batch[1]
            })

        # --- Process Valid Data ---
        v_batches = epoch_data.get("valid_loss", [])
        n_v = len(v_batches)
        for i, batch in enumerate(v_batches):
            epoch_float = (epoch_num - 1) + (i + 1) / n_v
            valid_records.append({
                "epoch": epoch_float, 
                "TIS_loss": batch[0], 
                "TTS_loss": batch[1]
            })

    # Convert to DataFrame
    df_train = pd.DataFrame(train_records)
    df_valid = pd.DataFrame(valid_records)

    # 3. Calculate smoothed curves (EMA)
    # Training set smoothing
    df_train['TIS_smoothed'] = df_train['TIS_loss'].ewm(span=train_window, adjust=False).mean()
    df_train['TTS_smoothed'] = df_train['TTS_loss'].ewm(span=train_window, adjust=False).mean()

    # Validation set smoothing (since validation batches are usually fewer, use smaller window to avoid over-smoothing)
    val_window = max(5, train_window // 5) 
    df_valid['TIS_smoothed'] = df_valid['TIS_loss'].ewm(span=val_window, adjust=False).mean()
    df_valid['TTS_smoothed'] = df_valid['TTS_loss'].ewm(span=val_window, adjust=False).mean()

    # 4. Plot setup (1 row, 2 columns)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Helper function to plot loss for a task
    def plot_task_loss(ax, task_name, raw_col, smoothed_col):
        # Plot Training (blue tones)
        sns.scatterplot(
            data=df_train, x="epoch", y=raw_col, 
            color="lightblue", alpha=0.3, s=15, edgecolor=None, label="Train Batch", ax=ax
        )
        sns.lineplot(
            data=df_train, x="epoch", y=smoothed_col, 
            color="blue", linewidth=2, label="Train Smoothed", ax=ax
        )

        # Plot Validation (red/orange tones)
        sns.scatterplot(
            data=df_valid, x="epoch", y=raw_col, 
            color="lightcoral", alpha=0.5, s=20, edgecolor=None, label="Valid Batch", ax=ax
        )
        sns.lineplot(
            data=df_valid, x="epoch", y=smoothed_col, 
            color="red", linewidth=2.5, linestyle="--", label="Valid Smoothed", ax=ax
        )

        ax.set_title(f"{task_name} Prediction Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right")

        if log:
            ax.set_yscale("log")
    

    # Plot left: TIS (Index 0)
    plot_task_loss(axes[0], "TIS", "TIS_loss", "TIS_smoothed")

    # Plot right: TTS (Index 1)\
    plot_task_loss(axes[1], "TTS", "TTS_loss", "TTS_smoothed")

    # 5. Overall layout and save
    plt.suptitle("Fine-tuning Batch Loss Dynamics", y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path + f"finetune_tis_tts_loss.{suffix}.pdf", bbox_inches='tight')
    plt.show()
