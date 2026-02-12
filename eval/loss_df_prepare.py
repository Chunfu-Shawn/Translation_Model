import sys
import os
import json
import pandas as pd

def json2dataframe(file_path, tasks=["count", "cell"]):
    with open(file_path, 'r') as f_epoch:
        epoch_data = json.load(f_epoch)
    epoch_train_loss = [x['train_loss'] for x in epoch_data]
    epoch_val_loss = [x['valid_loss'] for x in epoch_data]
    task_order = ["count", "cell"]
    loss_df = {}

    # get loss
    for task in tasks:
        k = task_order.index(task)
        loss_df[task] = pd.DataFrame(
            data={
                "epoch": [i for i, data in enumerate(epoch_train_loss)],
                "train": [i[k] for i in epoch_train_loss],
                "valid": [i[k] for i in epoch_val_loss]}
        ).melt(
            id_vars='epoch', value_vars=['train', 'valid'],
            var_name='mode', value_name='loss'
        )

    return loss_df

def loss_dataframe_prepare(
        json_files: list = [],
        output_dir: str = "/home/user/data3/rbase/translation_model/results/pretrain/loss",
        hyper_p: dict = {
            "model_size": [],
            "learning_rate": []},
        prefix = "model",
        tasks: list = ["count", "cell"]
        ):
    df = {task: pd.DataFrame() for task in tasks}
    for i in range(len(json_files)):
        tmp_df = json2dataframe(json_files[i], tasks)
        for task in tasks:
            tmp_df[task]["model_size"] = hyper_p["model_size"][i]
            tmp_df[task]["learning_rate"] = hyper_p["learning_rate"][i]
            df[task] = pd.concat([df[task], tmp_df[task]], axis=0)
    
    # save df
    for task in tasks:
        df[task].to_csv(
            os.path.join(output_dir, f"{prefix}.loss({task})_train_valid_hyper_parameters.csv"),
            index=False)

if __name__ == "__main__":
    log_dir = '/home/user/data3/rbase/translation_model/models/log/pretrain/'
    loss_dataframe_prepare(
    json_files=[
        log_dir + "base_model_128d_4h_6l_1c_1drop-MaskedDensityHead-MaskedCellTypeHead.0.01_0.001_0.3.epoch_data.json"
    ],
    hyper_p={
            "model_size": ["128d_4h_6l"],
            "learning_rate": ["0.001"]
    },
    tasks=['count'])