# -*- coding: utf-8 -*-
# Usage:
#   1) Prepare a test DataLoader that yields batches in your usual format,
#      e.g. (cell_idxs, seq_embs_padded, count_embs_padded, coding_embs_padded, pad_masks)
#   2) Provide your model (DDP unwrap if needed) and device.
#   3) Call plot_pr_roc_on_test(...)

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc, roc_auc_score

__author__ = "Chunfu Xiao"
__email__ = "chunfushawn@gmail.com"

def _as_numpy(x):
    """Helper: convert torch tensor or numpy to numpy array on CPU."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _sigmoid_np(x):
    return 1 / (1 + np.exp(-x))

def collect_test_coding_predictions(model, dataloader, device=None, predict_method="predict", tasks=("start", "stop", "in_orf")):
    """
    Run model on dataloader and collect logits/probs and targets for tasks.
    Returns a dict with keys for each task: {'preds': np.array, 'targets': np.array, 'mask': np.array}
    - model: nn.Module (if wrapped in DDP, use unwrap_model(model) or model.module)
    - dataloader: yields batches; last element should be pad_mask (bool tensor)
    - predict_method: "predict" or "forward" depending on your model API; "predict" allows flexible inputs
    - tasks: tuple/list of task names: expect target tensor shape (bs, L, 3) in order start,stop,in_orf
    """
    device = device or (next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu"))
    model.to(device)
    model.eval()

    preds_by_task = {t: [] for t in tasks}
    targets_by_task = {t: [] for t in tasks}

    with torch.no_grad():
        for batch in tqdm(dataloader, mininterval=1):
            # unpack batch adaptively (match your collate)
            # expected: (cell_idxs, seq_embs_padded, count_embs_padded, coding_embs_padded, pad_masks)
            # but be tolerant: user may have other ordering — adjust if necessary
            cell_idxs, seq_embs_padded, count_embs_padded, coding_embs_padded, pad_masks = [b.to(device) for b in batch]

            # get model outputs: either dict {start,stop,in_orf} with logits or a single tensor (bs,L,3)
            try:
                if predict_method == "predict":
                    out = model.predict(seq_embs_padded, count_embs_padded, cell_idxs, pad_masks, 
                                        head_names=["coding"], move_inputs_to_device=False, return_numpy=False)
                else:
                    out = model(seq_embs_padded, count_embs_padded, cell_idxs, pad_masks, 
                                        head_names=["coding"])
            except Exception:
                # fallback to calling unwrapped forward (if DDP wrapped)
                model_unwrapped = model.module if hasattr(model, "module") else model
                out = model_unwrapped.predict(seq_embs_padded, count_embs_padded, cell_idxs, pad_masks, 
                                        head_names=["coding"], move_inputs_to_device=False, return_numpy=False)
            
            
            # out is tensor (bs, L, 3) or (bs, 3, L)
            out_np = _as_numpy(out["coding"])
            if out_np.ndim == 3 and out_np.shape[2] == 3:
                logits_batch = out_np  # (bs, L, 3)
            elif out_np.ndim == 3 and out_np.shape[1] == 3:
                logits_batch = np.moveaxis(out_np, 1, 2)  # from (bs,3,L) -> (bs,L,3)
            else:
                raise ValueError("Model output shape not recognized. Expect (bs,L,3) or dict with keys start/stop/in_orf.")

            # targets and mask for this batch
            targets_batch = _as_numpy(coding_embs_padded)  # (bs, L, 3)
            mask_batch = _as_numpy(pad_masks).astype(bool)  # (bs, L)

            # For each task, select valid positions in this batch and append 1D arrays
            for idx, t in enumerate(tasks):
                preds_t = logits_batch[..., idx]  # shape (bs, L)
                tgts_t = targets_batch[..., idx]  # shape (bs, L)
                # select valid positions (boolean mask) -> 1D array
                valid_preds = preds_t[mask_batch]   # 1D
                valid_targets = tgts_t[mask_batch]  # 1D
                preds_by_task[t].append(valid_preds)
                targets_by_task[t].append(valid_targets)

    # concatenate per-task lists into long 1D arrays
    results = {}
    for t in tasks:
        if len(preds_by_task[t]) == 0:
            results[t] = {"logits": np.array([], dtype=np.float32), "targets": np.array([], dtype=np.float32)}
        else:
            results[t] = {
                "logits": np.concatenate(preds_by_task[t], axis=0),
                "targets": np.concatenate(targets_by_task[t], axis=0)
            }

    return results


def plot_pr_roc_on_test(results: dict,
                        out_dir: str = "figures",
                        model_name: str = "test",
                        tasks: tuple = ("start", "stop", "in_orf"),
                        dpi: int = 200,
                        show: bool = False):
    """
    Save one image per task. Each image contains two subplots:
      - left: Precision-Recall curve (AUPR)
      - right: ROC curve (AUROC)
    Args:
      results: dict mapping task -> {"logits": np.ndarray, "targets": np.ndarray}
      out_dir: directory to save images
      model_name: filename prefix of model_name
      tasks: ordered list/tuple of tasks to plot
      dpi: saved image DPI
      show: if True, plt.show() the figure (useful for local debugging)
    Returns:
      list of saved file paths
    """
    os.makedirs(out_dir, exist_ok=True)
    saved_paths = []

    for t in tasks:
        if t not in results:
            print(f"[save_plots] Warning: task '{t}' not found in results, skipping.")
            continue

        logits = results[t].get("logits", None)
        targets = results[t].get("targets", None)

        if logits is None or targets is None or logits.size == 0 or targets.size == 0:
            print(f"[save_plots] Task '{t}' has no samples (empty), skipping.")
            continue

        # convert logits -> probs if needed
        if np.abs(logits).max() > 1e-6:
            probs = _sigmoid_np(logits)
        else:
            # small-magnitude values -> assume already probabilities or near-prob
            probs = np.clip(logits, 0.0, 1.0)

        y_true = targets.astype(np.float64)

        # compute metrics and curves
        try:
            ap = average_precision_score(y_true, probs)
        except Exception:
            ap = float("nan")
        try:
            roc_auc = roc_auc_score(y_true, probs)
        except Exception:
            roc_auc = float("nan")

        precision, recall, _ = precision_recall_curve(y_true, probs)
        fpr, tpr, _ = roc_curve(y_true, probs)

        # draw figure for this task (one file per task)
        fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(10, 4.5))
        # PR
        ax_pr.plot(recall, precision, lw=2)
        ax_pr.set_title(f"{t} — Precision-Recall (AUPR={ap:.4f})")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_xlim([0.0, 1.0])
        ax_pr.set_ylim([0.0, 1.05])
        ax_pr.grid(True)

        # ROC
        ax_roc.plot(fpr, tpr, lw=2)
        ax_roc.plot([0,1],[0,1], color="gray", linestyle="--")
        ax_roc.set_title(f"{t} — ROC (AUROC={roc_auc:.4f})")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.grid(True)

        plt.tight_layout()
        filename = f"{model_name}.{t}_PR_ROC.pdf"
        savepath = os.path.join(out_dir, filename)
        fig.savefig(savepath, dpi=dpi)
        if show:
            plt.show()
        plt.close(fig)

        saved_paths.append(savepath)
        print(f"[save_plots] Saved {t} PR/ROC to {savepath}")

    return saved_paths
