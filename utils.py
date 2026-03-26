import os
import sys
import torch
import torch.nn as nn
import numpy as np
import gc

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def print_param_counts(model: nn.Module) -> None:
    """
    Print total parameter count and number of trainable parameters.
    Useful to verify replacement and trainability settings after freezing/unfreezing.
    """
    model = unwrap_model(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: trainable {trainable:,} / total {total:,} ({100.0 * trainable / total:.2f}% trainable)")
    
# -------------------------
# Utilities for DDP wrappers
# -------------------------

def unwrap_model(model: nn.Module) -> nn.Module:
    """
    If model is wrapped in DistributedDataParallel or DataParallel, return the underlying module,
    otherwise return model itself. This is useful because named_modules() on wrappers includes the wrapper.
    """
    # DDP wrappers in torch usually expose `.module`
    return getattr(model, "module", model)


def clean_up_memory():
    if 'dataset' in globals(): del dataset
    if 'saved_data' in globals(): del saved_data
    if 'dataloader' in globals(): del dataloader
    
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("Memory clean up finished.")


def inspect_adaln_weights(model):
    print("=== 🕵️ 细胞类型 Embedding 权重诊断报告 ===")
    
    # 获取细胞类型到 Index 的映射，并反转为 Index -> Cell Type
    mapping = model.cell_type_mapping
    idx_to_cell = {v: k for k, v in mapping.items()}
    idx_to_cell[0] = "Unknown/Padding" # 0 通常是预留的 Padding 或未知细胞类型
    
    # ---------------------------------------------------------
    # 检查目标 1: 检查第一层 Transformer Block 内部的 AdaLN 嵌入
    # ---------------------------------------------------------
    print("\n[1] 检查 Layer 0 -> MHA Sublayer -> cell_embed ...")
    try:
        # 定位到: model.encoder.encoder_layers[0].sublayers[0].cell_embed
        block0_emb = model.encoder.encoder_layers[0].sublayers[0].cell_embed.weight.detach().cpu().numpy()
        
        for idx in range(block0_emb.shape[0]):
            cell_name = idx_to_cell.get(idx, f"Unused-{idx}")
            w = block0_emb[idx]
            l2_norm = np.linalg.norm(w)
            has_nan = np.isnan(w).any()
            has_inf = np.isinf(w).any()
            is_zero = np.all(w == 0)
            
            # 高亮显示可能存在问题的状态
            status = "✅ 正常"
            if has_nan or has_inf: status = "🚨 崩溃 (NaN/Inf)"
            elif is_zero: status = "💀 死亡 (All Zero)"
            elif l2_norm > 100: status = "⚠️ 范数过大 (可能梯度爆炸)"
            elif l2_norm < 1e-4: status = "⚠️ 范数过小 (可能梯度消失)"
                
            print(f"  [{idx}] {cell_name:<18}: L2 Norm = {l2_norm:8.4f} | {status}")
    except Exception as e:
        print(f"  ❌ 无法读取第一层 AdaLN 权重: {e}")

    # ---------------------------------------------------------
    # 检查目标 2: 检查 Encoder 最后的 final_embed
    # ---------------------------------------------------------
    print("\n[2] 检查 Encoder -> final_embed ...")
    try:
        final_emb = model.encoder.final_embed.weight.detach().cpu().numpy()
        
        for idx in range(final_emb.shape[0]):
            cell_name = idx_to_cell.get(idx, f"Unused-{idx}")
            w = final_emb[idx]
            l2_norm = np.linalg.norm(w)
            has_nan = np.isnan(w).any()
            has_inf = np.isinf(w).any()
            is_zero = np.all(w == 0)
            
            status = "✅ 正常"
            if has_nan or has_inf: status = "🚨 崩溃 (NaN/Inf)"
            elif is_zero: status = "💀 死亡 (All Zero)"
            elif l2_norm > 100: status = "⚠️ 范数过大"
            elif l2_norm < 1e-4: status = "⚠️ 范数过小"
                
            print(f"  [{idx}] {cell_name:<18}: L2 Norm = {l2_norm:8.4f} | {status}")
    except Exception as e:
        print(f"  ❌ 无法读取 final_embed 权重: {e}")
        
    print("\n==============================================")

# 调用方式：
# inspect_adaln_weights(your_loaded_model)