import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import json
import os

def plot_scaling_law_curves(models_config, global_seq_len=1024, save_path=None):
    """
    绘制训练计算量(FLOPs)与性能(Loss)的关系曲线。
    修改点：相同 method 的配置将使用相同的颜色绘制。
    """
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(9, 6))
    
    # 颜色库
    colors = ['#4e8558', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#333333', '#17becf']
    markers = ['o', 's', '^', 'v', 'D', 'x', '*', 'p', '+']
    
    # --- [修改核心] 1. 预先构建 Method -> Color 的映射字典 ---
    # 提取所有唯一的 method 名称并排序，确保颜色分配稳定
    unique_methods = sorted(list(set(conf.get('method', 'Unknown') for conf in models_config)))
    
    # 为每个 method 分配一个颜色
    method_color_map = {method: colors[i % len(colors)] for i, method in enumerate(unique_methods)}
    
    print(f"颜色分配方案: {method_color_map}")

    for i, config in enumerate(models_config):
        # 1. 获取模型基本信息
        params = config.get('params')
        method = config.get('method', 'Unknown')
        lr = config.get('lr', '')
        
        # 2. 确定 Epoch 数据量 (Tokens)
        if 'tokens_per_epoch' in config:
            tokens_per_epoch = config['tokens_per_epoch']
        else:
            samples = config.get('dataset_samples', 0)
            tokens_per_epoch = samples * global_seq_len
            
        if tokens_per_epoch == 0:
            print(f"警告: 模型 {method} 的数据量为 0，无法计算 FLOPs，跳过。")
            continue

        # 3. 加载 Loss 数据
        data = []
        if 'loss_data' in config:
            data = config['loss_data']
        elif 'loss_path' in config and os.path.exists(config['loss_path']):
            try:
                with open(config['loss_path'], 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"读取文件失败: {config['loss_path']}, 错误: {e}")
                continue
        else:
            print(f"警告: 找不到模型 {method} 的 Loss 数据。")
            continue

        # 4. 解析数据
        epochs = []
        val_losses = []
        
        for entry in data:
            epochs.append(entry['epoch'])
            val_loss_val = entry['valid_loss'][0] if isinstance(entry['valid_loss'], list) else entry['valid_loss']
            val_losses.append(val_loss_val)
        
        epochs = np.array(epochs)
        val_losses = np.array(val_losses)

        # 5. 计算 Cumulative FLOPs
        flops_per_epoch = 6 * params * tokens_per_epoch
        cumulative_flops = epochs * flops_per_epoch

        # 6. 生成图例标签
        label = f"{method}"
        if lr:
            label += f" (lr={lr})"

        # 7. 绘图
        # --- [修改核心] 使用映射字典获取颜色 ---
        assigned_color = method_color_map[method]
        
        plt.semilogy(cumulative_flops, val_losses,
                     label=label,
                     color=assigned_color,          # 使用由 method 决定的颜色
                     marker=markers[i % len(markers)], # Marker 依然随 Index 变化，以便区分同一 method 下的不同实验(如不同lr)
                     markersize=5,
                     linewidth=1.5,
                     alpha=0.9)

    # ================= 美化部分 =================
    plt.xlabel('Training Compute (FLOPs)', fontsize=14, labelpad=10)
    plt.ylabel('Validation Loss', fontsize=14, labelpad=10)

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-1, 1))
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=11, frameon=True, framealpha=0.9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"图表已保存至: {save_path}")
    plt.show()