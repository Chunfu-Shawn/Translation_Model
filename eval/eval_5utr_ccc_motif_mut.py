import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from plotnine import *
from eval.calculate_te import *

class CCC_Motif_Evaluator:
    def __init__(self, model, out_dir="."):
        """
        In-silico evaluation of 5' UTR CCC motif dose-dependency.
        """
        self.device = model.device
        self.out_dir = out_dir
        
        # --- Model Predict Wrapper ---
        def model_predict_wrapper(seq_emb_input, cell_type_emb_input):
            # 1. 堆叠 Numpy Array
            if isinstance(seq_emb_input, list):
                # 这里的 stack 要求 list 中所有 array 形状相同
                # 我们通过在 evaluate 循环中逐样本预测来保证这一点
                seq_batch_np = np.stack(seq_emb_input)
                cell_batch_np = np.stack(cell_type_emb_input)
            else:
                seq_batch_np = seq_emb_input
                cell_batch_np = cell_type_emb_input

            # 2. 转 Tensor
            seq_tensor = torch.from_numpy(seq_batch_np).float().to(self.device)
            cell_tensor = torch.from_numpy(cell_batch_np).float().to(self.device)
            
            # 3. 预测
            model.eval()
            with torch.no_grad():
                output_dict = model.predict(seq_batch=seq_tensor, cell_type=cell_tensor, head_names=["count"]) 
                pred = output_dict['count']
                
            pred_np = pred.cpu().numpy()
            pred_np = np.expm1(pred_np) # Linear space
            return pred_np
        
        self.predict = model_predict_wrapper
        
        # --- Base Maps ---
        self.BASES_MAP = {
            'A': np.array([1, 0, 0, 0], dtype=np.float32),
            'C': np.array([0, 1, 0, 0], dtype=np.float32),
            'G': np.array([0, 0, 1, 0], dtype=np.float32),
            'T': np.array([0, 0, 0, 1], dtype=np.float32), # DNA T / RNA U
            'N': np.array([0, 0, 0, 0], dtype=np.float32)
        }
        self.REVERSE_MAP = {tuple(v): k for k, v in self.BASES_MAP.items()}


    def decode_seq(self, emb):
        """将 One-hot 解码为字符串"""
        seq = []
        limit = min(len(emb), 100)
        for i in range(limit):
            vec = emb[i]
            key = tuple(vec)
            seq.append(self.REVERSE_MAP.get(key, 'N'))
        return "".join(seq)

    def inject_ccc(self, base_emb, count, start_offset=3, gap=4):
        """在前 50nt 内植入 CCC (替换模式，保持长度不变)"""
        new_emb = base_emb.copy()
        c_vec = self.BASES_MAP['C']
        
        current_pos = start_offset
        
        for _ in range(count):
            if current_pos + 3 > 50: 
                return None
            
            for i in range(3):
                new_emb[current_pos + i] = c_vec
            
            current_pos += 3 + gap
            
        return new_emb

    def evaluate_ccc_dose(self, samples):
        """
        执行筛选和突变分析。
        修改点：不再批量收集所有序列，而是逐样本处理，避免长度不一致导致的 np.stack 报错。
        """
        # 1. 筛选 Clean Transcripts (假设传入的 samples 已经是筛选过的列表)
        # 如果需要再次确保，可以在这里加判断，这里直接使用
        selected_samples = samples
        print(f"Step 1: Evaluating {len(selected_samples)} samples...")
        
        results = []
        counts_to_test = [0, 1, 2, 3, 4, 5]
        
        # 2. 逐样本处理 (Per-Sample Processing)
        # 这样确保每次 predict 的 batch 中序列长度完全一致
        for sample in tqdm(selected_samples, desc="Processing"):
            base_emb = sample['seq_emb']
            uuid = sample['uuid']
            cell_type = sample['cell_type']
            
            # 2.1 为当前样本生成所有突变体
            current_seqs = []
            current_metas = [] # 记录对应的 count
            
            for cnt in counts_to_test:
                mut_emb = self.inject_ccc(base_emb, cnt)
                if mut_emb is not None:
                    current_seqs.append(mut_emb)
                    current_metas.append(cnt)
            
            if not current_seqs: continue
            
            # 2.2 立即预测当前样本这组突变 (Batch Size ≈ 6)
            # 因为 current_seqs 都源自同一个 base_emb，长度一致，可以 stack
            try:
                # 构造对应的 cell_type batch
                current_cells = [cell_type] * len(current_seqs)
                
                preds = self.predict(current_seqs, current_cells)
                
                # 2.3 计算 TE 并暂存
                sample_tes = {}
                
                for j, pred_arr in enumerate(preds):
                    cnt = current_metas[j]
                    te = calculate_morf_efficiency(pred_arr, sample['morf_start'], sample['morf_end'])
                    sample_tes[cnt] = te
                
                # 2.4 计算 Relative TE (相对于 Count=0)
                if 0 in sample_tes:
                    wt_te = sample_tes[0]
                    if wt_te > 1e-6: # 过滤掉底噪太大的样本
                        for cnt, abs_te in sample_tes.items():
                            results.append({
                                'UUID': uuid,
                                'CCC_Count': cnt,
                                'Relative_TE': abs_te / wt_te
                            })
                            
            except ValueError as e:
                # 理论上不应再出现 shape error
                print(f"Error processing {uuid}: {e}")
                continue

        return pd.DataFrame(results)

    def plot_ccc_dose_effect(self, df, suffix=""):
        """
        使用 Plotnine 绘制 Violin Plot
        """
        if df.empty:
            print("No data to plot.")
            return
        
        # 将 Count 转为 Categorical 以便绘图颜色区分
        df['CCC_Count_Cat'] = pd.Categorical(df['CCC_Count'], ordered=True)
        
        # 统计中位数
        stats = df.groupby('CCC_Count')['Relative_TE'].median().reset_index()
        
        p = (
            ggplot(df, aes(x='CCC_Count_Cat', y='Relative_TE', fill='CCC_Count_Cat'))
            + geom_violin(trim=False, alpha=0.6, show_legend=False)
            + geom_boxplot(width=0.15, fill="white", alpha=0.9, outlier_size=0, show_legend=False)
            + geom_hline(yintercept=1.0, linetype="dashed", color="gray", size=0.8)
            + theme_bw()
            + theme(
                figure_size=(8, 6),
                axis_text=element_text(size=12),
                axis_title=element_text(size=14, face="bold"),
                panel_grid_minor=element_blank()
            )
            + labs(
                x="Number of inserted CCC motifs",
                y="Relative mORF TE (Normalized to 0 CCC)",
                title=f"In-silico Dose-Dependent Effect of 5' UTR CCC ({suffix})"
            )
            + scale_fill_brewer(type="seq", palette="Blues")
        )
        
        os.makedirs(self.out_dir, exist_ok=True)
        save_path = os.path.join(self.out_dir, f"ccc_dose_effect_plotnine.{suffix}.pdf")
        p.save(save_path, width=8, height=6, dpi=300)
        print(f"Saved plot to {save_path}")