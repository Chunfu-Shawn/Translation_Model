import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from eval.calculate_te import *

class uORFEvaluatorEmb:
    def __init__(self, model, out_dir="."):
        """
        Args:
            model: 你的模型对象，需包含 model.device 和 model.predict
        """
        self.device = model.device # 修正：保存 device 到 self

        def model_predict_wrapper(seq_emb_input, cell_type_emb_input):
            """
            Args:
                seq_emb_input: List of Arrays [(L, 4)...] or Batch Array (B, L, 4)
                cell_type_emb_input: List of Arrays or Batch Array (B, D)
            Returns:
                preds_np: Numpy array shape (B, L)
            """
            # 1. 统一转换为 Batch Numpy
            if isinstance(seq_emb_input, list):
                seq_batch_np = np.stack(seq_emb_input) # (B, L, 4)
                cell_batch_np = np.stack(cell_type_emb_input) # (B, D)
            else:
                seq_batch_np = seq_emb_input
                cell_batch_np = cell_type_emb_input

            # 2. 转 Tensor
            seq_tensor = torch.from_numpy(seq_batch_np).float().to(self.device) # 使用 self.device
            cell_tensor = torch.from_numpy(cell_batch_np).float().to(self.device)
            
            model.eval()
            with torch.no_grad():
                output_dict = model.predict(seq_batch=seq_tensor, cell_type=cell_tensor, head_names=["count"]) 
                pred = output_dict['count'] # (B, L)
                
            # 3. 后处理 (保持 Batch 维度)
            pred_np = pred.cpu().numpy()
            pred_np = np.expm1(pred_np)
                
            return pred_np
        
        self.predict = model_predict_wrapper
        self.out_dir = out_dir
        
        self.BASES_MAP = {
            'A': np.array([1, 0, 0, 0], dtype=np.float32),
            'C': np.array([0, 1, 0, 0], dtype=np.float32),
            'G': np.array([0, 0, 1, 0], dtype=np.float32),
            'T': np.array([0, 0, 0, 1], dtype=np.float32),
            'N': np.array([0, 0, 0, 0], dtype=np.float32)
        }


    def inject_uorf_embedding(self, base_emb, morf_start, distance=None, uorf_len=21, kozak_type='strong'):
        """在 One-hot Embedding 上直接修改序列 (保持不变)"""
        new_emb = base_emb.copy()
        uorf_end_pos = morf_start - distance
        uorf_start_pos = uorf_end_pos - uorf_len
        
        if uorf_start_pos < 3:
            # raise ValueError("No space for uORF") 
            # 修改：返回 None 以便外部捕获，而不是中断
            return None

        if kozak_type == 'none':
            for i, base in enumerate(['A', 'T', 'C'],):
                new_emb[uorf_start_pos + i] = self.BASES_MAP[base]
            for i, base in enumerate(['C', 'G', 'A']):
                new_emb[uorf_end_pos - 3 + i] = self.BASES_MAP[base]
        else:
            if kozak_type == 'weak':
                start_codon, up_context, down_context, stop_codon = ['A', 'T', 'G'], ['G', 'G', 'G'], ['C', 'C'], ['T', 'G', 'A']
            elif kozak_type == 'strong':
                start_codon, up_context, down_context, stop_codon = ['A', 'T', 'G'], ['A', 'C', 'C'], ['G', 'G'], ['T', 'G', 'A']
            else:
                raise ValueError("Unknown kozak_type")
            
            for i, base in enumerate(start_codon):
                new_emb[uorf_start_pos + i] = self.BASES_MAP[base]
            for i, base in enumerate(up_context):
                new_emb[uorf_start_pos - 3 + i] = self.BASES_MAP[base]
            for i, base in enumerate(down_context):
                new_emb[uorf_start_pos + 3 + i] = self.BASES_MAP[base]
            for i, base in enumerate(stop_codon):
                new_emb[uorf_end_pos - 3 + i] = self.BASES_MAP[base]
            
        return new_emb

    def evaluate_batch_distance(self, samples, distances=None, fixed_uorf_len=30, batch_size=60):
        if distances is None:
            g = 0; d = 9; distances = [d]
            for i in range(1, 20):
                g = i * 3
                d = d + g
                distances.append(d)
        
        results = []
        print(f"Evaluating Distance Effect (Batch Size: {batch_size}) of {distances}")
        
        for sample in tqdm(samples, desc="Processing Samples"):
            cell_type_emb = sample['cell_type']
            base_emb = sample['seq_emb']
            m_start = sample['morf_start'] # 0-based
            m_end = sample['morf_end']
            
            # 1. 预测 WT (Base)
            try:
                pred_wt_batch = self.predict([base_emb], [cell_type_emb])
                pred_wt = pred_wt_batch[0]
                te_wt = calculate_morf_efficiency(pred_wt, m_start, m_end)
                if te_wt < 1e-6: continue 
            except Exception as e:
                continue

            # 2. 收集该样本的所有突变序列
            seqs_buffer = []
            cells_buffer = []
            meta_buffer = []

            for dist in distances:
                # Condition 1: Strong in-frame
                emb_in = self.inject_uorf_embedding(base_emb, m_start, distance=dist, uorf_len=fixed_uorf_len, kozak_type='strong')
                if emb_in is not None:
                    seqs_buffer.append(emb_in)
                    cells_buffer.append(cell_type_emb)
                    meta_buffer.append({'dist': dist, 'cond': 'Strong in-frame uORF'})

                # Condition 2: Strong out-frame
                emb_out = self.inject_uorf_embedding(base_emb, m_start, distance=dist + 1, uorf_len=fixed_uorf_len, kozak_type='strong')
                if emb_out is not None:
                    seqs_buffer.append(emb_out)
                    cells_buffer.append(cell_type_emb)
                    meta_buffer.append({'dist': dist + 1, 'cond': 'Strong out-frame uORF'})

                # Condition 3: Control
                emb_mut = self.inject_uorf_embedding(base_emb, m_start, distance=dist, uorf_len=fixed_uorf_len, kozak_type='none')
                if emb_mut is not None:
                    seqs_buffer.append(emb_mut)
                    cells_buffer.append(cell_type_emb)
                    meta_buffer.append({'dist': dist, 'cond': 'Control (broken uORF)'})

            # 3. 批量预测
            if not seqs_buffer: continue

            for i in range(0, len(seqs_buffer), batch_size):
                batch_seqs = seqs_buffer[i : i + batch_size]
                batch_cells = cells_buffer[i : i + batch_size]
                batch_preds = self.predict(batch_seqs, batch_cells)
                
                for j, pred_mut in enumerate(batch_preds):
                    meta = meta_buffer[i + j]
                    te_mut = calculate_morf_efficiency(pred_mut, m_start, m_end)
                    results.append({
                        'UUID': sample['uuid'],
                        'Distance': meta['dist'], 
                        'Condition': meta['cond'], 
                        'Relative_TE': te_mut / te_wt
                    })
                    
        return pd.DataFrame(results)

    def evaluate_batch_length(self, samples, uorf_lengths=None, fixed_distance=21, batch_size=60):
        # ... (Length 代码保持不变，略微省略以节省篇幅，逻辑同上) ...
        # 为了完整性，这里可以保持原样，只是注意 inject_uorf_embedding 返回 None 的处理
        if uorf_lengths is None: 
            g = 0; l = 9; uorf_lengths = [l]
            for i in range(1, 20):
                g = i * 3
                l = l + g
                uorf_lengths.append(l)
        
        results = []
        print(f"Evaluating Length Effect (Batch Size: {batch_size}) of {uorf_lengths}")
        
        for sample in tqdm(samples, desc="Processing Samples"):
            cell_type_emb = sample['cell_type']
            base_emb = sample['seq_emb']
            m_start = sample['morf_start']
            m_end = sample['morf_end']
            
            try:
                pred_wt_batch = self.predict([base_emb], [cell_type_emb])
                pred_wt = pred_wt_batch[0]
                te_wt = calculate_morf_efficiency(pred_wt, m_start, m_end)
                if te_wt < 1e-6: continue
            except: continue
                
            seqs_buffer = []
            cells_buffer = []
            meta_buffer = []

            for length in uorf_lengths:
                emb_in = self.inject_uorf_embedding(base_emb, m_start, distance=fixed_distance, uorf_len=length, kozak_type='strong')
                if emb_in is not None:
                    seqs_buffer.append(emb_in)
                    cells_buffer.append(cell_type_emb)
                    meta_buffer.append({'len': length, 'cond': 'Strong in-frame uORF'})

                emb_out = self.inject_uorf_embedding(base_emb, m_start, distance=fixed_distance + 1, uorf_len=length, kozak_type='strong')
                if emb_out is not None:
                    seqs_buffer.append(emb_out)
                    cells_buffer.append(cell_type_emb)
                    meta_buffer.append({'len': length, 'cond': 'Strong out-frame uORF'})

                emb_mut = self.inject_uorf_embedding(base_emb, m_start, distance=fixed_distance, uorf_len=length, kozak_type='none')
                if emb_mut is not None:
                    seqs_buffer.append(emb_mut)
                    cells_buffer.append(cell_type_emb)
                    meta_buffer.append({'len': length, 'cond': 'Control (broken uORF)'})
            
            if not seqs_buffer: continue

            for i in range(0, len(seqs_buffer), batch_size):
                batch_seqs = seqs_buffer[i : i + batch_size]
                batch_cells = cells_buffer[i : i + batch_size]
                batch_preds = self.predict(batch_seqs, batch_cells)
                
                for j, pred_mut in enumerate(batch_preds):
                    meta = meta_buffer[i + j]
                    te_mut = calculate_morf_efficiency(pred_mut, m_start, m_end)
                    results.append({
                        'UUID': sample['uuid'],
                        'uORF_Length': meta['len'],
                        'Condition': meta['cond'],
                        'Relative_TE': te_mut / te_wt
                    })
                
        return pd.DataFrame(results)

    def evaluate_batch_count(self, samples, max_count=20, uorf_len=21, batch_size=60):
        """
        评估 uORF 数量对 mORF 翻译的影响
        uORF 长度固定为 21nt
        数量从 1 到 max_count
        分组：
          1. Control: Broken uORF
          2. in-frame: 间隔 15nt，全部同框
          3. out-frame: 间隔 15nt，全部异框
          4. in/out-frame: 间隔交替，实现 In-Out-In-Out 排列
        """
        counts = list(range(1, max_count + 1))
        results = []
        print(f"Evaluating Count Effect (Batch Size: {batch_size}) from 1 to {max_count} uORFs")

        for sample in tqdm(samples, desc="Processing Samples"):
            cell_type_emb = sample['cell_type']
            base_emb = sample['seq_emb']
            m_start = sample['morf_start']
            m_end = sample['morf_end']

            # 1. WT Baseline
            try:
                pred_wt_batch = self.predict([base_emb], [cell_type_emb])
                pred_wt = pred_wt_batch[0]
                te_wt = calculate_morf_efficiency(pred_wt, m_start, m_end)
                if te_wt < 1e-6: continue
            except: continue

            seqs_buffer = []
            cells_buffer = []
            meta_buffer = []

            # 遍历每一个数量 N
            for cnt in counts:
                # --- Condition 1: In-frame ---
                # 全部间隔 15nt，起始距离 21nt (In-frame)
                emb_in = base_emb.copy()
                valid_in = True
                curr_dist = 21 # Closest uORF distance
                
                for k in range(cnt):
                    # 每次基于上一个状态继续叠加，或者每次基于 base_emb 叠加多个
                    # 这里的 inject_uorf_embedding 每次只加一个，我们需要循环调用
                    emb_in = self.inject_uorf_embedding(emb_in, m_start, distance=curr_dist, uorf_len=uorf_len, kozak_type='strong')
                    if emb_in is None: 
                        valid_in = False; break
                    # 下一个 uORF 更靠上游：当前距离 + 长度 + 间隔
                    curr_dist += uorf_len + 15
                
                if valid_in:
                    seqs_buffer.append(emb_in)
                    cells_buffer.append(cell_type_emb)
                    meta_buffer.append({'count': cnt, 'cond': 'In-frame'})

                # --- Condition 2: Out-frame ---
                # 全部间隔 15nt，起始距离 22nt (Out-frame)
                emb_out = base_emb.copy()
                valid_out = True
                curr_dist = 22 # Closest uORF distance (21+1)
                
                for k in range(cnt):
                    emb_out = self.inject_uorf_embedding(emb_out, m_start, distance=curr_dist, uorf_len=uorf_len, kozak_type='strong')
                    if emb_out is None:
                        valid_out = False; break
                    curr_dist += uorf_len + 15
                
                if valid_out:
                    seqs_buffer.append(emb_out)
                    cells_buffer.append(cell_type_emb)
                    meta_buffer.append({'count': cnt, 'cond': 'Out-frame'})

                # --- Condition 3: In/Out-frame Alternating ---
                
                emb_alt = base_emb.copy()
                valid_alt = True
                curr_dist = 21 # 第一个为 In
                
                for k in range(cnt):
                    # 当前是第 k 个 (0-based)
                    # 注入当前 uORF
                    emb_alt = self.inject_uorf_embedding(emb_alt, m_start, distance=curr_dist, uorf_len=uorf_len, kozak_type='strong')
                    if emb_alt is None:
                        valid_alt = False; break

                    # 使用固定 14nt 间隔，会产生 0 -> 2 -> 1 -> 0 循环。
                    # 这满足 "In/Out" 混合且主要由 "Out" 组成，且符合 "使用 14nt 实现 Out-frame" 的描述。
                    curr_dist += uorf_len + 14
                
                if valid_alt:
                    seqs_buffer.append(emb_alt)
                    cells_buffer.append(cell_type_emb)
                    meta_buffer.append({'count': cnt, 'cond': 'In/Out-frame'})

                # --- Condition 4: Control (Broken) ---
                # 结构同 In-frame，但 Kozak='none'
                emb_ctrl = base_emb.copy()
                valid_ctrl = True
                curr_dist = 21
                
                for k in range(cnt):
                    emb_ctrl = self.inject_uorf_embedding(emb_ctrl, m_start, distance=curr_dist, uorf_len=uorf_len, kozak_type='none')
                    if emb_ctrl is None:
                        valid_ctrl = False; break
                    curr_dist += uorf_len + 15 # 保持 15nt 间隔
                
                if valid_ctrl:
                    seqs_buffer.append(emb_ctrl)
                    cells_buffer.append(cell_type_emb)
                    meta_buffer.append({'count': cnt, 'cond': 'Control (Broken uORF)'})

            # 3. Batch Predict
            if not seqs_buffer: continue

            for i in range(0, len(seqs_buffer), batch_size):
                batch_seqs = seqs_buffer[i : i + batch_size]
                batch_cells = cells_buffer[i : i + batch_size]
                batch_preds = self.predict(batch_seqs, batch_cells)

                for j, pred_mut in enumerate(batch_preds):
                    meta = meta_buffer[i + j]
                    te_mut = calculate_morf_efficiency(pred_mut, m_start, m_end)
                    results.append({
                        'UUID': sample['uuid'],
                        'uORF_Count': meta['count'],
                        'Condition': meta['cond'],
                        'Relative_TE': te_mut / te_wt
                    })

        return pd.DataFrame(results)

    def plot_distance_effect(self, df, suffix=""):
        plt.figure(figsize=(6, 5))
        sns.lineplot(data=df, x='Distance', y='Relative_TE', hue='Condition', 
                     style='Condition', markers=True, dashes=False, errorbar='se')
        plt.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
        plt.title("Distance Effect on Re-initiation (Batch Prediction)")
        plt.xlabel("Intercistronic Distance (nt)")
        plt.ylabel("Relative mORF Efficiency")
        plt.xlim(729, 8)
        plt.xscale('log', base=3)
        save_file = os.path.join(self.out_dir, f"uORF_distance_effect.{suffix}.pdf")
        plt.savefig(save_file)
        plt.show()

    def plot_length_effect(self, df, suffix=""):
        plt.figure(figsize=(6, 5))
        sns.lineplot(data=df, x='uORF_Length', y='Relative_TE', hue='Condition',
                     style='Condition', markers=True, dashes=False, errorbar='se')
        plt.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
        plt.title(f"uORF Length Effect (Batch Prediction)")
        plt.xlabel("uORF Length (nt)")
        plt.ylabel("Relative mORF Efficiency")
        plt.xlim(729, 8)
        plt.xscale('log', base=3)
        save_file = os.path.join(self.out_dir, f"uORF_length_effect.{suffix}.pdf")
        plt.savefig(save_file)
        plt.show()

    def plot_count_effect(self, df, suffix=""):
        plt.figure(figsize=(6, 5))
        sns.lineplot(data=df, x='uORF_Count', y='Relative_TE', hue='Condition',
                     style='Condition', markers=True, dashes=False, errorbar='se')
        plt.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
        plt.title(f"uORF Count Effect (Batch Prediction)")
        plt.xlabel("Number of uORFs")
        plt.ylabel("Relative mORF Efficiency")
        # x轴设为整数刻度
        max_c = df['uORF_Count'].max()
        plt.xticks(range(1, int(max_c)+1, 2)) 
        save_file = os.path.join(self.out_dir, f"uORF_count_effect.{suffix}.pdf")
        plt.savefig(save_file)
        plt.show()