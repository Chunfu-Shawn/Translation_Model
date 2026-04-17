import os
import pandas as pd
import numpy as np
import torch

def build_expression_dict_end_to_end(file_path, output_pt_path, tpm_threshold=5.0, cv_threshold=1.0):
    """
    一站式处理流水线：
    读取Counts -> 计算全集TPM -> 过滤高变基因 -> Log2平滑 -> Z-score标准化 -> 打包为 float16 PT文件
    """
    print(f"正在读取文件: {file_path}")
    df = pd.read_csv(file_path, sep='\t', comment='#')
    
    # 1. 清理列名并识别样本列
    bam_cols = df.columns[6:]
    rename_dict = {col: os.path.basename(col).replace('.bam', '') for col in bam_cols}
    df = df.rename(columns=rename_dict)
    sample_cols = list(rename_dict.values())
    
    print(f"检测到 {len(sample_cols)} 个细胞系样本: {sample_cols}")
    
    # 2. 计算全集 TPM (必须在全局进行以保证 Scaling Factor 准确)
    print("正在计算全集 TPM...")
    tpm_cols = []
    for col in sample_cols:
        rpk = df[col] / (df['Length'] / 1000.0)
        scaling_factor = rpk.sum() / 1e6
        
        tpm_col_name = f"{col}_TPM"
        df[tpm_col_name] = rpk / scaling_factor
        tpm_cols.append(tpm_col_name)
        
    # 3. 双重过滤 (寻找高变基因)
    print("正在根据 TPM 和 CV 阈值过滤高变基因...")
    max_tpm = df[tpm_cols].max(axis=1)
    mask_expressed = max_tpm > tpm_threshold
    
    tpm_mean = df[tpm_cols].mean(axis=1)
    tpm_std = df[tpm_cols].std(axis=1)
    cv = tpm_std / (tpm_mean + 1e-8)  # 加上1e-8防除0
    mask_variation = cv > cv_threshold
    
    # 得到只包含高变基因的 DataFrame 子集
    filtered_df = df[mask_expressed & mask_variation].copy()
    
    print(f"-> 过滤前基因总数: {len(df)}")
    print(f"-> 过滤后高变基因数量: {len(filtered_df)}")
    
    # 4. 提取干净的 Gene ID，这将决定后续输入张量的排列顺序
    filtered_df['Clean_Geneid'] = filtered_df['Geneid'].str.split('.').str[0]
    gene_ids = filtered_df['Clean_Geneid'].tolist()
    d_expr_size = len(gene_ids)
    print(f"最终模型的输入基因维度 (d_expr): {d_expr_size}")
    
    # 5. 在高变子集上计算 Log2 和 Z-score，并打包字典
    print("正在对高变子集执行 Log2 平滑与 Z-score 标准化...")
    expr_dict = {}
    
    for col in sample_cols:
        # 提取当前细胞系的 TPM 子集
        tpm = filtered_df[f"{col}_TPM"]
        log_tpm = np.log2(tpm + 1.0)
        
        # 在当前细胞系内部对特征进行 Z-score 标准化
        z_score = (log_tpm - log_tpm.mean()) / (log_tpm.std() + 1e-8)
        
        # 转为 float16 以极大地节省模型显存
        tensor_fp16 = torch.tensor(z_score.values, dtype=torch.float16)
        expr_dict[col] = tensor_fp16
        
    # 6. 保存为 .pt 文件
    os.makedirs(os.path.dirname(output_pt_path) or '.', exist_ok=True)
    torch.save(expr_dict, output_pt_path)
    
    print(f"\n✅ 成功保存精简版表达量字典至: {output_pt_path}")
    print(f"   字典键值 (Keys): {list(expr_dict.keys())}")
    print(f"   向量形状 (Shape): {expr_dict[sample_cols[0]].shape}")
    print(f"   数据类型 (Dtype): {expr_dict[sample_cols[0]].dtype}")
    
    return expr_dict, gene_ids


if __name__ == "__main__":
    file_path = "/home/user/data3/yaoc/translation_model/rna-seq/counts_gene/matched_samples_gene_counts.txt"
    output_pt = "/home/user/data3/rbase/translation_model/models/src/config/cell_expression_dict.pt"

    # 运行一站式构建管道
    expr_dict, gene_list = build_expression_dict_end_to_end(
        file_path=file_path, 
        output_pt_path=output_pt, 
        tpm_threshold=5.0,   # 最大 TPM 至少为 5
        cv_threshold=1.0     # 变异系数大于 1
    )

    # 打印查看前 10 个提取的高变 ID
    print("\n提取的高变 ID 向量前 10 个示例:")
    print(gene_list[:10])

    # 保存基因顺序字典，以便未来核查或对齐新数据
    order_file = output_pt.replace('.pt', '_gene_order.txt')
    with open(order_file, 'w') as f:
        f.write("\n".join(gene_list))
    print(f"✅ 基因排列顺序已单独保存至: {order_file}")