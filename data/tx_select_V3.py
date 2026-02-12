import os, sys, argparse, pickle, pysam
import numpy as np
from scipy.sparse import csr_matrix
from numba import njit
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
# ---------- Step 1: Prepare for transcript selection ----------
def prepare_for_tx_select(annotation, bam_file, chunk_size=5_000_000, dest_name=None, cache_dir=None):
    if not annotation: raise ValueError("annotation required")
    if not os.path.exists(bam_file): raise FileNotFoundError(bam_file)

    # build junction index for fast lookup
    junc_index = {}
    for i, j in enumerate(annotation.get("junctions", [])):
        key = (j["Chromosome"], j["Start"], j["End"], j["Strand"])
        junc_index[key] = i
        j["reads"] = 0
        j["unique_reads"] = 0

    keep_chroms = {g["Chromosome"] for g in annotation.get("genes", [])}
    sam = pysam.AlignmentFile(bam_file, "rb")

    P_sites_all = {}
    for read in sam.fetch(until_eof=True):
        if read.is_secondary or read.is_duplicate: continue
        chrom = read.reference_name
        if keep_chroms and chrom not in keep_chroms: continue
        if not read.cigartuples or any(op in (1,2) for op,_ in read.cigartuples): continue

        strand = "-" if read.is_reverse else "+"
        pos = read.reference_end - 1 if read.is_reverse else read.reference_start
        key = (chrom, pos, strand)
        P_sites_all[key] = P_sites_all.get(key, 0) + 1

        if any(op == 3 for op,_ in read.cigartuples):  # junction
            ref = read.reference_start
            for op, l in read.cigartuples:
                if op in (0,7,8): ref += l
                elif op == 3:
                    jkey = (chrom, ref, ref+l, strand)
                    if jkey in junc_index:
                        j = annotation["junctions"][junc_index[jkey]]
                        j["reads"] += 1
                        if read.mapping_quality > 50: j["unique_reads"] += 1
                    ref += l

    out = {"P_sites_all": P_sites_all, "junctions": annotation["junctions"]}
    save_base = dest_name + "_for_tx_select" if dest_name else os.path.basename(bam_file) + "_for_tx_select"
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        save_base = os.path.join(cache_dir, os.path.basename(save_base))
    with open(save_base, "wb") as fh:
        pickle.dump(out, fh)
    return save_base


# ---------- Step 2: Numba-accelerated exon read counting ----------
@njit
def compute_exon_reads(exons_starts, exons_ends, sites_pos, sites_count):
    n = len(exons_starts)
    res = np.zeros(n, dtype=np.int64)
    for i in range(n):
        start = exons_starts[i]
        end = exons_ends[i]
        total = 0
        for j in range(len(sites_pos)):
            pos = sites_pos[j]
            if start <= pos < end:
                total += sites_count[j]
        res[i] = total
    return res

# ---------- Step 3: Transcript selection for one gene ----------
def select_txs_rstyle_vec(gene_entry, annotation, pre):
    chrom, start, end, strand, gene_id = (
        gene_entry['Chromosome'],
        gene_entry['Start'],
        gene_entry['End'],
        gene_entry['Strand'],
        gene_entry['gene_id']
    )

    # --- Junctions ---
    juncs = [j for j in annotation['junctions']
             if j['Chromosome']==chrom and j['End']>=start and j['Start']<=end and gene_id in j['gene_id']]
    
    # 构建 junction dict 加速查询
    pre_junc_dict = { (j['Chromosome'], j['Start'], j['End'], j['Strand']): j['reads']
                      for j in pre['junctions'] }
    
    junc_reads = np.array([
        pre_junc_dict.get((j['Chromosome'], j['Start'], j['End'], j['Strand']), 0)
        for j in juncs
    ])
    junc_txs = [j['tx_name'] for j in juncs]

    # --- Exons ---
    exons = [e for e in annotation['exons_bins']
             if e['Chromosome']==chrom and e['Strand']==strand and e['End']>=start and e['Start']<=end and gene_id in e['gene_id']]
    
    # 提取当前染色体+链的 P-sites
    sites = [(pos, count) for (c,s,pos), count in pre['P_sites_all'].items() if c==chrom and s==strand]
    if sites:
        positions, counts = zip(*sites)
        exon_reads = compute_exon_reads(np.array([e['Start'] for e in exons]),
                                        np.array([e['End'] for e in exons]),
                                        np.array(positions),
                                        np.array(counts))
    else:
        exon_reads = np.zeros(len(exons), dtype=int)
    exon_txs = [e['tx_name'] for e in exons]

    # --- Combine reads and features ---
    all_reads = np.concatenate([junc_reads, exon_reads])
    all_txs = junc_txs + exon_txs
    n_feat = len(all_reads)
    if n_feat == 0:
        return gene_id, []

    # Build feature x transcript sparse matrix
    txs_gene = sorted({tx for tx_list in all_txs for tx in tx_list if isinstance(tx, str)})
    if len(txs_gene) == 0:
        return gene_id, []

    tx2col = {tx:i for i, tx in enumerate(txs_gene)}
    row, col = [], []
    for i, tx_list in enumerate(all_txs):
        for tx in tx_list:
            if tx in tx2col:
                row.append(i)
                col.append(tx2col[tx])
    mat = csr_matrix((np.ones(len(row), dtype=int), (row, col)),
                     shape=(n_feat, len(txs_gene)), dtype=int)

    # --- Good/Bad features ---
    good_idx = np.where(all_reads>0)[0]
    if len(good_idx) == 0:
        return gene_id, []

    # --- Nested / identical transcripts ---
    nest, ident = set(), set()
    mat_dense = mat.toarray()
    for i in range(mat_dense.shape[1]):
        yes_i = set(np.where(mat_dense[:,i])[0])
        for j in range(mat_dense.shape[1]):
            if i == j: continue
            yes_j = set(np.where(mat_dense[:,j])[0])
            if yes_i == yes_j:
                ident.add(frozenset([txs_gene[i], txs_gene[j]]))
            elif yes_j.issubset(yes_i) and len(yes_j) < len(yes_i):
                nest.add(txs_gene[j])
    if ident:
        firsts = {list(x)[0] for x in ident}
        nest = nest - firsts
    txs_sofar = [tx for tx in txs_gene if tx not in nest]

    # --- Iterative selection ---
    change = True
    while change:
        cols = [tx2col[tx] for tx in txs_sofar]
        mat_sub = mat_dense[:, cols]
        good_mask = np.zeros_like(mat_sub)
        good_mask[good_idx, :] = mat_sub[good_idx, :]
        good_counts = good_mask.sum(axis=0)
        bad_counts = mat_sub.sum(axis=0) - good_counts

        txs_good = []
        expl_good = set()
        for i, tx in enumerate(txs_sofar):
            feat_idx = np.where(mat_sub[:, i])[0]
            tx_good = [idx for idx in feat_idx if idx in good_idx]
            tx_bad  = [idx for idx in feat_idx if idx not in good_idx]
            if len(tx_good) == 0:
                continue

            new_good = [idx for idx in tx_good if idx not in expl_good]
            if new_good:
                to_remove = []
                for sel_tx in txs_good:
                    sel_col = txs_sofar.index(sel_tx)
                    sel_idx = np.where(mat_sub[:, sel_col])[0]
                    sel_good = [idx for idx in sel_idx if idx in good_idx]
                    if all(idx in tx_good for idx in sel_good):
                        to_remove.append(sel_tx)
                for rtx in to_remove:
                    txs_good.remove(rtx)
                txs_good.append(tx)
                expl_good.update(tx_good)
            else:
                for sel_tx in txs_good.copy():
                    sel_col = txs_sofar.index(sel_tx)
                    sel_idx = np.where(mat_sub[:, sel_col])[0]
                    sel_good  = [idx for idx in sel_idx if idx in good_idx]
                    sel_bad   = [idx for idx in sel_idx if idx not in good_idx]
                    if all(idx in tx_good for idx in sel_good):
                        fi, la = tx_good[0], tx_good[-1]
                        int_tx_bad = [idx for idx in tx_bad if fi<=idx<=la]
                        sel_fi, sel_la = sel_good[0], sel_good[-1]
                        sel_int_bad = [idx for idx in sel_bad if sel_fi<=idx<=sel_la]
                        if len(int_tx_bad) <= len(sel_int_bad):
                            if len(int_tx_bad) < len(sel_int_bad):
                                txs_good.remove(sel_tx)
                            if tx not in txs_good:
                                txs_good.append(tx)
                                expl_good.update(tx_good)

        prev_set = set(txs_sofar)
        txs_sofar = sorted(set(txs_good))
        change = len(txs_sofar) != len(prev_set)

    return gene_id, txs_sofar

# ---------- Step added: Transforming final results ----------
# def merge_transcripts(results):
#     all_txs = []
#     for tx_list in results.values():
#         if tx_list:  # 非空列表
#             all_txs.extend(tx_list)

#     unique_txs = sorted(set(all_txs))
#     return {"transcripts": unique_txs}

# ---------- Step 4: Multi-processing wrapper ----------
def process_gene(gene_entry, annotation, pre):
    try:
        gene_id, selected_txs = select_txs_rstyle_vec(gene_entry, annotation, pre)
        return gene_id, selected_txs
    except Exception as e:
        return gene_entry['gene_id'], f"ERROR: {e}"

def run_all_genes(annotation, pre, output_pickle, n_proc=16):
    genes = annotation["genes"]
    print(f"Running transcript selection on {len(genes)} genes using {n_proc} cores...")
    results = {}
    func = partial(process_gene, annotation=annotation, pre=pre)
    with mp.Pool(processes=n_proc) as pool:
        for gene_id, txs in tqdm(pool.imap_unordered(func, genes), total=len(genes)):
            results[gene_id] = txs

    # merge_results = merge_transcripts(results)
    with open(output_pickle, "wb") as f:
        # pickle.dump(merge_results, f)
        pickle.dump(results, f)
    print(f"Done! Results saved to: {output_pickle}")
    # return merge_results
    return results

# ---------- Step 5: Command-line interface ----------
def main():
    parser = argparse.ArgumentParser(description="Transcript selection pipeline")
    parser.add_argument("--annotation", required=True, help="Input annotation pickle")
    parser.add_argument("--bam", required=True, help="Input BAM file")
    parser.add_argument("--cache_dir", default="cache", help="Cache directory")
    parser.add_argument("--output", required=True, help="Output pickle file")
    parser.add_argument("--nproc", type=int, default=16, help="Number of CPU cores")
    args = parser.parse_args()

    print("## Loading annotation...")
    with open(args.annotation, "rb") as f:
        annotation = pickle.load(f)

    print("## Preparing precomputed data from BAM...")
    pre_file = prepare_for_tx_select(annotation, args.bam, cache_dir=args.cache_dir)
    with open(pre_file, "rb") as f:
        pre = pickle.load(f)

    print("## Running transcript selection...")
    run_all_genes(annotation, pre, args.output, n_proc=args.nproc)


if __name__ == "__main__":
    main()