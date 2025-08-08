import os
import numpy as np
import pickle
import gffutils
from numba import njit
from collections import defaultdict
from intervaltree import IntervalTree

__author__ = "Chunfu Xiao"
__contributor__="..."
__version__="1.1.0"
__maintainer__ = "Chunfu Xiao"
__email__ = "chunfushawn@126.com"

def array_defaultdict():
    return []

def nested_defaultdict():
    return defaultdict(array_defaultdict)

@njit
def build_exon_index_np(exon_starts, exon_ends, strand):
    """
    把 GTF 读出的 exon_starts/exon_ends (1-based inclusive, 转录本顺序)
    直接转换成 0-based half-open, 并按照基因组坐标升序返回:
      starts0: np.array([...])
      ends0:   np.array([...])
    如果 strand == '+': 假定 exon_starts 已经按 genomic 升序，
      直接做 O(E) 的转换；如果 strand == '-': 假定它是 genomic 降序，
      只需一次反转即可。
    """
    # 1) 生成 (start0, end0, exon_idx) 三元组列表
    exons = [
        (s - 1, e, idx)
        for idx, (s, e) in enumerate(zip(exon_starts, exon_ends))
    ]
    
    if strand == '+':
        # 已经是 genomic 升序：直接拆包
        seq = exons
    else:
        # 对于负链，GTF 转录本顺序通常是 genomic 降序 → 反转一次
        seq = exons[::-1]
    
    # 2) 拆成三个 ndarray
    starts0 = np.array([s for s, _, _ in seq], dtype=np.int32)
    ends0   = np.array([e for _, e, _ in seq], dtype=np.int32)
    
    return starts0, ends0

@njit
def convert_position(
    genome_pos: int,
    exon_starts: np.ndarray,
    exon_ends: np.ndarray,
    tx_starts: np.ndarray,
    tx_ends: np.ndarray,
    strand):
    # all input need 1-based
    # return: pos [1, tx_len] 1-based
    if strand == '+':
        idx = np.searchsorted(exon_starts, genome_pos, side='right') - 1
    else:
        idx = exon_starts.size - np.searchsorted(np.sort(exon_starts), genome_pos, side='right')
    
    if idx >= 0 and exon_starts[idx] <= genome_pos <= exon_ends[idx]:
        if strand == '+':
            return tx_starts[idx] + (genome_pos - exon_starts[idx])
        else:
            return tx_ends[idx] - (genome_pos - exon_starts[idx])
    return -1

def create_optimized_index(gtf_file, db_file):
    """
    Build the index of transcripts and exons for each chromosome
    input gtf is 1-based
    """
    print("Construct the index of transcript by chromosome and strand")
    if os.path.isfile(db_file):
        db = gffutils.interface.FeatureDB(db_file, keep_order=True)
    else:
        db = gffutils.create_db(gtf_file, dbfn=db_file, disable_infer_genes=True,
                                    disable_infer_transcripts=True, keep_order=True)
    
    # global function to define data structure
    chrom_index = nested_defaultdict()
    chrom_strand_index = defaultdict(nested_defaultdict)
    transcript_meta_dict = {} 
    transcript_cds_dict = {}

    for transcript in db.features_of_type('transcript'):
        tid = transcript.id
        chrom = transcript.chrom
        strand = transcript.strand

        # exclude mitochondrial genes
        if chrom == "chrM":
            continue
        
        # sort by strand
        exons = sorted(db.children(transcript, featuretype='exon'), 
                      key=lambda x: x.start, reverse=(strand == '-')) # 1-based
        cds_regions = sorted(db.children(transcript, featuretype='CDS'), 
                      key=lambda x: x.start, reverse=(strand == '-')) # 1-based

        print("--- " + tid + " ---")
        # exon starts < exon ends
        exon_starts = np.array([e.start for e in exons], dtype=int) # 1-based
        exon_ends = np.array([e.end for e in exons], dtype=int) # 1-based
        exon_starts0_sorted, exon_ends0_sorted = build_exon_index_np(exon_starts, exon_ends, strand) # 0-based
        cds_starts = np.array([c.start for c in cds_regions], dtype=int) # 1-based
        cds_ends = np.array([c.end for c in cds_regions], dtype=int) # 1-based
        cds_frames = np.array([c.frame for c in cds_regions], dtype=int) # 0, 1, 2 frame
        
        tx_pos = np.cumsum([0] + [e.end - e.start + 1 for e in exons])[:-1] # 0-based
        tx_starts = tx_pos + 1 # 1-based
        tx_ends = tx_pos + (exon_ends - exon_starts + 1) # 1-based

        if cds_starts.size == 0:
            cds_start_pos = -1
            cds_end_pos = -1
        elif strand == "+":
            cds_start_pos = convert_position(cds_starts[0], exon_starts, exon_ends, tx_starts, tx_ends, strand) # 1-based
            cds_end_pos = convert_position(cds_ends[-1], exon_starts, exon_ends, tx_starts, tx_ends, strand) # 1-based
        elif strand == "-":
            cds_start_pos = convert_position(cds_ends[0], exon_starts, exon_ends, tx_starts, tx_ends, strand) # 1-based
            cds_end_pos = convert_position(cds_starts[-1], exon_starts, exon_ends, tx_starts, tx_ends, strand) # 1-based
        

        transcript_meta_dict[tid] = {
            'chrom': chrom,
            'strand': strand,
            'exon_starts': exon_starts,
            'exon_starts0_sorted': exon_starts0_sorted,
            'exon_ends': exon_ends,
            'exon_ends0_sorted': exon_ends0_sorted,
            'tx_starts': tx_starts,
            'tx_ends': tx_ends
        }
        transcript_cds_dict[tid] = {
            'cds_starts': cds_starts, # tx location (1-based) of first nt of start codon
            'cds_ends': cds_ends, # tx location (1-based) of last nt of CDS
            'cds_frames': cds_frames,
            'cds_start_pos': cds_start_pos, # tx pos (1-based) of first nt of start codon
            'cds_end_pos': cds_end_pos # tx pos (1-based) of last nt of CDS
        }

        # constrcut genome and exons index
        interval = np.array([[e.start, e.end, tid] for e in exons])
        chrom_index[chrom].append(interval)
        chrom_strand_index[chrom][strand].append(interval)
    
    # order exons by genomic positions and build tree interval
    print("Order the intervals (exons) and build tree interval for each chromosome and strand")
    tree_index = {}
    tree_strand_index = {}
    for chrom in chrom_index:
        all_intervals = np.vstack(chrom_index[chrom])
        chrom_index[chrom] = all_intervals[
                # sort exon start (int value)
                all_intervals[:, 0].astype(int).argsort()
            ]
        # build tree intervals: numpy array [[start,end,tid],...]
        tree = IntervalTree()
        for start, end, tid in chrom_index[chrom]:
            tree[int(start):int(end)+1] = tid  # [start, end]
        tree_index[chrom] = tree
        
    for chrom in chrom_strand_index:
        tree_strand_index[chrom] = {}
        for strand in chrom_strand_index[chrom]:
            all_intervals = np.vstack(chrom_strand_index[chrom][strand])
            chrom_strand_index[chrom][strand] = all_intervals[
                # sort exon start (int value)
                all_intervals[:, 0].astype(int).argsort()
            ]
            # build tree intervals: numpy array [[start,end,tid],...]
            tree = IntervalTree()
            for start, end, tid in chrom_strand_index[chrom][strand]:
                tree[int(start):int(end)+1] = tid  # [start, end]
            tree_strand_index[chrom][strand] = tree

    # return chrom_index, chrom_strand_index, tree_index, tree_strand_index, transcript_meta_dict
    return tree_index, tree_strand_index, transcript_meta_dict, transcript_cds_dict

def save_index(index, filename):
    """ save optimized index """
    # transfer numpy arrays to serializable format
    serializable = {}
    for chrom in index:
        serializable[chrom] = {}
        for strand in index[chrom]:
            arr = index[chrom][strand]
            # transfer numpy arrays to tuple: (start, end, tid)
            nested_tuples = [tuple(row) for row in arr.tolist()]
            serializable[chrom][strand] = nested_tuples

    # pickle
    with open(filename, 'wb') as f:
        pickle.dump(serializable, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_index(filename):
    """ load saved index """
    with open(filename, 'rb') as f:
        serializable = pickle.load(f)
    
    # 转换回numpy数组格式
    index = defaultdict(nested_defaultdict)
    for chrom in serializable:
        for strand in serializable[chrom]:
            nested_tuples = serializable[chrom][strand]
            arr = np.array(nested_tuples)
            index[chrom][strand] = arr
    return index

if __name__=="__main__":
    gtf_file = '/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/gencode.v48.comp_annotation_chro.gtf'
    tree_index_file = '/home/user/data3/rbase/translation_pred/models/lib/genome_index_tree.pkl'
    tx_meta_file = '/home/user/data3/rbase/translation_pred/models/lib/transcript_meta.pkl'
    tx_cds_file = '/home/user/data3/rbase/translation_pred/models/lib/transcript_cds.pkl'
    
    # build index
    tree_index, tree_strand_index, tx_meta, tx_cds = create_optimized_index(gtf_file, 'temp.db')

    # save
    with open(tree_index_file, 'wb') as f:
        pickle.dump(tree_index, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(tx_meta_file, 'wb') as f:
        pickle.dump(tx_meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(tx_cds_file, 'wb') as f:
        pickle.dump(tx_cds, f, protocol=pickle.HIGHEST_PROTOCOL)

    # load
    with open(tree_index_file, 'rb') as f:
        loaded_tree_index = pickle.load(f)
    with open(tx_meta_file, 'rb') as f:
        loaded_tx_meta = pickle.load(f)
    with open(tx_cds_file, 'rb') as f:
        loaded_tx_cds = pickle.load(f)

    for tid in loaded_tx_cds:
        print(tid)
        print(loaded_tx_cds[tid])