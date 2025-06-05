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
def convert_position(
    genome_pos: int,
    exon_starts: np.ndarray,
    exon_ends: np.ndarray,
    tx_starts: np.ndarray,
    tx_ends: np.ndarray,
    strand):

    idx = np.searchsorted(exon_starts, genome_pos, side='right') - 1
    if idx >= 0 and exon_starts[idx] <= genome_pos <= exon_ends[idx]:
        if strand == '+':
            return tx_starts[idx] + (genome_pos - exon_starts[idx])
        else:
            return tx_ends[idx] - (genome_pos - exon_starts[idx])
    return -1

def create_optimized_index(gtf_file):
    """构建基于NumPy的索引结构"""
    print("Construct the index of transcript by chromosome and strand")
    db = gffutils.create_db(gtf_file, dbfn='temp.db', disable_infer_genes=True,
                            disable_infer_transcripts=True, force=True, keep_order=True)
    
    # global function to define data structure
    chrom_strand_index = defaultdict(nested_defaultdict)
    transcript_arrays = {}  # 存储转录本的NumPy数组

    for transcript in db.features_of_type('transcript'):
        tid = transcript.id
        chrom = transcript.chrom
        strand = transcript.strand
        # sort by strand
        exons = sorted(db.children(transcript, featuretype='exon'), 
                      key=lambda x: x.start, reverse=(strand == '-'))
        cds_regions = sorted(db.children(transcript, featuretype='CDS'), 
                      key=lambda x: x.start, reverse=(strand == '-'))
        # cds_start_codon = db.children(transcript, featuretype='start_codon')
        # cds_stop_codon = db.children(transcript, featuretype='stop_codon')

        # 转换为NumPy数组加速查询
        print("--- " + tid + " ---")
        # exon starts < exon ends
        exon_starts = np.array([e.start for e in exons], dtype=np.int32)
        exon_ends = np.array([e.end for e in exons], dtype=np.int32)
        cds_starts = np.array([c.start for c in cds_regions], dtype=np.int32)
        cds_ends = np.array([c.end for c in cds_regions], dtype=np.int32)
        
        # cds_start_codon_starts = np.array([c.start for c in cds_start_codon], dtype=np.int32)
        # cds_stop_codon_ends = np.array([c.end for c in cds_stop_codon], dtype=np.int32)
        tx_pos = np.cumsum([0] + [e.end - e.start + 1 for e in exons])[:-1]
        tx_starts = tx_pos + 1
        tx_ends = tx_pos + (exon_ends - exon_starts + 1)
        cds_start = convert_position(
            cds_starts[0] if strand=="+" else cds_ends[-1],
            exon_starts, exon_ends, tx_starts, tx_ends, strand) if len(cds_starts) !=0 else -1
        cds_end = convert_position(
            cds_ends[0] if strand=="+" else cds_starts[-1] - 2,
            exon_starts, exon_ends, tx_starts, tx_ends, strand) if len(cds_ends) !=0 else -1

        transcript_arrays[tid] = {
            'chrom': chrom,
            'strand': strand,
            'exon_starts': exon_starts,
            'exon_ends': exon_ends,
            'tx_starts': tx_starts,
            'tx_ends': tx_ends,
            'cds_start': cds_start,
            'cds_end': cds_end,
        }

        # constrcut genome and exons index
        interval = np.array([[e.start, e.end, tid] for e in exons])
        chrom_strand_index[chrom][strand].append(interval)
    
    # order exons by genomic positions and build tree interval
    print("Order the intervals (exons) and build tree interval for each chromosome and strand")
    tree_index = {}
    for chrom in chrom_strand_index:
        tree_index[chrom] = {}
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
            tree_index[chrom][strand] = tree

    return chrom_strand_index, tree_index, transcript_arrays

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
    gtf_file = '/home/user/data3/rbase/translation_pred/models/test/gencode.v48.comp_annotation_chro.pc.no_chrM.gtf'
    index_file = '/home/user/data3/rbase/translation_pred/models/lib/genome_index.pkl'
    tree_index_file = '/home/user/data3/rbase/translation_pred/models/lib/genome_index_tree.pkl'
    tx_arrays_file = '/home/user/data3/rbase/translation_pred/models/lib/transcript_arrays.pkl'
    
    # build index
    cs_index, tree_index, tx_arrays = create_optimized_index(gtf_file)
    print("index_to_save")
    print(cs_index["chr1"]['-'])
    # save
    save_index(cs_index, index_file)
    with open(tree_index_file, 'wb') as f:
        pickle.dump(tree_index, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(tx_arrays_file, 'wb') as f:
        pickle.dump(tx_arrays, f, protocol=pickle.HIGHEST_PROTOCOL)
    # load
    loaded_index = load_index(index_file)
    with open(tree_index_file, 'rb') as f:
        loaded_tree_index = pickle.load(f)
    with open(tx_arrays_file, 'rb') as f:
        loaded_tx_arrays = pickle.load(f)
    # print
    print("loaded_index")
    for index in loaded_index:
        print(index)
        print(loaded_index[index]['+'])
    for tid in loaded_tx_arrays:
        print(tid)
        print(loaded_tx_arrays[tid])
    print([iv.data for iv in loaded_tree_index['chr1']['+'][248917280]])