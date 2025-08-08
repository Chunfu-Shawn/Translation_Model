import sys, time
sys.path.append("/home/user/data3/rbase/translation_pred/models/src")
import multiprocessing
import numpy as np
import pickle
from numba import njit
# multiprocessing.set_start_method('fork')
mp_ctx = multiprocessing.get_context('fork')
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import defaultdict
from intervaltree import IntervalTree
import pysam
from data.transcript_exon_index import convert_position

__author__ = "Chunfu Xiao"
__contributor__="..."
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__="1.0.0"
__maintainer__ = "Chunfu Xiao"
__email__ = "chunfushawn@126.com"

def zero():
    return 0

def nested_zero_defaultdict():
    return defaultdict(zero)

def double_nested_zero_defaultdict():
    return defaultdict(nested_zero_defaultdict)


def _init_worker(bam_path):
    """初始化工作进程，只执行一次"""
    global bam_cache
    bam_cache = pysam.AlignmentFile(bam_path, 'rb', threads=5)


@njit
def is_compatible(starts, ends, exon_starts0, exon_ends0, tol=2):
    """
    Junction_reads evaluation 边界兼容性检查：
    - blocks[0].start 只要在 exon 内部 (>= exon_start - tol) 即可
    - blocks[-1].end   只要在 exon 内部 (<= exon_end   + tol) 即可
    - 中间 blocks 必须严格对齐到边界 (±tol)
    - exon-order 连续不跳跃
    """

    # 1) calculate gaps to find real breaks of alignment, excluding the case of CIGAR: M1DM
    if len(starts) > 1:
        gaps = starts[1:] - ends[:-1]
        breaks_idx = np.nonzero(gaps > tol)[0]
        # 每段的第一个/最后一个 block 索引
        seg_starts = np.concatenate((
            np.array([0], dtype=breaks_idx.dtype),
            breaks_idx + 1
        ))
        seg_ends   = np.concatenate((
            breaks_idx,
            np.array([len(starts) - 1], dtype=breaks_idx.dtype)
        ))
    else:
        seg_starts = np.array([0])
        seg_ends   = np.array([0])

    # merged blocks ignoring 2 nt gap
    m_starts = starts[seg_starts]
    m_ends = ends[seg_ends]
    B = len(m_starts)

    # 2) 整体二分查找：先找到每个 block 属于哪个 exon (with tol)
    idxs = np.searchsorted(exon_starts0 - tol, m_starts , side='right') - 1
    if np.any(idxs < 0):
        return False

    # 3) 所有 block 中间的 start 和 end（如果 B>2）必须满足边界对齐
    if B > 2:
        starts_without_first = m_starts[1:]
        ends_without_last = m_ends[:-1]
        
        # start 对齐或 end 对齐（±tol）
        start_ok = np.abs(starts_without_first - exon_starts0[idxs[1:]]) <= tol
        end_ok = np.abs(ends_without_last - exon_ends0[idxs[:-1]]) <= tol
        if not np.all(start_ok & end_ok):
            print("Junction Error", idxs, m_starts, exon_starts0, m_ends, exon_ends0)
            return False

    # 4) exon-order 连续性检查
    diffs = np.diff(idxs)
    if np.any(diffs < 0) or np.any(diffs > 1):
        print("Exon-order Error", idxs, diffs, m_starts, exon_starts0, m_ends, exon_ends0)
        return False

    return True


class RPF_Counter:
    def __init__(self, chroms, tree_index_file, tx_meta_file, min_readlen = 25, max_readlen = 34, tol = 2):
        # load optimized index
        with open(tree_index_file, 'rb') as f:
            tree_index = pickle.load(f)
        with open(tx_meta_file, 'rb') as f:
            tx_meta = pickle.load(f)

        self.chroms = chroms
        self.tree_index = tree_index
        self.filterd_tree_index = tree_index
        self.tx_meta = tx_meta
        # 按染色体组织转录本数据
        self.tx_meta_by_chrom = defaultdict(dict)
        for tid, meta in self.tx_meta.items():
            chrom = meta['chrom']
            self.tx_meta_by_chrom[chrom][tid] = meta
        # 预计算染色体长度
        self.chrom_lengths = {
            chrom: max(meta['exon_ends0_sorted'][-1] for meta in self.tx_meta.values() if meta['chrom']==chrom)
            for chrom in self.chroms
        }
        self.min_readlen = min_readlen
        self.max_readlen = max_readlen
        self.tol = tol

    def save_count(self, final_counts, count_file):
        ''' save count data as .pkl file'''
        # save data by pickle
        with open(count_file, 'wb') as f_RPF:
            pickle.dump(final_counts, f_RPF, protocol=pickle.HIGHEST_PROTOCOL)

    def process_chrom(self, args):
        """process for one chromosome"""
        chrom = args
        print("### " + chrom + " ###")
        
        counts = defaultdict(double_nested_zero_defaultdict)
        tree_index = self.filterd_tree_index[chrom]
        tx_meta = self.tx_meta_by_chrom[chrom]
        # chromosome-specific reads
        print(f'Executor {chrom} start time: {time.time() - start_time} seconds')
        for read in bam_cache.fetch(chrom):
            ##### read in bam file is 0-based #####
            # mapped reads 
            blk = np.array(read.get_blocks(), dtype=int)
            if blk.size == 0:
                return False
            # reasonable RPF length
            read_len = read.query_length
            if read_len < self.min_readlen or read_len > self.max_readlen:
                continue
            
            # 5'end and 3'end genomic position, 0-base to 1-base
            starts = blk[:,0]
            ends   = blk[:,1]
            left_prime = starts[0] + 1
            right_prime = ends[-1]
            
            # find all transcripts overlapping reads
            left_overlaps = tree_index[left_prime]
            right_overlaps = tree_index[right_prime]
            if len(left_overlaps) == 0 or len(right_overlaps) == 0:
                continue
            left_tids = [iv.data for iv in left_overlaps]
            right_tids = [iv.data for iv in right_overlaps]
            ## both left and right prime of read located in the transcript
            tids = [tid for tid in left_tids if tid in right_tids]

            # transfer genomic position to tx position
            meta_data = {tid: meta for (tid, meta) in tx_meta.items() if tid in tids}
            for tid, meta in meta_data.items():
                # compatible with transcript exon structure ?
                if not is_compatible(starts, ends, meta['exon_starts0_sorted'], meta['exon_ends0_sorted'], tol=self.tol):
                    continue
                
                # transfer position and count
                pos = convert_position(
                    left_prime if meta['strand'] == "+" else right_prime,
                    meta['exon_starts'],
                    meta['exon_ends'],
                    meta['tx_starts'],
                    meta['tx_ends'],
                    meta['strand']
                )
                if pos >= 1:
                    counts[tid][read_len][pos] += 1

        print(f'Executor {chrom} end time: {time.time() - start_time} seconds')
        return counts

    def parallel_count_by_chrom(self, 
                                bam_path, 
                                tid_list: list = [], 
                                max_workers: int = 20):
        """ process all chromosones parallelly """
        # creare new tree for target transcripts
        if tid_list:
            print("--- Only count the RPFs of given transctipts ---")
            for chrom in self.tree_index:
                filterd_ivs = [iv for iv in self.tree_index[chrom] if iv.data in tid_list]
                self.filterd_tree_index[chrom] = IntervalTree(filterd_ivs)

        # create tasks
        print("--- Create parallel tasks ---")
        final_counts = {}
        with ThreadPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(bam_path,)) as executor:
            # chromosome-specific arguments
            tasks = [chrom for chrom in self.chroms]
            end_time = time.time()
            print(f'exec time: {end_time - start_time} seconds')
            futures = {executor.submit(self.process_chrom, t): t for t in tasks}

            # combine results within the executor
            for fut in as_completed(futures):
                local = fut.result()
                for tid, counts in local.items():
                    final_counts[tid] = counts
            return final_counts
    
    def process_window(self, args):
        chrom, start, end = args # , tree_index, tx_meta
        print("### " + chrom + " ###")

        counts = defaultdict(double_nested_zero_defaultdict)
        tree_index = self.filterd_tree_index[chrom]
        tx_meta = self.tx_meta_by_chrom[chrom]
        print(f'Executor {chrom}:{start}-{end} start time: {time.time() - start_time} seconds')

        # window-specific reads
        for read in bam_cache.fetch(chrom, start-1, end):
            blk = np.array(read.get_blocks(), dtype=int)
            if blk.size == 0:
                return False
            # reasonable RPF length
            read_len = read.query_length
            if read_len < self.min_readlen or read_len > self.max_readlen:
                continue
            
            # 5'end and 3'end genomic position, 0-base to 1-base
            starts = blk[:,0]
            ends   = blk[:,1]
            left_prime = starts[0] + 1
            right_prime = ends[-1]

            # find all transcripts overlapping reads
            left_overlaps = tree_index[left_prime]
            right_overlaps = tree_index[right_prime]
            if len(left_overlaps) == 0 or len(right_overlaps) == 0:
                continue
            left_tids = [iv.data for iv in left_overlaps]
            right_tids = [iv.data for iv in right_overlaps]
            ## both left and right prime of read located in the transcript
            tids = [tid for tid in left_tids if tid in right_tids]
    
            # transfer genomic position to tx position
            blk = np.array(read.get_blocks(), dtype=int)
            meta_data = {tid: meta for (tid, meta) in tx_meta.items() if tid in tids}
            for tid, meta in meta_data.items():
                # compatible with transcript exon structure ?
                if not is_compatible(starts, ends, meta['exon_starts0_sorted'], meta['exon_ends0_sorted'], tol=self.tol):
                    continue
                # transfer to transcript position (input 1-based)
                pos = convert_position(
                    left_prime if meta['strand'] == "+" else right_prime,
                    meta['exon_starts'],
                    meta['exon_ends'],
                    meta['tx_starts'],
                    meta['tx_ends'],
                    meta['strand']
                )
                # count the read
                if pos >= 1:
                    counts[tid][read_len][pos] += 1
                        
        print(f'Executor {chrom}:{start}-{end} end time: {time.time() - start_time} seconds')
        return counts

    def parallel_count_by_windows(self, 
                                  bam_path, 
                                  tid_list: list = [], 
                                  window_size: int = 20000000, 
                                  max_workers: int = 20):
        global start_time
        start_time = time.time()
        
        """ process all windows parallelly """
        # creare new tree for target transcripts
        if tid_list:
            print("--- Only count the RPFs of given transctipts ---")
            for chrom in self.tree_index:
                filterd_ivs = [iv for iv in self.tree_index[chrom] if iv.data in tid_list]
                self.filterd_tree_index[chrom] = IntervalTree(filterd_ivs)

        # create tasks
        print("--- Create parallel tasks ---")
        tasks = []
        for chrom, length in self.chrom_lengths.items():
            for start in range(1, length+1, window_size):
                end = min(start + window_size - 1, length)
                tasks.append((chrom, start, end))
        end_time = time.time()
        print(f'exec time: {end_time - start_time} seconds')

        # 2) 并行处理
        final_counts = {}
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(bam_path,), mp_context=mp_ctx) as exe:
            futures = {exe.submit(self.process_window, task): task for task in tasks} # tasks[::-1] reverse chromosomes for saving time
            
            for future in as_completed(futures):
                result = future.result()
                # combine results
                for tid, count_data in result.items():
                    final_counts[tid] = count_data
        return final_counts


if __name__=="__main__":
    tree_index_file = '/home/user/data3/rbase/translation_pred/models/lib/genome_index_tree.pkl'
    tx_meta_file = '/home/user/data3/rbase/translation_pred/models/lib/transcript_meta.pkl'
    bam_file = '/home/user/data3/rbase/translation_pred/models/test/SRR15513158.uniq.sorted.pc.bam'
    RPF_count_file = '/home/user/data3/rbase/translation_pred/models/test/SRR15513158.read_count.pkl'
    chroms = ["chr17", "chr19", "chr11"] + \
        [f'chr{i}' for i in range(1, 11)] + \
        [f'chr{i}' for i in range(12, 17)] + \
        ['chr18', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY']

    # count parallelly
    counter = RPF_Counter(chroms, tree_index_file, tx_meta_file, 25, 34, 3)
    # final_counts = counter.parallel_count_by_chrom(bam_file, max_workers=24)
    final_counts = counter.parallel_count_by_windows(bam_file, [], window_size=83000000, max_workers=20)
    counter.save_count(final_counts, RPF_count_file)