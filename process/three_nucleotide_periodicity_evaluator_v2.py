import sys, time
sys.path.append("/home/user/data3/rbase/translation_pred/models/src")
import numpy as np
import pickle
from numba import njit
from concurrent.futures import ProcessPoolExecutor

__author__ = "Chunfu Xiao"
__contributor__="..."
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__="1.0.0"
__maintainer__ = "Chunfu Xiao"
__email__ = "chunfushawn@126.com"


@njit(cache=True, nogil=True, fastmath=True)
def calc_periodicity_rrs(count_pos, starts, ends, eps):
    n_orf = starts.size
    sum_count_orf = np.empty(n_orf, dtype=np.int32)
    periodicity = np.empty(n_orf, dtype=np.float32)
    rrs = np.empty(n_orf, dtype=np.float32)
    # for each ORF
    for k in range(n_orf):
        start, end = starts[k], ends[k] # 0-based
        orf_len = end - start

        # 3nt periodicity
        sum_count_3nt = np.array([count_pos[start+j : end : 3].sum() for j in range(3)])
        periodicity[k] = (max(sum_count_3nt) + eps) / (sum_count_3nt.sum() + eps)

        # ribosome release score
        sum_count_orf[k] = count_pos[start : end].sum()
        sum_count_3UTR = count_pos[end : end + orf_len].sum()
        rrs[k] = (sum_count_orf[k] + eps) / (sum_count_3UTR + eps)

    return sum_count_orf, periodicity, rrs


class PeriodicityEvaluator:
    def __init__(self, chroms, ORF_info_file, offset = 12, eps=5):
        self.chroms = chroms
        self.ORF_info_file = ORF_info_file
        self.offset = offset
        self.eps = eps


    def process_chrom(self, args):
        """process for one chromosome"""
        chrom, RPF_count_file = args
        with open(self.ORF_info_file, 'rb') as f_info:
            ORF_info_all = pickle.load(f_info)
            ORF_info = {tid: v for tid, v in ORF_info_all.items()
                        if v["chrom"] == chrom}
        with open(RPF_count_file, 'rb') as f_count:
            RPF_all = pickle.load(f_count)
            RPF_count = {tid: v for tid, v in RPF_all.items()
                         if tid in ORF_info}
        print("### " + chrom + " ###")
        
        # chromosome-specific reads
        print(f'Executor {chrom} start time: {time.time() - start_time} seconds')
        ORF_results = {}
        for tid in RPF_count:
            print(f"--- for {tid} ---")
            tx_len = ORF_info[tid]["tx_len"]

            # numpy matrix
            count_pos = np.zeros(tx_len)
            count = RPF_count[tid]
            for read_l in count:
                if read_l in range(27, 32): # only consider reads of 27-31 nt
                    for pos in count[read_l]:
                        # add offset to infer P-site
                        count_pos[pos - 1 + self.offset] += count[read_l][pos]
            
            ids = np.array([o['id'] for o in ORF_info[tid]["ORFs"]])
            starts = np.array([int(o['start']) - 1 for o in ORF_info[tid]["ORFs"]], dtype=np.int32)
            ends = np.array([int(o['end']) - 1 for o in ORF_info[tid]["ORFs"]], dtype=np.int32)

            sum_count_orf, periodicity, rrs = calc_periodicity_rrs(count_pos, starts, ends, self.eps)

            ORF_results[tid] = [
                {
                    'id': str(i),
                    'read_count_ORF': int(s),
                    'periodicity': float(p),
                    'RRS': float(r)
                } for i, s, p, r in zip(ids, sum_count_orf, periodicity, rrs)
            ]
               
        print(f'Executor {chrom} end time: {time.time() - start_time} seconds')
        return ORF_results

    def parallel_eval_by_chrom(self, RPF_count_file):
        """ process all chromosones parallelly """

        # create tasks
        print("--- Create parallel tasks ---")
        final_results = {}
        global start_time
        start_time = time.time()
        with ProcessPoolExecutor() as executor:
            # chromosome-specific arguments
            args = [
                (
                    chrom, RPF_count_file
                ) for chrom in self.chroms
                ]
            results_iterator = executor.map(self.process_chrom, args)

            # combine results within the executor
            final_results = {}
            for chrom_result in results_iterator:
                for k, v in chrom_result.items():
                    final_results[k] = v
            return final_results



if __name__=="__main__":
    ORF_info_dict_file = '/home/user/data3/rbase/translation_pred/models/lib/ORF/candidate_ORFs/candidateORF.60nt.longest.tx_pos.pkl'
    RPF_count_file = '/home/user/data3/yaoc/translation_model/model/SRR15513148_49_50_51_52.read_count.pkl'
    result_file = '/home/user/data3/rbase/translation_pred/models/lib/ORF/SRR15513148_49_50_51_52.ORF_period_rrs.pkl'
    chroms = ["chr17", "chr19", "chr11"] + \
        [f'chr{i}' for i in range(1, 11)] + \
        [f'chr{i}' for i in range(12, 17)] + \
        ['chr18', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY']

    # count parallelly
    print("--- Create Evaluator ---")
    counter = PeriodicityEvaluator(chroms, ORF_info_dict_file, offset=12, eps=5)

    # for each Ribo-seq sample
    results = counter.parallel_eval_by_chrom(RPF_count_file)
    print(results["ENST00000574003.1"])
    with open(result_file, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)