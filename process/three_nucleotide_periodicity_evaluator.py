import sys, time
import csv
sys.path.append("/home/user/data3/rbase/translation_pred/models/src")
import multiprocessing
import numpy as np
import pickle

mp_ctx = multiprocessing.get_context('fork')
from concurrent.futures import ProcessPoolExecutor

__author__ = "Chunfu Xiao"
__contributor__="..."
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__="1.0.0"
__maintainer__ = "Chunfu Xiao"
__email__ = "chunfushawn@126.com"


def tsv_to_dicts(file_path, output_file):
    """ transfer tsv file to dictionary files"""
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t',
                                fieldnames=["tid", "chrom", "strand", "id", "tx_len", "start", "end", "orf_type", "start_codon"])
        result = {}
        print("transfer tsv to dict file")
        for row in reader:
            chrom = row["chrom"]
            row.pop("chrom")
            tid = row["tid"]
            row.pop("tid")
            tx_len = row["tx_len"]
            row.pop("tx_len")
            strand = row["strand"]
            row.pop("strand")
            if chrom not in result:
                result[chrom] = {}
            if tid not in result[chrom]:
                result[chrom][tid] = {
                    "tx_len": np.int32(tx_len),
                    "strand": strand,
                    "ORFs": []
                    }
            result[chrom][tid]["ORFs"].append(row)
        
        # save data by pickle'
        print("save dict file")
        with open(output_file, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)


class PeriodicityEvaluator:
    def __init__(self, chroms, ORF_info_file, offset = 12, eps=5):
        self.chroms = chroms
        with open(ORF_info_file, 'rb') as f_info:
            self.ORF_info = pickle.load(f_info)
        self.offset = offset
        self.eps = eps
    

    def getitem(self, chrom, tid):
        return(self.ORF_info[chrom][tid])


    def save_periodicity(self, periodicity, file):
        ''' save count data as .pkl file'''
        # save data by pickle
        with open(file, 'wb') as f:
            pickle.dump(periodicity, f, protocol=pickle.HIGHEST_PROTOCOL)


    def process_chrom(self, args):
        """process for one chromosome"""
        chrom, ORF_info, RPF_count = args
        print("### " + chrom + " ###")
        
        # chromosome-specific reads
        print(f'Executor {chrom} start time: {time.time() - start_time} seconds')
        ORF_results = {}
        for tid in RPF_count:
            ORF_results[tid] = []
            tx_len = ORF_info[tid]["tx_len"]

            count_pos = np.zeros(tx_len)
            count = RPF_count[tid]
            for read_l in count:
                if read_l in range(27, 32): # only consider reads of 27-31 nt
                    for pos in count[read_l]:
                        # add offset to infer P-site
                        count_pos[pos - 1 + self.offset] += count[read_l][pos]
            # for each ORF
            for orf in ORF_info[tid]["ORFs"]:
                start = int(orf['start']) - 1 # 0-based
                end = int(orf['end']) - 1 # 0-based
                orf_len = end - start

                # 3nt periodicity
                sum_count_3nt = np.array([np.sum(count_pos[start+j : end : 3], axis=0) for j in range(3)])
                periodicity = np.max(sum_count_3nt) / np.sum(sum_count_3nt)

                # ribosome release score
                sum_count_orf = np.sum(count_pos[start : end])
                rrs = (sum_count_orf + self.eps) / (np.sum(count_pos[end : end + orf_len]) + self.eps)

                ORF_results[tid].append({
                    'read_count_ORF': sum_count_orf,
                    'periodicity': periodicity,
                    'RRS': rrs
                })

        print(f'Executor {chrom} end time: {time.time() - start_time} seconds')
        return ORF_results

    def parallel_eval_by_chrom(self, RPF_count_file):
        """ process all chromosones parallelly """

        with open(RPF_count_file, 'rb') as f_count:
            RPF_count = pickle.load(f_count)

        # create tasks
        print("--- Create parallel tasks ---")
        final_results = {}
        global start_time
        start_time = time.time()
        with ProcessPoolExecutor() as executor:
            # chromosome-specific arguments
            args = [
                (
                    chrom, self.ORF_info[chrom], {tid: v for tid, v in RPF_count.items() if tid in self.ORF_info[chrom]}, 
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
    tx_meta_file = '/home/user/data3/rbase/translation_pred/models/lib/transcript_meta.pkl'
    ORF_info_tsv_file = '/home/user/data3/rbase/translation_pred/models/lib/ORF/candidate_ORFs/candidateORF.60nt.filtered.tx_pos.txt'
    ORF_info_dict_file = '/home/user/data3/rbase/translation_pred/models/lib/ORF/candidate_ORFs/candidateORF.60nt.filtered.tx_pos.pkl'
    RPF_count_file = '/home/user/data3/yaoc/translation_model/model/SRR15513148_49_50_51_52.read_count.pkl'
    chroms = ["chr17", "chr19", "chr11"] + \
        [f'chr{i}' for i in range(1, 11)] + \
        [f'chr{i}' for i in range(12, 17)] + \
        ['chr18', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY']
    
    # generate
    # tsv_to_dicts(ORF_info_tsv_file, ORF_info_dict_file)
    # count parallelly
    print("--- Create Evaluator ---")
    counter = PeriodicityEvaluator(chroms, ORF_info_dict_file, offset=12)

    # for each Ribo-seq sample
    results = counter.parallel_eval_by_chrom(RPF_count_file)
    print(results["ENST00000574003.1"])