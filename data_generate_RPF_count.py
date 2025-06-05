# from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
import os
import pysam
from collections import defaultdict
from genome_tx_exon_index import *

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

def split_bam(bam_file, chrs, thread_count=20):
    out_file_prefix = bam_file.strip('bam')
    # splited file path
    bam_files_dic = {}
    for chr in chrs:
        print("---- split raw bam file for " + chr + " ----")
        out_file = out_file_prefix + chr + ".bam"
        os.system("samtools view -@ {tn} -b {bam} {chr} | samtools sort -@ {tn} -o {out}".format(tn=thread_count, bam=bam_file, chr=chr, out=out_file))
        os.system("samtools index {out} {out}.bai".format(out=out_file))
        bam_files_dic[chr] = out_file
    return bam_files_dic

def remove_splited_bam(raw_bam_file, chrs):
    raw_file_prefix = raw_bam_file.strip('bam')
    for chr in chrs:
        print("---- remove splited bam file for " + chr + " ----")
        target_file = raw_file_prefix + chr + ".bam"
        os.system("rm {file}".format(file=target_file))
        os.system("rm {file}.bai".format(file=target_file))

def process_chrom(args):
    """process for one chromosome"""
    chrom, bam_file, tree_index, tx_arrays = args
    print("### " + chrom + " ###")
    # define count data structure
    counts = defaultdict(double_nested_zero_defaultdict)
    bam = pysam.AlignmentFile(bam_file, 'rb')
    # chromosome-specific reads
    for read in bam.fetch(chrom):
        print(read.query_name)
        # mapped reads
        if read.is_unmapped:
            continue
        # reasonable RPF length
        read_len = len(read.query_sequence)
        if read_len < 25 or read_len > 34:
            continue
        # for transcipts in both strand
        for strand in ["+", "-"]:
            # 5'end and 3'end genomic position
            five_prime = read.reference_start + 1 if strand == "+" else read.reference_end
            
            # find all transcripts overlapping reads
            overlaps = tree_index[strand][five_prime]
            # mask = (tx_index[strand][:,0].astype(int) <= five_prime) & (five_prime <= tx_index[strand][:,1].astype(int))
            # metches = tx_index[strand][mask]
            if len(overlaps) == 0:
                continue
            
            # tx id
            tids = [iv.data for iv in overlaps]

            # transfer genomic position to tx position
            arr_data = {k:v for (k,v) in tx_arrays.items() if k in tids}
            for tid in arr_data:
                pos = convert_position(
                    five_prime,
                    arr_data[tid]['exon_starts'],
                    arr_data[tid]['exon_ends'],
                    arr_data[tid]['tx_starts'],
                    arr_data[tid]['tx_ends'],
                    strand
                )
                if pos >= 0:
                    counts[tid][read_len][pos] += 1
    if len(counts.keys()) == 0:
        print("No reads in " + chrom)
    return counts

def parallel_count(bam_file, tree_index, tx_arrays):
    """ process all chromosones parallelly """
    print("Count the 5'end of RPF for each transcript")

    with ProcessPoolExecutor() as executor:
        # chromosome-specific arguments
        args = [(chrom, bam_file, tree_index[chrom], {k:v for (k,v) in tx_arrays.items() if v["chrom"]==chrom})
                for chrom in tree_index.keys()]
        results_iterator = executor.map(process_chrom, args)

        # combine results within the executor
        final_counts = {}
        for chrom_result in results_iterator:
            for k, v in chrom_result.items():
                final_counts[k] = v
        return final_counts


if __name__=="__main__":
    tree_index_file = '/home/user/data3/rbase/translation_pred/models/lib/genome_index_tree.pkl'
    tx_arrays_file = '/home/user/data3/rbase/translation_pred/models/lib/transcript_arrays.pkl'
    bam_file = '/home/user/data3/rbase/translation_pred/models/test/SRR15513158.uniq.sorted.pc.bam'
    RPF_count_file = '/home/user/data3/rbase/translation_pred/models/test/SRR15513158.read_count.pkl'

    # load optimized index
    with open(tree_index_file, 'rb') as f:
        tree_index = pickle.load(f)
    with open(tx_arrays_file, 'rb') as f:
        tx_arrays = pickle.load(f)
    print(tree_index.keys())

    # count parallelly
    final_counts = parallel_count(bam_file, tree_index, tx_arrays)

    # save data by pickle
    with open(RPF_count_file, 'wb') as f_RPF:
        pickle.dump(final_counts, f_RPF, protocol=pickle.HIGHEST_PROTOCOL)
    
    # load results
    with open(RPF_count_file, 'rb') as f_RPF:
        loaded_counts = pickle.load(f_RPF)
    for tid in loaded_counts:
        for pos in sorted(loaded_counts[tid]):
            print(f"{tid}\t{pos}\t{loaded_counts[tid][pos]}")