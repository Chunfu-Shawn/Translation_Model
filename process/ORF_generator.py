import copy, sys
sys.path.append("/home/user/data3/rbase/translation_pred/models/src")
import csv
import numpy as np
import pickle
from collections import defaultdict
from data.transcript_sequence_generate import fasta_iter


__author__ = "Chunfu Xiao"
__contributor__="..."
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__="1.0.0"
__maintainer__ = "Chunfu Xiao"
__email__ = "chunfushawn@126.com"

def list():
    return []

def nested_list_defaultdict():
    return defaultdict(list)

def filter_fasta_data(merged_ORFs, fasta_in_file, fasta_out_file, orf_seq_pkl_file):
    # process fasta data
    orf_seq = {}
    print("filter out fasta data of merged ORFs")
    with open(fasta_out_file, 'w+') as fout: 
        for h, s in fasta_iter(fasta_in_file):
            # parse header
            keys = h.split("|")
            tid = keys[0].split(":")[0]
            orf_id = keys[1]

            # all merged ORFs
            if not tid in merged_ORFs:
                continue
            rows = merged_ORFs[tid]["ORFs"]

            # filter by merged ORF list
            if orf_id in [r['id'] for r in rows]:
                if tid not in orf_seq:
                    orf_seq[tid] = []
                orf_seq[tid].append(s)
            
                # write into fasta file
                fout.write(f">{h}\n")
                fout.write(f"{s}\n")


    with open(orf_seq_pkl_file, 'wb') as f_seq:
        pickle.dump(orf_seq, f_seq, protocol=pickle.HIGHEST_PROTOCOL)

def group_ORF(file_path, output_file):
    groups = defaultdict(nested_list_defaultdict)
    results = {}
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t',
                                fieldnames=["tid", "chrom", "strand", "id", "tx_len", "start", "end", "orf_type", "start_codon"])
        print("transfer tsv to dictionary data, and group ORF by end position")
        for row in reader:
            tid = row["tid"]
            row.pop("tid")

            # group by end pos
            row['tx_len'] = int(row['tx_len'])
            row['start'] = int(row['start'])
            row['end'] = int(row['end'])
            row_cp = copy.deepcopy(row)
            groups[tid][row['end']].append(row_cp)

            # save all ORFs
            chrom = row["chrom"]
            row.pop("chrom")
            tx_len = row["tx_len"]
            row.pop("tx_len")
            strand = row["strand"]
            row.pop("strand")

            if tid not in results:
                results[tid] = {
                    "chrom": chrom,
                    "tx_len": int(tx_len),
                    "strand": strand,
                    "ORFs": []
                    }
            results[tid]["ORFs"].append(row)
        
    # save data by pickle
    print("save all ORFs as dict file")
    print(results["ENST00000574003.1"])
    with open(output_file, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    return groups

def merge_ORF(ORF_groups, output_file):
    """ transfer tsv file to dictionary files"""
   
    print("merge ORF with ATG start codon or longest")
    merged_results = {}
    for tid in ORF_groups:
        # for ORFs with the same end position
        for end, rows in ORF_groups[tid].items():
            atg_rows = [r for r in rows if r['start_codon'] == 'ATG']
            atg_rows.sort(key=lambda r: r['end'] - r['start'])

            if atg_rows:
                # multi ORF with ATG, choose longest one
                chosen = atg_rows[-1]
            else:
                # no ATG, choose longest one
                chosen = min(rows, key=lambda r: r['start'])

            chrom = chosen["chrom"]
            chosen.pop("chrom")
            tx_len = chosen["tx_len"]
            chosen.pop("tx_len")
            strand = chosen["strand"]
            chosen.pop("strand")

            if tid not in merged_results:
                merged_results[tid] = {
                    "chrom": chrom,
                    "tx_len": tx_len,
                    "strand": strand,
                    "ORFs": []
                    }
            merged_results[tid]["ORFs"].append(chosen)
    
    # save data by pickle
    print("save merged ORF as dict file")
    print(merged_results["ENST00000574003.1"])
    with open(output_file, 'wb') as f:
        pickle.dump(merged_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return merged_results


if __name__=="__main__":
    out_dir = '/home/user/data3/rbase/translation_pred/models/lib/ORF/candidate_ORFs/'
    tx_meta_file = '/home/user/data3/rbase/translation_pred/models/lib/transcript_meta.pkl'
    ORF_info_tsv_file = out_dir + 'candidateORF.60nt.filtered.tx_pos.txt'
    ORF_info_dict_file1 = out_dir + 'candidateORF.60nt.filtered.tx_pos.pkl'
    ORF_info_dict_file2 = out_dir + 'candidateORF.60nt.longest.tx_pos.pkl'
    ORF_fasta_file = out_dir + 'candidateORF.60nt.fa'
    ORF_fasta_out_file = out_dir + 'candidateORF.60nt.longest.fa'
    ORF_fasta_pkl_file = out_dir + 'candidateORF.60nt.longest.seq.pkl'
    
    # generate
    print("--- Create ORF generator ---")
    groups = group_ORF(ORF_info_tsv_file, ORF_info_dict_file1)
    merged_results = merge_ORF(groups, ORF_info_dict_file2)
    filter_fasta_data(merged_results, ORF_fasta_file, ORF_fasta_out_file, ORF_fasta_pkl_file)
