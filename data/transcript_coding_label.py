import pickle
import math
import numpy as np

__author__ = "Chunfu Xiao"
__contributor__="..."
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__="1.0.0"
__maintainer__ = "Chunfu Xiao"
__email__ = "chunfushawn@126.com"

class CodingLabeler:
    def __init__(self, tx_seq_file, tx_cds_file):
        # load optimized index
        with open(tx_seq_file, 'rb') as f:
            tx_seqs = pickle.load(f)
        with open(tx_cds_file, 'rb') as f:
            tx_cds = pickle.load(f)

        self.tx_cds = tx_cds
        self.tx_seqs = tx_seqs

    def _start_stop_position_embedding(self, tid, vaild_value = 1.0):
        L = len(self.tx_seqs[tid])
        start_stop_embs = np.zeros((L, 3), dtype=np.float32) # 3 dimensions for TIS, TTS and in_ORF score (relative translation intensity)
        start_stop_embs[:, 2] = vaild_value # assume all non-start/stop codon
 
        start_tx_pos = self.tx_cds[tid]['cds_start_pos'] # 1-based
        end_tx_pos = self.tx_cds[tid]['cds_end_pos'] # 1-based
        start_codon_exist = self.tx_cds[tid]['start_codon'] # 1-based
        stop_codon_exist = self.tx_cds[tid]['stop_codon'] # 1-based
        print(start_tx_pos, end_tx_pos)

        # Cautious:
        ## cds_starts or cds_ends don't represent the real start or end of in-frame coding ORF  especially in case of lacking star/stop codon annotation
        if start_codon_exist:
            # for tis in start_tx_pos:
            start_stop_embs[start_tx_pos - 1, 0] = vaild_value
            start_stop_embs[start_tx_pos - 1, 2] = 0

        if stop_codon_exist:
            # for tts in start_tx_pos:
            # last nt of CDS in 1-based is equal to first nt of TTS in 0-based
            start_stop_embs[end_tx_pos, 1] = vaild_value
            start_stop_embs[end_tx_pos, 2] = 0

        return start_stop_embs
    
    def _coding_embedding(self, tid, coding_value = 1.0):
        L = len(self.tx_seqs[tid])
        coding_embs = np.zeros((L, 3), dtype=np.float32) # 3 dimensions for TIS, TTS and in_ORF score (relative translation intensity)
 
        start_tx_pos = self.tx_cds[tid]['cds_start_pos'] # 1-based
        end_tx_pos = self.tx_cds[tid]['cds_end_pos'] # 1-based
        start_codon_exist = self.tx_cds[tid]['start_codon'] # 1-based
        stop_codon_exist = self.tx_cds[tid]['stop_codon'] # 1-based
        print(start_tx_pos, end_tx_pos)

        # Cautious:
        ## cds_starts or cds_ends don't represent the real start or end of in-frame coding ORF  especially in case of lacking star/stop codon annotation
        coding_flag = False
        if start_codon_exist:
            # for tis in start_tx_pos:
            coding_embs[start_tx_pos - 1, 0] = 1.0
            coding_flag = True

        if stop_codon_exist:
            # for tts in start_tx_pos:
            # last nt of CDS in 1-based is equal to first nt of TTS in 0-based
            coding_embs[end_tx_pos, 1] = 1.0
            coding_flag = True

        if coding_flag:
            s = start_tx_pos - 1 if start_codon_exist else 0
            e = end_tx_pos if stop_codon_exist else L
            # CDS region (include start codon and exclude stop codon)
            coding_embs[s:e, 2] = coding_value # could replace with relatively intensity
            # print(coding_embs)

        return coding_embs
    
    def iteration(self):
        coding_emb_dict = {}
        for tid in self.tx_cds:
            print(f"--- generate coding embedding of {tid} ---")
            coding_emb_dict[tid] = self._start_stop_position_embedding(tid)

        return coding_emb_dict


if __name__=="__main__":
    tx_seq_file = '/home/user/data3/rbase/translation_pred/models/lib/tx_seq.v48.pkl'
    tx_cds_file = '/home/user/data3/rbase/translation_pred/models/lib/transcript_cds.pkl'
    tx_coding_emb_file = '/home/user/data3/rbase/translation_pred/models/lib/transcript_start_stop_embedding.pkl'

    # generate CDS embedding
    np.set_printoptions(threshold=np.inf)
    labeler = CodingLabeler(tx_seq_file, tx_cds_file)
    coding_emb_dict = labeler.iteration()

    # save
    with open(tx_coding_emb_file, 'wb') as f:
        pickle.dump(coding_emb_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

