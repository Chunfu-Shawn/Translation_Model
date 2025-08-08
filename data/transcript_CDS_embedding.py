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

class CDS_Embedding:
    def __init__(self, tx_cds_file, tx_seq_file):
        # load optimized index
        with open(tx_cds_file, 'rb') as f:
            tx_cds = pickle.load(f)
        with open(tx_seq_file, 'rb') as f:
            tx_seqs = pickle.load(f)

        self.tx_cds = tx_cds
        self.tx_seqs = tx_seqs

    
    def _coding_embedding(self, tid, coding_value = 1):
        L = len(self.tx_seqs[tid])

        # split trinucleotides as a codon for 3 frames
        max_num_codon = math.floor(L / 3)
        coding_embs = np.zeros((max_num_codon, 3))

        start_tx_pos = self.tx_cds[tid]['cds_start_pos'] # 1-based
        end_tx_pos = self.tx_cds[tid]['cds_end_pos'] # 1-based
        cds_frames = self.tx_cds[tid]['cds_frames']
        print(start_tx_pos, end_tx_pos)

        # be cautious for the cds_ends
        ## cds_starts or cds_ends don't represent the real start or end of in-frame coding ORF
        ### especially in case of lacking star/stop codon annotation
        if start_tx_pos != -1:

            ##############################################################
            ### if lack of start codon annotation, infer real start pos ##
            ##############################################################
            start_tx_shift = cds_frames[0]
            start_tx_pos = start_tx_pos + start_tx_shift # in-frame start position of the transcript

            # determinate frame of coding region in the transcript (first nt is frame 0)
            transcript_frame = (start_tx_pos - 1) % 3
            # determinate the start position in embedding
            start_emb_pos = (start_tx_pos - 1) / 3 # 0-based, 1-3 nt are assigned to (0, frame)
            # transfer emb pos to tx pos: start_emb_pos * 3 + transcript_frame + 1

            #################################################################
            ### if lack of stop codon annotation, exclude incomplete codon ##
            #################################################################
            num_codon = (end_tx_pos - start_tx_pos + 1) / 3 # so take the largest integer less than (CDS length / 3)

            print("Frame shift", start_tx_shift ,"Frame in transcript: ", transcript_frame, "Start emb pos: ", start_emb_pos, "NO. codons: ", num_codon)

            # mask coding ORF as coding value (excluding stop codon)
            coding_embs[math.floor(start_emb_pos) : math.floor(start_emb_pos) + math.floor(num_codon), transcript_frame] = coding_value

        return coding_embs
    
    def coding_embedding_generate(self):
        coding_emb_dict = {}
        for tid in self.tx_cds:
            print(f"--- generate coding embedding of {tid} ---")
            coding_emb_dict[tid] = self._coding_embedding(tid)

        return coding_emb_dict


if __name__=="__main__":
    tx_seq_file = '/home/user/data3/rbase/translation_pred/models/lib/tx_seq.v48.pkl'
    tx_cds_file = '/home/user/data3/rbase/translation_pred/models/lib/transcript_cds.pkl'
    tx_coding_emb_file = '/home/user/data3/rbase/translation_pred/models/lib/transcript_coding_embedding.pkl'

    # generate CDS embedding
    cds_embedding = CDS_Embedding(tx_cds_file, tx_seq_file)
    coding_emb_dict = cds_embedding.coding_embedding_generate()

    # save
    with open(tx_coding_emb_file, 'wb') as f:
        pickle.dump(coding_emb_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

