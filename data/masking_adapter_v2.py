import numpy as np
from numpy import random
import ahocorasick
import copy

def load_motifs_from_file(motif_file_path):
    motifs = []
    with open(motif_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            columns = line.split()
            if len(columns) < 3:
                continue
            motif = columns[2].strip().upper()
            if motif:
                motifs.append(motif)
    return motifs

def build_automaton(motifs):
    """Build Aho-Corasick automaton with caching"""
    
    automaton = ahocorasick.Automaton()
    for motif in motifs:
        # 跳过空字符串
        if motif:
            automaton.add_word(motif, motif)
    automaton.make_automaton()

    return automaton


class MaskingAdapter:
    def __init__(self, seq_dict, mask_value, motif_file_path, mask_perc=[0.15, 0.15]):
        """
        Masking adapter for RNA sequences
        
        Args:
            mask_perc (float): Maximum masking percentage (0-1)
            mask_value (float): Value to use for masking
        """
        self.seq_dict = seq_dict
        self.mask_value = mask_value
        self.motifs_list = load_motifs_from_file(motif_file_path)
        self.motif_automaton = build_automaton(self.motifs_list)
        self.mask_perc = mask_perc
    
    def mask_random_single_base(self, seq, embeddings, tx_cds):
        """
        Single-base masking for RNA sequences
        
        Args:
            seq (str): RNA sequence
            embedding (list[np.ndarray]): Original embedding matrix (max_src_len, d_model)
            
        Returns:
            masked_emb (list[np.ndarray]): Masked embedding matrix
            emb_mask (list[np.ndarray]): Boolean (True) mask indicating masked positions
        """

        src_len = len(seq)
        
        masked_emb = copy.deepcopy(embeddings)
        num = len(embeddings)
        emb_mask = [np.zeros(src_len, dtype=bool) for _ in range(num)] # mask indicator
        # for seq or count
        for i in range(num):
            d_model = embeddings[i].shape[1]
            mask_amount = round(src_len * self.mask_perc[i])
            mask_pos = random.choice(range(src_len - 1), mask_amount, replace=False)
            for p in mask_pos:
                perc = random.random()
                if perc < 0.8:
                    MASK = self.mask_value # 80% of the time, replace with [MASK]
                elif perc < 0.9:
                    MASK = random.rand(d_model) # replace with random embedding
                else:
                    MASK = embeddings[i][p, :] # replace with raw embedding
                # replace with MASK
                masked_emb[i][p, :] = MASK
                emb_mask[i][p] = True

        return masked_emb, emb_mask
    
    def mask_random_trinucleotide(self, seq, embeddings, tx_cds):
        """
        Trinucleotide (codon) masking for RNA sequences
        
        Args:
            seq (str): RNA sequence
            tx_idx (str): Transcript ID in Ensembl
            embedding (np.ndarray): Original embedding matrix (max_src_len, d_model)
            
        Returns:
            masked_emb (np.ndarray): Masked embedding matrix
            emb_mask (np.ndarray): Boolean (True) mask indicating masked positions
        """

        src_len = len(seq)

        start = 0 # frame-shift
        masked_emb = copy.deepcopy(embeddings)
        num = len(embeddings)
        emb_mask = [np.zeros(src_len, dtype=bool) for _ in range(num)]

        # for seq or count
        for i in range(num):
            mask_tri_amount = round(src_len * self.mask_perc[i] // 3) # number of trinucleotide needed to mask
            d_model = embeddings[i].shape[1]

            # mask random or mask in-frame trinucleotide
            if tx_cds['cds_start_pos'] == -1:
                tri_len = src_len // 3
            else:
                cds_start = tx_cds['cds_start_pos'] - 1
                start = cds_start % 3 # in-frame
                tri_len = (src_len - start) // 3
            
            # mask trinucleotide by index
            mask_pos = random.choice(tri_len, mask_tri_amount, replace=False)
            mask_start = start + mask_pos * 3
            mask_end = start + mask_pos * 3 + 3

            for n in range(mask_tri_amount):
                perc = random.random()
                if perc < 0.8:
                    MASK = self.mask_value # 80% of the time, replace with [MASK]
                elif perc < 0.9:
                    MASK = random.rand(3, d_model) # replace with random embedding
                else:
                    MASK = embeddings[i][mask_start[n]:mask_end[n], :] # replace with raw embedding
                # replace with MASK
                masked_emb[i][mask_start[n]:mask_end[n], :] = MASK
                emb_mask[i][mask_start[n]:mask_end[n]] = True

        return masked_emb, emb_mask

    def mask_random_motif(self, seq, embeddings, tx_cds):
        """
        Fast motif masking for RNA sequences using Aho-Corasick algorithm.
        
        Args:
            seq (str): RNA sequence
            embedding (np.ndarray): Original embedding matrix (max_src_len, d_model)
            
        Returns:
            masked_emb (np.ndarray): Masked embedding matrix
            emb_mask (np.ndarray): Boolean mask (True) indicating masked positions
        """

        src_len = len(seq)
        masked_emb = copy.deepcopy(embeddings)
        num = len(embeddings)
        emb_mask = [np.zeros(src_len, dtype=bool) for _ in range(num)]

        # for seq or count
        for i in range(num):
            d_model = embeddings[i].shape[1]

            # If no motifs provided, return random single base mask
            if not self.motif_automaton:
                return self.mask_random_single_base(seq, embeddings, tx_cds)
            else:
                # 1. Find all motif occurrences in the sequence
                seq_upper = seq.upper()
                motif_occurrences = []
                for end_idx, motif in self.motif_automaton.iter_long(seq_upper):
                    start_idx = end_idx - len(motif) + 1
                    motif_occurrences.append((start_idx, end_idx + 1))  # (start, end) where end is exclusive
                
                # 2. Calculate mask limits
                max_mask_tokens = round(src_len * self.mask_perc[i])
                if not motif_occurrences or max_mask_tokens == 0:
                    return masked_emb, emb_mask
                
                # 3. Select motifs to mask without overlaps
                # Randomize order of motif occurrences
                random.shuffle(motif_occurrences)
                
                for j in range(len(motif_occurrences)):
                    total_num_masked = sum([e - s for s, e in motif_occurrences[j:]])
                    # Skip if selected motif would not exceed the mask limit
                    if total_num_masked <= max_mask_tokens:
                        motif_occurrences = motif_occurrences[j:]
                        break
                
                # 4. Apply masking to the embedding
                for start, end in motif_occurrences:
                    perc = random.random()
                    if perc < 0.8:
                        MASK = self.mask_value # 80% of the time, replace with [MASK]
                    elif perc < 0.9:
                        MASK = random.rand(end - start, d_model) # 10% replace with random embedding
                    else:
                        MASK = embeddings[i][start:end, :] # 10% replace with raw embedding

                    # replace with MASK
                    masked_emb[i][start:end, :] = MASK
                    emb_mask[i][start:end] = True
        
        return masked_emb, emb_mask
    
    def get_random_mask_function(self, seq, embeddings, tx_cds):
        mask_func = random.choice([
            self.mask_random_single_base,
            self.mask_random_trinucleotide,
            self.mask_random_motif
        ])
        return mask_func(seq, embeddings, tx_cds)

if __name__ == "__main__":
    seq = "ACGATCGTAGCTAGCTACGTACGTAGC"
    embs = [np.ones((len(seq),1))] * 2
    adp = MaskingAdapter({}, -1, "/home/user/data3/rbase/translation_pred/models/lib/RBP_motif_annotation.v1.tsv")
    masked_emb, emb_mask = adp.mask_random_single_base(seq, embs, {'cds_start_pos': -1})

    print(len(seq))
    print(emb_mask)
    print(emb_mask[0].sum())