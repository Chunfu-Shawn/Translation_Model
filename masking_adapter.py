import numpy as np
from numpy import random
import ahocorasick

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
    def __init__(self, seq_dict, mask_perc, mask_value, motif_file_path):
        """
        Masking adapter for RNA sequences
        
        Args:
            mask_perc (float): Maximum masking percentage (0-1)
            mask_value (float): Value to use for masking
        """
        self.seq_dict = seq_dict
        self.mask_perc = mask_perc
        self.mask_value = mask_value
        self.motifs_list = load_motifs_from_file(motif_file_path)
        self.motif_automaton = build_automaton(self.motifs_list)

    def mask_random_single_base(self, seq, embedding, tx_idx):
        """
        Single-base masking for RNA sequences
        
        Args:
            seq (str): RNA sequence
            embedding (np.ndarray): Original embedding matrix (max_src_len, d_model)
            
        Returns:
            masked_embedding (np.ndarray): Masked embedding matrix
            inverse_embedding_mask (np.ndarray): Boolean mask indicating masked positions
        """

        src_len = len(seq)
        max_src_len = embedding.shape[0]
        d_model = embedding.shape[1]
        masked_embedding = embedding.copy()
        inverse_embedding_mask = np.ones(src_len, dtype=bool)

        mask_amount = round(src_len * self.mask_perc)
        for _ in range(mask_amount):
            i = random.randint(0, src_len - 1)
            perc = random.random()
            if perc < 0.8:
                MASK = self.mask_value # 80% of the time, replace with [MASK]
            elif perc < 0.9:
                MASK = random.rand(d_model) # replace with random embedding
            else:
                MASK = embedding[i, :] # replace with raw embedding
            # replace with MASK
            masked_embedding[i, :] = MASK
            inverse_embedding_mask[i] = False

        # padding to maximum length
        if max_src_len > src_len:
            pad_mask = np.full(max_src_len - src_len, True)
            inverse_embedding_mask = np.hstack((inverse_embedding_mask, pad_mask))

        return masked_embedding, inverse_embedding_mask
    
    def mask_random_trinucleotide(self, seq, embedding, tx_idx):
        """
        Trinucleotide (codon) masking for RNA sequences
        
        Args:
            seq (str): RNA sequence
            tx_idx (str): Transcript ID in Ensembl
            embedding (np.ndarray): Original embedding matrix (max_src_len, d_model)
            
        Returns:
            masked_embedding (np.ndarray): Masked embedding matrix
            inverse_embedding_mask (np.ndarray): Boolean mask indicating masked positions
        """
        src_len = len(seq)
        start = 0 # frame-shift
        mask_tri_amount = int(src_len * self.mask_perc // 3) # number of trinucleotide needed to mask
        inverse_embedding_mask = np.ones(src_len, dtype=bool) # mask indicator
        max_src_len = embedding.shape[0]
        d_model = embedding.shape[1]
        masked_embedding = embedding.copy()

        # mask random or mask in-frame trinucleotide
        if tx_idx['cds_start'] == -1:
            tri_len = src_len // 3
        else:
            cds_start = tx_idx['cds_start'] - 1
            start = cds_start % 3 # in-frame
            tri_len = (src_len - start) // 3
        
        # mask trinucleotide by index
        i = random.choice(tri_len, mask_tri_amount, replace=False)
        mask_start = start + i * 3
        mask_end = start + i * 3 + 3

        for n in range(mask_tri_amount):
            perc = random.random()
            if perc < 0.8:
                MASK = self.mask_value # 80% of the time, replace with [MASK]
            elif perc < 0.9:
                MASK = random.rand(3, d_model) # replace with random embedding
            else:
                MASK = embedding[mask_start[n]:mask_end[n], :] # replace with raw embedding
            # replace with MASK
            masked_embedding[mask_start[n]:mask_end[n], :] = MASK
            inverse_embedding_mask[mask_start[n]:mask_end[n]] = False

        # padding to maximum length
        if max_src_len > src_len:
            pad_mask = np.full(max_src_len - src_len, True)
            inverse_embedding_mask = np.hstack((inverse_embedding_mask, pad_mask))

        return masked_embedding, inverse_embedding_mask

    def mask_random_motif(self, seq, embedding, tx_idx):
        """
        Fast motif masking for RNA sequences using Aho-Corasick algorithm.
        
        Args:
            seq (str): RNA sequence
            embedding (np.ndarray): Original embedding matrix (max_src_len, d_model)
            
        Returns:
            masked_embedding (np.ndarray): Masked embedding matrix
            inverse_embedding_mask (np.ndarray): Boolean mask indicating masked positions
        """
        src_len = len(seq)
        d_model = embedding.shape[1]
        max_src_len = embedding.shape[0]
        masked_embedding = embedding.copy()
        inverse_embedding_mask = np.ones(src_len, dtype=bool)
        # If no motifs provided, return original sequence
        if not self.motif_automaton:
            return masked_embedding, inverse_embedding_mask
        else:

            # 1. Find all motif occurrences in the sequence
            seq_upper = seq.upper()
            motif_occurrences = []
            for end_idx, motif in self.motif_automaton.iter_long(seq_upper):
                start_idx = end_idx - len(motif) + 1
                motif_occurrences.append((start_idx, end_idx + 1))  # (start, end) where end is exclusive
            
            # 2. Calculate mask limits
            max_mask_tokens = int(src_len * self.mask_perc)
            if not motif_occurrences or max_mask_tokens == 0:
                return masked_embedding, inverse_embedding_mask
            
            # 3. Select motifs to mask without overlaps
            # Randomize order of motif occurrences
            random.shuffle(motif_occurrences)
            
            for i in range(len(motif_occurrences)):
                total_masked = sum([e - s for s, e in motif_occurrences[i:]])
                # Skip if selected motif would not exceed the mask limit
                if total_masked <= max_mask_tokens:
                    motif_occurrences = motif_occurrences[i:]
                    break
            
            # 4. Apply masking to the embedding
            for start, end in motif_occurrences:
                perc = random.random()
                if perc < 0.8:
                    MASK = self.mask_value # 80% of the time, replace with [MASK]
                elif perc < 0.9:
                    MASK = random.rand(end - start, d_model) # 10% replace with random embedding
                else:
                    MASK = embedding[start:end, :] # 10% replace with raw embedding

                # replace with MASK
                masked_embedding[start:end, :] = MASK
                inverse_embedding_mask[start:end] = False
        
        # 5. Handle padding if needed
        if max_src_len > src_len:
            pad_mask = np.full(max_src_len - src_len, True)
            inverse_embedding_mask = np.hstack((inverse_embedding_mask, pad_mask))
        
        return masked_embedding, inverse_embedding_mask
    
    def get_random_mask_function(self, seq, embedding, tx_idx):
        mask_func = random.choice([
            self.mask_random_single_base,
            self.mask_random_trinucleotide,
            self.mask_random_motif
        ])
        return mask_func(seq, embedding, tx_idx)
