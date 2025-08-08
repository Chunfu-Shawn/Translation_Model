import pickle
from itertools import groupby


__author__ = "Chunfu Xiao"
__contributor__="..."
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__="1.0.0"
__maintainer__ = "Chunfu Xiao"
__email__ = "chunfushawn@126.com"


# load fasta files
def fasta_iter(fasta_file):
    """
    given a fasta file, yield tuples of header, sequence
    """
    with open(fasta_file) as file:
        # ditch the boolean (x[0]) and just keep the header or sequence since
        faiter = (x[1] for x in groupby(file, lambda line: line[0] == ">"))
        for header in faiter:
            header = header.__next__()[1:].strip() # drop the ">"
            seq = "".join(s.strip() for s in faiter.__next__()) # join all sequences
            yield header, seq

if __name__=="__main__":
    tx_meta_file = '/home/user/data3/rbase/translation_pred/models/lib/transcript_meta.pkl'
    fasta_tx_file = '/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/fasta/transcripts/gencode.v48.transcripts.fa'
    tx_seq_file = '/home/user/data3/rbase/translation_pred/models/lib/tx_seq.v48.pkl'

    # load tx index
    with open(tx_meta_file, 'rb') as f:
        tx_meta = pickle.load(f)

    # process fasta data
    fasta_tx = fasta_iter(fasta_tx_file)
    # save pickle data
    tx_seq = {}
    for h, s in fasta_tx:
        tx_id = h.split("|")[0]
        if tx_id in tx_meta:
            tx_seq[tx_id] = s

    with open(tx_seq_file, 'wb') as f_seq:
        pickle.dump(tx_seq, f_seq, protocol=pickle.HIGHEST_PROTOCOL)
