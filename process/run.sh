################################################
#File Name: run.sh
#Author: rbase    
#Mail: xiaochunfu@stu.pku.edu.cn
#Created Time: Mon 04 Aug 2025 05:44:28 PM CST
################################################

#!/bin/sh 

WORK_DIR=/home/user/data3/rbase/translation_pred/models/lib/ORF
GENOME_FILE=/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/fasta/GRCh38.primary_assembly.genome.fa
GTF_FILE=/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/gencode.v48.comp_annotation_chro.gtf
GPE_FILE=/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/gencode.v48.comp_annotation_chro.genePred.txt
RibORF=/home/user/data3/rbase/opt/RibORF/RibORF.2.0
CANDIDATE_ORF_DIR=$WORK_DIR/candidate_ORFs
UNIPROT_FA=/home/user/data3/lit/project/sORFs/01-ribo-seq/annot/uniprot/human/uniprotkb_taxonomy_id_9606_AND_reviewed_2025_03_24.1.fasta


echo "--- Get gpe annotations of transcripts ---"
# gtfToGenePred $GTF_FILE $GPE_FILE

echo "--- Get candidate ORFs longer than 6 aa (18 + 3 nt) in transcripts ---"
# $RibORF/ORFannotate.pl:
# -g genomeSequenceFile: the genome assembly file in fasta format;
# -t transcriptomeFile: the reference transcriptome annotation file in genePred format;
# -o outputDir: output directory;
# -s startCodon [optional]: start codon types to be considered separated by “/”, default: ATG/CTG/GTG/TTG/ACG;
# -l orfLengthCutoff [optional]: cutoff of minimum candidate ORF length, default: 6nt.

# [ -f $CANDIDATE_ORF_DIR/candidateORF.20aa.genepred.txt ] || perl $RibORF/ORFannotate.pl \
#     -g $GENOME_FILE -t $GPE_FILE -l 63 -o $CANDIDATE_ORF_DIR
[ -f $CANDIDATE_ORF_DIR/candidateORF.6aa.fa ] || ( perl $RibORF/ORFannotate.pl \
    -g $GENOME_FILE -t $GPE_FILE -l 21 -o $CANDIDATE_ORF_DIR && \
    mv $CANDIDATE_ORF_DIR/candidateORF.genepred.txt $CANDIDATE_ORF_DIR/candidateORF.6aa.genepred.txt && \
    mv $CANDIDATE_ORF_DIR/candidateORF.fa $CANDIDATE_ORF_DIR/candidateORF.6aa.fa )

# There will be 2 files generated in the output directory, including “candidateORF.genepred.txt” with 
# candidate ORFs in genePred format, and “candidateORF.fa” with candidate ORF sequences in Fasta format.

# -rw-r--r-- 1 rbase bgm  12G Aug  4 21:51 candidateORF.6aa.fa
# -rw-r--r-- 1 rbase bgm 6.8G Aug  4 21:51 candidateORF.6aa.genepred.txt
# startCodonPosition 1-based
# stopCodonPosition 0-based

# get positions of ORFs in transcripts
[ -f $CANDIDATE_ORF_DIR/candidateORF.6aa.tx_pos.txt ] || cut -f 1 $CANDIDATE_ORF_DIR/candidateORF.6aa.genepred.txt \
    | sed 's/|/\t/g' | sed 's/:/\t/g' > $CANDIDATE_ORF_DIR/candidateORF.6aa.tx_pos.txt

# stats 1
echo "--- Count ORF numbers per ORF type --- "
[ -f $CANDIDATE_ORF_DIR/candidateORF.6aa.orf_type.txt ] || awk 'BEGIN{OFS=FS="\t"}
        {count[$8]++} 
     END{for (grp in count) print grp, count[grp] }' $CANDIDATE_ORF_DIR/candidateORF.6aa.tx_pos.txt \
    > $CANDIDATE_ORF_DIR/candidateORF.6aa.orf_type.txt

# filtering
echo "--- Retain specified ORF type --- "
[ -f $CANDIDATE_ORF_DIR/candidateORF.6aa.tx_pos.filtered.txt ] || \
    grep -v 'extension\|readthrough\|truncation\|seqerror' $CANDIDATE_ORF_DIR/candidateORF.6aa.tx_pos.txt \
    > $CANDIDATE_ORF_DIR/candidateORF.6aa.tx_pos.filtered.txt

# Collapse ORFs with the same stop codon and stats 2
echo "--- Collapse ORFs with the same stop codon, stats and generate fasta data --- "
[ -f $CANDIDATE_ORF_DIR/candidateORF.6aa.long.fa ] || python ./ORF_combiner_generator.py \
    --tx_pos_file $CANDIDATE_ORF_DIR/candidateORF.6aa.tx_pos.filtered.txt \
    --orf_fasta_file $CANDIDATE_ORF_DIR/candidateORF.6aa.fa \
    --out_dir $CANDIDATE_ORF_DIR

# Get protein sequences of ORFs
echo "--- Generate protein sequences of ORFs --- "
[ -f $CANDIDATE_ORF_DIR/candidateORF.6aa.long.pep.fa ] || faTrans -stop \
    $CANDIDATE_ORF_DIR/candidateORF.6aa.long.fa $CANDIDATE_ORF_DIR/candidateORF.6aa.long.pep.fa

# connect chopped fasta sequence and replace first amino acid as Met
echo "--- Connect chopped protein fasta sequences and translate non-canonical start codons as Met --- "
[ -f $CANDIDATE_ORF_DIR/candidateORF.6aa.long.M.pep.fa ] || awk 'BEGIN{RS=">"; ORS=""}
    NR>1{
    n = index($0, "\n")
    header = substr($0, 1, n-1)
    seq = substr($0, n+1)
    gsub(/\r/,"",seq)
    gsub(/\n/,"",seq)            # 合并多行为单行
    if(length(seq)>0) seq = "M" substr(seq,2)  # 无条件把首位替成 M
    else seq = ""               # 若没有序列则留空
    print ">" header "\n" seq "\n"
    }' $CANDIDATE_ORF_DIR/candidateORF.6aa.long.pep.fa \
    > $CANDIDATE_ORF_DIR/candidateORF.6aa.long.M.pep.fa

# Add uniprot reviewed proteins in the last
echo "--- Add uniprot reviewed proteins in the last --- "
cat $UNIPROT_FA >> $CANDIDATE_ORF_DIR/candidateORF.6aa.long.M.pep.fa

# Unique fasta seq of ORF and save duplicated ORFs
echo "--- Unique fasta seq of ORFs and save duplicated ORFs --- "
# -s, --by-seq           by seq
# -w, --line-width int   line width when outputting FASTA format (0 for no wrap) (default 60)
# -o, --out-file string  out file ("-" for stdout, suffix .gz for gzipped out) (default "-")
# -t, --seq-type string  sequence type (dna|rna|protein|unlimit|auto) (for auto, it automatically detect by the first sequence) (default "auto")
# -j, --threads int      number of CPUs. can also set with environment variable
[ -f $CANDIDATE_ORF_DIR/candidateORF.6aa.long.M.rmdup.pep.fa ] || seqkit rmdup -s -t protein -w 0 -j 20 \
    -D $CANDIDATE_ORF_DIR/candidateORF.6aa.long.M.dup.txt \
    -o $CANDIDATE_ORF_DIR/candidateORF.6aa.long.M.rmdup.pep.fa \
    $CANDIDATE_ORF_DIR/candidateORF.6aa.long.M.pep.fa
# first item of candidateORF.6aa.long.M.dup.txt was retained in candidateORF.6aa.long.M.rmdup.pep.fa

# stat3
echo "--- Count ORF numbers per ORF type --- "
awk '/^>/{ 
    h = substr($0, 2);            # 去掉 >
    n = split(h, a, "\\|");      # 以 | 分割
    t = (n == 5 ? a[n-1] : "Uniprot Reviewed");
    counts[t]++
}
END {
    for (k in counts) print k "\t" counts[k]
}' $CANDIDATE_ORF_DIR/candidateORF.6aa.long.M.rmdup.pep.fa | sort -k2,2nr \
    > $CANDIDATE_ORF_DIR/candidateORF.6aa.long.M.rmdup.orf_type.txt

# evaluate periodicity
# echo "--- Evaluate periodicity and RRS for all candidate ORFs --- "
# [ -f ./evaluator.log ] || python ./three_nucleotide_periodicity_evaluator_v2.py > ./evaluator.log