################################################
#File Name: run.sh
#Author: rbase    
#Mail: xiaochunfu@stu.pku.edu.cn
#Created Time: Mon 04 Aug 2025 05:44:28 PM CST
################################################

#!/bin/sh 

##并发运行脚本，并控制并发数
# 设置并发的进程数
thread_num=4
a=$(date +%H%M%S)
# mkfifo
tempfifo="my_temp_fifo"
mkfifo ${tempfifo}
# 使文件描述符为非阻塞式
exec 6<>${tempfifo}
rm -f ${tempfifo}

# 为文件描述符创建占位信息
for ((i=1;i<=${thread_num};i++))
do
{
    echo 
}
done >&6 #事实上就是在fd6中放置了$thread个回车符

WORK_DIR=/home/user/data3/rbase/translation_pred/models/lib/ORF
GENOME_FILE=/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/fasta/GRCh38.primary_assembly.genome.fa
GTF_FILE=/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/gencode.v48.comp_annotation_chro.gtf
GPE_FILE=/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/gencode.v48.comp_annotation_chro.genePred.txt
RibORF=/home/user/data3/rbase/opt/RibORF/RibORF.2.0
CANDIDATE_ORF_DIR=$WORK_DIR/candidate_ORFs


# 1. RibORF workflow
echo "### RibORF workflow ###"
# Get candidate ORFs in transcripts
echo "--- Get candidate ORFs in transcripts ---"
# -g genomeSequenceFile: the genome assembly file in fasta format;
# -t transcriptomeFile: the reference transcriptome annotation file in genePred format;
# -o outputDir: output directory;
# -s startCodon [optional]: start codon types to be considered separated by “/”, default: ATG/CTG/GTG/TTG/ACG;
# -l orfLengthCutoff [optional]: cutoff of minimum candidate ORF length, default: 6nt.

echo "--- Get candidate ORFs longer than 60 nt in transcripts ---"
[ -f $CANDIDATE_ORF_DIR/candidateORF.60nt.genepred.txt ] || perl $RibORF/ORFannotate.pl \
    -g $GENOME_FILE -t $GPE_FILE -l 60 -o $CANDIDATE_ORF_DIR
# echo "--- Get candidate ORFs longer than 90 nt in transcripts ---"
# [ -f $CANDIDATE_ORF_DIR/candidateORF.90nt.genepred.txt ] || perl $RibORF/ORFannotate.pl \
#     -g $GENOME_FILE -t $GPE_FILE -l 90 -o $CANDIDATE_ORF_DIR

# There will be 2 files generated in the output directory, including “candidateORF.genepred.txt” with 
# candidate ORFs in genePred format, and “candidateORF.fa” with candidate ORF sequences in Fasta format.

# -rw-r--r-- 1 rbase bgm  12G Aug  4 21:51 candidateORF.60nt.fa
# -rw-r--r-- 1 rbase bgm 6.8G Aug  4 21:51 candidateORF.60nt.genepred.txt
# -rw-r--r-- 1 rbase bgm  12G Aug  4 21:10 candidateORF.90nt.fa
# -rw-r--r-- 1 rbase bgm 5.3G Aug  4 21:10 candidateORF.90nt.genepred.txt
# startCodonPosition 1-based
# stopCodonPosition 0-based

[ -f $CANDIDATE_ORF_DIR/candidateORF.60nt.tx_pos.txt ] || cut -f 1 $CANDIDATE_ORF_DIR/candidateORF.60nt.genepred.txt \
    | sed 's/|/\t/g' | sed 's/:/\t/g' > $CANDIDATE_ORF_DIR/candidateORF.60nt.tx_pos.txt
# stats
echo "--- Count stats of ORF type and filter --- "
[ -f $CANDIDATE_ORF_DIR/candidateORF.60nt.orf_type.txt ] || awk 'BEGIN{OFS=FS="\t"}
        {count[$8]++} 
     END{for (grp in count) print grp, count[grp] }' $CANDIDATE_ORF_DIR/candidateORF.60nt.tx_pos.txt \
    > $CANDIDATE_ORF_DIR/candidateORF.60nt.orf_type.txt
# filtering
[ -f $CANDIDATE_ORF_DIR/candidateORF.60nt.filtered.tx_pos.txt ] || \
    grep -v 'ouORF\|odORF\|extension\|readthrough\|truncation\|seqerror' $CANDIDATE_ORF_DIR/candidateORF.60nt.tx_pos.txt \
    > $CANDIDATE_ORF_DIR/candidateORF.60nt.filtered.tx_pos.txt

# generate non-redundant ORFs
echo "--- Generate non-redundant ORFs and fasta data --- "
[ -f ./ORF_generator.log ] || python ./ORF_generator.py > ./ORF_generator.log

# evaluate periodicity
echo "--- Evaluate periodicity and RRS for all candidate ORFs --- "
[ -f ./evaluator.log ] || python ./three_nucleotide_periodicity_evaluator_v2.py > ./evaluator.log