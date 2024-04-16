#!/bin/bash
#BSUB -n 32
#BSUB -q new_gpu
#BSUB -gpu "num=2:mode=shared:mps=yes"
#BSUB -m "gpu12"
#BSUB -W 24:00
#BSUB -J PuzzleFussionVoronoi24k
#BSUB -o stdout.%J
#BSUB -e stderr.%J
source ~/.bashrc
conda activate /usr/local/usrapps/lsmsmart/regarry/puzzle
cd /share/lsmsmart/regarry/ForkedPuzzleFussion/scripts
bash script.sh
conda deactivate