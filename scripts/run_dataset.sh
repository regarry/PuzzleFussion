#!/bin/bash
#BSUB -n 32
#BSUB -W 2:00
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -q new_gpu
#BSUB -J PuzzleFussionVoronoi24k
#BSUB -o stdout.%J
#BSUB -e stderr.%J
source ~/.bashrc
conda activate /usr/local/usrapps/lsmsmart/regarry/puzzle
cd /share/lsmsmart/regarry/ForkedPuzzleFussion/scripts
mpiexec -n 32 python make_dataset.py
conda deactivate