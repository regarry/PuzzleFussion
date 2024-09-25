#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p volta-gpu
#SBATCH --mem=32g
#SBATCH -t 12:00:00
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

# Record the start time
start_time=$(date +%s)
echo "Start time: $(date)"

module purge
module load anaconda
conda activate puzzle
echo "Active conda environment:"
conda info --envs | grep '*' | awk '{print $1}'

module load cuda/11.8
hostname
lscpu
#free -h
#df -h
top -b | head -n 20
nvidia-smi
nvcc --version

export CUDA_VISIBLE_DEVICES=0
conda run -n puzzle python -c "import torch; print(torch.cuda.is_available())"
conda run -n puzzle python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.current_device())"
cd /work/users/r/e/regarry/PuzzleFussion/scripts
SAMPLE_FLAGS="--num_samples 2000 --dataset voronoi --batch_size 1024 --set_name test --use_image_features False"
stdbuf -oL conda run -n puzzle python image_sample.py  --set_name test --model_path ckpts/preds/model300000.pt  $SAMPLE_FLAGS

# Record the finish time
finish_time=$(date +%s)
echo "Finish time: $(date)"

# Calculate and print the elapsed time
elapsed_time=$((finish_time - start_time))
echo "Time elapsed: $(date -ud "@$elapsed_time" +'%H:%M:%S')"