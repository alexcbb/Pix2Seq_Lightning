#!/bin/bash
#SBATCH --job-name=pix2seq
#SBATCH --output=logs/pix2seq.%j.out
#SBATCH --error=logs/pix2seq.%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=180G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 80:00:00
#SBATCH --mail-user=alexandre.chapin@ec-lyon.fr
#SBATCH --mail-typ=FAIL

echo ${SLURM_NODELIST}

module purge

source ~/.bashrc

conda activate mttoc

srun python train.py