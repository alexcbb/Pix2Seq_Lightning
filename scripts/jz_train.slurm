#!/bin/bash
#SBATCH --job-name=pix2seq
#SBATCH --output=logs/pix2seq.%j.out
#SBATCH --error=logs/pix2seq.%j.err

#SBATCH -A uli@v100
#SBATCH -C v100-32g
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4 #number of MPI tasks per node (=number of GPUs per node)
#SBATCH --hint=nomultithread
#SBATCH -t 20:00:00
#SBATCH --mail-user=alexandre.chapin@ec-lyon.fr
#SBATCH --mail-typ=FAIL
#SBATCH --qos=qos_gpu-t3

echo ${SLURM_NODELIST}

module purge

# module load cpuarch/amd # To be compatible with a100 nodes
module load pytorch-gpu/py3/2.0.0

srun python train.py