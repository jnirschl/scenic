#!/bin/bash
# -*- coding: utf-8 -*-
set -o pipefail

#SBATCH --job-name=test_scenic_gpu
#SBATCH --partition=syyeung --qos=normal
#SBATCH --output=./test_scenic_%j.out # STDOUT logs %j=job id
#SBATCH --error=./test_scenic_%j.err # Error logs %j=job id 
#SBATCH --time=0:05:00 # Wall time limit days-HH:mm:ss
#SBATCH --nodes=1 # Maximum number of nodes to be allocated
#SBATCH --mem=4gb # Memory (e.g., RAM) in gb 
#SBATCH --cpus-per-task=2 # Number of cores per task
#SBATCH --gres=gpu:1


# print information
echo -e "SLURM_JOBID:\t$SLURM_JOBID
SLURM_JOB_NODELIST:\t$SLURM_JOB_NODELIST
SLURM_NNODES:\t$SLURM_NNODES
SLURMTMPDIR:\t$SLURMTMPDIR
Working directory:\t$SLURM_SUBMIT_DIR

Allocating resources..."

# sample process (list hostnames of the nodes you've requested)
NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l` 
echo -e "NPROCS:\t$NPROCS"
#echo -e node_feat -p gpu | grep GPU_
echo -e "NVIDIA-SMI:\t\n\t$(nvidia-smi)"
echo -e "NVCC version:\t\n\t$(nvcc --version)"

# load modules
echo -e "Loading modules cudnn, py-tensorflow, and py-tensorboardx"
module load cudnn
module load py-tensorflow/2.9.1_py39 # sherlock tensorflow module
module load py-tensorboardx # sherlock tensorboardx module

# list the allocated gpu, if desired
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

echo -e "Resources are allocated (with GPU)."
if [[ -f $HOME/miniconda3/etc/profile.d/conda.sh ]]; then
	echo -e "Activating conda environment scenic."
	source $HOME/miniconda3/etc/profile.d/conda.sh
	conda activate scenic
else
	echo -e 'FileNotFoundError: $HOME/miniconda3/etc/profile.d/conda.sh'
exit 1
fi

## set env var
export TF_CPP_MIN_LOG_LEVEL=0

 ## run dvc
echo -e "Running Scenic MNIST example with GPU..."
cd "$HOME/GitHub/scenic"
python3 scenic/main.py --config=scenic/projects/baselines/configs/mnist/mnist_config.py --workdir=scenic/projects/mnist/

echo -e "Exit code:	$?"
