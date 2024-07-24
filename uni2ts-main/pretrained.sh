#!/bin/bash
#SBATCH -p ampere
#SBATCH --account BRINTRUP-SL3-GPU
#SBATCH -D /home/yl892/rds/hpc-work/Time-Series-with-Uncertainty/uni2ts-main
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=80G    # RAM memory. Default: 1G
#SBATCH -t 12:00:00 # time for the job HH:MM:SS. Default: 1 min
python -m cli.train \
  -cp conf/pretrain \
  run_name=first_run \
  model=moirai_small \
  data=lotsa_v1_unweighted