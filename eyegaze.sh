#!/bin/bash
#SBATCH -A rohit.gajawada
#SBATCH --gres=gpu:1
#SBATCH -n 20
#SBATCH --mincpus=20
#SBATCH --mem-per-cpu=4096
#SBATCH --time=72:00:00

python3 main.py
