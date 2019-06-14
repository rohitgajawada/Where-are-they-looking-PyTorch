#!/bin/bash
#SBATCH -A rohit.gajawada
#SBATCH --gres=gpu:1
#SBATCH -n 20
#SBATCH --mincpus=18
#SBATCH --mem-per-cpu=2048
#SBATCH --time=72:00:00

python3 -W ignore main.py 
