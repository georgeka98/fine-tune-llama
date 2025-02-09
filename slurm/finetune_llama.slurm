#!/bin/bash
#SBATCH --job-name=llama_finetune
#SBATCH --output=logs/training.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

module load python/3.10
source ~/venvs/llama/bin/activate
python scripts/train.py --config configs/config.yaml
