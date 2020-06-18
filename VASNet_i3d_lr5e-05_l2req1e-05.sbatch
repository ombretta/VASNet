#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2000
#SBATCH --gres=gpu:1
module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.6.0.64
srun python main.py --train  --datasets datasets/datasets_list.txt --output-dir=i3d_features --lr=5e-05 --l2_req=1e-05