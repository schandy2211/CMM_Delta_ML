#!/bin/bash
#SBATCH --job-name=test2
#SBATCH --account=co_armada2
#SBATCH --partition=savio3_gpu
#SBATCH --time=20:00:00
#SBATCH --qos=savio_lowprio
#SBATCH --requeue
#SBATCH --nodes=1
python main_eda_dipole.py