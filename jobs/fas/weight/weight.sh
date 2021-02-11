#!/usr/bin/env bash

#SBATCH --job-name=weight
#SBATCH --output=array_%A_%a.out
#SBATCH --error=array_%A_%a.err
#SBATCH --array=0-99
#SBATCH --time=01:00:00
#SBATCH --partition=arguelles_delgado
#SBATCH --ntasks=1
#SBATCH --mem=2G
#SBATCH --cpus-per-task 1


source /n/home09/aschneider/programs/LIDUNE/setup_local.sh
export HDF5_USE_FILE_LOCKING=FALSE
python ../../../weighting/calc_gen.py --propagated "/n/holyscratch01/arguelles_delgado_lab/Lab/DUNEAnalysis/store/simulation/propagated/${SLURM_ARRAY_TASK_ID}.json" --lic '/n/holyscratch01/arguelles_delgado_lab/Lab/DUNEAnalysis/store/simulation/injected/*.lic' --outdir '/n/holyscratch01/arguelles_delgado_lab/Lab/DUNEAnalysis/store/simulation/weighted/' --output weighted_${SLURM_ARRAY_TASK_ID}
