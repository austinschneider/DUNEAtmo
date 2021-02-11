#!/usr/bin/env bash

#SBATCH --job-name=inject
#SBATCH --output=array_%A_%a.out
#SBATCH --error=array_%A_%a.err
#SBATCH --array=1-10
#SBATCH --time=08:00:00
#SBATCH --partition=arguelles_delgado
#SBATCH --ntasks=1
#SBATCH --mem=2G
#SBATCH --cpus-per-task 1


#source /n/holyscratch01/arguelles_delgado_lab/Lab/DUNEAnalysis/programs/LIDUNE/setup.sh
source /n/home09/aschneider/programs/LIDUNE/setup_local.sh
export HDF5_USE_FILE_LOCKING=FALSE
python -u $GOLEMSOURCEPATH/DUNEAtmo/likelihood/flux_pre_gen/gen_lv.py --chunks 200 --chunk-number ${SLURM_ARRAY_TASK_ID}
