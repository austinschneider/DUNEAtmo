#!/usr/bin/env bash

#SBATCH --job-name=inject
#SBATCH --output=array_%A_%a.out
#SBATCH --error=array_%A_%a.err
#SBATCH --array=1-10
#SBATCH --time=01:00:00
#SBATCH --partition=arguelles_delgado
#SBATCH --ntasks=1
#SBATCH --mem=2G
#SBATCH --cpus-per-task 1


#source /n/holyscratch01/arguelles_delgado_lab/Lab/DUNEAnalysis/programs/LIDUNE/setup.sh
source /n/home09/aschneider/programs/LIDUNE/setup_local.sh
export HDF5_USE_FILE_LOCKING=FALSE
rm /n/holyscratch01/arguelles_delgado_lab/Lab/DUNEAnalysis/store/simulation/injected/${SLURM_ARRAY_TASK_ID}.lic
rm /n/holyscratch01/arguelles_delgado_lab/Lab/DUNEAnalysis/store/simulation/injected/${SLURM_ARRAY_TASK_ID}.h5
$GOLEMSOURCEPATH/DUNEAtmo/injection/inject_muons --numu --numubar --cc --output /n/holyscratch01/arguelles_delgado_lab/Lab/DUNEAnalysis/store/simulation/injected/${SLURM_ARRAY_TASK_ID} --n-ranged 1e4 --n-volume 1e3 --seed ${SLURM_ARRAY_TASK_ID}
python -u $GOLEMSOURCEPATH/DUNEAtmo/propagation/propagate.py --config ${GOLEMSOURCEPATH}/DUNEAtmo/proposal_config/config.json --h5 /n/holyscratch01/arguelles_delgado_lab/Lab/DUNEAnalysis/store/simulation/injected/${SLURM_ARRAY_TASK_ID}.h5 --output /n/holyscratch01/arguelles_delgado_lab/Lab/DUNEAnalysis/store/simulation/propagated/${SLURM_ARRAY_TASK_ID}.json
