#!/bin/bash

###############################################
###      SLURM script for F&B Mapmaking    ####
###                                        ####
###     Tested on Uni Bonn Marvin Cluster  ####
###             v2.0 , 18.09.2025          ####
###############################################

### This script run the F&B-pipeline for a EoRSpec
### This launches an array job for all the channels
### and steps as listed in params.csv
### Params must be set in params.csv
### Format: [PARAM_ID,CHANNEL,STEP,N_DETS] 
### NOTE: CONFIG parameters must be set in Section 4
### NOTE: Job name must be set in launcher.sh

### section 1 - SLURM params

#SBATCH --partition=intelsr_short   ###intelsr_short / intelsr_devel
### cluster specific / intelsr_short, lm_short, lm_devel
### https://wiki.hpc.uni-bonn.de/en/running_jobs

#SBATCH --ntasks-per-node 48 # tasks per node ; num of processes per node
#SBATCH --cpus-per-task 1 # number of cores per task ; threads per process

#SBATCH --nodes 1  # number of nodes

#SBATCH --array=0-21 #0-21

### #ntasks = #nodes x #ntasks-per-node
### #ncores_total = #ntasks x #cpus-per-task
### #ncores_total_pernode = #ntasks-per-node x #cpus-per-task

#SBATCH --mem=550G           # max mem requested, per node, baseline 550G
### Maximum requested time (days-hrs:min:sec)
#SBATCH --time 0-01:30:00    #estimated runtime max, baseline 0-01:30:00 

### #SBATCH --mail-type=ALL   # notifications for job done & fail
### #SBATCH --mail-user=adev@astro.uni-bonn.de #user email for updates

#SBATCH --output=./logs_map/%A/%A_%a_%x.out
#SBATCH --error=./logs_map/%A/%A_%a_%x.err


### section 2 - Runtime env set-up
module purge
### load all modules needed for Job
module load mpi4py
module load HDF5/1.14.5-gompi-2024a
### initiate conda
### source /opt/software/easybuild-INTEL/software/Anaconda3/2022.05/etc/profile.d/conda.sh
source /opt/software/easybuild-INTEL/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda deactivate
conda activate latest-toast
### conda activate toast3
echo "===== Loaded Modules ====="
module list 2>&1
echo "=========================="
set -euo pipefail

### section 3 - Job Logging
echo ""
echo "*************"
echo "Running Job..."
echo "Starting at `date`"
echo "Hostname $HOSTNAME"
echo "Job Name: $SLURM_JOB_NAME"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Slurm Ntasks: $SLURM_NTASKS"
echo "Number of Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "Number of CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Cores per Node: $SLURM_CPUS_ON_NODE"
echo "Total Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Current working directory is `pwd`"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Using mpi4py: $(python -c 'import mpi4py; print(mpi4py.__file__)' 2>/dev/null || echo '!! mpi4py not found !!')"
echo "TOAST version: $(python -c 'import toast; print(toast.__version__)' 2>/dev/null || echo '!! toast not found !!')"
echo "Using MPI lib: $(which mpirun)"
echo "Using mpicc lib: $(which mpicc)"
echo "Using GCC lib: $(which gcc)"
echo "=========================="
echo ""

### section 4 - Set Job parameters
####  CONFIG   ####
###################
# params.csv: array_id,channel,step
# Params file (relative to submission dir)
PARAMS_FILE="${PARAMS_FILE:-params.csv}"
echo "Loading params file: $PARAMS_FILE"
if [[ ! -f "$PARAMS_FILE" ]]; then
  echo "ERROR: params file not found: $PARAMS_FILE" >&2
  exit 2
fi

# Select the current array task’s row from PARAMS_FILE (skip header, then pick Nth line)
# Split the row by commas and assign 2nd → CHANNEL, 3rd → STEP (ignore 1st column)
# N_DETS is set but unused
IFS=',' read -r _ CHANNEL STEP N_DETS < <(tail -n +2 "$PARAMS_FILE" | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")

INDIR='data_CII_tomo_NOSIG'
OUTDIR='outmaps_fbv2_NOSIG'
GRP_SIZE=2 #4
NOTES='no input sig, wn + atm noise; 2 groups; Steps: Poly Detrend, AZ Template corr, No CM removal, PCA 3 comp., jy/sr units'

echo "***** CONFIG *****"
echo "CHANNEL: $CHANNEL"
echo "STEP: $STEP"
echo "INDIR: $INDIR"
echo "OUTDIR: $OUTDIR"
echo "GRP_SIZE: $GRP_SIZE"
echo "NOTES: $NOTES"
echo ""
###################


### section 5 - Job Run
echo "***** EXEC SCRIPT *****"
echo `date '+%F %H:%M:%S'`
echo "***********************"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

### Filter and bin
mpirun -np $SLURM_NTASKS python write_toast_maps.py -c $CHANNEL --step $STEP -in $INDIR -out $OUTDIR -g $GRP_SIZE -n "$NOTES"

echo ""
echo "******** DONE *********"
echo `date '+%F %H:%M:%S'`
echo "***********************"