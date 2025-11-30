#!/bin/bash

###############################################
###     SLURM script for eorspec_sims      ####
### ARRAY JOB runs all schedules in a step ####
###                                        ####
###     Tested on Uni Bonn Marvin Cluster  ####
###             v2.0 , 18.09.2025          ####
###############################################

### This script runs through all the schedule files
### launching SLURM ARRAY jobs for each schedule independently
### This launches array jobs for all schedules
### for the channel and step as listed in params.csv
### for the passed PARAM_ID
### Params must be set in params.csv
### Format: [PARAM_ID,CHANNEL,STEP,N_DETS] 
### NOTE: export PARAM_ID value must be set
### NOTE: Job name must be set in launcher.sh

### section 1 - SLURM params

#SBATCH --partition=intelsr_short   ###intelsr_short / intelsr_devel
### cluster specific / intelsr_short, lm_short, lm_devel
### https://wiki.hpc.uni-bonn.de/en/running_jobs

### #SBATCH --ntasks 8      # number of procs to start
### or
#SBATCH --ntasks-per-node 12 # tasks per node ; num of processes per node
#SBATCH --cpus-per-task 4 # number of cores per task ; threads per process

#SBATCH --nodes 1     # number of nodes
#SBATCH --array=0-138 ##0-138, Define the array size (Default set max)
### step210: 0-136
### step216: 0-137
### step222: 0-137
### step228: 0-138
### step235: 0-138
### basename -a input_files/step_schedules/step235/*.txt | wc -l

### #SBATCH --job-name=f357_step235_sim_atm
### Job name, f#chnl_step#XYZ_sim_atm/wn

### #ntasks = #nodes x #ntasks-per-node
### #ncores_total = #ntasks x #cpus-per-task
### #ncores_total_pernode = #ntasks-per-node x #cpus-per-task

#SBATCH --mem=300G                # max mem requested, per node
### Maximum requested time (days-hrs:min:sec)
#SBATCH --time 0-00:30:00 #estimated runtime max 0-02:00:00, nominal 0-00:30:00

### #SBATCH --mail-type=ALL   # notifications for job done & fail
### #SBATCH --mail-user=adev@astro.uni-bonn.de #user email for updates
#SBATCH --output=./logs_sims/%A/%x_%A_%a.out
#SBATCH --error=./logs_sims/%A/%x_%A_%a.err


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
echo "TOAST version: $(python -c 'import toast; print(toast.__version__)')"
echo "Using MPI lib: $(which mpirun)"
echo "Using mpicc lib: $(which mpicc)"
echo "Using GCC lib: $(which gcc)"
echo "=========================="
echo ""

### section 4 - Set Job parameters
###################
####  CONFIG   ####
###################
# Read CHANNEL, STEP, NDETS from params file using exported PARAM_ID
PARAMS_FILE="${PARAMS_FILE:-params.csv}"  
echo "Loading params file: $PARAMS_FILE"
if [[ ! -f "$PARAMS_FILE" ]]; then
  echo "ERROR: params file not found: $PARAMS_FILE" >&2
  exit 2
fi

# params CSV header: ARRAY_ID,CHANNEL,STEP,N_dets
# Use PARAM_ID to pick the (PARAM_ID)-th data row (0-based)
IFS=',' read -r _ CHANNEL STEP NDETS < <(tail -n +2 "$PARAMS_FILE" | sed -n "$((PARAM_ID+1))p")

### section 5 - Job Run
echo "***** EXEC SCRIPT *****"
echo `date '+%F %H:%M:%S'`
echo "***********************"
echo ""

### Collect all schedule files from input dir
SCHEDULE_DIR="input_files/step_schedules"
schedule_files_list=($(basename -a "$SCHEDULE_DIR"/"$STEP"/*.txt))
# Skip tasks beyond the number of schedules
if (( SLURM_ARRAY_TASK_ID >= ${#schedule_files_list[@]} )); then
  echo "SKIP: $STEP has only ${#schedule_files_list[@]} schedules"
  echo "Array ID $SLURM_ARRAY_TASK_ID out of range"
  echo "Param ID: $PARAM_ID"
  exit 0
fi

### Select schedule based on array index
schedule_file=${schedule_files_list[$SLURM_ARRAY_TASK_ID]}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo ""
echo "*** CONFIG ***"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running schedule file: $schedule_file"
echo "Param ID: $PARAM_ID"
echo "FPI Channel: $CHANNEL"
echo "FPI Step: $STEP"
echo "Number of dets: $NDETS"
echo "**************"
echo ""

### Simulating timestream data from schedules
mpirun -np $SLURM_NTASKS python3 sim_data_eorspec_mpi.py -s $schedule_file -c $CHANNEL --step $STEP -d $NDETS

echo ""
echo "******** DONE *********"
echo `date '+%F %H:%M:%S'`
echo "***********************"

