#!/bin/bash

###############################################
###       SLURM script for ML Mapmaking    ####
###                                        ####
###     Tested on Uni Bonn Marvin Cluster  ####
###             v1.0 , 18.10.2024          ####
###############################################

### This script run the F&B-pipeline for a EoRSpec
### NOTE: PARENT_DIR parameter must be set in Section 4

### section 1 - SLURM params

#SBATCH --partition=intelsr_short   ###intelsr_short / intelsr_devel
### cluster specific / intelsr_short, lm_short, lm_devel
### https://wiki.hpc.uni-bonn.de/en/running_jobs

#SBATCH --ntasks-per-node 48 # tasks per node ; num of processes per node
#SBATCH --cpus-per-task 2 # number of cores per task ; threads per process

#SBATCH --nodes 1  # number of nodes

#SBATCH --job-name=det1000_n16c6nd2_fb # Job name, #ntaskstotal#corespertask#n_nodes
#f350_step222_fb_atm 

### #ntasks = #nodes x #ntasks-per-node
### #ncores_total = #ntasks x #cpus-per-task
### #ncores_total_pernode = #ntasks-per-node x #cpus-per-task

#SBATCH --mem=500G           # max mem requested, per node, baseline 400G
### Maximum requested time (days-hrs:min:sec)
#SBATCH --time 0-01:00:00 #estimated runtime max, baseline 0-15:00:00 

### #SBATCH --mail-type=ALL   # notifications for job done & fail
### #SBATCH --mail-user=adev@astro.uni-bonn.de #user email for updates
#SBATCH --output ./logs_map/%j_%x.out   # stdout file (overwrite)
#SBATCH --error ./logs_map/%j_%x.err     # stdout file (overwrite)


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

### section 3 - Job Logging
echo ""
echo "*************"
echo "Running Job..."
echo "Starting at `date`"
echo "Hostname $HOSTNAME"
echo "Job Name: $SLURM_JOB_NAME"
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
echo "Using MPI lib: $(which mpirun)"
echo "Using mpicc lib: $(which mpicc)"
echo "Using GCC lib: $(which gcc)"
echo "=========================="
echo ""

### section 4 - Set Job parameters
####  CONFIG   ####
###################
CHANNEL=340
STEP='step216'
INDIR='data_CII_tomo_ATM'
OUTDIR='outmaps_fb_ATM'
NOTES=''
###################


### section 5 - Job Run
echo "***** EXEC SCRIPT *****"
echo `date '+%F %H:%M:%S'`
echo "***********************"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

### Filter and bin
mpirun -np 48 python write_toast_maps.py -c $CHANNEL --step $STEP -in $INDIR -out $OUTDIR -n "$NOTES"

echo ""
echo "******** DONE *********"
echo `date '+%F %H:%M:%S'`
echo "***********************"