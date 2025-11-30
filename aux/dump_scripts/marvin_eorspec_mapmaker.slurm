#!/bin/bash

###############################################
###      SLURM script for F&B Mapmaking    ####
###                                        ####
###     Tested on Uni Bonn Marvin Cluster  ####
###             v1.1 , 12.05.2025          ####
###############################################

### This script run the F&B-pipeline for a EoRSpec
### NOTE: CONFIG parameters must be set in Section 4

### section 1 - SLURM params

#SBATCH --partition=intelsr_short   ###intelsr_short / intelsr_devel
### cluster specific / intelsr_short, lm_short, lm_devel
### https://wiki.hpc.uni-bonn.de/en/running_jobs

#SBATCH --ntasks-per-node 48 # tasks per node ; num of processes per node
#SBATCH --cpus-per-task 1 # number of cores per task ; threads per process

#SBATCH --nodes 1  # number of nodes

#SBATCH --job-name=f350_step228_fbv2_atm
### Job name, f#chnl_step#XYZ_fb(v)_atm/wn

### #ntasks = #nodes x #ntasks-per-node
### #ncores_total = #ntasks x #cpus-per-task
### #ncores_total_pernode = #ntasks-per-node x #cpus-per-task

#SBATCH --mem=550G           # max mem requested, per node, baseline 550G
### Maximum requested time (days-hrs:min:sec)
#SBATCH --time 0-01:00:00    #estimated runtime max, baseline 0-01:00:00 

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
### conda activate toast3
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
echo "TOAST version: $(python -c 'import toast; print(toast.__version__)' 2>/dev/null || echo '!! toast not found !!')"
echo "Using MPI lib: $(which mpirun)"
echo "Using mpicc lib: $(which mpicc)"
echo "Using GCC lib: $(which gcc)"
echo "=========================="
echo ""

### section 4 - Set Job parameters
####  CONFIG   ####
###################
CHANNEL=350
STEP='step228'
INDIR='data_CII_tomo_ATM'
OUTDIR='outmaps_fbv2_ATM'
GRP_SIZE=2 #4
NOTES='2 groups; Steps: Poly Detrend, AZ Template corr, No CM removal, PCA 3 comp., jy/sr units'

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
set -e

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

### Filter and bin
mpirun -np $SLURM_NTASKS python write_toast_maps.py -c $CHANNEL --step $STEP -in $INDIR -out $OUTDIR -g $GRP_SIZE -n "$NOTES"

echo ""
echo "******** DONE *********"
echo `date '+%F %H:%M:%S'`
echo "***********************"