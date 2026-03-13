#!/bin/bash

### For individual runs/ debugging:
###SBATCH_SCRIPT="marvin_eorspec_schedules.slurm"
###SBATCH_SCRIPT="marvin_eorspec_mapmaker.slurm"

#######################################################

### For the schedules simulations:
### Job name, param_sim_atm/wn/nosig/sigonly
# PARAM_ID="${PARAM_ID:?PARAM_ID not set}";
# SBATCH_SCRIPT="marvin_eorspec_schedules_param.slurm"; JOB_NAME="param_${PARAM_ID}_sim_sigonly"
# JOB_SUBMIT_OUTPUT=$(sbatch -J "$JOB_NAME" --export=ALL,PARAM_ID="$PARAM_ID" "$SBATCH_SCRIPT")

### Submit the Slurm job and capture the job ID and Param ID
### run as: for i in $(seq 0 21); do PARAM_ID=$i ./slurm_job_launcher.sh; done

#######################################################

### For the map-maker:
### Job name, fb(v)_atm/nosig/sigonly
SBATCH_SCRIPT="marvin_eorspec_mapmaker_param.slurm"; JOB_NAME="fb_sigonly"
JOB_SUBMIT_OUTPUT=$(sbatch -J "$JOB_NAME" "$SBATCH_SCRIPT")

### run as: ./slurm_job_launcher.sh

#######################################################

JOB_ID=$(echo "$JOB_SUBMIT_OUTPUT" | awk '{print $4}')

# Display job ID and name on screen
echo "Submitted job ID: $JOB_ID"

### Start a background process to wait for the job to finish and append resource usage
(
    # Wait for the job to finish
    while squeue -j $JOB_ID > /dev/null 2>&1; do
        sleep 60
    done 

    # Define the log file name based on the Job ID and Job Name
    LOG_FILE="./logs/${JOB_ID}.res"

    # Append resource usage to the dynamically named log file
    sacct -j $JOB_ID --format=JobID,JobName,Partition,AllocCPUS,Elapsed,State,ExitCode,NodeList,MaxRSS,MaxVMSize,TotalCPU,CPUTime,ReqMem,AveRSS,AveVMSize \
        >> "$LOG_FILE"
) &

# Detach from the login shell after submitting
exit 0

