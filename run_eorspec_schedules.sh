#!/bin/bash
# export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH
export OMP_NUM_THREADS=4

echo "$(which python)"
echo "$(python --version)"

###################
####  CONFIG   ####
###################
CHANNEL=333
STEP='step210'
NDETS=50
###################

# Directory containing the schedule files
SCHEDULE_DIR="input_files/step_schedules"
SCHEDULE_FILES=("$SCHEDULE_DIR"/"$STEP"/*.txt)

# Default values for start and stop indices
START_IDX=0
STOP_IDX=${#SCHEDULE_FILES[@]}  # By default, go to the end of the array

# Read optional start and stop indices from command-line arguments
if [[ ! -z $1 ]]; then
    START_IDX=$1
fi

if [[ ! -z $2 ]]; then
    STOP_IDX=$2
fi

# EoR-Spec Channel IDs
# 350 GHz Band
# 333, 337, 340, 343, 347, 350,
# 354, 357, 361, 365, 368


# Simulation info from user
echo "Running with 244 Hz sampling rate, scanning CES 0.5 deg/s, 1 deg/s^2 acc"

# Loop through the specified range of schedule files
for ((i=START_IDX; i<STOP_IDX; i++))
do
    # Get the current schedule file name
    SCH_NAME=$(basename "${SCHEDULE_FILES[$i]}")
    echo ""
    echo "****************"
    echo ""
    echo "Running simulation for schedule file: $SCH_NAME (Index: $i)"
    echo ""
    echo "****************"
    # Run the MPI command with the current schedule file
    # (nice -n 10 bash -c "echo -e '\n****************\n' ; /usr/bin/time -v mpirun -np 16 python sim_data_primecam_mpi.py -s \"$SCH_NAME\"") 2>&1 | tee -a toast_270924_arc10.log
    mpirun -n 32 python sim_data_eorspec_mpi.py -s $SCH_NAME -c $CHANNEL --step $STEP -d $NDETS

    # -g GRP_SIZE sets the number of processes per group
    # N_GRP = N_TASKS / GRP_SIZE
    # Max N_GRP must be less than N_OBS
    # Smaller GRP_SIZE may be more efficient

    # Sleep for 5 sec before next simulation
    sleep 5
    # # Break loop if N schedules have been run
    # if (( i - START_IDX + 1 >= 3 )); then
    #     echo "Exiting Schedules loop."
    #     break
    # fi
done

### End of script ###
### Notes:
# ./run_primecam_schedules.sh START_IDX STOP_IDX
# ./run_primecam_schedules.sh 2 5
# This will run the schedule files at indices 2, 3, and 4 in the array of schedule files.
# Default START_IDX=0 and STOP_IDX=length of the array of schedule files.
#
# Run as:
# /usr/bin/time -v ./run_primecam_schedules.sh 0 1 2>&1 | tee -a logs/toast_270924_arc10.log
