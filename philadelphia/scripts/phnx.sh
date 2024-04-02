#!/bin/bash

# Read the configuration file
source config.sh

# Check if there is a job running with the job name PXMASTER
PXMASTER_RUNNING=$(squeue -u $USER --name=PXMASTER | wc -l)

# If PXMASTER is running, exit the script
if [ $PXMASTER_RUNNING -gt 2 ]; then
    echo "PXMASTER is already running. Exiting."
    exit 1
fi

# Create the output directory with a timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTDIR="./phnxrun_${TIMESTAMP}"
mkdir -p $OUTDIR

# Store the original number of tasks and the start time if this is the first run
if [ ! -f "${OUTDIR}/phnx_info.txt" ]; then
    ORIG_TASKS=$(wc -l < $QUEUE)
    START_TIME=$(date +%s)
    echo "${ORIG_TASKS}:${START_TIME}" > "${OUTDIR}/phnx_info.txt"
fi

# Get the number of running jobs for the user
RUNNINGJOBS=$(squeue -u $USER | wc -l)

# Adjust for the header line in squeue output
RUNNINGJOBS=$((RUNNINGJOBS - 1))

# Calculate the number of jobs to launch
JOBSTOLAUNCH=$((MAXJOBS - RUNNINGJOBS))

# If there are jobs to launch, read the first MAXJOBS lines from QUEUE
if [ $JOBSTOLAUNCH -gt 0 ]; then
    JOBS=$(head -n $JOBSTOLAUNCH $QUEUE)

    # Launch the jobs using sbatch with the specified settings
    COUNTER=1
    echo "$JOBS" | while read -r JOB; do
        JOBNAME="PHX${COUNTER}"
        OUTFILE="${OUTDIR}/${JOBNAME}.out"
        ERRFILE="${OUTDIR}/${JOBNAME}.err"
        sbatch --job-name=$JOBNAME --output=$OUTFILE --error=$ERRFILE --time=$TIME --partition=$PARTITION --mem=$MEMORY --cpus-per-task=$CPUS --ntasks=$CORES --wrap="date; $JOB; date"
        COUNTER=$((COUNTER + 1))
    done

    # Remove the launched jobs from the QUEUE file
    sed -i "1,${JOBSTOLAUNCH}d" $QUEUE
fi

# Wait 10 minutes and launch a copy of itself with jobname PXMASTER only if there are still tasks in the queue
CURR_TASKS=$(wc -l < $QUEUE)
if [ $CURR_TASKS -gt 0 ]; then
    PXMASTER_OUTFILE="${OUTDIR}/PXMASTER.out"
    PXMASTER_ERRFILE="${OUTDIR}/PXMASTER.err"
    sleep 10m
    sbatch --job-name=PXMASTER --output=$PXMASTER_OUTFILE --error=$PXMASTER_ERRFILE --time=15:00 --partition=$PARTITION --mem=1G --cpus-per-task=1 --ntasks=1 --wrap="./$0"
fi
