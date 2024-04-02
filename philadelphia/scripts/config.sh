#!/bin/bash

# Set sbatch parameters
TIME="11:00:00"
PARTITION="broadwl"
MEMORY="5G"
CPUS=1
CORES=28

# Set the maximum number of jobs allowed
MAXJOBS=98

# Set the QUEUE file path
QUEUE="program_calls.txt"
