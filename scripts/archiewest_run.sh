#!/bin/bash

 #======================================================
 #
 # Job script for running a serial job on a single core
 #
 #======================================================

 #======================================================
 # Propogate environment variables to the compute node
 #SBATCH --export=ALL
 ## Distribute processes in round-robin fashion
 ##SBATCH --distribution=cyclic
 # Run in the standard partition (queue)
 #SBATCH --partition=standard
 #
 # Specify project account
 #SBATCH --account=wrb15144
 #
 # No. of tasks required (ntasks=1 for a single-core job)
 #SBATCH --ntasks=20
 #
 # Specify (hard) runtime (HH:MM:SS)
 #SBATCH --time=72:00:00
 #
 # Job name
 #SBATCH --job-name=extracting_info
 #
 # Output file
 #SBATCH --output=slurm-%j.out
 #======================================================

 #module purge

 #example module load command (foss 2018a contains the gcc 6.4.0 toolchain & openmpi 2.12)
 #module load foss/2018a

 #======================================================
 # Prologue script to record job details
 # Do not change the line below
 #======================================================
 /opt/software/scripts/job_prologue.sh
 #------------------------------------------------------

 # Modify the line below to run your program
 python ../util/construct_dataset_mat_to_pickle_v1.py -t task1-SR -v "v2" -d "/users/wrb15144/temp_data/osfstorage-archive/" -o "/users/wrb15144/temp_data/preprocessed_zuco_1/task1/" -m "archie-west"

 #======================================================
 # Epilogue script to record job endtime and runtime
 # Do not change the line below
 #======================================================
 /opt/software/scripts/job_epilogue.sh
 #------------------------------------------------------
