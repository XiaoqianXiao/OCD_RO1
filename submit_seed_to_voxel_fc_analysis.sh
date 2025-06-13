#!/bin/bash
#SBATCH --job-name=seed_to_voxel_analysis
#SBATCH --account=def-jfeusner
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=STV%A_%a.out
#SBATCH --error=STV_%A_%a.err
#SBATCH --array=0-%N


module load apptainer

# Get subject from array
SUBJECTS=($(find /project/6079231/dliang55/R01_AOCD/derivatives/fmriprep-1.4.1 -maxdepth 1 -type d -name "sub-*" -exec basename {} \; | sed 's/sub-//g' | sort))
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

# Path to Apptainer container
CONTAINER="/scratch/xxqian/repo/image/STV_1st_level_1.0.sif"

# Run the Apptainer container for one subject
apptainer run \
    --bind /project:/project,/scratch:/scratch \
    $CONTAINER --subject $SUBJECT