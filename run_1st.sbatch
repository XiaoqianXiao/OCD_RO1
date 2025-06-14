#!/bin/bash
#SBATCH --job-name=seed_to_voxel_analysis
#SBATCH --account=def-jfeusner
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/xxqian/logs/STV_%A_%a.out
#SBATCH --error=/scratch/xxqian/logs/STV_%A_%a.err
#SBATCH --array=0-%N

# Load Apptainer module
module load apptainer

# Get subject from array
SUBJECTS=($(find /project/6079231/dliang55/R01_AOCD/derivatives/fmriprep-1.4.1 -maxdepth 1 -type d -name "sub-*" -exec basename {} \; | sed 's/sub-//g' | sort))
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

# Path to Apptainer container
CONTAINER="/scratch/xxqian/repo/image/STV_1st_and_group_level.sif"

# Set environment variables
export FSLOUTPUTTYPE=NIFTI_GZ
export APPTAINERENV_TEMPLATEFLOW_HOME=/templateflow
export OMP_NUM_THREADS=8

# Bind directories
APPTAINER_BIND="/project/6079231/dliang55/R01_AOCD:/project/6079231/dliang55/R01_AOCD,/scratch/xxqian/work_flow:/scratch/xxqian/work_flow,/scratch/xxqian/OCD:/scratch/xxqian/OCD,/scratch/xxqian/roi:/scratch/xxqian/roi"

# Run the Apptainer container for one subject
if [ -n "$SUBJECT" ]; then
  apptainer run --bind ${APPTAINER_BIND} ${CONTAINER} first-level --subject sub-${SUBJECT}
else
  echo "Error: No subject found for array index $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi