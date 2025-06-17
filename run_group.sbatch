#!/bin/bash
#SBATCH --job-name=mixed_anova_pcc
#SBATCH --output=STVGroup%A_%a.out
#SBATCH --error=STVGroup_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Define paths
PROJECT_DIR="/project/6079231/dliang55/R01_AOCD"
BIDS_DIR="${PROJECT_DIR}/derivatives/fmriprep-1.4.1"
SCRATCH_DIR="/scratch/xxqian"
OUTPUT_DIR="${SCRATCH_DIR}/OCD"
WORK_DIR="${SCRATCH_DIR}/work_flow"
CONTAINER="${SCRATCH_DIR}/repo/image/STV_1st_and_group_level.sif"
SUBJECTS_CSV="${PROJECT_DIR}/metadata/shared_demographics.csv"
CLINICAL_CSV="${SCRATCH_DIR}/OCD/behav/clinical.csv"

# Set environment variables for FSL
export FSLOUTPUTTYPE=NIFTI_GZ
export OMP_NUM_THREADS=16


# Bind necessary directories to the container
APPTAINER_BIND="${WORK_DIR}:/scratch,${OUTPUT_DIR}:/output,${CLINICAL_CSV}:/clinical.csv,${SUBJECTS_CSV}:/subjects.csv:ro"

# Run the Python script inside the Apptainer container
apptainer exec --bind ${APPTAINER_BIND} ${CONTAINER} \
  python3 /app/STV_group.py group-level \
    --subjects_csv /subjects.csv \
    --clinical_csv /clinical.csv \
    --output_dir /output \
    --work_dir /scratch