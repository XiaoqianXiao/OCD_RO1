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
project_dir = '/project/6079231/dliang55/R01_AOCD'
bids_dir = os.path.join(project_dir, 'derivatives/fmriprep-1.4.1')
scratch_dir = '/scratch/xxqian'
OUTPUT_DIR = os.path.join(scratch_dir, 'OCD')
WORK_DIR = os.path.join(scratch_dir, 'work_flow')
CONTAINER="/scratch/xxqian/repo/image/STV_1st_and_group_level.sif"
SUBJECTS_CSV="/project/6079231/dliang55/R01_AOCD/metadata/shared_demographics.csv"
Clinical_CSV="/scratch/xxqian/OCD/behav/clinical.csv"

# Set environment variables for FSL
export FSLOUTPUTTYPE=NIFTI_GZ
export OMP_NUM_THREADS=16


# Bind necessary directories to the container
APPTAINER_BIND="${WORK_DIR}:/scratch,${OUTPUT_DIR}:/output,${Clinical_CSV}:/clinical.csv,${SUBJECTS_CSV}:/subjects.csv:ro"

# Run the Python script inside the Apptainer container
apptainer exec --bind ${APPTAINER_BIND} ${CONTAINER} \
  python /output/STV_group.py group-level \
    --subjects_csv /subjects.csv \
    --clinical_csv /clinical.csv \
    --output_dir /output \
    --work_dir /scratch