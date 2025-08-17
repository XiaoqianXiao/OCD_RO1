#!/bin/bash
#SBATCH --job-name=network_fc_group
#SBATCH --output=/scratch/xxqian/logs/NWGroup_%A_%a.out
#SBATCH --error=/scratch/xxqian/logs/NWGroup_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G

# Load Apptainer module
module load apptainer

# Path to Apptainer container
CONTAINER="/scratch/xxqian/repo/image/OCD.sif"

# Set environment variables
export OMP_NUM_THREADS=8

# Define directories and files
PROJECT_DIR="/project/6079231/dliang55/R01_AOCD"
SCRATCH_DIR="/scratch/xxqian"
OUTPUT_DIR="${SCRATCH_DIR}/OCD/NW_group"
SUBJECTS_CSV="${PROJECT_DIR}/metadata/shared_demographics.csv"
CLINICAL_CSV="${SCRATCH_DIR}/OCD/behav/clinical.csv"
INPUT_DIR="${SCRATCH_DIR}/OCD"

# Bind directories
APPTAINER_BIND="/scratch/xxqian/repo/OCD_RO1:/app, ${SCRATCH_DIR}/OCD:/output, ${PROJECT_DIR}/metadata:/metadata, ${INPUT_DIR}:/input"

# Verify bind paths
for path in "${SCRATCH_DIR}/OCD" "${PROJECT_DIR}/metadata" "${INPUT_DIR}"; do
    if [ ! -e "$path" ]; then
        echo "Error: Bind path does not exist: $path"
        exit 1
    fi
done

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run the Python script inside the Apptainer container
apptainer exec --bind "${APPTAINER_BIND}" ${CONTAINER} python3 /app/NW_group.py \
    --subjects_csv /metadata/shared_demographics.csv \
    --clinical_csv /output/behav/clinical.csv \
    --output_dir /output/NW_group \
    --input_dir /input \
    --auto-detect-atlas