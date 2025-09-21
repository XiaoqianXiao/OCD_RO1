#!/bin/bash

# =============================================================================
# SLURM Job Script for Seed-to-Voxel Functional Connectivity Group Analysis
# =============================================================================

set -euo pipefail

# =============================================================================
# SLURM JOB CONFIGURATION
# =============================================================================

#SBATCH --job-name=STV_Group_Analysis
#SBATCH --output=/scratch/xxqian/logs/STV_Group_%j.out
#SBATCH --error=/scratch/xxqian/logs/STV_Group_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=def-jfeusner

# =============================================================================
# CONFIGURATION
# =============================================================================

# Container and script paths
CONTAINER="/scratch/xxqian/repo/image/OCD.sif"
SCRIPT_PATH="/scratch/xxqian/repo/OCD_RO1/STV_group.py"

# Directory configuration
PROJECT_DIR="/project/6079231/dliang55/R01_AOCD"
SCRATCH_DIR="/scratch/xxqian"
OUTPUT_DIR="/scratch/xxqian/OCD/STV_group"
WORK_DIR="/scratch/xxqian/OCD/work"
ROI_DIR="/scratch/xxqian/roi"

# Input data files
SUBJECTS_CSV="${PROJECT_DIR}/metadata/shared_demographics.csv"
CLINICAL_CSV="${SCRATCH_DIR}/OCD/behav/clinical.csv"
INPUT_DIR="${SCRATCH_DIR}/OCD"

# =============================================================================
# MAIN EXECUTION
# =============================================================================

echo "Starting Seed-to-Voxel FC Group Analysis..."
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-local}"

# Load apptainer module
module load apptainer

# Create necessary directories
mkdir -p "$OUTPUT_DIR" "$WORK_DIR" "/scratch/xxqian/logs"

# Set environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

# Run the analysis
echo "Executing STV_group.py for group analysis..."

apptainer exec \
    --bind "${SCRIPT_PATH}:/app/STV_group.py" \
    --bind "${INPUT_DIR}:/input" \
    --bind "${OUTPUT_DIR}:/output" \
    --bind "${WORK_DIR}:/work" \
    --bind "${ROI_DIR}:/roi" \
    --bind "${SUBJECTS_CSV}:/subjects.csv" \
    --bind "${CLINICAL_CSV}:/clinical.csv" \
    --bind "${SCRATCH_DIR}:/scratch" \
    --bind "${PROJECT_DIR}:/project" \
    "$CONTAINER" \
    python3 /app/STV_group.py \
        --subjects_csv /subjects.csv \
        --clinical_csv /clinical.csv \
        --output_dir /output \
        --input_dir /input \
        --verbose

# Check exit status
if [ $? -eq 0 ]; then
    echo "Seed-to-Voxel FC group analysis completed successfully"
    echo "Generated output files:"
    ls -la "$OUTPUT_DIR"/* 2>/dev/null || echo "No output files found"
else
    echo "Seed-to-Voxel FC group analysis failed"
    exit 1
fi

echo "Job completed at $(date)"