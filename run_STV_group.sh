#!/bin/bash

# =============================================================================
# Submission Script for Seed-to-Voxel Functional Connectivity Group Analysis
# =============================================================================
# This script submits the STV_group.sbatch job to SLURM
# Usage: bash run_STV_group.sh
# =============================================================================

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Path to the SBATCH file
SBATCH_FILE="${SCRIPT_DIR}/submit_STV_group.sbatch"

# Check if the SBATCH file exists
if [ ! -f "$SBATCH_FILE" ]; then
    echo "Error: SBATCH file not found at $SBATCH_FILE"
    exit 1
fi

echo "Submitting Seed-to-Voxel FC Group Analysis job..."
echo "SBATCH file: $SBATCH_FILE"

# Submit the job
sbatch "$SBATCH_FILE"

# Check if submission was successful
if [ $? -eq 0 ]; then
    echo "Job submitted successfully!"
    echo "To check job status: squeue -u \$USER"
    echo "To view logs: tail -f /scratch/xxqian/logs/STV_Group_*.out"
else
    echo "Job submission failed!"
    exit 1
fi
