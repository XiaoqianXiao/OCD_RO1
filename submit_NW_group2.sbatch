#!/bin/bash

# =============================================================================
# SLURM Batch Job Script for NW_group2.py
# =============================================================================
# 
# This script runs network-level pairwise functional connectivity group analysis using
# NW_group2.py. It performs group-level statistical analysis on network-to-network
# FC data, comparing healthy controls vs. OCD patients and performing
# longitudinal analyses with condition confounders.
#
# USAGE:
#   sbatch submit_NW_group2.sbatch
#
# EXAMPLES:
#   1. Run with default settings:
#      sbatch submit_NW_group2.sbatch
#
#   2. Check job status:
#      squeue -u xxqian
#
#   3. Check job logs:
#      tail -f /scratch/xxqian/logs/NWGroup2_<JOB_ID>.out
#
#   4. Monitor job progress:
#      watch -n 10 'squeue -u xxqian'
#
#   5. Check specific job details:
#      scontrol show job <JOB_ID>
#
# REQUIRED FILES:
# ---------------
# - Container image: /scratch/xxqian/repo/image/OCD.sif
# - Python script: /scratch/xxqian/repo/OCD_RO1/NW_group2.py
# - Input directory: /scratch/xxqian/OCD/NW_1st (FC data from NW_1st.py)
# - Subjects CSV: /project/6079231/dliang55/R01_AOCD/metadata/shared_demographics.csv
# - Clinical CSV: /scratch/xxqian/OCD/behav/clinical.csv
#
# OUTPUT:
# -------
# - Group difference analysis results (HC vs OCD)
# - Baseline FC vs symptom change regression (with condition confounder)
# - FC change vs symptom change regression (with condition confounder)
# - Condition-based analysis (Treat vs Wait differences)
# - Statistical analysis reports
# - Results saved to: /scratch/xxqian/OCD/NW_group2
#
# SLURM RESOURCES:
# ----------------
# - Time: 01:00:00 (1 hour)
# - Memory: 8G
# - CPUs: 16
# - Nodes: 1
#
# TROUBLESHOOTING:
# ----------------
# 1. Check that all required directories exist
# 2. Verify CSV file formats and column names
# 3. Ensure container image is accessible
# 4. Check SLURM job status and logs
# 5. Verify FC data files exist in input directory
# 6. Check atlas naming consistency with NW_1st.py output
# 7. Ensure network-level FC files exist (not ROI-level)
#
# For more information, see NW_group2.py --help
# =============================================================================

#SBATCH --job-name=network_fc_group2
#SBATCH --output=/scratch/xxqian/logs/NWGroup2_%A_%a.out
#SBATCH --error=/scratch/xxqian/logs/NWGroup2_%A_%a.err
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
OUTPUT_DIR="${SCRATCH_DIR}/OCD/NW_group2"
SUBJECTS_CSV="${PROJECT_DIR}/metadata/shared_demographics.csv"
CLINICAL_CSV="${SCRATCH_DIR}/OCD/behav/clinical.csv"
INPUT_DIR="${SCRATCH_DIR}/OCD/NW_1st"

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

# =============================================================================
# USAGE INFORMATION
# =============================================================================

echo "=" * 80
echo "NW_group2.py Network-Level FC Group Analysis Job"
echo "=" * 80
echo "Job Configuration:"
echo "  Container: $CONTAINER"
echo "  Input Directory: $INPUT_DIR"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Subjects CSV: $SUBJECTS_CSV"
echo "  Clinical CSV: $CLINICAL_CSV"
echo "  Bind Paths: $APPTAINER_BIND"
echo ""
echo "Container Paths:"
echo "  --subjects_csv: /metadata/shared_demographics.csv (maps to: $SUBJECTS_CSV)"
echo "  --clinical_csv: /output/behav/clinical.csv (maps to: $CLINICAL_CSV)"
echo "  --output_dir: /output/NW_group2 (maps to: $OUTPUT_DIR)"
echo "  --input_dir: /input (maps to: $INPUT_DIR)"
echo "=" * 80

# Run the Python script inside the Apptainer container
echo "Starting NW_group2.py analysis..."
echo "Command: python3 /app/NW_group2.py --subjects_csv /metadata/shared_demographics.csv --clinical_csv /output/behav/clinical.csv --output_dir /output/NW_group2 --input_dir /input --auto-detect-atlas"
apptainer exec --bind "${APPTAINER_BIND}" ${CONTAINER} python3 /app/NW_group2.py \
    --subjects_csv /metadata/shared_demographics.csv \
    --clinical_csv /output/behav/clinical.csv \
    --output_dir /output/NW_group2 \
    --input_dir /input \
    --auto-detect-atlas

# Check exit status
exit_code=$?
if [[ $exit_code -eq 0 ]]; then
    echo "=" * 80
    echo "NW_group2.py Network-Level FC Analysis completed successfully at: $(date)"
    echo "Results saved to: $OUTPUT_DIR"
    echo "=" * 80
else
    echo "=" * 80
    echo "NW_group2.py Network-Level FC Analysis failed with exit code $exit_code at: $(date)"
    echo "Check the error logs above for details"
    echo "=" * 80
    exit $exit_code
fi