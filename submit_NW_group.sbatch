#!/bin/bash

# =============================================================================
# SLURM Batch Job Script for NW_group.py
# =============================================================================
# 
# This script runs ROI-to-network functional connectivity group analysis using
# NW_group.py. It performs group-level statistical analysis on ROI-to-network
# FC data, comparing healthy controls vs. OCD patients and performing
# longitudinal analyses.
#
# NOTE: By default, this script uses --auto-detect-atlas to automatically
# detect the atlas name from input FC files. For Power 2011 atlas, this will
# detect "power_2011" from files named power_2011_network_fc_avg.csv.
#
# USAGE:
#   sbatch submit_NW_group.sbatch
#
# EXAMPLES:
#   1. Run with default Power 2011 atlas (auto-detected):
#      sbatch submit_NW_group.sbatch
#
#   2. Run with specific Power 2011 atlas:
#      sbatch submit_NW_group.sbatch --atlas power_2011
#
#   3. Run with Schaefer 2018 atlas (400 ROIs, 7 networks):
#      sbatch submit_NW_group.sbatch --atlas schaefer_2018_400_7_2
#
#   4. Run with Schaefer 2018 atlas (1000 ROIs, 17 networks):
#      sbatch submit_NW_group.sbatch --atlas schaefer_2018_1000_17_1
#
#   5. Check job status:
#      squeue -u xxqian
#
#   6. Check job logs:
#      tail -f /scratch/xxqian/logs/NWGroup_<JOB_ID>.out
#
#   7. Monitor job progress:
#      watch -n 10 'squeue -u xxqian'
#
#   8. Check specific job details:
#      scontrol show job <JOB_ID>
#
# REQUIRED FILES:
# ---------------
# - Container image: /scratch/xxqian/repo/image/OCD.sif
# - Python script: /scratch/xxqian/repo/OCD_RO1/NW_group.py
# - Input directory: /scratch/xxqian/OCD/NW_1st (FC data from NW_1st.py)
# - Subjects CSV: /project/6079231/dliang55/R01_AOCD/metadata/shared_demographics.csv
# - Clinical CSV: /scratch/xxqian/OCD/behav/clinical.csv
#
# OUTPUT:
# -------
# - Group difference analysis results
# - Baseline FC vs symptom change regression
# - FC change vs symptom change regression
# - Statistical analysis reports
# - Results saved to: /scratch/xxqian/OCD/NW_group
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
# 1. Power 2011 Atlas Issues:
#    - Check that /scratch/xxqian/roi/power264/power264NodeNames.txt exists
#    - Verify the file has exactly 264 lines
#    - Ensure the file is readable by the container
#    - Check that input FC files follow naming: power_2011_network_fc_avg.csv
#
# 2. General Issues:
#    - Check that all required directories exist
#    - Verify CSV file formats and column names
#    - Ensure container image is accessible
#    - Check SLURM job status and logs
#    - Verify FC data files exist in input directory
#    - Check atlas naming consistency with NW_1st.py output
#
# ATLAS TYPES:
# ------------
# - power_2011: Power 2011 atlas (264 ROIs, 14 networks) - DEFAULT
#     * Automatically detected from input FC files
#     * Uses network labels from /scratch/xxqian/roi/power264/
#     * Best for: Standard functional connectivity analysis with established networks
# - schaefer_2018: Schaefer 2018 parcellation (100-1000 ROIs, 7/17 networks)
#     * Highly customizable: choose ROI count and network count
#     * Naming: schaefer_2018_{n_rois}_{yeo_networks}_{resolution_mm}
#     * Examples: schaefer_2018_400_7_2, schaefer_2018_1000_17_1
# - harvard_oxford: Harvard-Oxford cortical/subcortical atlases
#     * Anatomically defined regions
#     * Best for: Anatomical ROI analysis
# - aal: Automated Anatomical Labeling atlas (116 ROIs)
#     * Standard anatomical parcellation
#     * Best for: Traditional anatomical ROI analysis
# - talairach: Talairach atlas (1107 ROIs)
#     * High-resolution anatomical parcellation
#     * Best for: Detailed anatomical analysis
# - yeo_2011: Yeo 2011 network parcellation (7/17 networks)
#     * Pure network-based parcellation
#     * Best for: Network-focused analysis
#
# ATLAS DETECTION:
# ----------------
# The script automatically detects the atlas name from input FC files:
# - Power 2011: power_2011_network_fc_avg.csv
# - Schaefer 2018: schaefer_2018_{n_rois}_{yeo_networks}_{resolution_mm}_network_fc_avg.csv
# - Custom: custom_atlas_network_fc_avg.csv
#
# For more information, see NW_group.py --help
# =============================================================================

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
echo "NW_group.py Group Analysis Job"
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
echo "  --output_dir: /output/NW_group (maps to: $OUTPUT_DIR)"
echo "  --input_dir: /input (maps to: $INPUT_DIR)"
echo "=" * 80

# Run the Python script inside the Apptainer container
echo "Starting NW_group.py analysis..."
echo "Command: python3 /app/NW_group.py --subjects_csv /metadata/shared_demographics.csv --clinical_csv /output/behav/clinical.csv --output_dir /output/NW_group --input_dir /input --auto-detect-atlas"
apptainer exec --bind "${APPTAINER_BIND}" ${CONTAINER} python3 /app/NW_group.py \
    --subjects_csv /metadata/shared_demographics.csv \
    --clinical_csv /output/behav/clinical.csv \
    --output_dir /output/NW_group \
    --input_dir /input \
    --auto-detect-atlas

# Check exit status
exit_code=$?
if [[ $exit_code -eq 0 ]]; then
    echo "=" * 80
    echo "Analysis completed successfully at: $(date)"
    echo "Results saved to: $OUTPUT_DIR"
    echo "=" * 80
else
    echo "=" * 80
    echo "Analysis failed with exit code $exit_code at: $(date)"
    echo "Check the error logs above for details"
    echo "=" * 80
    exit $exit_code
fi