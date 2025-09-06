#!/bin/bash

# SLURM job submission script for ROI-to-Network FC Group Analysis
# This script analyzes ROI-to-Network similarity at the group level
# using outputs from submit_NW_1st.sh
#
# USAGE: sbatch submit_NW_group_roi_network.sh <atlas_name> [atlas_params]
#
# EXAMPLES FOR ROI-TO-NETWORK SIMILARITY ANALYSIS:
#
# 1. Schaefer 2018 Atlas (Network-based):
#    sbatch submit_NW_group_roi_network.sh schaefer_2018
#    sbatch submit_NW_group_roi_network.sh schaefer_2018 '{"n_rois": 100, "yeo_networks": 7, "resolution_mm": 2}'
#    sbatch submit_NW_group_roi_network.sh schaefer_2018 '{"n_rois": 400, "yeo_networks": 17, "resolution_mm": 1}'
#    sbatch submit_NW_group_roi_network.sh schaefer_2018 '{"n_rois": 1000, "yeo_networks": 7, "resolution_mm": 2}'
#
# 2. Power 2011 Atlas (Network-based):
#    sbatch submit_NW_group_roi_network.sh power_2011
#
# 3. Pauli 2017 Atlas (Network-based):
#    sbatch submit_NW_group_roi_network.sh pauli_2017
#
# 4. Yeo 2011 Atlas (Network-based, Network-level only):
#    sbatch submit_NW_group_roi_network.sh yeo_2011
#
# NOTE: Only network-based atlases support ROI-to-Network similarity analysis.
# Non-network atlases (harvard_oxford, aal, talairach) are not suitable for this analysis.

ATLAS=$1
ATLAS_PARAMS=${2:-'{}'}  # Default to empty JSON if not provided

# SLURM Configuration
#SBATCH --job-name=roi_network_fc_group_${ATLAS}
#SBATCH --output=/scratch/xxqian/logs/ROINetworkGroup_${ATLAS}_%A_%a.out
#SBATCH --error=/scratch/xxqian/logs/ROINetworkGroup_${ATLAS}_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --account=def-jfeusner

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
APPTAINER_BIND="/scratch/xxqian/repo/OCD_RO1/NW_group.py:/app/NW_group.py,${SCRATCH_DIR}/OCD:/output,${PROJECT_DIR}/metadata:/metadata,${CLINICAL_CSV}:/clinical.csv,${SUBJECTS_CSV}:/subjects.csv,${INPUT_DIR}:/input"

# Verify bind paths
for path in "${SCRATCH_DIR}/OCD" "${PROJECT_DIR}/metadata" "${CLINICAL_CSV}" "${SUBJECTS_CSV}" "${INPUT_DIR}"; do
    if [ ! -e "$path" ]; then
        echo "Error: Bind path does not exist: $path"
        exit 1
    fi
done

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Print job information
echo "=========================================="
echo "ROI-to-Network FC Group Analysis"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Atlas: $ATLAS"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# Run the Python script inside the Apptainer container
echo "Starting ROI-to-Network FC group analysis..."
apptainer exec --bind "${APPTAINER_BIND}" ${CONTAINER} python3 /app/NW_group.py \
    --subjects_csv /subjects.csv \
    --clinical_csv /clinical.csv \
    --output_dir /output \
    --input_dir /input \
    --atlas_name ${ATLAS} \
    --atlas_params '${ATLAS_PARAMS}'

