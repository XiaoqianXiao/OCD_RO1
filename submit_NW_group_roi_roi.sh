#!/bin/bash

# SLURM job submission script for ROI-to-ROI FC Group Analysis
# This script analyzes ROI-to-ROI similarity at the group level
# using outputs from submit_NW_1st.sh
#
# USAGE: sbatch submit_NW_group_roi_roi.sh <atlas_name> [atlas_params]
#
# EXAMPLES FOR ROI-to-ROI SIMILARITY ANALYSIS:
#
# 
#
# 2. Power 2011 Atlas:
#    sbatch submit_NW_group_roi_roi.sh power_2011
#
#
# 4. Yeo 2011 Atlas (roi-based, roi-level only):
#    sbatch submit_NW_group_roi_roi.sh yeo_2011
#
# 5. Custom Atlas Names (if using pre-configured atlas names):
#    sbatch submit_NW_group_roi_roi.sh schaefer_2018_100_7_2
#    sbatch submit_NW_group_roi_roi.sh schaefer_2018_400_17_1
#
# NOTE: Only roi-based atlases support ROI-to-ROI similarity analysis.
# Non-roi atlases (harvard_oxford, aal, talairach) are not suitable for this analysis.

ATLAS=$1
ATLAS_PARAMS=${2:-'{}'}  # Default to empty JSON if not provided

# SLURM Configuration
#SBATCH --job-name=roi_roi_fc_group_${ATLAS}
#SBATCH --output=/scratch/xxqian/logs/ROIROIGroup_${ATLAS}_%A.out
#SBATCH --error=/scratch/xxqian/logs/ROIROIGroup_${ATLAS}_%A.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
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
OUTPUT_DIR="${SCRATCH_DIR}/OCD/NW_group_roi_roi"
SUBJECTS_CSV="${PROJECT_DIR}/metadata/shared_demographics.csv"
CLINICAL_CSV="${SCRATCH_DIR}/OCD/behav/clinical.csv"
INPUT_DIR="${SCRATCH_DIR}/OCD/NW_1st"



# Bind directories
APPTAINER_BIND="/scratch/xxqian/repo/OCD_RO1/NW_group3.py:/app/NW_group3.py,${PROJECT_DIR}/metadata:/metadata,${CLINICAL_CSV}:/clinical.csv,${SUBJECTS_CSV}:/subjects.csv,${INPUT_DIR}:/input,${OUTPUT_DIR}:/output"

# Verify bind paths
for path in "${SCRATCH_DIR}/OCD" "${PROJECT_DIR}/metadata" "${CLINICAL_CSV}" "${SUBJECTS_CSV}" "${INPUT_DIR}" "${OUTPUT_DIR}"; do
    if [ ! -e "$path" ]; then
        echo "Error: Bind path does not exist: $path"
        exit 1
    fi
done

# Create output directory
mkdir -p "${OUTPUT_DIR}"


# Run the Python script inside the Apptainer container
echo "Starting ROI-to-ROI FC group analysis..."
apptainer exec --bind "${APPTAINER_BIND}" ${CONTAINER} python3 /app/NW_group3.py \
    --subjects_csv /subjects.csv \
    --clinical_csv /clinical.csv \
    --output_dir /output \
    --input_dir /input \
    --atlas_name "${ATLAS}"

