#!/bin/bash

# =============================================================================
# Dynamic SLURM Job Submission Script for NW_group3.py
# =============================================================================
# 
# This script dynamically generates and submits SLURM jobs for ROI-to-ROI
# functional connectivity group analysis using NW_group3.py. It creates
# atlas-specific SLURM scripts and job names.
#
# NOTE: This script works with ALL atlases that generate ROI-to-ROI connectivity
# data, including both network-based atlases (Power 2011, Schaefer 2018, YEO 2011)
# and anatomical atlases (Harvard-Oxford, AAL, Talairach).
#
# USAGE:
#   bash submit_NW_group3.sh [OPTIONS]
#
# EXAMPLES:
#   1. lea
#      bash submit_NW_group3.sh --atlas power_2011
#
#   2. Run with Schaefer 2018 atlas (network-based):
#      bash submit_NW_group3.sh --atlas schaefer_2018 --atlas-params '{"n_rois": 400, "yeo_networks": 7}'
#
#   3. Run with YEO 2011 atlas (network-based):
#      bash submit_NW_group3.sh --atlas yeo_2011 --atlas-params '{"n_networks": 7, "thickness": "thick"}'
#
#   4. Run with Harvard-Oxford atlas (anatomical):
#      bash submit_NW_group3.sh --atlas harvard_oxford_cort-maxprob-thr25-2mm
#
#   5. Run with AAL atlas (anatomical):
#      bash submit_NW_group3.sh --atlas aal
#
#   6. Run with Talairach atlas (anatomical):
#      bash submit_NW_group3.sh --atlas talairach
#
#   7. Auto-detect atlas:
#      bash submit_NW_group3.sh
#
# OPTIONS:
#   --atlas ATLAS            Atlas name (default: auto-detect)
#   --atlas-params PARAMS   JSON string of atlas parameters
#   --subjects-csv FILE     Subjects CSV file (default: group.csv)
#   --clinical-csv FILE     Clinical CSV file (default: clinical.csv)
#   --input-dir DIR         Input directory (default: /project/6079231/dliang55/R01_AOCD)
#   --output-dir DIR        Output directory (default: /project/6079231/dliang55/R01_AOCD/NW_group3)
#   --min-subjects N        Minimum subjects per group (default: 2)
#   --significance-threshold P  Significance threshold (default: 0.05)
#   --no-fdr                Disable FDR correction
#   --verbose               Enable verbose logging
#   --help                  Show this usage information
# =============================================================================

# Default values
ATLAS=""
ATLAS_PARAMS=""
SUBJECTS_CSV="group.csv"
CLINICAL_CSV="clinical.csv"
INPUT_DIR="/project/6079231/dliang55/R01_AOCD"
OUTPUT_DIR="/project/6079231/dliang55/R01_AOCD/NW_group3"
MIN_SUBJECTS=2
SIGNIFICANCE_THRESHOLD=0.05
NO_FDR=""
VERBOSE=""
AUTO_DETECT_ATLAS="--auto-detect-atlas"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --atlas)
            ATLAS="$2"
            AUTO_DETECT_ATLAS=""
            shift 2
            ;;
        --atlas-params)
            ATLAS_PARAMS="$2"
            shift 2
            ;;
        --subjects-csv)
            SUBJECTS_CSV="$2"
            shift 2
            ;;
        --clinical-csv)
            CLINICAL_CSV="$2"
            shift 2
            ;;
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --min-subjects)
            MIN_SUBJECTS="$2"
            shift 2
            ;;
        --significance-threshold)
            SIGNIFICANCE_THRESHOLD="$2"
            shift 2
            ;;
        --no-fdr)
            NO_FDR="--no-fdr"
            shift
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "  --atlas ATLAS            Atlas name (default: auto-detect)"
            echo "  --atlas-params PARAMS   JSON string of atlas parameters"
            echo "  --subjects-csv FILE     Subjects CSV file (default: group.csv)"
            echo "  --clinical-csv FILE     Clinical CSV file (default: clinical.csv)"
            echo "  --input-dir DIR         Input directory (default: /project/6079231/dliang55/R01_AOCD)"
            echo "  --output-dir DIR        Output directory (default: /project/6079231/dliang55/R01_AOCD/NW_group3)"
            echo "  --min-subjects N        Minimum subjects per group (default: 2)"
            echo "  --significance-threshold P  Significance threshold (default: 0.05)"
            echo "  --no-fdr                Disable FDR correction"
            echo "  --verbose               Enable verbose logging"
            echo "  --help                  Show this usage information"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Determine atlas name for job naming
if [ -n "$ATLAS" ]; then
    ATLAS_NAME="$ATLAS"
else
    ATLAS_NAME="auto"
fi

# Create job name with atlas information
JOB_NAME="roiroi_fc_group3_${ATLAS_NAME}"

# Create temporary directory for SLURM scripts
TEMP_JOB_DIR="/tmp/slurm_jobs_$$"
mkdir -p "$TEMP_JOB_DIR"

# Create SLURM job script
JOB_SCRIPT="$TEMP_JOB_DIR/${JOB_NAME}.slurm"

echo "Creating SLURM job script: $JOB_SCRIPT"
echo "Job name: $JOB_NAME"
echo "Atlas: $ATLAS_NAME"

# Create the SLURM job script
cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%A_%a.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=${MEMORY}

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
INPUT_DIR="${SCRATCH_DIR}/OCD/NW_1stLevel"

# Bind directories
APPTAINER_BIND="/scratch/xxqian/repo/OCD_RO1/NW_group3.py:/app/NW_group3.py,${SCRATCH_DIR}/OCD:/output,${PROJECT_DIR}/metadata:/metadata,${CLINICAL_CSV}:/clinical.csv,${SUBJECTS_CSV}:/subjects.csv,${INPUT_DIR}:/input"

# Verify bind paths
for path in "${SCRATCH_DIR}/OCD" "${PROJECT_DIR}/metadata" "${CLINICAL_CSV}" "${SUBJECTS_CSV}" "${INPUT_DIR}"; do
    if [ ! -e "\$path" ]; then
        echo "Error: Bind path does not exist: \$path"
        exit 1
    fi
done

# Create output directory
mkdir -p "\${OUTPUT_DIR}"

# Run the Python script inside the Apptainer container
apptainer exec --bind "\${APPTAINER_BIND}" \${CONTAINER} python3 /app/NW_group3.py \\
    --subjects_csv /subjects.csv \\
    --clinical_csv /clinical.csv \\
    --output_dir /output \\
    --input_dir /input \\
    $([ -n "$ATLAS" ] && echo "--atlas_name ${ATLAS}") \\
    $([ "$VERBOSE" = true ] && echo "--verbose")
EOF


# Submit the job
echo "Submitting job: $JOB_NAME"
sbatch "$JOB_SCRIPT"

# Check if submission was successful
if [ $? -eq 0 ]; then
    echo "Job submitted successfully!"
    echo "Job name: $JOB_NAME"
    echo "SLURM script: $JOB_SCRIPT"
    echo "Output files will be saved to: /scratch/xxqian/logs/NWGroup3_${ATLAS_NAME}_*.out"
    echo "Error files will be saved to: /scratch/xxqian/logs/NWGroup3_${ATLAS_NAME}_*.err"
else
    echo "ERROR: Failed to submit job"
    exit 1
fi

# Clean up temporary files
echo "Cleaning up temporary files..."
rm -rf "$TEMP_JOB_DIR"

echo "Done!"
