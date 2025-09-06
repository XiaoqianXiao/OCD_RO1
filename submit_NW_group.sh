#!/bin/bash

# =============================================================================
# SLURM job submission script for ROI-to-Network group functional connectivity
# analysis using NW_group.py
#
# This script dynamically generates and submits SLURM jobs for ROI-to-network
# functional connectivity group analysis. It creates atlas-specific SLURM
# scripts and job names, ensuring correct configuration for each atlas type.
#
# DEFAULT ATLAS: Power 2011 (264 ROIs, 13 networks)
#
# USAGE:
#   bash submit_NW_group.sh [OPTIONS]
#
# EXAMPLES:
#   1. Run with default Power 2011 atlas (auto-detected):
#      bash submit_NW_group.sh
#
#   2. Run with specific Power 2011 atlas:
#      bash submit_NW_group.sh --atlas power_2011
#
#   3. Run with Schaefer 2018 atlas (400 ROIs, 7 networks):
#      bash submit_NW_group.sh --atlas schaefer_2018 --atlas-params '{"n_rois": 400, "yeo_networks": 7}'
#
#   4. Run with Harvard-Oxford atlas (anatomical):
#      bash submit_NW_group.sh --atlas harvard_oxford --atlas-params '{"atlas_name": "cort-maxprob-thr25-2mm"}'
#
#   5. Run with Yeo 2011 atlas (7 networks):
#      bash submit_NW_group.sh --atlas yeo_2011 --atlas-params '{"n_networks": 7, "thickness": "thick"}'
#
# OPTIONS:
#   --atlas ATLAS             Atlas name (default: power_2011)
#   --atlas-params PARAMS     JSON string of atlas parameters
#   --subjects-csv FILE       Subjects CSV file (default: group.csv)
#   --clinical-csv FILE       Clinical CSV file (default: clinical.csv)
#   --input-dir DIR           Input directory (default: /project/6079231/dliang55/R01_AOCD)
#   --output-dir DIR          Output directory (default: /project/6079231/dliang55/R01_AOCD/NW_group)
#   --min-subjects N          Minimum subjects per group (default: 2)
#   --significance-threshold P Significance threshold (default: 0.05)
#   --no-fdr                  Disable FDR correction
#   --verbose                 Enable verbose logging
#   --help                    Show this usage information
#
# SLURM CONFIGURATION:
#   - Memory: dynamic based on atlas
#   - Time: 2 hours per job
#   - CPUs: 2 per job
#   - Account: def-jfeusner
#   - Logs: /scratch/xxqian/logs/
#   - SLURM scripts: /home/xxqian/scratch/slurm_jobs
# =============================================================================

CONTAINER="/scratch/xxqian/repo/image/OCD.sif"

# Default values
ATLAS="power_2011"
ATLAS_PARAMS=""
SUBJECTS_CSV="group.csv"
CLINICAL_CSV="clinical.csv"
INPUT_DIR="/project/6079231/dliang55/R01_AOCD"
OUTPUT_DIR="/project/6079231/dliang55/R01_AOCD/NW_group"
MIN_SUBJECTS=2
SIGNIFICANCE_THRESHOLD=0.05
NO_FDR=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --atlas) ATLAS="$2"; shift 2 ;;
        --atlas-params) ATLAS_PARAMS="$2"; shift 2 ;;
        --subjects-csv) SUBJECTS_CSV="$2"; shift 2 ;;
        --clinical-csv) CLINICAL_CSV="$2"; shift 2 ;;
        --input-dir) INPUT_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --min-subjects) MIN_SUBJECTS="$2"; shift 2 ;;
        --significance-threshold) SIGNIFICANCE_THRESHOLD="$2"; shift 2 ;;
        --no-fdr) NO_FDR=true; shift ;;
        --verbose) VERBOSE=true; shift ;;
        --help)
            grep "^# " "$0" | sed 's/^# //'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Directories
LOG_DIR="/scratch/xxqian/logs"
TEMP_JOB_DIR="/home/xxqian/scratch/slurm_jobs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$TEMP_JOB_DIR"

# Check container
if [ ! -f "$CONTAINER" ]; then
    echo "Error: Apptainer container $CONTAINER not found"
    exit 1
fi

# Set memory based on atlas
case "$ATLAS" in
    "schaefer_2018")
        if [[ "$ATLAS_PARAMS" == *'"n_rois": 100'* ]]; then
            MEMORY="16G"
        elif [[ "$ATLAS_PARAMS" == *'"n_rois": 400'* ]]; then
            MEMORY="24G"
        else
            MEMORY="32G"
        fi
        ;;
    "yeo_2011") MEMORY="16G" ;;
    "power_2011") MEMORY="16G" ;;
    "harvard_oxford"|"aal"|"talairach") MEMORY="8G" ;;
    *) MEMORY="16G" ;;
esac

# Job name
JOB_NAME="NWGroup_${ATLAS}"
JOB_SCRIPT="$TEMP_JOB_DIR/${JOB_NAME}.slurm"

# Create SLURM job script
cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --mem=${MEMORY}
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --account=def-jfeusner

module load apptainer

apptainer exec --bind /project/6079231:/project/6079231,/scratch/xxqian:/scratch/xxqian "${CONTAINER}" \\
    python3 /app/NW_group.py \\
    --atlas_name ${ATLAS} \\
    --subjects_csv ${SUBJECTS_CSV} \\
    --clinical_csv ${CLINICAL_CSV} \\
    --input_dir ${INPUT_DIR} \\
    --output_dir ${OUTPUT_DIR} \\
    --min_subjects ${MIN_SUBJECTS} \\
    --significance_threshold ${SIGNIFICANCE_THRESHOLD} \\
    $([ "$NO_FDR" = true ] && echo "--no_fdr") \\
    $([ "$VERBOSE" = true ] && echo "--verbose") \\
    $([ -n "$ATLAS_PARAMS" ] && echo "--atlas_params '${ATLAS_PARAMS}'")
EOF

# Submit job
sbatch "$JOB_SCRIPT"
echo "Submitted group job: $JOB_SCRIPT"
