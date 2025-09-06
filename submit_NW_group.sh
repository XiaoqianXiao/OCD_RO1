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
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%A_%a.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=${MEMORY}
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
INPUT_DIR="${SCRATCH_DIR}/OCD/NW_1stLevel"

# Bind directories
APPTAINER_BIND="/scratch/xxqian/repo/OCD_RO1/NW_group.py:/app/NW_group.py,\${SCRATCH_DIR}/OCD:/output,\${PROJECT_DIR}/metadata:/metadata,\${CLINICAL_CSV}:/clinical.csv,\${SUBJECTS_CSV}:/subjects.csv,\${INPUT_DIR}:/input"

# Verify bind paths
for path in "\${SCRATCH_DIR}/OCD" "\${PROJECT_DIR}/metadata" "\${CLINICAL_CSV}" "\${SUBJECTS_CSV}" "\${INPUT_DIR}"; do
    if [ ! -e "\$path" ]; then
        echo "Error: Bind path does not exist: \$path"
        exit 1
    fi
done

# Create output directory
mkdir -p "\${OUTPUT_DIR}"

# Run the Python script inside the Apptainer container
apptainer exec --bind "\${APPTAINER_BIND}" \${CONTAINER} python3 /app/NW_group.py \\
    --subjects_csv /subjects.csv \\
    --clinical_csv /clinical.csv \\
    --output_dir /output \\
    --input_dir /input \\
    $([ -n "$ATLAS" ] && echo "--atlas_name ${ATLAS}") \\
    $([ "$VERBOSE" = true ] && echo "--verbose")
EOF

# Submit job
sbatch "$JOB_SCRIPT"
echo "Submitted group job: $JOB_SCRIPT"
