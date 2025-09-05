#!/bin/bash

# SLURM job submission script for running NW_1st.py functional connectivity analysis
# 
# This script submits individual SLURM jobs for each subject to compute:
# - ROI-to-ROI functional connectivity matrices
# - ROI-to-Network functional connectivity
# - Network-to-Network functional connectivity
# 
# DEFAULT ATLAS: Power 2011 (264 ROIs, 13 networks) - local file-based
# 
# USAGE:
#   bash submit_NW_1st.sh [OPTIONS]
#
# NOTE: The Power 2011 atlas (default) requires the atlas file to exist at
# /scratch/xxqian/roi/power_2011_atlas.nii.gz and network labels from
# /scratch/xxqian/roi/power264/power264NodeNames.txt
#
# EXAMPLES:
#   1. Submit all subjects with default Power 2011 atlas:
#      bash submit_NW_1st.sh
#
#   2. Submit with Schaefer 2018 atlas (100 ROIs, 7 networks) - good balance:
#      bash submit_NW_1st.sh --atlas schaefer_2018 --atlas-params '{"n_rois": 100, "yeo_networks": 7}'
#
#   3. Submit with Schaefer 2018 atlas (400 ROIs, 7 networks) - high resolution:
#      bash submit_NW_1st.sh --atlas schaefer_2018 --atlas-params '{"n_rois": 400, "yeo_networks": 7}'
#
#   3. Submit with Schaefer 2018 atlas (1000 ROIs, 17 networks) - maximum resolution:
#      bash submit_NW_1st.sh --atlas schaefer_2018 --atlas-params '{"n_rois": 1000, "yeo_networks": 17}'
#
#   4. Submit with Harvard-Oxford atlas (anatomical):
#      bash submit_NW_1st.sh --atlas harvard_oxford --atlas-params '{"atlas_name": "cort-maxprob-thr25-2mm"}'
#
#   5. Submit with AAL atlas (standard anatomical):
#      bash submit_NW_1st.sh --atlas aal
#
#   6. Submit with custom label pattern for Power 2011:
#      bash submit_NW_1st.sh --atlas power_2011 --label-pattern custom --custom-regex "network_(\\d+)_(.+)"
#
#   7. Submit with Yeo 2011 atlas (uses default 7-network thick parcellation):
#      bash submit_NW_1st.sh --atlas yeo_2011
#
#   Note: YEO 2011 uses default parcellation in older Nilearn versions (0.10.4)
#   Parameters are ignored and the atlas uses 7-network thick parcellation
#
# OPTIONS:
#   --atlas ATLAS            Atlas name (default: power_2011)
#   --atlas-params PARAMS   JSON string of atlas parameters
#   --label-pattern PATTERN Label pattern (default: power)
#   --custom-regex REGEX    Custom regex pattern for label parsing (use with --label-pattern custom)
#   --help                   Show this usage information
#
# ATLAS TYPES:
#   - power_2011: Power 2011 atlas (264 ROIs, 13 networks) - DEFAULT
#     * Requires atlas file at /scratch/xxqian/roi/power_2011_atlas.nii.gz
#     * Uses network labels from /scratch/xxqian/roi/power264/
#     * Best for: Standard functional connectivity analysis with established networks
#   - schaefer_2018: Schaefer 2018 parcellation (100-1000 ROIs, 7/17 networks)
#     * Highly customizable: choose ROI count and network count
#     * Best for: High-resolution parcellation with Yeo network assignments
#   - harvard_oxford: Harvard-Oxford cortical/subcortical atlases
#     * Anatomically defined regions
#     * Best for: Anatomical ROI analysis
#   - aal: Automated Anatomical Labeling atlas (116 ROIs)
#     * Standard anatomical parcellation
#     * Best for: Traditional anatomical ROI analysis
#   - talairach: Talairach atlas (1107 ROIs)
#     * High-resolution anatomical parcellation
#     * Best for: Detailed anatomical analysis
#   - yeo_2011: Yeo 2011 network parcellation (7 networks, thick)
#     * Uses default 7-network thick parcellation (older Nilearn 0.10.4)
#     * Parameters are ignored - always uses default parcellation
#     * Best for: Network-focused analysis
#
# LABEL PATTERNS:
#   - power: Power 2011 label format (default) - automatically loads from /scratch/xxqian/roi/power264/
#   - nilearn: Built-in atlas label format (for Schaefer, Harvard-Oxford, AAL, etc.)
#   - custom: Custom regex pattern (use with --custom-regex)
#
# POWER 2011 ATLAS FEATURES:
#   - Requires pre-existing atlas file: /scratch/xxqian/roi/power_2011_atlas.nii.gz
#   - Uses real network labels from cluster directory: /scratch/xxqian/roi/power264/
#   - 264 ROIs organized into 13 functional networks
#   - Local file-based loading (no coordinate generation)
#   - Container automatically mounts /scratch/xxqian for access to files
#
# SLURM CONFIGURATION:
#   - Memory: 8G per job
#   - Time: 1 hour per job
#   - CPUs: 1 per job
#   - Account: def-jfeusner
#   - Logs: /scratch/xxqian/logs/
#
# For more information, see the script help or run with --help.

CONTAINER="/scratch/xxqian/repo/image/OCD.sif"

# Parse command line arguments
ATLAS="power_2011"
ATLAS_PARAMS=""
LABEL_PATTERN="power"

while [[ $# -gt 0 ]]; do
    case $1 in
        --atlas)
            ATLAS="$2"
            shift 2
            ;;
        --atlas-params)
            ATLAS_PARAMS="$2"
            shift 2
            ;;
        --label-pattern)
            LABEL_PATTERN="$2"
            shift 2
            ;;
        --custom-regex)
            CUSTOM_REGEX="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--atlas ATLAS] [--atlas-params PARAMS] [--label-pattern PATTERN] [--custom-regex REGEX]"
            echo "  --atlas ATLAS            Atlas name (default: power_2011)"
            echo "  --atlas-params PARAMS   JSON string of atlas parameters"
            echo "  --label-pattern PATTERN Label pattern (default: power)"
            echo "  --custom-regex REGEX    Custom regex pattern for label parsing (use with --label-pattern custom)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Define directories
BIDS_DIR="/project/6079231/dliang55/R01_AOCD/derivatives/fmriprep-1.4.1"
OUTPUT_DIR="/home/xxqian/scratch/OCD/NW_1st"
LOG_DIR="/scratch/xxqian/logs"
TEMP_JOB_DIR="/scratch/xxqian/slurm_jobs"

# Create necessary directories
mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$TEMP_JOB_DIR"

# Check if container exists
if [ ! -f "$CONTAINER" ]; then
    echo "Error: Apptainer container $CONTAINER not found"
    exit 1
fi

# Get list of subjects from BIDS derivatives directory
SUBJECTS=($(ls -d "$BIDS_DIR"/sub-* | xargs -n 1 basename | grep '^sub-'))
if [ ${#SUBJECTS[@]} -eq 0 ]; then
    echo "Error: No subjects found in $BIDS_DIR"
    exit 1
fi

# Define bind paths for Apptainer, avoiding redundant mounts
APPTAINER_BIND="/project/6079231/dliang55/R01_AOCD:/project/6079231/dliang55/R01_AOCD,/scratch/xxqian:/scratch/xxqian,/scratch/xxqian/repo/OCD_RO1:/app"

# Verify bind paths
for path in "/project/6079231/dliang55/R01_AOCD" "/scratch/xxqian"; do
    if [ ! -e "$path" ]; then
        echo "Error: Bind path does not exist: $path"
        exit 1
    fi
done

echo "Submitting NW_1st.py jobs to SLURM"
echo "Atlas: $ATLAS"
if [[ -n "$ATLAS_PARAMS" ]]; then
    echo "Atlas Parameters: $ATLAS_PARAMS"
fi
echo "Label Pattern: $LABEL_PATTERN"
echo "Subjects found: ${#SUBJECTS[@]}"

# Loop over subjects and submit a job for each
for SUBJECT in "${SUBJECTS[@]}"; do
    JOB_NAME="NW_1st_${SUBJECT}_${ATLAS}"
    JOB_SCRIPT="$TEMP_JOB_DIR/${JOB_NAME}.slurm"

    # Extract subject ID without 'sub-' prefix
    SUBJECT_ID=${SUBJECT#sub-}

    # Set optimal memory allocation based on atlas type and configuration
    case "$ATLAS" in
        "yeo_2011")
            MEMORY="16G"  # YEO 2011 needs 16GB due to label processing fix
            ;;
        "schaefer_2018")
            # Check if specific ROI count is provided in atlas-params
            if [[ "$ATLAS_PARAMS" == *"n_rois\": 100"* ]]; then
                MEMORY="8G"   # 100 ROIs: 8GB sufficient (proven successful)
            elif [[ "$ATLAS_PARAMS" == *"n_rois\": 200"* ]] || [[ "$ATLAS_PARAMS" == *"n_rois\": 300"* ]]; then
                MEMORY="16G"  # 200-300 ROIs: 16GB
            elif [[ "$ATLAS_PARAMS" == *"n_rois\": 400"* ]] || [[ "$ATLAS_PARAMS" == *"n_rois\": 500"* ]] || [[ "$ATLAS_PARAMS" == *"n_rois\": 600"* ]]; then
                MEMORY="24G"  # 400-600 ROIs: 24GB
            else
                MEMORY="32G"  # 700+ ROIs: 32GB (default for large configurations)
            fi
            ;;
        "power_2011")
            MEMORY="16G"  # Power 2011 (264 ROIs): 16GB sufficient
            ;;
        "harvard_oxford"|"aal"|"talairach")
            MEMORY="8G"   # Anatomical atlases: 8GB sufficient
            ;;
        *)
            MEMORY="16G"  # Default for unknown atlases
            ;;
    esac

    # Log memory allocation decision
    echo "Memory allocation for ${ATLAS}: ${MEMORY}"

    # Create SLURM job script
    cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --mem=${MEMORY}
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --account=def-jfeusner

# Load Apptainer module
module load apptainer

# Define variables
SUBJECT=${SUBJECT_ID}

# Run the Apptainer container for one subject
if [ -n "\$SUBJECT" ]; then
    apptainer exec --bind "${APPTAINER_BIND}" "${CONTAINER}" python3 /app/NW_1st.py --subject sub-\${SUBJECT} --atlas $ATLAS --label-pattern $LABEL_PATTERN$([ -n "$ATLAS_PARAMS" ] && echo " --atlas-params '$ATLAS_PARAMS'")
else
    echo "Error: No subject found" >&2
    exit 1
fi

EOF

    # Submit the job
    sbatch "$JOB_SCRIPT"
    echo "Submitted job for $SUBJECT (Job script: $JOB_SCRIPT)"
done

echo "All jobs submitted. Check status with 'squeue -u $USER'"

echo ""
echo "TROUBLESHOOTING:"
echo "================"
echo "1. If Power 2011 atlas fails:"
echo "   - Check that /scratch/xxqian/roi/power_2011_atlas.nii.gz exists"
echo "   - Check that /scratch/xxqian/roi/power264/power264NodeNames.txt exists"
echo "   - Verify the label file has exactly 264 lines"
echo "   - Ensure both files are readable by the container"
echo ""
echo "2. If other atlases fail:"
echo "   - Check network connectivity for Nilearn downloads"
echo "   - Verify container has sufficient disk space"
echo "   - Check SLURM logs for specific error messages"
echo ""
echo "3. For custom atlases:"
echo "   - Ensure --labels file exists and is accessible"
echo "   - Verify --custom-regex pattern matches your label format"
echo ""
echo "4. Check job status:"
echo "   - squeue -u $USER          # View running jobs"
echo "   - scancel <job_id>         # Cancel specific job"
echo "   - tail -f <log_file>       # Monitor job progress"