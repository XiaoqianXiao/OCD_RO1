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
#SBATCH --output=/scratch/xxqian/logs/NWGroup3_${ATLAS_NAME}_%j.out
#SBATCH --error=/scratch/xxqian/logs/NWGroup3_${ATLAS_NAME}_%j.err
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

# Print job information
echo "=========================================="
echo "ROI-to-ROI FC Group Analysis Job"
echo "=========================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Start time: \$(date)"
echo "Atlas: $ATLAS_NAME"
echo "=========================================="

# Run the analysis
echo "Starting ROI-to-ROI FC group analysis..."
echo "Command: apptainer exec \$CONTAINER python NW_group3.py \\"
echo "  --subjects_csv $SUBJECTS_CSV \\"
echo "  --clinical_csv $CLINICAL_CSV \\"
echo "  --input_dir $INPUT_DIR \\"
echo "  --output_dir $OUTPUT_DIR \\"
echo "  --min_subjects $MIN_SUBJECTS \\"
echo "  --significance_threshold $SIGNIFICANCE_THRESHOLD \\"
echo "  $AUTO_DETECT_ATLAS \\"
echo "  $NO_FDR \\"
echo "  $VERBOSE \\"
echo "  $([ -n "$ATLAS" ] && echo "--atlas_name $ATLAS") \\"
echo "  $([ -n "$ATLAS_PARAMS" ] && echo "--atlas_params '$ATLAS_PARAMS'")"
echo "=========================================="

# Execute the analysis
apptainer exec \$CONTAINER python NW_group3.py \\
  --subjects_csv "$SUBJECTS_CSV" \\
  --clinical_csv "$CLINICAL_CSV" \\
  --input_dir "$INPUT_DIR" \\
  --output_dir "$OUTPUT_DIR" \\
  --min_subjects $MIN_SUBJECTS \\
  --significance_threshold $SIGNIFICANCE_THRESHOLD \\
  $AUTO_DETECT_ATLAS \\
  $NO_FDR \\
  $VERBOSE \\
  \$([ -n "\$ATLAS" ] && echo "--atlas_name \$ATLAS") \\
  \$([ -n "\$ATLAS_PARAMS" ] && echo "--atlas_params '\$ATLAS_PARAMS'")

# Check exit status
EXIT_STATUS=\$?

echo "=========================================="
echo "Job completed at: \$(date)"
echo "Exit status: \$EXIT_STATUS"

if [ \$EXIT_STATUS -eq 0 ]; then
    echo "SUCCESS: ROI-to-ROI FC group analysis completed successfully"
    echo "Results saved to: $OUTPUT_DIR"
    echo "Log files saved to: $OUTPUT_DIR/roiroi_fc_analysis_*.log"
else
    echo "ERROR: ROI-to-ROI FC group analysis failed with exit status \$EXIT_STATUS"
    echo "Check log files for details: $OUTPUT_DIR/roiroi_fc_analysis_*.log"
fi

echo "=========================================="
exit \$EXIT_STATUS
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
