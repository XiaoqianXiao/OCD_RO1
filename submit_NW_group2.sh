#!/bin/bash

# =============================================================================
# Dynamic SLURM Job Submission Script for NW_group2.py
# =============================================================================
# 
# This script dynamically generates and submits SLURM jobs for network-level
# functional connectivity group analysis using NW_group2.py. It creates
# atlas-specific SLURM scripts and job names.
#
# USAGE:
#   bash submit_NW_group2.sh [OPTIONS]
#
# EXAMPLES:
#   1. Run with default Power 2011 atlas (auto-detected):
#      bash submit_NW_group2.sh
#
#   2. Run with specific Power 2011 atlas:
#      bash submit_NW_group2.sh --atlas power_2011
#
#   3. Run with Schaefer 2018 atlas (400 ROIs, 7 networks):
#      bash submit_NW_group2.sh --atlas schaefer_2018 --atlas-params '{"n_rois": 400, "yeo_networks": 7}'
#
#   4. Run with YEO 2011 atlas (7 networks):
#      bash submit_NW_group2.sh --atlas yeo_2011 --atlas-params '{"n_networks": 7, "thickness": "thick"}'
#
# OPTIONS:
#   --atlas ATLAS            Atlas name (default: auto-detect)
#   --atlas-params PARAMS   JSON string of atlas parameters
#   --subjects-csv FILE     Subjects CSV file (default: group.csv)
#   --clinical-csv FILE     Clinical CSV file (default: clinical.csv)
#   --input-dir DIR         Input directory (default: /project/6079231/dliang55/R01_AOCD)
#   --output-dir DIR        Output directory (default: /project/6079231/dliang55/R01_AOCD/NW_group2)
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
OUTPUT_DIR="/project/6079231/dliang55/R01_AOCD/NW_group2"
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
            echo "  --output-dir DIR        Output directory (default: /project/6079231/dliang55/R01_AOCD/NW_group2)"
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
JOB_NAME="network_fc_group2_${ATLAS_NAME}"

# Create temporary directory for SLURM scripts
TEMP_JOB_DIR="/tmp/slurm_jobs_$$"
mkdir -p "$TEMP_JOB_DIR"

# Create SLURM job script
JOB_SCRIPT="$TEMP_JOB_DIR/${JOB_NAME}.slurm"

echo "Creating SLURM job script: $JOB_SCRIPT"
echo "Job name: $JOB_NAME"
echo "Atlas: $ATLAS_NAME"

# Create the SLURM job script
cat > "$JOB_SCRIPT" << 'TEMPLATE_EOF'
#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH --output=/scratch/xxqian/logs/NWGroup2___ATLAS_NAME__%j.out
#SBATCH --error=/scratch/xxqian/logs/NWGroup2___ATLAS_NAME__%j.err
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

# Export runtime variables
export SUBJECTS_CSV="__SUBJECTS_CSV__"
export CLINICAL_CSV="__CLINICAL_CSV__"
export INPUT_DIR="__INPUT_DIR__"
export OUTPUT_DIR="__OUTPUT_DIR__"
export MIN_SUBJECTS=__MIN_SUBJECTS__
export SIGNIFICANCE_THRESHOLD=__SIGNIFICANCE_THRESHOLD__
export AUTO_DETECT_ATLAS="__AUTO_DETECT_ATLAS__"
export NO_FDR="__NO_FDR__"
export VERBOSE="__VERBOSE__"
export ATLAS="__ATLAS__"
export ATLAS_PARAMS='__ATLAS_PARAMS__'

# Print job information
echo "=========================================="
echo "Network-Level FC Group Analysis Job"
echo "=========================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Start time: \$(date)"
echo "Atlas: \$ATLAS"
echo "=========================================="

# Run the analysis
echo "Starting network-level FC group analysis..."
CMD="apptainer exec \$CONTAINER python NW_group2.py \
  --subjects_csv \$SUBJECTS_CSV \
  --clinical_csv \$CLINICAL_CSV \
  --input_dir \$INPUT_DIR \
  --output_dir \$OUTPUT_DIR \
  \$AUTO_DETECT_ATLAS \
  \$VERBOSE"

if [ -n "\$ATLAS" ]; then
  CMD+=" --atlas_name \$ATLAS"
fi

echo "Command: \$CMD"
eval \$CMD

# Check exit status
EXIT_STATUS=\$?

echo "=========================================="
echo "Job completed at: \$(date)"
echo "Exit status: \$EXIT_STATUS"

if [ \$EXIT_STATUS -eq 0 ]; then
fr    echo "SUCCESS: Network-level FC group analysis completed successfully"
    echo "Results saved to: $OUTPUT_DIR"
    echo "Log files saved to: $OUTPUT_DIR/network_level_group_analysis_*.log"
else
    echo "ERROR: Network-level FC group analysis failed with exit status \$EXIT_STATUS"
    echo "Check log files for details: $OUTPUT_DIR/network_level_group_analysis_*.log"
fi

echo "=========================================="
exit \$EXIT_STATUS
TEMPLATE_EOF

# Replace placeholders
sed -i "s|__JOB_NAME__|$JOB_NAME|g" "$JOB_SCRIPT"
sed -i "s|__ATLAS_NAME__|$ATLAS_NAME|g" "$JOB_SCRIPT"
sed -i "s|__SUBJECTS_CSV__|$SUBJECTS_CSV|g" "$JOB_SCRIPT"
sed -i "s|__CLINICAL_CSV__|$CLINICAL_CSV|g" "$JOB_SCRIPT"
sed -i "s|__INPUT_DIR__|$INPUT_DIR|g" "$JOB_SCRIPT"
sed -i "s|__OUTPUT_DIR__|$OUTPUT_DIR|g" "$JOB_SCRIPT"
sed -i "s|__MIN_SUBJECTS__|$MIN_SUBJECTS|g" "$JOB_SCRIPT"
sed -i "s|__SIGNIFICANCE_THRESHOLD__|$SIGNIFICANCE_THRESHOLD|g" "$JOB_SCRIPT"
sed -i "s|__AUTO_DETECT_ATLAS__|$AUTO_DETECT_ATLAS|g" "$JOB_SCRIPT"
sed -i "s|__NO_FDR__|$NO_FDR|g" "$JOB_SCRIPT"
sed -i "s|__VERBOSE__|$VERBOSE|g" "$JOB_SCRIPT"
sed -i "s|__ATLAS__|$ATLAS|g" "$JOB_SCRIPT"
sed -i "s|__ATLAS_PARAMS__|$ATLAS_PARAMS|g" "$JOB_SCRIPT"

# Submit the job
echo "Submitting job: $JOB_NAME"
sbatch "$JOB_SCRIPT"

# Check if submission was successful
if [ $? -eq 0 ]; then
    echo "Job submitted successfully!"
    echo "Job name: $JOB_NAME"
    echo "SLURM script: $JOB_SCRIPT"
    echo "Output files will be saved to: /scratch/xxqian/logs/NWGroup2_${ATLAS_NAME}_*.out"
    echo "Error files will be saved to: /scratch/xxqian/logs/NWGroup2_${ATLAS_NAME}_*.err"
else
    echo "ERROR: Failed to submit job"
    exit 1
fi

# Clean up temporary files
echo "Cleaning up temporary files..."
rm -rf "$TEMP_JOB_DIR"

echo "Done!"
