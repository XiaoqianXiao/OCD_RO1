#!/bin/bash

# =============================================================================
# Dynamic SLURM Job Submission Script for NW_group.py
# =============================================================================
# 
# This script dynamically generates and submits SLURM jobs for ROI-to-network
# functional connectivity group analysis using NW_group.py. It creates
# atlas-specific SLURM scripts and job names.
#
# USAGE:
#   bash submit_NW_group.sh [OPTIONS]
# =============================================================================

# Default values
ATLAS=""
ATLAS_PARAMS=""
SUBJECTS_CSV="group.csv"
CLINICAL_CSV="clinical.csv"
INPUT_DIR="/project/6079231/dliang55/R01_AOCD"
OUTPUT_DIR="/project/6079231/dliang55/R01_AOCD/NW_group"
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
JOB_NAME="network_fc_group_${ATLAS_NAME}"

# Create temporary directory for SLURM scripts
TEMP_JOB_DIR="/tmp/slurm_jobs_$$"
mkdir -p "$TEMP_JOB_DIR"

# Create SLURM job script
JOB_SCRIPT="$TEMP_JOB_DIR/${JOB_NAME}.slurm"

echo "Creating SLURM job script: $JOB_SCRIPT"
echo "Job name: $JOB_NAME"
echo "Atlas: $ATLAS_NAME"

# Create the SLURM job script using a template approach
cat > "$JOB_SCRIPT" << 'TEMPLATE_EOF'
#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH --output=/scratch/xxqian/logs/NWGroup___ATLAS_NAME__%j.out
#SBATCH --error=/scratch/xxqian/logs/NWGroup___ATLAS_NAME__%j.err
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
echo "ROI-to-Network FC Group Analysis Job"
echo "=========================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Start time: \$(date)"
echo "Atlas: \$ATLAS"
echo "=========================================="

# Run the analysis
echo "Starting ROI-to-network FC group analysis..."
CMD="apptainer exec \$CONTAINER python NW_group.py \
  --subjects_csv \$SUBJECTS_CSV \
  --clinical_csv \$CLINICAL_CSV \
  --input_dir \$INPUT_DIR \
  --output_dir \$OUTPUT_DIR \
  --min_subjects \$MIN_SUBJECTS \
  --significance_threshold \$SIGNIFICANCE_THRESHOLD \
  \$AUTO_DETECT_ATLAS \
  \$NO_FDR \
  \$VERBOSE"

if [ -n "\$ATLAS" ]; then
  CMD+=" --atlas_name \$ATLAS"
fi

if [ -n "\$ATLAS_PARAMS" ]; then
  CMD+=" --atlas_params '\$ATLAS_PARAMS'"
fi

echo "Command: \$CMD"
eval \$CMD

# Check exit status
EXIT_STATUS=$?

echo "=========================================="
echo "Job completed at: \$(date)"
echo "Exit status: \$EXIT_STATUS"

if [ \$EXIT_STATUS -eq 0 ]; then
    echo "SUCCESS: ROI-to-network FC group analysis completed successfully"
    echo "Results saved to: \$OUTPUT_DIR"
else
    echo "ERROR: ROI-to-network FC group analysis failed with exit status \$EXIT_STATUS"
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

if [ $? -eq 0 ]; then
    echo "Job submitted successfully!"
    echo "Job name: $JOB_NAME"
    echo "SLURM script: $JOB_SCRIPT"
else
    echo "ERROR: Failed to submit job"
    exit 1
fi

# Clean up temporary files
rm -rf "$TEMP_JOB_DIR"

echo "Done!"
