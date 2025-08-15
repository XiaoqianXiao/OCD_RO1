#!/bin/bash
#SBATCH --job-name=network_fc_group
#SBATCH --output=/scratch/xxqian/logs/NWGroup_%j.out
#SBATCH --error=/scratch/xxqian/logs/NWGroup_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --account=def-jfeusner
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=$USER@mcgill.ca

# =============================================================================
# SLURM JOB SCRIPT FOR NW_GROUP.PY ANALYSIS
# =============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Load required modules
module load apptainer

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to Apptainer container
CONTAINER="/scratch/xxqian/repo/image/OCD.sif"

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# Define directories and files
PROJECT_DIR="/project/6079231/dliang55/R01_AOCD"
SCRATCH_DIR="/scratch/xxqian"
OUTPUT_DIR="${SCRATCH_DIR}/OCD/NW_group"
WORK_DIR="${SCRATCH_DIR}/work_flow"
SUBJECTS_CSV="${PROJECT_DIR}/metadata/shared_demographics.csv"
CLINICAL_CSV="${SCRATCH_DIR}/OCD/behav/clinical.csv"
INPUT_DIR="${SCRATCH_DIR}/OCD"

# =============================================================================
# DEPENDENCY CHECKING
# =============================================================================

echo "Checking dependencies..."

# Check if container exists
if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Apptainer container $CONTAINER not found"
    exit 1
fi

# Check if NW_group.py exists
if [ ! -f "/scratch/xxqian/repo/OCD_RO1/NW_group.py" ]; then
    echo "ERROR: NW_group.py script not found at /scratch/xxqian/repo/OCD_RO1/NW_group.py"
    exit 1
fi

# Verify required paths and files
REQUIRED_PATHS=(
    "$SCRATCH_DIR/OCD"
    "$PROJECT_DIR/metadata"
    "$CLINICAL_CSV"
    "$SUBJECTS_CSV"
    "$INPUT_DIR"
)

for path in "${REQUIRED_PATHS[@]}"; do
    if [ ! -e "$path" ]; then
        echo "ERROR: Required path does not exist: $path"
        exit 1
    fi
done

echo "All dependencies verified successfully"

# =============================================================================
# DIRECTORY SETUP
# =============================================================================

echo "Setting up directories..."

# Create necessary directories
mkdir -p "$OUTPUT_DIR" "$WORK_DIR" "/scratch/xxqian/logs"

echo "Directories created successfully"

# =============================================================================
# APPTAINER BIND CONFIGURATION
# =============================================================================

echo "Configuring Apptainer bind paths..."

# Define bind paths for Apptainer
# Note: Using absolute paths to avoid bind path issues
APPTAINER_BIND="/scratch/xxqian/repo/OCD_RO1/NW_group.py:/app/NW_group.py,\
${SCRATCH_DIR}:/scratch,\
${PROJECT_DIR}:/project,\
${OUTPUT_DIR}:/output,\
${WORK_DIR}:/work"

echo "Bind paths configured:"
echo "  Script: /scratch/xxqian/repo/OCD_RO1/NW_group.py -> /app/NW_group.py"
echo "  Scratch: ${SCRATCH_DIR} -> /scratch"
echo "  Project: ${PROJECT_DIR} -> /project"
echo "  Output: ${OUTPUT_DIR} -> /output"
echo "  Work: ${WORK_DIR} -> /work"

# =============================================================================
# ANALYSIS EXECUTION
# =============================================================================

echo "Starting NW_group.py analysis..."
echo "Input subjects CSV: $SUBJECTS_CSV"
echo "Input clinical CSV: $CLINICAL_CSV"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Run the Python script inside the Apptainer container
if apptainer exec --bind "${APPTAINER_BIND}" "$CONTAINER" python3 /app/NW_group.py \
    --subjects_csv /project/metadata/shared_demographics.csv \
    --clinical_csv /scratch/OCD/behav/clinical.csv \
    --output_dir /output \
    --input_dir /scratch/OCD \
    --verbose; then
    
    echo "=========================================="
    echo "Analysis completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo "=========================================="
    exit_code=0
else
    echo "=========================================="
    echo "ERROR: Analysis failed!"
    echo "Check the error logs above for details"
    echo "=========================================="
    exit_code=1
fi

# =============================================================================
# CLEANUP AND EXIT
# =============================================================================

echo "Cleaning up temporary files..."
# Remove any temporary files if needed

echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code