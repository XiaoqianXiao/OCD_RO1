#!/bin/bash

# =============================================================================
# SLURM Job Script for Seed-to-Voxel Functional Connectivity Group Analysis
# =============================================================================
# This script runs STV_group.py inside an Apptainer container to perform
# group-level statistical analysis on Seed-to-Voxel FC maps using FSL Randomise.
# =============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# =============================================================================
# SLURM JOB CONFIGURATION
# =============================================================================

#SBATCH --job-name=STV_Group_Analysis
#SBATCH --output=/scratch/xxqian/logs/STV_Group_%j.out
#SBATCH --error=/scratch/xxqian/logs/STV_Group_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=def-jfeusner

# =============================================================================
# CONFIGURATION
# =============================================================================

# Container and script paths
CONTAINER="/scratch/xxqian/repo/image/OCD.sif"
SCRIPT_PATH="/scratch/xxqian/repo/OCD_RO1/STV_group.py"

# Directory configuration
PROJECT_DIR="/project/6079231/dliang55/R01_AOCD"
SCRATCH_DIR="/scratch/xxqian"
OUTPUT_DIR="/scratch/xxqian/OCD/STV_group"
WORK_DIR="/scratch/xxqian/OCD/work"
ROI_DIR="/scratch/xxqian/OCD/roi"

# Input data files
SUBJECTS_CSV="${PROJECT_DIR}/metadata/shared_demographics.csv"
CLINICAL_CSV="${SCRATCH_DIR}/OCD/behav/clinical.csv"
INPUT_DIR="${SCRATCH_DIR}/OCD"

# FSL and analysis configuration
FSL_CONFIG=(
    "FSLOUTPUTTYPE=NIFTI_GZ"
    "FSLDIR=/opt/fsl"
    "PATH=/opt/fsl/bin:\$PATH"
    "LD_LIBRARY_PATH=/opt/fsl/lib:\$LD_LIBRARY_PATH"
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

log_message() {
    local level="$1"
    shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${level}] $*"
}

log_info() {
    log_message "INFO" "$@"
}

log_warning() {
    log_message "WARNING" "$@"
}

log_error() {
    log_message "ERROR" "$@"
}

log_success() {
    log_message "SUCCESS" "$@"
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check if container exists
    if [ ! -f "$CONTAINER" ]; then
        log_error "Apptainer container not found: $CONTAINER"
        return 1
    fi
    
    # Check if script exists
    if [ ! -f "$SCRIPT_PATH" ]; then
        log_error "STV_group.py script not found: $SCRIPT_PATH"
        return 1
    fi
    
    # Check if required directories exist
    local required_dirs=("$PROJECT_DIR" "$SCRATCH_DIR" "$INPUT_DIR")
    for dir in "${required_dirs[@]}"; do
        if [ ! -e "$dir" ]; then
            log_error "Required directory does not exist: $dir"
            return 1
        fi
    done
    
    # Check if required files exist
    local required_files=("$SUBJECTS_CSV" "$CLINICAL_CSV")
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "Required file does not exist: $file"
            return 1
        fi
    done
    
    log_success "All dependencies verified"
    return 0
}

create_directories() {
    log_info "Creating necessary directories..."
    
    local dirs=("$OUTPUT_DIR" "$WORK_DIR" "/scratch/xxqian/logs")
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        else
            log_info "Directory already exists: $dir"
        fi
    done
}

validate_input_data() {
    log_info "Validating input data..."
    
    # Check if input directory contains STV FC files
    local stv_files
    stv_files=$(find "$INPUT_DIR" -name "*_pcc_seed_to_voxel_fc.nii.gz" 2>/dev/null | wc -l)
    
    if [ "$stv_files" -eq 0 ]; then
        log_warning "No Seed-to-Voxel FC files found in input directory: $INPUT_DIR"
        log_info "Directory contents:"
        ls -la "$INPUT_DIR" 2>/dev/null || echo "Cannot list directory contents"
    else
        log_success "Found $stv_files Seed-to-Voxel FC files in input directory"
    fi
    
    # Check CSV file contents
    if [ -f "$SUBJECTS_CSV" ]; then
        local subject_count
        subject_count=$(tail -n +2 "$SUBJECTS_CSV" | wc -l 2>/dev/null || echo "0")
        log_info "Subjects CSV contains $subject_count subjects"
    fi
    
    if [ -f "$CLINICAL_CSV" ]; then
        local clinical_count
        clinical_count=$(tail -n +2 "$CLINICAL_CSV" | wc -l 2>/dev/null || echo "0")
        log_info "Clinical CSV contains $clinical_count records"
    fi
}

setup_environment() {
    log_info "Setting up environment..."
    
    # Load required modules
    module load apptainer
    
    # Set environment variables for optimal performance
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export PYTHONUNBUFFERED=1
    
    # Set FSL environment variables
    for fsl_var in "${FSL_CONFIG[@]}"; do
        export "$fsl_var"
        log_info "Set FSL environment variable: $fsl_var"
    done
    
    log_info "Environment variables set:"
    log_info "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
    log_info "  MKL_NUM_THREADS: $MKL_NUM_THREADS"
    log_info "  PYTHONUNBUFFERED: $PYTHONUNBUFFERED"
    log_info "  FSLOUTPUTTYPE: $FSLOUTPUTTYPE"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    log_info "Starting Seed-to-Voxel FC Group Analysis..."
    log_info "Job ID: $SLURM_JOB_ID"
    log_info "Node: $SLURM_NODELIST"
    log_info "Working directory: $SLURM_SUBMIT_DIR"
    log_info "Container: $CONTAINER"
    log_info "Script: $SCRIPT_PATH"
    log_info "Input directory: $INPUT_DIR"
    log_info "Output directory: $OUTPUT_DIR"
    
    # Check dependencies
    if ! check_dependencies; then
        log_error "Dependency check failed. Exiting."
        exit 1
    fi
    
    # Create necessary directories
    create_directories
    
    # Validate input data
    validate_input_data
    
    # Setup environment
    setup_environment
    
    # Define Apptainer bind paths
    # Map host paths to container paths for proper file access
    log_info "Setting up Apptainer bind paths..."
    
    local bind_paths=(
        "${SCRIPT_PATH}:/app/STV_group.py"
        "${INPUT_DIR}:/input"
        "${OUTPUT_DIR}:/output"
        "${WORK_DIR}:/work"
        "${SUBJECTS_CSV}:/subjects.csv"
        "${CLINICAL_CSV}:/clinical.csv"
        "${SCRATCH_DIR}:/scratch"
        "${PROJECT_DIR}:/project"
    )
    
    # Convert bind paths array to comma-separated string
    local bind_string
    bind_string=$(IFS=','; echo "${bind_paths[*]}")
    
    log_info "Bind paths configured:"
    for path in "${bind_paths[@]}"; do
        log_info "  $path"
    done
    
    # Run the analysis
    log_info "Executing STV_group.py for group analysis..."
    
    apptainer exec \
        --bind "$bind_string" \
        "$CONTAINER" \
        python3 /app/STV_group.py \
            --subjects_csv /subjects.csv \
            --clinical_csv /clinical.csv \
            --output_dir /output \
            --input_dir /input \
            --verbose
    
    # Check exit status
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log_success "Seed-to-Voxel FC group analysis completed successfully"
        
        # List generated output files
        log_info "Generated output files:"
        if [ -d "$OUTPUT_DIR" ]; then
            ls -la "$OUTPUT_DIR"/* 2>/dev/null || log_warning "No output files found"
        fi
        
        # Show analysis summary
        log_info "Analysis summary:"
        if [ -f "$OUTPUT_DIR/group_diff_baseline_stv_fc.nii.gz" ]; then
            log_info "  Group difference results: group_diff_baseline_stv_fc.nii.gz"
        fi
        
        if [ -f "$OUTPUT_DIR/baselineFC_vs_deltaYBOCS_stv_fc.nii.gz" ]; then
            log_info "  Baseline FC vs delta YBOCS results: baselineFC_vs_deltaYBOCS_stv_fc.nii.gz"
        fi
        
        if [ -f "$OUTPUT_DIR/deltaFC_vs_deltaYBOCS_stv_fc.nii.gz" ]; then
            log_info "  Delta FC vs delta YBOCS results: deltaFC_vs_deltaYBOCS_stv_fc.nii.gz"
        fi
        
        # Check for FSL Randomise outputs
        local randomise_files
        randomise_files=$(find "$OUTPUT_DIR" -name "*_tfce_corrp_tstat*.nii.gz" 2>/dev/null | wc -l)
        if [ "$randomise_files" -gt 0 ]; then
            log_info "  FSL Randomise TFCE results: $randomise_files files"
        fi
        
    else
        log_error "Seed-to-Voxel FC group analysis failed with exit code $exit_code"
        
        # Show error logs if available
        if [ -f "/scratch/xxqian/logs/STV_Group_${SLURM_JOB_ID}.err" ]; then
            log_error "Error log contents:"
            tail -20 "/scratch/xxqian/logs/STV_Group_${SLURM_JOB_ID}.err"
        fi
        
        exit $exit_code
    fi
    
    log_info "Job completed at $(date)"
}

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

# Set up error handling
trap 'log_error "Script interrupted. Exiting..."; exit 1' INT TERM

# Run main function
if main; then
    log_success "Seed-to-Voxel FC Group Analysis completed successfully"
    exit 0
else
    log_error "Seed-to-Voxel FC Group Analysis failed"
    exit 1
fi