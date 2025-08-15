#!/bin/bash

# =============================================================================
# SLURM Job Submission Script for ROI-to-ROI Functional Connectivity Analysis
# =============================================================================
# This script submits individual SLURM jobs for each subject to run ROI_1st.py
# using the Harvard-Oxford Atlas for ROI-to-ROI functional connectivity analysis.
# =============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# =============================================================================
# CONFIGURATION
# =============================================================================

# Container and script paths
CONTAINER="/scratch/xxqian/repo/image/OCD.sif"
SCRIPT_PATH="/scratch/xxqian/repo/OCD_RO1/ROI_1st.py"

# Directory configuration
BIDS_DIR="/project/6079231/dliang55/R01_AOCD/derivatives/fmriprep-1.4.1"
OUTPUT_DIR="/scratch/xxqian/OCD"
LOG_DIR="/scratch/xxqian/logs"
TEMP_JOB_DIR="/scratch/xxqian/slurm_jobs"
WORK_DIR="/scratch/xxqian/work_flow"

# SLURM job configuration
SLURM_CONFIG=(
    "--mem=16G"
    "--time=02:00:00"
    "--cpus-per-task=2"
    "--account=def-jfeusner"
    "--mail-type=FAIL"
    "--mail-user=$USER"
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
        log_error "ROI_1st.py script not found: $SCRIPT_PATH"
        return 1
    fi
    
    # Check if required directories exist
    local required_dirs=("$BIDS_DIR" "/scratch/xxqian" "/project/6079231/dliang55/R01_AOCD")
    for dir in "${required_dirs[@]}"; do
        if [ ! -e "$dir" ]; then
            log_error "Required directory does not exist: $dir"
            return 1
        fi
    done
    
    log_success "All dependencies verified"
    return 0
}

create_directories() {
    log_info "Creating necessary directories..."
    
    local dirs=("$OUTPUT_DIR" "$LOG_DIR" "$TEMP_JOB_DIR" "$WORK_DIR")
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        else
            log_info "Directory already exists: $dir"
        fi
    done
}

get_subjects() {
    log_info "Scanning for subjects in BIDS directory..."
    
    if [ ! -d "$BIDS_DIR" ]; then
        log_error "BIDS directory does not exist: $BIDS_DIR"
        return 1
    fi
    
    # Find all subject directories
    local subjects=()
    while IFS= read -r -d '' subject; do
        if [ -d "$subject" ]; then
            subjects+=("$(basename "$subject")")
        fi
    done < <(find "$BIDS_DIR" -maxdepth 1 -type d -name "sub-*" -print0)
    
    if [ ${#subjects[@]} -eq 0 ]; then
        log_error "No subjects found in BIDS directory: $BIDS_DIR"
        return 1
    fi
    
    log_success "Found ${#subjects[@]} subjects: ${subjects[*]}"
    printf '%s\n' "${subjects[@]}"
}

generate_job_script() {
    local subject="$1"
    local job_name="$2"
    local job_script="$3"
    
    log_info "Generating SLURM job script for $subject..."
    
    # Extract subject ID without 'sub-' prefix
    local subject_id="${subject#sub-}"
    
    # Create SLURM job script
    cat > "$job_script" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${LOG_DIR}/${job_name}_%j.out
#SBATCH --error=${LOG_DIR}/${job_name}_%j.err
$(printf '%s\n' "${SLURM_CONFIG[@]}")
#SBATCH --workdir=${WORK_DIR}

# =============================================================================
# ROI-to-ROI Functional Connectivity Analysis Job
# =============================================================================
# Job Information:
#   Subject: ${subject} (${subject_id})
#   Script: ROI_1st.py
#   Atlas: Harvard-Oxford Cortical Atlas
#   Analysis: ROI-to-ROI Functional Connectivity
# =============================================================================

echo "Starting ROI-to-ROI FC analysis for ${subject} at \$(date)"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Working directory: \$SLURM_SUBMIT_DIR"

# Load required modules
module load apptainer

# Set up environment
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK

# Create subject-specific work directory
SUBJECT_WORK_DIR="${WORK_DIR}/${subject_id}"
mkdir -p "\$SUBJECT_WORK_DIR"

# Define Apptainer bind paths
# Map host paths to container paths for proper file access
BIND_PATHS=(
    "${SCRIPT_PATH}:/app/ROI_1st.py"
    "${BIDS_DIR}:/project"
    "${OUTPUT_DIR}:/output"
    "\$SUBJECT_WORK_DIR:/work"
    "/scratch/xxqian:/scratch"
)

# Convert bind paths array to comma-separated string
BIND_STRING=\$(IFS=','; echo "\${BIND_PATHS[*]}")

# Run the analysis
echo "Executing ROI_1st.py for ${subject}..."
apptainer exec \\
    --bind "\$BIND_STRING" \\
    "${CONTAINER}" \\
    python3 /app/ROI_1st.py \\
        --subject ${subject} \\
        --output_dir /output \\
        --work_dir /work \\
        --verbose

# Check exit status
if [ \$? -eq 0 ]; then
    echo "ROI-to-ROI FC analysis completed successfully for ${subject} at \$(date)"
    
    # List generated output files
    echo "Generated output files:"
    ls -la /output/*${subject}* 2>/dev/null || echo "No output files found"
    
    # Clean up temporary files
    echo "Cleaning up temporary files..."
    rm -rf /work/*
    
else
    echo "ROI-to-ROI FC analysis failed for ${subject} at \$(date)" >&2
    exit 1
fi

echo "Job completed at \$(date)"
EOF

    log_success "Generated job script: $job_script"
}

submit_job() {
    local job_script="$1"
    local subject="$2"
    
    log_info "Submitting SLURM job for $subject..."
    
    # Submit the job
    local job_id
    job_id=$(sbatch "$job_script" | awk '{print $4}')
    
    if [ -n "$job_id" ]; then
        log_success "Submitted job for $subject (Job ID: $job_id)"
        return 0
    else
        log_error "Failed to submit job for $subject"
        return 1
    fi
}

cleanup_job_scripts() {
    log_info "Cleaning up temporary job scripts..."
    
    if [ -d "$TEMP_JOB_DIR" ]; then
        rm -rf "$TEMP_JOB_DIR"/*
        log_info "Cleaned up temporary job scripts"
    fi
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    log_info "Starting ROI-to-ROI FC job submission process..."
    log_info "Container: $CONTAINER"
    log_info "Script: $SCRIPT_PATH"
    log_info "BIDS directory: $BIDS_DIR"
    log_info "Output directory: $OUTPUT_DIR"
    
    # Check dependencies
    if ! check_dependencies; then
        log_error "Dependency check failed. Exiting."
        exit 1
    fi
    
    # Create necessary directories
    create_directories
    
    # Get list of subjects
    local subjects
    subjects=$(get_subjects)
    if [ $? -ne 0 ]; then
        log_error "Failed to get subjects. Exiting."
        exit 1
    fi
    
    # Process each subject
    local total_subjects=0
    local submitted_jobs=0
    local failed_jobs=0
    
    while IFS= read -r subject; do
        [ -z "$subject" ] && continue
        
        total_subjects=$((total_subjects + 1))
        log_info "Processing subject $total_subjects: $subject"
        
        # Generate job name and script path
        local job_name="ROI_1st_${subject}"
        local job_script="$TEMP_JOB_DIR/${job_name}.slurm"
        
        # Generate job script
        if generate_job_script "$subject" "$job_name" "$job_script"; then
            # Submit job
            if submit_job "$job_script" "$subject"; then
                submitted_jobs=$((submitted_jobs + 1))
            else
                failed_jobs=$((failed_jobs + 1))
            fi
        else
            log_error "Failed to generate job script for $subject"
            failed_jobs=$((failed_jobs + 1))
        fi
        
        # Small delay between submissions
        sleep 1
        
    done <<< "$subjects"
    
    # Summary
    log_info "Job submission process completed"
    log_info "Total subjects: $total_subjects"
    log_info "Successfully submitted: $submitted_jobs"
    log_info "Failed: $failed_jobs"
    
    if [ $submitted_jobs -gt 0 ]; then
        log_success "Submitted $submitted_jobs jobs successfully"
        log_info "Check job status with: squeue -u $USER"
        log_info "Monitor logs in: $LOG_DIR"
    fi
    
    if [ $failed_jobs -gt 0 ]; then
        log_warning "$failed_jobs jobs failed to submit"
        exit 1
    fi
}

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

# Set up error handling
trap 'log_error "Script interrupted. Cleaning up..."; cleanup_job_scripts; exit 1' INT TERM

# Run main function
if main; then
    log_success "ROI-to-ROI FC job submission completed successfully"
    exit 0
else
    log_error "ROI-to-ROI FC job submission failed"
    exit 1
fi