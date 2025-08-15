#!/bin/bash

# SLURM job submission script for running ROI-to-ROI FC analysis for each subject
# Updated to work with the revised ROI_1st.py script

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# =============================================================================
# CONFIGURATION
# =============================================================================

# Apptainer container
CONTAINER="/scratch/xxqian/repo/image/OCD.sif"

# Directory configuration (matching ROI_1st.py DEFAULT_CONFIG)
BIDS_DIR="/project/6079231/dliang55/R01_AOCD/derivatives/fmriprep-1.4.1"
OUTPUT_DIR="/scratch/xxqian/OCD"
WORK_DIR="/scratch/xxqian/work_flow"
ROI_DIR="/scratch/xxqian/roi"
LOG_DIR="/scratch/xxqian/logs"
TEMP_JOB_DIR="/scratch/xxqian/slurm_jobs"

# SLURM job configuration
MEMORY="16G"           # Increased memory for better performance
TIME_LIMIT="02:00:00"  # Increased time limit for complex analyses
CPUS_PER_TASK=2        # Allow some parallel processing
ACCOUNT="def-jfeusner"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

error_exit() {
    log_message "ERROR: $1" >&2
    exit 1
}

check_dependencies() {
    log_message "Checking dependencies..."
    
    # Check if container exists
    if [ ! -f "$CONTAINER" ]; then
        error_exit "Apptainer container $CONTAINER not found"
    fi
    
    # Check if ROI_1st.py exists in the expected location
    if [ ! -f "/scratch/xxqian/repo/OCD_RO1/ROI_1st.py" ]; then
        error_exit "ROI_1st.py script not found at /scratch/xxqian/repo/OCD_RO1/ROI_1st.py"
    fi
    
    # Verify bind paths
    for path in "$BIDS_DIR" "$OUTPUT_DIR" "$WORK_DIR" "$ROI_DIR"; do
        if [ ! -e "$path" ]; then
            error_exit "Required path does not exist: $path"
        fi
    done
    
    log_message "All dependencies verified successfully"
}

create_directories() {
    log_message "Creating necessary directories..."
    
    for dir in "$OUTPUT_DIR" "$WORK_DIR" "$ROI_DIR" "$LOG_DIR" "$TEMP_JOB_DIR"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_message "Created directory: $dir"
        fi
    done
}

get_subjects() {
    log_message "Scanning for subjects in BIDS directory..."
    
    if [ ! -d "$BIDS_DIR" ]; then
        error_exit "BIDS directory does not exist: $BIDS_DIR"
    fi
    
    # Get list of subjects from BIDS derivatives directory
    SUBJECTS=($(ls -d "$BIDS_DIR"/sub-* 2>/dev/null | xargs -n 1 basename | grep '^sub-' || true))
    
    if [ ${#SUBJECTS[@]} -eq 0 ]; then
        error_exit "No subjects found in $BIDS_DIR"
    fi
    
    log_message "Found ${#SUBJECTS[@]} subjects: ${SUBJECTS[*]}"
}

# =============================================================================
# JOB SCRIPT GENERATION
# =============================================================================

create_job_script() {
    local subject="$1"
    local subject_id="${subject#sub-}"
    local job_name="PCC_ROI_1st_${subject}"
    local job_script="$TEMP_JOB_DIR/${job_name}.slurm"
    
    log_message "Creating job script for $subject..."
    
    # Define bind paths for Apptainer
    local apptainer_bind="/project/6079231/dliang55/R01_AOCD:/project/6079231/dliang55/R01_AOCD,/scratch/xxqian:/scratch/xxqian,/scratch/xxqian/repo/OCD_RO1/ROI_1st.py:/app/ROI_1st.py"
    
    # Create SLURM job script
    cat > "$job_script" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${LOG_DIR}/${job_name}_%j.out
#SBATCH --error=${LOG_DIR}/${job_name}_%j.err
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --account=${ACCOUNT}
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=\$USER@mcgill.ca

# Job information
echo "=========================================="
echo "Job started at: \$(date)"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Subject: ${subject}"
echo "=========================================="

# Load required modules
module load apptainer

# Set up environment
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK

# Create subject-specific work directory
SUBJECT_WORK_DIR="${WORK_DIR}/\${SLURM_JOB_ID}_${subject_id}"
mkdir -p "\$SUBJECT_WORK_DIR"

# Run the analysis
echo "Starting ROI_1st.py analysis for ${subject}..."
echo "Work directory: \$SUBJECT_WORK_DIR"

if apptainer exec --bind "${apptainer_bind}" "${CONTAINER}" python3 /app/ROI_1st.py --subject ${subject} --verbose; then
    echo "Analysis completed successfully for ${subject}"
    exit_code=0
else
    echo "Analysis failed for ${subject}" >&2
    exit_code=1
fi

# Clean up temporary files
echo "Cleaning up temporary files..."
rm -rf "\$SUBJECT_WORK_DIR"

echo "=========================================="
echo "Job finished at: \$(date)"
echo "Exit code: \$exit_code"
echo "=========================================="

exit \$exit_code
EOF

    log_message "Created job script: $job_script"
    echo "$job_script"
}

submit_jobs() {
    local subjects=("$@")
    local submitted_count=0
    local failed_count=0
    
    log_message "Submitting jobs for ${#subjects[@]} subjects..."
    
    for subject in "${subjects[@]}"; do
        local job_script
        job_script=$(create_job_script "$subject")
        
        if sbatch "$job_script" >/dev/null 2>&1; then
            log_message "Submitted job for $subject"
            ((submitted_count++))
        else
            log_message "Failed to submit job for $subject" >&2
            ((failed_count++))
        fi
        
        # Small delay between submissions to avoid overwhelming the scheduler
        sleep 0.5
    done
    
    log_message "Job submission summary:"
    log_message "  Successfully submitted: $submitted_count"
    log_message "  Failed submissions: $failed_count"
    
    if [ $failed_count -gt 0 ]; then
        log_message "WARNING: Some jobs failed to submit. Check the logs above."
        return 1
    fi
    
    return 0
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    log_message "Starting PCC ROI_1st.py job submission process..."
    
    # Check dependencies
    check_dependencies
    
    # Create directories
    create_directories
    
    # Get list of subjects
    get_subjects
    
    # Submit jobs
    if submit_jobs "${SUBJECTS[@]}"; then
        log_message "All jobs submitted successfully!"
        log_message "Check job status with: squeue -u \$USER"
        log_message "Monitor logs in: $LOG_DIR"
        log_message "Output files will be in: $OUTPUT_DIR"
    else
        error_exit "Some jobs failed to submit. Check the logs above."
    fi
}

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

# Only run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi