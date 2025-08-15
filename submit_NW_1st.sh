#!/bin/bash
# =============================================================================
# SLURM Job Submission Script for NW_1st.py
# =============================================================================
# 
# This script submits individual subject/session functional connectivity analysis
# jobs to the SLURM queue using NW_1st.py. It processes ROI-to-ROI and 
# ROI-to-network connectivity for resting-state fMRI data.
#
# USAGE:
#   bash submit_NW_1st.sh [OPTIONS]
#
# EXAMPLES:
#   1. Submit all subjects with default settings:
#      bash submit_NW_1st.sh
#
#   2. Submit specific subjects:
#      bash submit_NW_1st.sh --subjects sub-AOCD001,sub-AOCD002
#
#   3. Submit with custom atlas:
#      bash submit_NW_1st.sh --atlas schaefer_2018 --atlas-params '{"n_rois": 400}'
#
#   4. Submit with custom output directory:
#      bash submit_NW_1st.sh --output-dir /custom/output/path
#
#   5. Submit with custom work directory:
#      bash submit_NW_1st.sh --work-dir /custom/work/path
#
#   6. Submit with verbose logging:
#      bash submit_NW_1st.sh --verbose
#
#   7. Submit with custom SLURM parameters:
#      bash submit_NW_1st.sh --time 4:00:00 --mem 32G --cpus 4
#
# HELP OPTIONS:
#   --help          Show this help message
#   --usage         Show detailed usage examples
#
# For more information, run with --usage or --help.
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================



# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Safe SLURM environment variable access
get_slurm_var() {
    local var_name="$1"
    local fallback="$2"
    
    if [[ -n "${!var_name:-}" ]]; then
        echo "${!var_name}"
    else
        echo "$fallback"
    fi
}

check_slurm_issues() {
    local subject="$1"
    local job_script="$2"
    
    echo "Checking for common SLURM submission issues..."
    
    # Check SLURM daemon status
    echo "Checking SLURM daemon status..."
    if [[ "$SQUEUE_AVAILABLE" == true ]]; then
        if ! squeue -u $USER >/dev/null 2>&1; then
            echo "ERROR: Cannot connect to SLURM daemon. SLURM may be down or you may not have access."
            return 1
        fi
    else
        echo "Skipping SLURM daemon check (squeue not available)"
    fi
    
    # Check partition availability
    echo "Checking partition availability..."
    if [[ "$SBATCH_AVAILABLE" == true ]]; then
        if ! sinfo >/dev/null 2>&1; then
            echo "ERROR: Cannot get partition information. SLURM may be down or you may not have access."
            return 1
        fi
    else
        echo "Skipping partition check (sinfo not available)"
    fi
    
    # Check account access
    if [[ -n "$SLURM_ACCOUNT" ]]; then
        echo "Checking account access for: $SLURM_ACCOUNT"
        if [[ "$SBATCH_AVAILABLE" == true ]]; then
            if ! sacctmgr show account $SLURM_ACCOUNT >/dev/null 2>&1; then
                echo "WARNING: Cannot verify account $SLURM_ACCOUNT. You may not have access or the account may not exist."
            fi
        else
            echo "Skipping account verification (sacctmgr not available)"
        fi
    fi
    
    # Check resource limits
    echo "Checking resource limits..."
    if [[ -n "$SLURM_MEM" ]]; then
        echo "Memory request: $SLURM_MEM"
    fi
    if [[ -n "$SLURM_CPUS" ]]; then
        echo "CPU request: $SLURM_CPUS"
    fi
    if [[ -n "$SLURM_TIME" ]]; then
        echo "Time request: $SLURM_TIME"
    fi
    
    # Check if user has pending jobs
    if [[ "$SQUEUE_AVAILABLE" == true ]]; then
        local pending_jobs=$(squeue -u $USER -h 2>/dev/null | wc -l)
        echo "Current pending jobs for user $USER: $pending_jobs"
    else
        echo "Skipping pending jobs check (squeue not available)"
    fi
    
    # Check job script syntax
    echo "Checking job script syntax..."
    if ! bash -n "$job_script" 2>/dev/null; then
        echo "ERROR: Job script has syntax errors: $job_script"
        return 1
    fi
    
    echo "SLURM environment check completed"
    return 0
}

validate_job_script() {
    local job_script="$1"
    local subject="$2"
    
    echo "Validating job script for $subject..."
    
    # Check if job script exists and is executable
    if [[ ! -f "$job_script" ]]; then
        echo "ERROR: Job script does not exist: $job_script"
        return 1
    fi
    
    if [[ ! -x "$job_script" ]]; then
        echo "ERROR: Job script is not executable: $job_script"
        return 1
    fi
    
    # Check job script size
    local script_size=$(wc -c < "$job_script")
    if [[ $script_size -lt 100 ]]; then
        echo "ERROR: Job script seems too small ($script_size bytes): $job_script"
        return 1
    fi
    
    # Check for required SLURM directives
    if ! grep -q "^#SBATCH" "$job_script"; then
        echo "ERROR: Job script missing SLURM directives: $job_script"
        return 1
    fi
    
    # Check for required content
    if ! grep -q "apptainer exec" "$job_script"; then
        echo "ERROR: Job script missing apptainer command: $job_script"
        return 1
    fi
    
    if ! grep -q "python.*$PYTHON_SCRIPT" "$job_script"; then
        echo "ERROR: Job script missing Python execution: $job_script"
        return 1
    fi
    
    echo "Job script validation passed for $subject"
    return 0
}

print_help() {
    cat << 'EOF'
SLURM Job Submission Script for NW_1st.py
==========================================

This script submits individual subject/session functional connectivity analysis
jobs to the SLURM queue using NW_1st.py. It processes ROI-to-ROI and 
ROI-to-network connectivity for resting-state fMRI data.

BASIC USAGE:
  bash submit_NW_1st.sh [OPTIONS]

OPTIONS:
  --help                    Show this help message
  --usage                   Show detailed usage examples
  --subjects SUBJECTS       Comma-separated list of subjects (default: all)
  --atlas ATLAS            Atlas name or path (default: power_2011)
  --atlas-params PARAMS    JSON string of atlas parameters
  --labels LABELS          Path to labels file (default: power264 labels)
  --label-pattern PATTERN  Label pattern type (default: power)
  --custom-regex REGEX     Custom regex for label parsing
  --atlas-name NAME        Custom atlas name for output files
          --output-dir DIR         Output directory (default: /scratch/xxqian/OCD)
        --work-dir DIR           Work directory (default: /scratch/xxqian/work_flow)
        --slurm-scripts-dir DIR  SLURM scripts directory (default: /scratch/xxqian/slurm_jobs)
          --bids-dir DIR           BIDS directory (default: /project/6079231/dliang55/R01_AOCD/derivatives/fmriprep-1.4.1)
          --roi-dir DIR            ROI directory (default: /scratch/xxqian/roi)
  --time TIME              SLURM time limit (default: 2:00:00)
  --mem MEM                SLURM memory limit (default: 16G)
  --cpus CPUS              SLURM CPUs per task (default: 2)
  --account ACCOUNT        SLURM account (default: xxqian)
  --mail-type TYPE         SLURM mail type (default: END)
  --mail-user USER         SLURM mail user (default: xxqian@stanford.edu)
  --verbose                Enable verbose logging
  --dry-run                Show what would be submitted without submitting

EXAMPLES:
  1. Submit all subjects with default settings:
     bash submit_NW_1st.sh

  2. Submit specific subjects:
     bash submit_NW_1st.sh --subjects sub-AOCD001,sub-AOCD002

  3. Submit with Schaefer 2018 atlas:
     bash submit_NW_1st.sh --atlas schaefer_2018 --atlas-params '{"n_rois": 400}'

  4. Submit with custom atlas:
     bash submit_NW_1st.sh --atlas /path/to/atlas.nii.gz --labels /path/to/labels.txt

  5. Submit with custom output directory:
     bash submit_NW_1st.sh --output-dir /custom/output/path

  6. Submit with custom SLURM parameters:
     bash submit_NW_1st.sh --time 4:00:00 --mem 32G --cpus 4

  7. Submit with verbose logging:
     bash submit_NW_1st.sh --verbose

  8. Dry run to see what would be submitted:
     bash submit_NW_1st.sh --dry-run

REQUIRED FILES:
---------------
- BIDS directory with fMRI data
- ROI directory with atlas files
- Container image (OCD.sif)
- Python script (NW_1st.py)

OUTPUT:
-------
- Individual subject FC matrices
- ROI-to-ROI connectivity files
- ROI-to-network connectivity files
- Network summary files

For more information, run with --usage.
EOF
}

print_usage() {
    cat << 'EOF'
DETAILED USAGE EXAMPLES
=======================

1. DEFAULT ANALYSIS (Power 2011 Atlas)
   ------------------------------------
   Submit all subjects using the default Power 2011 atlas:
   
   bash submit_NW_1st.sh
   
   This will:
   - Process all subjects in the BIDS directory
   - Use Power 2011 atlas with 264 ROIs
   - Submit jobs with 2:00:00 time limit and 16G memory
           - Save results to /scratch/xxqian/OCD

2. SPECIFIC SUBJECTS
   ------------------
   Submit only specific subjects:
   
   bash submit_NW_1st.sh --subjects sub-AOCD001,sub-AOCD002,sub-AOCD003
   
   This is useful for:
   - Testing the pipeline on a few subjects
   - Re-processing failed subjects
   - Processing new subjects incrementally

3. NILEARN BUILT-IN ATLASES
   -------------------------
   Submit using Nilearn's built-in atlases:
   
   # Schaefer 2018 (400 ROIs, 7 networks)
   bash submit_NW_1st.sh --atlas schaefer_2018 --atlas-params '{"n_rois": 400, "yeo_networks": 7}'
   
   # Harvard-Oxford (cortical regions)
   bash submit_NW_1st.sh --atlas harvard_oxford --atlas-params '{"atlas_name": "cort-maxprob-thr25-2mm"}'
   
   # AAL atlas
   bash submit_NW_1st.sh --atlas aal

4. CUSTOM ATLAS FILES
   -------------------
   Submit using custom atlas files:
   
   bash submit_NW_1st.sh \
     --atlas /path/to/custom_atlas.nii.gz \
     --labels /path/to/custom_labels.txt \
     --label-pattern simple \
     --atlas-name custom_atlas

5. CUSTOM OUTPUT PATHS
   --------------------
   Submit with custom output and work directories:
   
   bash submit_NW_1st.sh \
     --output-dir /scratch/user/custom_output \
     --work-dir /scratch/user/custom_work \
     --bids-dir /scratch/user/custom_bids

6. CUSTOM SLURM PARAMETERS
   -------------------------
   Submit with custom resource allocation:
   
   bash submit_NW_1st.sh \
     --time 4:00:00 \
     --mem 32G \
     --cpus 4 \
     --account custom_account

7. VERBOSE LOGGING
   ----------------
   Submit with detailed logging for debugging:
   
   bash submit_NW_1st.sh --verbose
   
   This enables:
   - Detailed SLURM job information
   - Python script verbose logging
   - More informative error messages

8. DRY RUN
   ---------
   See what would be submitted without actually submitting:
   
   bash submit_NW_1st.sh --dry-run
   
   This shows:
   - Which subjects would be processed
   - What SLURM parameters would be used
   - What commands would be executed

ATLAS TYPES AND PARAMETERS:
---------------------------
Built-in Nilearn Atlases:
- power_2011: Power 2011 atlas (264 ROIs, 13 networks)
- schaefer_2018: Schaefer 2018 parcellation (100-1000 ROIs, 7/17 networks)
- harvard_oxford: Harvard-Oxford cortical/subcortical atlases
- aal: Automated Anatomical Labeling atlas (116 ROIs)
- talairach: Talairach atlas (1107 ROIs)
- yeo_2011: Yeo 2011 network parcellation (7/17 networks)

Custom Atlases:
- File-based: Provide path to .nii.gz file and labels file
- Label patterns: simple, power, harvard_oxford, custom

SLURM RESOURCE RECOMMENDATIONS:
-------------------------------
- Small datasets (< 100 subjects): 2:00:00, 16G, 2 CPUs
- Medium datasets (100-500 subjects): 4:00:00, 32G, 4 CPUs
- Large datasets (> 500 subjects): 8:00:00, 64G, 8 CPUs

TROUBLESHOOTING:
----------------
1. Check that BIDS directory contains valid fMRI data
2. Verify ROI directory has required atlas files
3. Ensure container image exists and is accessible
4. Check SLURM account and partition access
5. Use --verbose for detailed error information
6. Use --dry-run to verify configuration before submission

COMMON SLURM ISSUES:
--------------------
1. "Failed to submit job" errors:
   - Check SLURM daemon status: squeue -u $USER
   - Verify account access: sacctmgr show account $SLURM_ACCOUNT
   - Check partition availability: sinfo
   - Verify resource limits are reasonable

2. "Permission denied" errors:
   - Ensure job script is executable: chmod +x job_script.sh
   - Check file permissions in work and output directories
   - Verify SLURM account has sufficient permissions

3. "Container not found" errors:
   - Verify container path: ls -la $CONTAINER
   - Check container file permissions
   - Ensure container is accessible from compute nodes

4. "Python script not found" errors:
   - Verify script exists: ls -la $PYTHON_SCRIPT
   - Check script path in container bind mounts
   - Ensure script is accessible from container

5. "Memory/CPU limit exceeded" errors:
   - Reduce SLURM resource requests: --mem 8G --cpus 1
   - Check cluster resource limits: sinfo -o "%20P %5D %14F"
   - Use smaller resource requests for testing

For more information, see the script help or run with --help.
EOF
}

print_quick_help() {
    cat << 'EOF'
QUICK HELP - SLURM Job Submission for NW_1st.py
================================================

BASIC USAGE:
  bash submit_NW_1st.sh [OPTIONS]

QUICK EXAMPLES:
  1. Default (Power 2011 Atlas):
     bash submit_NW_1st.sh

  2. Specific Subjects:
     bash submit_NW_1st.sh --subjects sub-AOCD001,sub-AOCD002

  3. Schaefer 2018 Atlas:
     bash submit_NW_1st.sh --atlas schaefer_2018 --atlas-params '{"n_rois": 400}'

  4. Custom Atlas:
     bash submit_NW_1st.sh --atlas /path/to/atlas.nii.gz --labels /path/to/labels.txt

HELP OPTIONS:
  --help          Show full help with all arguments
  --usage         Show detailed usage examples

For more information, run with --usage or --help.
EOF
}

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h|help)
            print_help
            exit 0
            ;;
        --usage)
            print_usage
            exit 0
            ;;
        --subjects)
            SUBJECTS="$2"
            shift 2
            ;;
        --atlas)
            ATLAS="$2"
            shift 2
            ;;
        --atlas-params)
            ATLAS_PARAMS="$2"
            shift 2
            ;;
        --labels)
            LABELS="$2"
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
        --atlas-name)
            ATLAS_NAME="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --slurm-scripts-dir)
            SLURM_SCRIPTS_DIR="$2"
            shift 2
            ;;
        --bids-dir)
            BIDS_DIR="$2"
            shift 2
            ;;
        --roi-dir)
            ROI_DIR="$2"
            shift 2
            ;;
        --time)
            SLURM_TIME="$2"
            shift 2
            ;;
        --mem)
            SLURM_MEM="$2"
            shift 2
            ;;
        --cpus)
            SLURM_CPUS="$2"
            shift 2
            ;;
        --account)
            SLURM_ACCOUNT="$2"
            shift 2
            ;;
        --mail-type)
            SLURM_MAIL_TYPE="$2"
            shift 2
            ;;
        --mail-user)
            SLURM_MAIL_USER="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Initialize variables with defaults
BIDS_DIR="/project/6079231/dliang55/R01_AOCD/derivatives/fmriprep-1.4.1"
OUTPUT_DIR="/scratch/xxqian/OCD"
WORK_DIR="/scratch/xxqian/work_flow"
SLURM_SCRIPTS_DIR="/scratch/xxqian/slurm_jobs"
ROI_DIR="/scratch/xxqian/roi"
CONTAINER="/scratch/xxqian/repo/image/OCD.sif"
PYTHON_SCRIPT="/scratch/xxqian/repo/OCD_RO1/NW_1st.py"
SLURM_TIME="4:00:00"
SLURM_MEM="16G"
SLURM_CPUS="2"
SLURM_ACCOUNT="xxqian"
SLURM_MAIL_TYPE="END"
SLURM_MAIL_USER="xxqian@stanford.edu"
ATLAS="power_2011"
LABELS=""
LABEL_PATTERN="power"
ATLAS_PARAMS=""
CUSTOM_REGEX=""
ATLAS_NAME=""
SUBJECTS=""
VERBOSE=""
DRY_RUN=false

# =============================================================================
# VALIDATION AND SETUP
# =============================================================================

# Check SLURM environment
echo "Checking SLURM environment..."
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Running inside SLURM job: $SLURM_JOB_ID"
    SLURM_MODE=true
else
    echo "Running outside SLURM (interactive mode)"
    SLURM_MODE=false
fi

# Check SLURM commands availability
echo "Checking SLURM commands..."
if command -v sbatch >/dev/null 2>&1; then
    echo "sbatch command available: $(which sbatch)"
    sbatch --version 2>/dev/null | head -1 || echo "Could not get sbatch version"
    SBATCH_AVAILABLE=true
else
    echo "ERROR: sbatch command not found"
    if [[ "$SLURM_MODE" == true ]]; then
        echo "Cannot run SLURM job submission without sbatch command"
        exit 1
    else
        echo "WARNING: sbatch not available - will only perform validation checks"
        SBATCH_AVAILABLE=false
    fi
fi

if command -v squeue >/dev/null 2>&1; then
    echo "squeue command available: $(which squeue)"
    SQUEUE_AVAILABLE=true
else
    echo "WARNING: squeue command not found"
    SQUEUE_AVAILABLE=false
fi

# Validate required directories
if [[ ! -d "$BIDS_DIR" ]]; then
    echo "Error: BIDS directory does not exist: $BIDS_DIR"
    exit 1
fi

if [[ ! -d "$ROI_DIR" ]]; then
    echo "Error: ROI directory does not exist: $ROI_DIR"
    exit 1
fi

if [[ ! -f "$CONTAINER" ]]; then
    echo "Error: Container image does not exist: $CONTAINER"
    exit 1
fi

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: Python script does not exist: $PYTHON_SCRIPT"
    exit 1
fi

# Create output and work directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$WORK_DIR"
mkdir -p "$SLURM_SCRIPTS_DIR"

# Check SLURM account and partition access
echo "Checking SLURM account access..."
if [[ -n "$SLURM_ACCOUNT" ]]; then
    echo "SLURM account: $SLURM_ACCOUNT"
    # Try to check account status (this might fail on some systems)
    if [[ "$SBATCH_AVAILABLE" == true ]]; then
        sacctmgr show account $SLURM_ACCOUNT 2>/dev/null | head -5 || echo "Could not check account status"
    else
        echo "Skipping account check (sbatch not available)"
    fi
else
    echo "WARNING: No SLURM account specified"
fi

echo "Checking available partitions..."
if [[ "$SBATCH_AVAILABLE" == true ]]; then
    sinfo -o "%20P %5D %14F" 2>/dev/null | head -10 || echo "Could not check partition status"
else
    echo "Skipping partition check (sbatch not available)"
fi

# =============================================================================
# SUBJECT DISCOVERY
# =============================================================================

# Get list of subjects
if [[ -z "$SUBJECTS" ]]; then
    echo "Discovering subjects in BIDS directory: $BIDS_DIR"
    
    # Find subjects and check if any were found
    found_subjects=$(find "$BIDS_DIR" -maxdepth 1 -type d -name "sub-*" 2>/dev/null | sort)
    
    if [[ -z "$found_subjects" ]]; then
        echo "Error: No subjects found in BIDS directory: $BIDS_DIR"
        echo "Please check that:"
        echo "  1. The BIDS directory exists and is accessible"
        echo "  2. The directory contains sub-* folders"
        echo "  3. You have read permissions for the directory"
        exit 1
    fi
    
    # Convert to comma-separated list
    SUBJECTS=$(echo "$found_subjects" | xargs -n 1 basename | tr '\n' ',' | sed 's/,$//')
    
    if [[ -z "$SUBJECTS" ]]; then
        echo "Error: Failed to process subject names from: $found_subjects"
        exit 1
    fi
    
    echo "Found subjects: $SUBJECTS"
fi

# Convert comma-separated list to array
IFS=',' read -ra SUBJECT_ARRAY <<< "$SUBJECTS"

# =============================================================================
# JOB SUBMISSION
# =============================================================================

# Initialize counters
SUCCESSFUL_SUBMISSIONS=0
FAILED_SUBMISSIONS=0
SKIPPED_SUBMISSIONS=0

echo "=" * 80
echo "Submitting NW_1st.py jobs to SLURM"
echo "=" * 80
echo "BIDS Directory: $BIDS_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Work Directory: $WORK_DIR"
echo "SLURM Scripts Directory: $SLURM_SCRIPTS_DIR"
echo "ROI Directory: $ROI_DIR"
echo "Atlas: $ATLAS"
if [[ -n "$ATLAS_PARAMS" ]]; then
    echo "Atlas Parameters: $ATLAS_PARAMS"
fi
if [[ -n "$LABELS" ]]; then
    echo "Labels: $LABELS"
fi
echo "Label Pattern: $LABEL_PATTERN"
if [[ -n "$CUSTOM_REGEX" ]]; then
    echo "Custom Regex: $CUSTOM_REGEX"
fi
if [[ -n "$ATLAS_NAME" ]]; then
    echo "Atlas Name: $ATLAS_NAME"
fi
echo "SLURM Time: $SLURM_TIME"
echo "SLURM Memory: $SLURM_MEM"
echo "SLURM CPUs: $SLURM_CPUS"
echo "Subjects: $SUBJECTS"
echo "=" * 80

# Submit jobs for each subject
for subject in "${SUBJECT_ARRAY[@]}"; do
    echo "Processing subject: $subject"
    
    # Create subject-specific work directory
    subject_work_dir="$WORK_DIR/$subject"
    mkdir -p "$subject_work_dir"
    
    # Build Python command
    python_cmd="python /scripts/NW_1st.py --subject $subject --atlas $ATLAS --label-pattern $LABEL_PATTERN"
    
    if [[ -n "$ATLAS_PARAMS" ]]; then
        python_cmd="$python_cmd --atlas-params '$ATLAS_PARAMS'"
    fi
    
    if [[ -n "$LABELS" ]]; then
        python_cmd="$python_cmd --labels $LABELS"
    fi
    
    if [[ -n "$CUSTOM_REGEX" ]]; then
        python_cmd="$python_cmd --custom-regex '$CUSTOM_REGEX'"
    fi
    
    if [[ -n "$ATLAS_NAME" ]]; then
        python_cmd="$python_cmd --atlas-name $ATLAS_NAME"
    fi
    
    if [[ -n "$VERBOSE" ]]; then
        python_cmd="$python_cmd $VERBOSE"
    fi
    
    # Create SLURM job script
    job_script="$SLURM_SCRIPTS_DIR/submit_${subject}.sh"
    
    cat > "$job_script" << EOF
#!/bin/bash
#SBATCH --job-name=NW_1st_${subject}
#SBATCH --output=${subject_work_dir}/%j.out
#SBATCH --error=${subject_work_dir}/%j.err
#SBATCH --time=${SLURM_TIME}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --mail-type=${SLURM_MAIL_TYPE}
#SBATCH --mail-user=${SLURM_MAIL_USER}

# Job information
echo "Job started at: \$(date)"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Subject: $subject"
echo "Working directory: \$SLURM_SUBMIT_DIR"

# Load modules (if needed)
# module load fsl

# Run the analysis
echo "Starting NW_1st.py analysis for $subject"
apptainer exec \\
  --bind $BIDS_DIR:/input \\
  --bind $OUTPUT_DIR:/output \\
  --bind $subject_work_dir:/work \\
  --bind $ROI_DIR:/roi \\
  --bind /scratch/xxqian/repo/OCD_RO1:/scripts \\
  $CONTAINER \\
  $python_cmd

# Check exit status
exit_code=\$?
if [[ \$exit_code -eq 0 ]]; then
    echo "Job completed successfully at: \$(date)"
else
    echo "Job failed with exit code \$exit_code at: \$(date)"
fi

exit \$exit_code
EOF

    # Make job script executable
    chmod +x "$job_script"
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "DRY RUN: Would submit job for $subject"
        echo "Job script: $job_script"
        echo "Python command: $python_cmd"
        echo "---"
    else
        # Validate job script
        if ! validate_job_script "$job_script" "$subject"; then
            echo "Skipping job for $subject due to validation errors."
            ((SKIPPED_SUBMISSIONS++))
            continue
        fi

        # Validate SLURM environment
        if [[ "$SBATCH_AVAILABLE" == true ]]; then
            if ! check_slurm_issues "$subject" "$job_script"; then
                echo "Skipping job for $subject due to SLURM environment issues."
                ((SKIPPED_SUBMISSIONS++))
                continue
            fi
        else
            echo "Skipping SLURM environment check (sbatch not available)"
        fi

        # Submit job to SLURM
        echo "Submitting job for $subject..."
        
        if [[ "$SBATCH_AVAILABLE" == false ]]; then
            echo "ERROR: Cannot submit job - sbatch command not available"
            echo "This script requires SLURM to be installed and accessible"
            echo "You can use --dry-run to validate the configuration without submitting jobs"
            ((FAILED_SUBMISSIONS++))
            continue
        fi
        
        # Submit the job and capture output
        sbatch_output=$(sbatch "$job_script" 2>&1)
        sbatch_exit_code=$?
        
        if [[ $sbatch_exit_code -eq 0 ]]; then
            # Extract job ID from successful submission
            job_id=$(echo "$sbatch_output" | awk '{print $4}')
            if [[ -n "$job_id" ]]; then
                echo "Submitted job ID: $job_id for $subject"
                ((SUCCESSFUL_SUBMISSIONS++))
            else
                echo "Warning: Could not extract job ID from sbatch output: $sbatch_output"
                ((FAILED_SUBMISSIONS++))
            fi
        else
            echo "ERROR: Failed to submit job for $subject"
            echo "sbatch exit code: $sbatch_exit_code"
            echo "sbatch output: $sbatch_output"
            echo "Job script: $job_script"
            echo "Python command: $python_cmd"
            
            # Check if the job script exists and is executable
            if [[ -f "$job_script" ]]; then
                echo "Job script exists and is executable"
                echo "Job script permissions: $(ls -la "$job_script")"
            else
                echo "ERROR: Job script does not exist: $job_script"
            fi
            
            # Check SLURM status
            echo "Checking SLURM status..."
            squeue -u $USER 2>/dev/null || echo "Could not check SLURM queue"
            
            # Continue with next subject instead of exiting
            echo "Continuing with next subject..."
            ((FAILED_SUBMISSIONS++))
            continue
        fi
        
        # Wait a moment between submissions to avoid overwhelming the scheduler
        sleep 1
    fi
done

if [[ "$DRY_RUN" == true ]]; then
    echo "DRY RUN COMPLETED"
    echo "No jobs were actually submitted to SLURM"
    echo "Review the configuration above and run without --dry-run to submit jobs"
else
    echo "=" * 80
    echo "Job submission completed"
    echo "=" * 80
    echo "Summary:"
    echo "  Total subjects processed: ${#SUBJECT_ARRAY[@]}"
    echo "  Successful submissions: $SUCCESSFUL_SUBMISSIONS"
    echo "  Failed submissions: $FAILED_SUBMISSIONS"
    echo "  Skipped submissions: $SKIPPED_SUBMISSIONS"
    echo "=" * 80
    
    if [[ $SUCCESSFUL_SUBMISSIONS -gt 0 ]]; then
        echo "Check job status with: squeue -u $USER"
        echo "Monitor jobs with: watch -n 10 'squeue -u $USER'"
        echo "Check logs in: $WORK_DIR/*/slurm-*.out"
    fi
    
    if [[ $FAILED_SUBMISSIONS -gt 0 ]]; then
        echo "WARNING: $FAILED_SUBMISSIONS jobs failed to submit"
        echo "Check the error messages above for details"
        echo "You may need to fix issues before re-running"
    fi
    
    if [[ $SKIPPED_SUBMISSIONS -gt 0 ]]; then
        echo "NOTE: $SKIPPED_SUBMISSIONS subjects were skipped"
        echo "Check the validation messages above for details"
    fi
fi