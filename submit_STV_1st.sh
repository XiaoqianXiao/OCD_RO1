#!/bin/bash
# =============================================================================
# SLURM Job Submission Script for STV_1st.py
# =============================================================================
# 
# This script submits individual subject/session Seed-to-Voxel functional connectivity
# analysis jobs to the SLURM queue using STV_1st.py. It processes seed-to-voxel
# connectivity maps for resting-state fMRI data using a PCC seed.
#
# USAGE:
#   bash submit_STV_1st.sh [OPTIONS]
#
# EXAMPLES:
#   1. Submit all subjects with default settings:
#      bash submit_STV_1st.sh
#
#   2. Submit specific subjects:
#      bash submit_STV_1st.sh --subjects sub-AOCD001,sub-AOCD002
#
#   3. Submit with custom output directory:
#      bash submit_STV_1st.sh --output-dir /custom/output/path
#
#   4. Submit with verbose logging:
#      bash submit_STV_1st.sh --verbose
#
#   5. Submit with custom SLURM parameters:
#      bash submit_STV_1st.sh --time 4:00:00 --mem 32G --cpus 4
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

# Default directories
BIDS_DIR="/project/6079231/dliang55/R01_AOCD/derivatives/fmriprep-1.4.1"
OUTPUT_DIR="/scratch/xxqian/OCD"
WORK_DIR="/scratch/xxqian/work_flow"
ROI_DIR="/scratch/xxqian/roi"

# Default SLURM parameters
SLURM_TIME="2:00:00"
SLURM_MEM="16G"
SLURM_CPUS="2"
SLURM_ACCOUNT="xxqian"
SLURM_MAIL_TYPE="END"
SLURM_MAIL_USER="xxqian@stanford.edu"

# Default analysis parameters
VERBOSE=""

# Container and Python script
CONTAINER="/scratch/xxqian/repo/image/OCD.sif"
PYTHON_SCRIPT="/scratch/xxqian/repo/OCD_RO1/STV_1st.py"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

print_help() {
    cat << 'EOF'
SLURM Job Submission Script for STV_1st.py
===========================================

This script submits individual subject/session Seed-to-Voxel functional connectivity
analysis jobs to the SLURM queue using STV_1st.py. It processes seed-to-voxel
connectivity maps for resting-state fMRI data using a PCC seed.

The analysis computes connectivity between a seed region (Posterior Cingulate Cortex)
and all voxels in the brain, creating whole-brain connectivity maps for each subject.

BASIC USAGE:
  bash submit_STV_1st.sh [OPTIONS]

OPTIONS:
  --help                    Show this help message
  --usage                   Show detailed usage examples
  --subjects SUBJECTS       Comma-separated list of subjects (default: all)
          --output-dir DIR          Output directory (default: /scratch/xxqian/OCD)
        --work-dir DIR            Work directory (default: /scratch/xxqian/work_flow)
          --bids-dir DIR            BIDS directory (default: /project/6079231/dliang55/R01_AOCD/derivatives/fmriprep-1.4.1)
          --roi-dir DIR             ROI directory (default: /scratch/xxqian/roi)
  --time TIME               SLURM time limit (default: 2:00:00)
  --mem MEM                 SLURM memory limit (default: 16G)
  --cpus CPUS               SLURM CPUs per task (default: 2)
  --account ACCOUNT         SLURM account (default: xxqian)
  --mail-type TYPE          SLURM mail type (default: END)
  --mail-user USER          SLURM mail user (default: xxqian@stanford.edu)
  --verbose                 Enable verbose logging
  --dry-run                 Show what would be submitted without submitting

EXAMPLES:
  1. Submit all subjects with default settings:
     bash submit_STV_1st.sh

  2. Submit specific subjects:
     bash submit_STV_1st.sh --subjects sub-AOCD001,sub-AOCD002

  3. Submit with custom output directory:
     bash submit_STV_1st.sh --output-dir /custom/output/path

  4. Submit with custom SLURM parameters:
     bash submit_STV_1st.sh --time 4:00:00 --mem 32G --cpus 4

  5. Submit with verbose logging:
     bash submit_STV_1st.sh --verbose

  6. Dry run to see what would be submitted:
     bash submit_STV_1st.sh --dry-run

REQUIRED FILES:
---------------
- BIDS directory with fMRI data
- ROI directory with seed region files
- Container image (OCD.sif)
- Python script (STV_1st.py)

OUTPUT:
-------
- Individual subject seed-to-voxel connectivity maps
- Z-score normalized connectivity maps
- Correlation coefficient maps
- Statistical significance maps

For more information, run with --usage.
EOF
}

print_usage() {
    cat << 'EOF'
DETAILED USAGE EXAMPLES
=======================

1. DEFAULT ANALYSIS (PCC Seed-to-Voxel)
   -------------------------------------
   Submit all subjects using the default PCC seed region:
   
   bash submit_STV_1st.sh
   
   This will:
   - Process all subjects in the BIDS directory
   - Use PCC seed region for connectivity analysis
   - Submit jobs with 2:00:00 time limit and 16G memory
   - Save results to /scratch/xxqian/OCD

2. SPECIFIC SUBJECTS
   ------------------
   Submit only specific subjects:
   
   bash submit_STV_1st.sh --subjects sub-AOCD001,sub-AOCD002,sub-AOCD003
   
   This is useful for:
   - Testing the pipeline on a few subjects
   - Re-processing failed subjects
   - Processing new subjects incrementally

3. CUSTOM OUTPUT PATHS
   --------------------
   Submit with custom output and work directories:
   
   bash submit_STV_1st.sh \
     --output-dir /scratch/user/custom_output \
     --work-dir /scratch/user/custom_work \
     --bids-dir /scratch/user/custom_bids

4. CUSTOM SLURM PARAMETERS
   -------------------------
   Submit with custom resource allocation:
   
   bash submit_STV_1st.sh \
     --time 4:00:00 \
     --mem 32G \
     --cpus 4 \
     --account custom_account

5. VERBOSE LOGGING
   ----------------
   Submit with detailed logging for debugging:
   
   bash submit_STV_1st.sh --verbose
   
   This enables:
   - Detailed SLURM job information
   - Python script verbose logging
   - More informative error messages

6. DRY RUN
   ---------
   See what would be submitted without actually submitting:
   
   bash submit_STV_1st.sh --dry-run
   
   This shows:
   - Which subjects would be processed
   - What SLURM parameters would be used
   - What commands would be executed

SEED-TO-VOXEL ANALYSIS FEATURES:
--------------------------------
The STV analysis includes:
- PCC seed region definition and extraction
- Whole-brain voxel-wise connectivity computation
- Correlation coefficient calculation
- Z-score normalization
- Statistical significance testing
- Multi-run averaging and quality control

SLURM RESOURCE RECOMMENDATIONS:
-------------------------------
- Small datasets (< 100 subjects): 2:00:00, 16G, 2 CPUs
- Medium datasets (100-500 subjects): 4:00:00, 32G, 4 CPUs
- Large datasets (> 500 subjects): 8:00:00, 64G, 8 CPUs

TROUBLESHOOTING:
----------------
1. Check that BIDS directory contains valid fMRI data
2. Verify ROI directory has required seed region files
3. Ensure container image exists and is accessible
4. Check SLURM account and partition access
5. Use --verbose for detailed error information
6. Use --dry-run to verify configuration before submission

For more information, see the script help or run with --help.
EOF
}

print_quick_help() {
    cat << 'EOF'
QUICK HELP - SLURM Job Submission for STV_1st.py
=================================================

BASIC USAGE:
  bash submit_STV_1st.sh [OPTIONS]

QUICK EXAMPLES:
  1. Default (PCC Seed-to-Voxel):
     bash submit_STV_1st.sh

  2. Specific Subjects:
     bash submit_STV_1st.sh --subjects sub-AOCD001,sub-AOCD002

  3. Custom Output Directory:
     bash submit_STV_1st.sh --output-dir /custom/output/path

  4. Verbose Logging:
     bash submit_STV_1st.sh --verbose

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
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
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

# Set default values for unset variables
: ${SUBJECTS:=""}
: ${DRY_RUN:=false}

# =============================================================================
# VALIDATION AND SETUP
# =============================================================================

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

# =============================================================================
# SUBJECT DISCOVERY
# =============================================================================

# Get list of subjects
if [[ -z "$SUBJECTS" ]]; then
    echo "Discovering subjects in BIDS directory: $BIDS_DIR"
    SUBJECTS=$(find "$BIDS_DIR" -maxdepth 1 -type d -name "sub-*" | sort | xargs -n 1 basename | tr '\n' ',' | sed 's/,$//')
    if [[ -z "$SUBJECTS" ]]; then
        echo "Error: No subjects found in BIDS directory: $BIDS_DIR"
        exit 1
    fi
    echo "Found subjects: $SUBJECTS"
fi

# Convert comma-separated list to array
IFS=',' read -ra SUBJECT_ARRAY <<< "$SUBJECTS"

# =============================================================================
# JOB SUBMISSION
# =============================================================================

echo "=" * 80
echo "Submitting STV_1st.py jobs to SLURM"
echo "=" * 80
echo "BIDS Directory: $BIDS_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Work Directory: $WORK_DIR"
echo "ROI Directory: $ROI_DIR"
echo "Analysis Type: Seed-to-Voxel (PCC seed)"
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
    python_cmd="python /scripts/STV_1st.py --subject $subject"
    
    if [[ -n "$VERBOSE" ]]; then
        python_cmd="$python_cmd $VERBOSE"
    fi
    
    # Create SLURM job script
    job_script="$subject_work_dir/submit_${subject}.sh"
    
    cat > "$job_script" << EOF
#!/bin/bash
#SBATCH --job-name=STV_1st_${subject}
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
echo "Starting STV_1st.py analysis for $subject"
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
        # Submit job to SLURM
        echo "Submitting job for $subject..."
        job_id=$(sbatch "$job_script" | awk '{print $4}')
        echo "Submitted job ID: $job_id for $subject"
        
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
    echo "Submitted $((${#SUBJECT_ARRAY[@]})) jobs to SLURM"
    echo "Check job status with: squeue -u $USER"
    echo "Monitor jobs with: watch -n 10 'squeue -u $USER'"
    echo "Check logs in: $WORK_DIR/*/slurm-*.out"
fi