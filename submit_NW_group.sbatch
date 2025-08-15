#!/bin/bash
# =============================================================================
# SLURM Batch Job Script for NW_group.py
# =============================================================================
# 
# This script runs ROI-to-network functional connectivity group analysis using
# NW_group.py. It performs group-level statistical analysis on ROI-to-network
# FC data, comparing healthy controls vs. OCD patients and performing
# longitudinal analyses.
#
# USAGE:
#   sbatch submit_NW_group.sbatch [OPTIONS]
#
# EXAMPLES:
#   1. Run with default settings:
#      sbatch submit_NW_group.sbatch
#
#   2. Run with custom input directory:
#      sbatch submit_NW_group.sbatch --input-dir /custom/input/path
#
#   3. Run with custom output directory:
#      sbatch submit_NW_group.sbatch --output-dir /custom/output/path
#
#   4. Run with specific atlas name:
#      sbatch submit_NW_group.sbatch --atlas-name schaefer_2018_400_7_2
#
#   5. Run with auto-detect atlas:
#      sbatch submit_NW_group.sbatch --auto-detect-atlas
#
#   6. Run with verbose logging:
#      sbatch submit_NW_group.sbatch --verbose
#
# HELP OPTIONS:
#   --help          Show this help message
#   --usage         Show detailed usage examples
#
# For more information, run with --usage or --help.
# =============================================================================

#SBATCH --job-name=NW_group_analysis
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --account=xxqian
#SBATCH --mail-type=END
#SBATCH --mail-user=xxqian@stanford.edu

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default directories and files
DEFAULT_INPUT_DIR="/scratch/xxqian/OCD"
DEFAULT_OUTPUT_DIR="/scratch/xxqian/OCD/NW_group"
DEFAULT_SUBJECTS_CSV="/scratch/xxqian/OCD/group.csv"
DEFAULT_CLINICAL_CSV="/scratch/xxqian/OCD/clinical.csv"

# Default analysis parameters
DEFAULT_ATLAS_NAME=""
AUTO_DETECT_ATLAS=""
VERBOSE=""

# Container and Python script
CONTAINER="/scratch/xxqian/repo/image/OCD.sif"
PYTHON_SCRIPT="NW_group.py"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

print_help() {
    cat << 'EOF'
SLURM Batch Job Script for NW_group.py
=======================================

This script runs ROI-to-network functional connectivity group analysis using
NW_group.py. It performs group-level statistical analysis on ROI-to-network
FC data, comparing healthy controls vs. OCD patients and performing
longitudinal analyses.

BASIC USAGE:
  sbatch submit_NW_group.sbatch [OPTIONS]

OPTIONS:
  --help                    Show this help message
  --usage                   Show detailed usage examples
  --input-dir DIR           Input directory with FC data (default: /scratch/xxqian/OCD)
  --output-dir DIR          Output directory for results (default: /scratch/xxqian/OCD/NW_group)
  --subjects-csv FILE       Group metadata CSV file (default: /scratch/xxqian/OCD/group.csv)
  --clinical-csv FILE       Clinical data CSV file (default: /scratch/xxqian/OCD/clinical.csv)
  --atlas-name NAME         Specific atlas name to use
  --auto-detect-atlas       Automatically detect atlas name from input files
  --verbose                 Enable verbose logging

EXAMPLES:
  1. Run with default settings:
     sbatch submit_NW_group.sbatch

  2. Run with custom input directory:
     sbatch submit_NW_group.sbatch --input-dir /custom/input/path

  3. Run with custom output directory:
     sbatch submit_NW_group.sbatch --output-dir /custom/output/path

  4. Run with specific atlas name:
     sbatch submit_NW_group.sbatch --atlas-name schaefer_2018_400_7_2

  5. Run with auto-detect atlas:
     sbatch submit_NW_group.sbatch --auto-detect-atlas

  6. Run with verbose logging:
     sbatch submit_NW_group.sbatch --verbose

REQUIRED FILES:
---------------
- Input directory with FC data files from NW_1st.py
- Group metadata CSV with 'sub' and 'group' columns
- Clinical data CSV with 'sub', 'ybocs_baseline', 'ybocs_followup' columns
- Container image (OCD.sif)
- Python script (NW_group.py)

OUTPUT:
-------
- Group difference analysis results
- Baseline FC vs symptom change regression
- FC change vs symptom change regression
- Statistical analysis reports

For more information, run with --usage.
EOF
}

print_usage() {
    cat << 'EOF'
DETAILED USAGE EXAMPLES
=======================

1. DEFAULT ANALYSIS
   -----------------
   Run with default settings using Power 2011 atlas:
   
   sbatch submit_NW_group.sbatch
   
   This will:
   - Use default input directory: /scratch/xxqian/OCD
   - Use default output directory: /scratch/xxqian/OCD/NW_group
   - Use default group and clinical CSV files
   - Use default atlas name (Power 2011)
   - Submit job with 4:00:00 time limit and 32G memory

2. CUSTOM INPUT DIRECTORY
   ------------------------
   Run with custom input directory:
   
   sbatch submit_NW_group.sbatch --input-dir /scratch/user/custom_fc_data
   
   This is useful when:
   - FC data is stored in a different location
   - You want to analyze data from a different experiment
   - You have multiple FC datasets to compare

3. CUSTOM OUTPUT DIRECTORY
   -------------------------
   Run with custom output directory:
   
   sbatch submit_NW_group.sbatch --output-dir /scratch/user/custom_results
   
   This is useful when:
   - You want to organize results by experiment
   - You have limited space in the default directory
   - You want to share results with other users

4. SPECIFIC ATLAS NAME
   ---------------------
   Run with a specific atlas name:
   
   sbatch submit_NW_group.sbatch --atlas-name schaefer_2018_400_7_2
   
   This is useful when:
   - You know the exact atlas name from NW_1st.py output
   - You want to ensure consistency across analyses
   - You're comparing results from different atlases

5. AUTO-DETECT ATLAS
   -------------------
   Automatically detect atlas name from input files:
   
   sbatch submit_NW_group.sbatch --auto-detect-atlas
   
   This is useful when:
   - You're not sure what atlas was used
   - You want to avoid manual atlas name specification
   - You're processing data from different sources

6. VERBOSE LOGGING
   ----------------
   Enable detailed logging for debugging:
   
   sbatch submit_NW_group.sbatch --verbose
   
   This enables:
   - Detailed Python script logging
   - More informative error messages
   - Better debugging information

7. COMBINED OPTIONS
   -----------------
   Combine multiple options for custom analysis:
   
   sbatch submit_NW_group.sbatch \\
     --input-dir /scratch/user/custom_fc_data \\
     --output-dir /scratch/user/custom_results \\
     --auto-detect-atlas \\
     --verbose

ANALYSIS TYPES:
---------------
The script performs several types of analysis:

1. Group Difference Analysis:
   - Compares HC vs OCD groups at baseline
   - Uses t-tests with FDR correction
   - Identifies significantly different connections

2. Baseline FC vs Symptom Change:
   - Correlates baseline FC with YBOCS change
   - Uses linear regression analysis
   - Identifies predictive connections

3. FC Change vs Symptom Change:
   - Correlates FC change with symptom change
   - Uses longitudinal data
   - Identifies treatment-responsive connections

SLURM RESOURCE RECOMMENDATIONS:
-------------------------------
- Small datasets (< 100 subjects): 2:00:00, 16G, 4 CPUs
- Medium datasets (100-500 subjects): 4:00:00, 32G, 8 CPUs
- Large datasets (> 500 subjects): 8:00:00, 64G, 16 CPUs

TROUBLESHOOTING:
----------------
1. Check that input directory contains valid FC data files
2. Verify group and clinical CSV files have correct format
3. Ensure container image exists and is accessible
4. Check SLURM account and partition access
5. Use --verbose for detailed error information
6. Verify atlas naming consistency with NW_1st.py output

For more information, see the script help or run with --help.
EOF
}

print_quick_help() {
    cat << 'EOF'
QUICK HELP - SLURM Batch Job for NW_group.py
=============================================

BASIC USAGE:
  sbatch submit_NW_group.sbatch [OPTIONS]

QUICK EXAMPLES:
  1. Default Analysis:
     sbatch submit_NW_group.sbatch

  2. Custom Input Directory:
     sbatch submit_NW_group.sbatch --input-dir /custom/input/path

  3. Auto-detect Atlas:
     sbatch submit_NW_group.sbatch --auto-detect-atlas

  4. Verbose Logging:
     sbatch submit_NW_group.sbatch --verbose

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
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
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
        --atlas-name)
            ATLAS_NAME="$2"
            shift 2
            ;;
        --auto-detect-atlas)
            AUTO_DETECT_ATLAS="--auto-detect-atlas"
            shift
            ;;
        --verbose)
            VERBOSE="--verbose"
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
: ${INPUT_DIR:="$DEFAULT_INPUT_DIR"}
: ${OUTPUT_DIR:="$DEFAULT_OUTPUT_DIR"}
: ${SUBJECTS_CSV:="$DEFAULT_SUBJECTS_CSV"}
: ${CLINICAL_CSV:="$DEFAULT_CLINICAL_CSV"}
: ${ATLAS_NAME:="$DEFAULT_ATLAS_NAME"}

# =============================================================================
# VALIDATION AND SETUP
# =============================================================================

# Validate required directories and files
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

if [[ ! -f "$SUBJECTS_CSV" ]]; then
    echo "Error: Subjects CSV file does not exist: $SUBJECTS_CSV"
    exit 1
fi

if [[ ! -f "$CLINICAL_CSV" ]]; then
    echo "Error: Clinical CSV file does not exist: $CLINICAL_CSV"
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

# Create output directory
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# JOB EXECUTION
# =============================================================================

echo "=" * 80
echo "Starting NW_group.py Analysis"
echo "=" * 80
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Input Directory: $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Subjects CSV: $SUBJECTS_CSV"
echo "Clinical CSV: $CLINICAL_CSV"
if [[ -n "$ATLAS_NAME" ]]; then
    echo "Atlas Name: $ATLAS_NAME"
fi
if [[ -n "$AUTO_DETECT_ATLAS" ]]; then
    echo "Auto-detect Atlas: Enabled"
fi
if [[ -n "$VERBOSE" ]]; then
    echo "Verbose Logging: Enabled"
fi
echo "SLURM Time: $SLURM_JOB_PARTITION"
echo "SLURM Memory: $SLURM_MEM_PER_NODE"
echo "SLURM CPUs: $SLURM_CPUS_PER_TASK"
echo "=" * 80

# Build Python command
python_cmd="python $PYTHON_SCRIPT --subjects_csv $SUBJECTS_CSV --clinical_csv $CLINICAL_CSV --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR"

if [[ -n "$ATLAS_NAME" ]]; then
    python_cmd="$python_cmd --atlas_name $ATLAS_NAME"
fi

if [[ -n "$AUTO_DETECT_ATLAS" ]]; then
    python_cmd="$python_cmd $AUTO_DETECT_ATLAS"
fi

if [[ -n "$VERBOSE" ]]; then
    python_cmd="$python_cmd $VERBOSE"
fi

echo "Executing: $python_cmd"
echo "=" * 80

# Run the analysis
apptainer exec \
  --bind $INPUT_DIR:/input \
  --bind $OUTPUT_DIR:/output \
  --bind $(pwd):/scripts \
  $CONTAINER \
  $python_cmd

# Check exit status
exit_code=$?
if [[ $exit_code -eq 0 ]]; then
    echo "=" * 80
    echo "Analysis completed successfully at: $(date)"
    echo "Results saved to: $OUTPUT_DIR"
    echo "=" * 80
else
    echo "=" * 80
    echo "Analysis failed with exit code $exit_code at: $(date)"
    echo "Check the error logs above for details"
    echo "=" * 80
    exit $exit_code
fi