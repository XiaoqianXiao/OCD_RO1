#!/bin/bash
#
# This script dynamically generates and submits SLURM jobs for ROI-to-network
# functional connectivity group analysis using NW_group.py. It creates
# atlas-specific SLURM scripts and job names.
#
# =============================================================================
# USAGE:
#   bash submit_NW_group.sh [OPTIONS]
#
# EXAMPLES:
#   1. Run with default Power 2011 atlas (auto-detected):
#      bash submit_NW_group.sh
#
#   2. Run with specific Power 2011 atlas:
#      bash submit_NW_group.sh --atlas power_2011
#
#   3. Run with Schaefer 2018 atlas (400 ROIs, 7 networks):
#      bash submit_NW_group.sh --atlas schaefer_2018 --atlas-params '{"n_rois": 400, "yeo_networks": 7}'
#
#   4. Run with Harvard-Oxford atlas (anatomical):
#      bash submit_NW_group.sh --atlas harvard_oxford --atlas-params '{"atlas_name": "cort-maxprob-thr25-2mm"}'
#
#   5. Run with YEO 2011 atlas (7 networks):
#      bash submit_NW_group.sh --atlas yeo_2011 --atlas-params '{"n_networks": 7, "thickness": "thick"}'
#
# OPTIONS:
#   --atlas ATLAS                 Atlas name (default: power_2011)
#   --atlas-params PARAMS         JSON string of atlas parameters
#   --subjects-csv FILE           Subjects CSV file (default: group.csv)
#   --clinical-csv FILE           Clinical CSV file (default: clinical.csv)
#   --input-dir DIR               Input directory (default: /project/6079231/dliang55/R01_AOCD)
#   --output-dir DIR              Output directory (default: /project/6079231/dliang55/R01_AOCD/NW_group)
#   --min-subjects N              Minimum subjects per group (default: 2)
#   --significance-threshold P    Significance threshold (default: 0.05)
#   --no-fdr                      Disable FDR correction
#   --verbose                     Enable verbose logging
#   --help                        Show this usage information
# =============================================================================

# ----------------------------
# Default values
# ----------------------------
ATLAS="power_2011"
ATLAS_PARAMS=""
SUBJECTS_CSV="group.csv"
CLINICAL_CSV="clinical.csv"
INPUT_DIR="/project/6079231/dliang55/R01_AOCD"
OUTPUT_DIR="/project/6079231/dliang55/R01_AOCD/NW_group"
MIN_SUBJECTS=2
SIGNIFICANCE_THRESHOLD=0.05
FDR="--fdr"
VERBOSE=""

CONTAINER="/scratch/xxqian/repo/OCD_RO1/OCD.sif"
JOB_DIR="/home/xxqian/scratch/slurm_jobs"
mkdir -p "$JOB_DIR"

# ----------------------------
# Parse arguments
# ----------------------------
print_help() {
  grep '^# ' "$0" | sed 's/^# //'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --atlas) ATLAS="$2"; shift 2 ;;
    --atlas-params) ATLAS_PARAMS="$2"; shift 2 ;;
    --subjects-csv) SUBJECTS_CSV="$2"; shift 2 ;;
    --clinical-csv) CLINICAL_CSV="$2"; shift 2 ;;
    --input-dir) INPUT_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --min-subjects) MIN_SUBJECTS="$2"; shift 2 ;;
    --significance-threshold) SIGNIFICANCE_THRESHOLD="$2"; shift 2 ;;
    --no-fdr) FDR=""; shift ;;
    --verbose) VERBOSE="--verbose"; shift ;;
    --help) print_help ;;
    *) echo "Unknown option: $1"; print_help ;;
  esac
done

# ----------------------------
# Generate SLURM job script
# ----------------------------
JOB_SCRIPT="$JOB_DIR/NWGroup_${ATLAS}_$(date +%s).slurm"

cat <<EOT > "$JOB_SCRIPT"
#!/bin/bash
#SBATCH --job-name=NWGroup_${ATLAS}
#SBATCH --output=${JOB_DIR}/NWGroup_${ATLAS}_%j.out
#SBATCH --error=${JOB_DIR}/NWGroup_${ATLAS}_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xxqian@uw.edu

echo "Job \$SLURM_JOB_ID started on \$(date)"

CMD="apptainer exec $CONTAINER python NW_group.py \
  --subjects_csv $SUBJECTS_CSV \
  --clinical_csv $CLINICAL_CSV \
  --input_dir $INPUT_DIR \
  --output_dir $OUTPUT_DIR \
  --min_subjects $MIN_SUBJECTS \
  --significance_threshold $SIGNIFICANCE_THRESHOLD \
  $FDR $VERBOSE"

if [ -n "$ATLAS" ]; then
  CMD+=" --atlas_name $ATLAS"
fi
if [ -n "$ATLAS_PARAMS" ]; then
  CMD+=" --atlas_params '$ATLAS_PARAMS'"
fi

echo "Running command: \$CMD"
eval "\$CMD"
EXIT_STATUS=\$?

if [ \$EXIT_STATUS -eq 0 ]; then
  echo "Job \$SLURM_JOB_ID completed successfully."
else
  echo "Job \$SLURM_JOB_ID failed with exit status \$EXIT_STATUS."
fi

echo "Job finished on \$(date)"
exit \$EXIT_STATUS
EOT

# ----------------------------
# Submit the job
# ----------------------------
sbatch "$JOB_SCRIPT"
echo "Submitted job script: $JOB_SCRIPT"
