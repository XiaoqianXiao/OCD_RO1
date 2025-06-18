#!/bin/bash

# SLURM job submission script for running NW_1st.py for each subject

# Define directories
BIDS_DIR="/project/6079231/dliang55/R01_AOCD/derivatives/fmriprep-1.4.1"
OUTPUT_DIR="/scratch/xxqian/OCD"
CONTAINER_PATH="$(pwd)/run_roi_to_roi.sif"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if container exists
if [ ! -f "$CONTAINER_PATH" ]; then
    echo "Error: Apptainer container $CONTAINER_PATH not found"
    exit 1
fi

# Get list of subjects from BIDS derivatives directory
SUBJECTS=($(ls -d $BIDS_DIR/sub-* | xargs -n 1 basename | grep '^sub-'))
if [ ${#SUBJECTS[@]} -eq 0 ]; then
    echo "Error: No subjects found in $BIDS_DIR"
    exit 1
fi

# Create a temporary directory for SLURM job scripts
TEMP_JOB_DIR="$OUTPUT_DIR/slurm_jobs"
mkdir -p "$TEMP_JOB_DIR"

# Loop over subjects and submit a job for each
for SUBJECT in "${SUBJECTS[@]}"; do
    JOB_NAME="NW_1st_${SUBJECT}"
    JOB_SCRIPT="$TEMP_JOB_DIR/${JOB_NAME}.slurm"

    # Create SLURM job script
    cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$OUTPUT_DIR/slurm_logs/%x_%j.out
#SBATCH --error=$OUTPUT_DIR/slurm_logs/%x_%j.err
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --account=def-youraccount

# Load Apptainer module (adjust module name if needed)
module load apptainer/1.2.2

# Create output log directory
mkdir -p $OUTPUT_DIR/slurm_logs

# Run the Apptainer container
apptainer run \\
    --bind /project:/project,/scratch:/scratch \\
    $CONTAINER_PATH roi-to-roi --subject $SUBJECT

# Check if the job was successful
if [ \$? -eq 0 ]; then
    echo "Job for $SUBJECT completed successfully" >> $OUTPUT_DIR/slurm_logs/${SUBJECT}_status.log
else
    echo "Job for $SUBJECT failed" >> $OUTPUT_DIR/slurm_logs/${SUBJECT}_status.log
fi
EOF

    # Submit the job
    sbatch "$JOB_SCRIPT"
    echo "Submitted job for $SUBJECT (Job script: $JOB_SCRIPT)"
done

echo "All jobs submitted. Check status with 'squeue -u $USER'"
echo "Logs will be in $OUTPUT_DIR/slurm_logs/"