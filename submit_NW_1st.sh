#!/bin/bash

# SLURM job submission script for running roi-to-roi for each subject

# Define directories
BIDS_DIR="/project/6079231/dliang55/R01_AOCD/derivatives/fmriprep-1.4.1"
OUTPUT_DIR="/scratch/xxqian/OCD"
CONTAINER_PATH="/scratch/xxqian/OCD.sif"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if container exists
if [ ! -f "$CONTAINER_PATH" ]; then
    echo "Error: Apptainer container $CONTAINER_PATH not found"
    exit 1
fi

# Get list of subjects from BIDS derivatives directory
SUBJECTS=($(ls -d "$BIDS_DIR"/sub-* | xargs -n 1 basename | grep '^sub-'))
if [ ${#SUBJECTS[@]} -eq 0 ]; then
    echo "Error: No subjects found in $BIDS_DIR"
    exit 1
fi

# Create a temporary directory for SLURM job scripts
TEMP_JOB_DIR="/scratch/xxqian/slurm_jobs"
mkdir -p "$TEMP_JOB_DIR"

# Define bind paths for Apptainer
APPTAINER_BIND="/project:/project,/scratch:/scratch"

# Loop over subjects and submit a job for each
for SUBJECT in "${SUBJECTS[@]}"; do
    JOB_NAME="NW_1st_${SUBJECT}"
    JOB_SCRIPT="$TEMP_JOB_DIR/${JOB_NAME}.slurm"

    # Extract subject ID without 'sub-' prefix
    SUBJECT_ID=${SUBJECT#sub-}

    # Create SLURM job script with quoted heredoc
    cat > "$JOB_SCRIPT" << 'EOF'
#!/bin/bash
#SBATCH --job-name=NW_1st_SUBJECT
#SBATCH --output=/scratch/xxqian/logs/NW_1st_%x_%j.out
#SBATCH --error=/scratch/xxqian/logs/NW_1st_%x_%j.err
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --account=def-jfeusner

# Load Apptainer module
module load apptainer

# Create output log directory
mkdir -p /scratch/xxqian/OCD/slurm_logs

# Define variables
SUBJECT=SUBJECT_ID
APPTAINER_BIND=APPTAINER_BINDPATH
CONTAINER=/scratch/xxqian/OCD.sif

# Run the Apptainer container for one subject
if [ -n "$SUBJECT" ]; then
  apptainer exec --bind ${APPTAINER_BIND} ${CONTAINER} roi-to-roi --subject sub-${SUBJECT}
else
  echo "Error: No subject found" >&2
  exit 1
fi

# Check if the job was successful
if [ $? -eq 0 ]; then
    echo "Job for sub-${SUBJECT} completed successfully" >> /scratch/xxqian/OCD/slurm_logs/sub-${SUBJECT}_status.log
else
    echo "Job for sub-${SUBJECT} failed" >> /scratch/xxqian/OCD/slurm_logs/sub-${SUBJECT}_status.log
fi
EOF

    # Replace placeholders with actual values
    sed -i "s|SUBJECT_ID|${SUBJECT_ID}|g" "$JOB_SCRIPT"
    sed -i "s|APPTAINER_BINDPATH|${APPTAINER_BIND}|g" "$JOB_SCRIPT"
    sed -i "s|NW_1st_SUBJECT|${JOB_NAME}|g" "$JOB_SCRIPT"

    # Submit the job
    sbatch "$JOB_SCRIPT"
    echo "Submitted job for $SUBJECT (Job script: $JOB_SCRIPT)"
done

echo "All jobs submitted. Check status with 'squeue -u $USER'"
echo "Logs will be in $OUTPUT_DIR/slurm_logs/"