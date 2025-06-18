#!/bin/bash

# SLURM job submission script for running roi-to-roi for each subject
CONTAINER="/scratch/xxqian/repo/image/OCD.sif"

# Define directories
BIDS_DIR="/project/6079231/dliang55/R01_AOCD/derivatives/fmriprep-1.4.1"
OUTPUT_DIR="/scratch/xxqian/OCD"
LOG_DIR="/scratch/xxqian/logs"
TEMP_JOB_DIR="/scratch/xxqian/slurm_jobs"

# Create necessary directories
mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$TEMP_JOB_DIR"

# Check if container exists
if [ ! -f "$CONTAINER" ]; then
    echo "Error: Apptainer container $CONTAINER not found"
    exit 1
fi

# Get list of subjects from BIDS derivatives directory
SUBJECTS=($(ls -d "$BIDS_DIR"/sub-* | xargs -n 1 basename | grep '^sub-'))
if [ ${#SUBJECTS[@]} -eq 0 ]; then
    echo "Error: No subjects found in $BIDS_DIR"
    exit 1
fi

# Define bind paths for Apptainer, avoiding redundant mounts
APPTAINER_BIND="/project/6079231/dliang55/R01_AOCD:/project/6079231/dliang55/R01_AOCD,/scratch/xxqian:/scratch/xxqian, /scratch/xxqian/repo/OCD_RO1/NW_1st.py:/app/NW_1st.py"

# Verify bind paths
for path in "/project/6079231/dliang55/R01_AOCD" "/scratch/xxqian"; do
    if [ ! -e "$path" ]; then
        echo "Error: Bind path does not exist: $path"
        exit 1
    fi
done

# Loop over subjects and submit a job for each
for SUBJECT in "${SUBJECTS[@]}"; do
    JOB_NAME="NW_1st_${SUBJECT}"
    JOB_SCRIPT="$TEMP_JOB_DIR/${JOB_NAME}.slurm"

    # Extract subject ID without 'sub-' prefix
    SUBJECT_ID=${SUBJECT#sub-}

    # Create SLURM job script
    cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --account=def-jfeusner

# Load Apptainer module
module load apptainer

# Define variables
SUBJECT=${SUBJECT_ID}

# Run the Apptainer container for one subject
if [ -n "\$SUBJECT" ]; then
    apptainer exec --bind "${APPTAINER_BIND}" "${CONTAINER}" python3 /app/NW_1st.py --subject sub-\${SUBJECT}
else
    echo "Error: No subject found" >&2
    exit 1
fi

EOF

    # Submit the job
    sbatch "$JOB_SCRIPT"
    echo "Submitted job for $SUBJECT (Job script: $JOB_SCRIPT)"
done

echo "All jobs submitted. Check status with 'squeue -u $USER'"