#!/bin/bash

# Calculate number of subjects
SUBJECTS=($(find /project/6079231/dliang55/R01_AOCD/derivatives/fmriprep-1.4.1 -maxdepth 1 -type d -name "sub-*" ! -name "*_orig" -exec basename {} \; | sed 's/^sub-//' | sort))
N=$((${#SUBJECTS[@]} - 1))

echo "Submitting jobs for $((N + 1)) subjects: ${SUBJECTS[@]}"

# Submit the job with dynamic array
sbatch --array=0-$N << 'EOF'
#!/bin/bash
#SBATCH --job-name=seed_to_voxel_analysis
#SBATCH --account=def-jfeusner
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/xxqian/logs/STV_%A_%a.out
#SBATCH --error=/scratch/xxqian/logs/STV_%A_%a.err

# Load Apptainer module
module load apptainer

# Get subject from array
SUBJECTS=($(find /project/6079231/dliang55/R01_AOCD/derivatives/fmriprep-1.4.1 -maxdepth 1 -type d -name "sub-*" -exec basename {} \; | sed 's/sub-//g' | sort))
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

# Path to Apptainer container
CONTAINER="/scratch/xxqian/repo/image/OCD.sif"

# Set environment variables
export FSLOUTPUTTYPE=NIFTI_GZ
export OMP_NUM_THREADS=8

# Bind directories
APPTAINER_BIND="/project/6079231/dliang55/R01_AOCD:/project/6079231/dliang55/R01_AOCD,/scratch/xxqian/work_flow:/scratch/xxqian/work_flow,/scratch/xxqian/OCD:/scratch/xxqian/OCD,/scratch/xxqian/roi:/scratch/xxqian/roi"

# Run the Apptainer container for one subject
if [ -n "$SUBJECT" ]; then
  apptainer exec --bind ${APPTAINER_BIND} ${CONTAINER} python3 /app/STV_1st.py --subject sub-${SUBJECT}
else
  echo "Error: No subject found for array index $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi
EOF