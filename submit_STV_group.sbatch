#!/bin/bash
#SBATCH --job-name=mixed_anova_pcc
#SBATCH --output=/scratch/xxqian/logs/STVGroup_%A_%a.out
#SBATCH --error=/scratch/xxqian/logs/STVGroup_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Load Apptainer module
module load apptainer

# Path to Apptainer container
CONTAINER="/scratch/xxqian/repo/image/OCD.sif"

# Set environment variables
export FSLOUTPUTTYPE=NIFTI_GZ
export OMP_NUM_THREADS=8

PROJECT_DIR="/project/6079231/dliang55/R01_AOCD"
SCRATCH_DIR="/scratch/xxqian"
OUTPUT_DIR="${SCRATCH_DIR}/OCD"
SUBJECTS_CSV="${PROJECT_DIR}/metadata/shared_demographics.csv"
CLINICAL_CSV="${SCRATCH_DIR}/OCD/behav/clinical.csv"


# Bind directories
APPTAINER_BIND="/scratch/xxqian/work_flow:/scratch, /scratch/xxqian/OCD:/output, /scratch/xxqian/roi:/scratch/xxqian/roi, ${CLINICAL_CSV}:/clinical.csv, ${SUBJECTS_CSV}:/subjects.csv"

# Verify bind paths
for path in "${SCRATCH_DIR}/work_flow" "${SCRATCH_DIR}/OCD" "${SCRATCH_DIR}/roi" "${CLINICAL_CSV}" "${SUBJECTS_CSV}"; do
    if [ ! -e "$path" ]; then
        echo "Error: Bind path does not exist: $path"
        exit 1
    fi
done

# Verify mask file
MASK_FILE="${SCRATCH_DIR}/roi/tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz"
if [ ! -f "${MASK_FILE}" ]; then
    echo "Error: Mask file not found: ${MASK_FILE}"
    exit 1
fi
# Run the Python script inside the Apptainer container
apptainer exec --bind "${APPTAINER_BIND}" ${CONTAINER} python3 /app/STV_group.py \
    --subjects_csv /subjects.csv \
    --clinical_csv /clinical.csv \
    --output_dir /output \
    --work_dir /scratch