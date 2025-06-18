#!/bin/bash

# Calculate number of subjects
SUBJECTS=($(find /project/6079231/dliang55/R01_AOCD/derivatives/fmriprep-1.4.1 -maxdepth 1 -type d -name "sub-*" ! -name "*_orig" -exec basename {} \; | sed 's/^sub-//' | sort))
N=$((${#SUBJECTS[@]} - 1))

echo "Submitting jobs for $((N + 1)) subjects: ${SUBJECTS[@]}"

# Submit the job with dynamic array
sbatch --array=0-$N run_1st.sbatch