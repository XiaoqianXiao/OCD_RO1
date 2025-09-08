#!/bin/bash
# Script to generate a SLURM job script for ROI-to-ROI FC Group Analysis
# USAGE: bash generate_slurm_script.sh <atlas_name> [output_script_name]
# EXAMPLE: bash generate_slurm_script.sh power_2011 submit_NW_group_roi_roi.sh

# Check if atlas name is provided
if [ -z "$1" ]; then
    echo "Error: Atlas name required. Usage: $0 <atlas_name> [output_script_name]" >&2
    exit 1
fi

ATLAS=$1
OUTPUT_SCRIPT=${2:-"submit_NW_group_roi_roi.sh"}  # Default output script name

# Define SLURM script content
cat > "${OUTPUT_SCRIPT}" << EOL
#!/bin/bash
# SLURM job submission script for ROI-to-ROI FC Group Analysis
# This script analyzes ROI-to-ROI similarity at the group level
# using outputs from submit_NW_1st.sh
#
# USAGE: sbatch ${OUTPUT_SCRIPT} <atlas_name> [atlas_params]
#
# EXAMPLES FOR ROI-to-ROI SIMILARITY ANALYSIS:
#    sbatch ${OUTPUT_SCRIPT} power_2011
#    sbatch ${OUTPUT_SCRIPT} yeo_2011
#    sbatch ${OUTPUT_SCRIPT} schaefer_2018_100_7_2
#
# NOTE: Only roi-based atlases support ROI-to-ROI similarity analysis.
# Non-roi atlases (harvard_oxford, aal, talairach) are not suitable.

ATLAS=\$1
ATLAS_PARAMS=\${2:-'{}'}  # Default to empty JSON if not provided

# SLURM Configuration
#SBATCH --job-name=roi_roi_fc_group_\${ATLAS}
#SBATCH --output=/scratch/xxqian/logs/ROIROIGroup_\${ATLAS}_%j.out
#SBATCH --error=/scratch/xxqian/logs/ROIROIGroup_\${ATLAS}_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=2G  # Covers ~231 MB + overhead; increase to 32G if QoS allows
#SBATCH --account=def-jfeusner
#SBATCH --partition=cpubase_bycore_b5  # 514 GB+, 7-day limit
#SBATCH --qos=normal  # Adjust after checking sacctmgr

# Create log directory
mkdir -p /scratch/xxqian/logs

# Load Apptainer module
module load apptainer
echo "Loaded modules: \$(module list)"

# Path to Apptainer container
CONTAINER="/scratch/xxqian/repo/image/OCD.sif"

# Set environment variables
export OMP_NUM_THREADS=8

# Define directories and files
PROJECT_DIR="/project/6079231/dliang55/R01_AOCD"
SCRATCH_DIR="/scratch/xxqian"
OUTPUT_DIR="\${SCRATCH_DIR}/OCD/NW_group_roi_roi"
SUBJECTS_CSV="\${PROJECT_DIR}/metadata/shared_demographics.csv"
CLINICAL_CSV="\${SCRATCH_DIR}/OCD/behav/clinical.csv"
INPUT_DIR="\${SCRATCH_DIR}/OCD/NW_1st"

# Bind directories
APPTAINER_BIND="/scratch/xxqian/repo/OCD_RO1/NW_group3.py:/app/NW_group3.py,\${PROJECT_DIR}/metadata:/metadata,\${CLINICAL_CSV}:/clinical.csv,\${SUBJECTS_CSV}:/subjects.csv,\${INPUT_DIR}:/input,\${OUTPUT_DIR}:/output"

# Verify bind paths and inputs
echo "SLURM Resources: Account=\${SLURM_JOB_ACCOUNT}, Partition=\${SLURM_JOB_PARTITION}, Mem=\${SLURM_MEM_PER_NODE}M, CPUs=\${SLURM_CPUS_PER_TASK}"
for path in "\${SCRATCH_DIR}/OCD" "\${PROJECT_DIR}/metadata" "\${CLINICAL_CSV}" "\${SUBJECTS_CSV}" "\${INPUT_DIR}" "\${OUTPUT_DIR}"; do
    if [ ! -e "\$path" ]; then
        echo "Error: Bind path does not exist: \$path" >&2
        exit 1
    fi
done
echo "Input files for \${ATLAS}:"
ls -lh \${INPUT_DIR}/*\${ATLAS}*roiroi_fc_avg.csv || { echo "Error: No input files for \${ATLAS}" >&2; exit 1; }
echo "Available memory before run:"
free -m

# Create output directory
mkdir -p "\${OUTPUT_DIR}"

# Run the Python script
echo "Starting ROI-to-ROI FC group analysis..."
apptainer exec --bind "\${APPTAINER_BIND}" \${CONTAINER} python3 /app/NW_group3.py \\
    --subjects_csv /subjects.csv \\
    --clinical_csv /clinical.csv \\
    --output_dir /output \\
    --input_dir /input \\
    --atlas_name "\${ATLAS}" \\
    --verbose
EOL

# Make the generated script executable
chmod +x "${OUTPUT_SCRIPT}"

echo "Generated SLURM script: ${OUTPUT_SCRIPT}"
echo "Submit with: sbatch ${OUTPUT_SCRIPT} ${ATLAS}"