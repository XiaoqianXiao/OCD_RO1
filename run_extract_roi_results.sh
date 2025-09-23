#!/bin/bash

# Example script to run ROI results extraction
# Based on the structure from NW_roi_roi.py

echo "--- ROI Results Extraction Script ---"

# Configuration
SUBJECTS_CSV="/project/6079231/dliang55/R01_AOCD/metadata/shared_demographics.csv"
CLINICAL_CSV="/Users/xiaoqianxiao/projects/OCD/results/behav/clinical.csv"
INPUT_DIR="/Users/xiaoqianxiao/projects/OCD/results/NW_1st"
OUTPUT_DIR="./roi_results"
ATLAS_NAME="power_2011"  # or "schaefer_2018_400_7_2", "harvard_oxford_cort-maxprob-thr25-2mm", etc.

echo "Configuration:"
echo "  Subjects CSV: $SUBJECTS_CSV"
echo "  Clinical CSV: $CLINICAL_CSV"
echo "  Input Directory: $INPUT_DIR"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Atlas Name: $ATLAS_NAME"
echo ""

# Check if files exist
if [ ! -f "$SUBJECTS_CSV" ]; then
    echo "ERROR: Subjects CSV file not found: $SUBJECTS_CSV"
    exit 1
fi

if [ ! -f "$CLINICAL_CSV" ]; then
    echo "ERROR: Clinical CSV file not found: $CLINICAL_CSV"
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory not found: $INPUT_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Running ROI results extraction..."

# Run the extraction script
python3 extract_roi_results.py \
    --subjects_csv "$SUBJECTS_CSV" \
    --clinical_csv "$CLINICAL_CSV" \
    --input_dir "$INPUT_DIR" \
    --atlas_name "$ATLAS_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --verbose

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "SUCCESS: ROI results extraction completed!"
    echo ""
    echo "Output files created in $OUTPUT_DIR:"
    if [ "$ATLAS_NAME" = "power_2011" ]; then
        echo "  - power_atlas_roi_fc_summary.csv (Power atlas specific format)"
        echo "    Columns: subject_ID, group, ybocs_baseline, ybocs_followup, ROI1, ROI2, FC_baseline, FC_followup"
    fi
    echo "  - all_subjects_roi_fc_data.csv (generic summary with all subjects)"
    echo "  - successful_subjects.csv (list of subjects with data)"
    echo "  - sub-XXX_roi_fc_data.csv (individual subject files)"
    echo "  - extract_roi_results.log (extraction log)"
    echo ""
    echo "Power Atlas Format:"
    echo "  Each row represents one subject's FC value for one ROI pair"
    echo "  Column meanings:"
    echo "    - subject_ID: Subject identifier"
    echo "    - group: HC or OCD"
    echo "    - ybocs_baseline, ybocs_followup: YBOCS clinical scores"
    echo "    - ROI1, ROI2: ROI pair names"
    echo "    - FC_baseline, FC_followup: Functional connectivity values"
else
    echo ""
    echo "ERROR: ROI results extraction failed with exit code $exit_code"
    echo "Check the log file for details: $OUTPUT_DIR/extract_roi_results.log"
fi

echo "------------------------------------------"
