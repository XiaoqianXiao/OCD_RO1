#!/bin/bash

# Local version of submit_NW_1st.sh for macOS testing
# This script runs NW_1st.py directly without SLURM or containers

# Parse command line arguments
ATLAS="power_2011"
ATLAS_PARAMS=""
LABEL_PATTERN="power"

while [[ $# -gt 0 ]]; do
    case $1 in
        --atlas)
            ATLAS="$2"
            shift 2
            ;;
        --atlas-params)
            ATLAS_PARAMS="$2"
            shift 2
            ;;
        --label-pattern)
            LABEL_PATTERN="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--atlas ATLAS] [--atlas-params PARAMS] [--label-pattern PATTERN]"
            echo "  --atlas ATLAS            Atlas name (default: power_2011)"
            echo "  --atlas-params PARAMS   JSON string of atlas parameters"
            echo "  --label-pattern PATTERN Label pattern (default: power)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Define local directories
OUTPUT_DIR="./results/NW_1st"
LOG_DIR="./logs"

# Create necessary directories
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "Running NW_1st.py locally (no SLURM)"
echo "Atlas: $ATLAS"
if [[ -n "$ATLAS_PARAMS" ]]; then
    echo "Atlas Parameters: $ATLAS_PARAMS"
fi
echo "Label Pattern: $LABEL_PATTERN"
echo "Output Directory: $OUTPUT_DIR"

# Check if NW_1st.py exists
if [ ! -f "NW_1st.py" ]; then
    echo "Error: NW_1st.py not found in current directory"
    exit 1
fi

# For testing, run with a sample subject (you'll need to modify this)
echo ""
echo "NOTE: This is a test run. You need to:"
echo "1. Have BIDS-formatted fMRI data available"
echo "2. Modify the subject ID below to match your data"
echo "3. Update the BIDS directory path in NW_1st.py"
echo ""

# Example command (modify subject ID as needed)
echo "Example command:"
echo "python NW_1st.py --subject sub-AOCD001 --atlas $ATLAS --label-pattern $LABEL_PATTERN --output-dir $OUTPUT_DIR"

# Uncomment the line below when you have data and want to run:
# python NW_1st.py --subject sub-AOCD001 --atlas $ATLAS --label-pattern $LABEL_PATTERN --output-dir $OUTPUT_DIR

echo ""
echo "To run with real data:"
echo "1. Update the subject ID above"
echo "2. Ensure you have BIDS data available"
echo "3. Uncomment the python command above"
echo "4. Run: bash submit_NW_1st_local.sh"

