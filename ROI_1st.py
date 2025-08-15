#!/usr/bin/env python3
"""
ROI-to-ROI Functional Connectivity Analysis using Customizable Atlas

This script computes ROI-to-ROI functional connectivity matrices using any user-specified
atlas for resting-state fMRI data. It processes individual subjects and sessions,
generating correlation matrices and pairwise connectivity measures.

Author: [Your Name]
Date: [Current Date]

USAGE EXAMPLES:
==============

1. Harvard-Oxford Atlas (Cortical Regions):
   python ROI_1st.py \
     --subject sub-AOCD001 \
     --atlas /scratch/xxqian/roi/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz \
     --labels /scratch/xxqian/roi/HarvardOxford_cort_labels.txt \
     --label-pattern harvard_oxford

2. Power 2011 Atlas (264 ROIs):
   python ROI_1st.py \
     --subject sub-AOCD001 \
     --atlas /scratch/xxqian/roi/power_2011_atlas.nii.gz \
     --labels /scratch/xxqian/roi/power264/power264NodeNames.txt \
     --label-pattern power

3. Nilearn Built-in Atlas (Schaefer 2018):
   python ROI_1st.py \
     --subject sub-AOCD001 \
     --atlas schaefer_2018 \
     --atlas-params '{"n_rois": 400, "yeo_networks": 7, "resolution_mm": 2}' \
     --label-pattern nilearn

4. Nilearn Built-in Atlas (Harvard-Oxford):
   python ROI_1st.py \
     --subject sub-AOCD001 \
     --atlas harvard_oxford \
     --atlas-params '{"atlas_name": "cort-maxprob-thr25-2mm"}' \
     --label-pattern nilearn

5. Custom Atlas with Simple Labels:
   python ROI_1st.py \
     --subject sub-AOCD001 \
     --atlas /path/to/custom_atlas.nii.gz \
     --labels /path/to/custom_labels.txt \
     --label-pattern simple \
     --atlas-name custom_atlas

6. Custom Atlas with Complex Label Format:
   python ROI_1st.py \
     --subject sub-AOCD001 \
     --atlas /path/to/custom_atlas.nii.gz \
     --labels /path/to/custom_labels.txt \
     --label-pattern custom \
     --custom-regex "ROI_(\d+)_(.+)" \
     --atlas-name custom_atlas

7. With Custom Output and Work Directories:
   python ROI_1st.py \
     --subject sub-AOCD001 \
     --atlas /path/to/atlas.nii.gz \
     --labels /path/to/labels.txt \
     --label-pattern simple \
     --output-dir /custom/output/path \
     --work-dir /custom/work/path

ATLAS TYPES:
============

Built-in Nilearn Atlases:
- schaefer_2018: Schaefer 2018 parcellation (100-1000 ROIs, 7/17 networks)
- harvard_oxford: Harvard-Oxford cortical/subcortical atlases
- aal: Automated Anatomical Labeling atlas
- talairach: Talairach atlas
- power_2011: Power 2011 atlas (264 ROIs)
- coords_power_2012: Power 2012 coordinate-based atlas
- pauli_2017: Pauli 2017 subcortical atlas
- yeo_2011: Yeo 2011 7/17 network parcellation

Custom Atlases:
- File-based: Provide path to .nii.gz file and labels file
- Label patterns: simple, power, harvard_oxford, custom

LABEL PATTERN TYPES:
===================

- nilearn: Built-in atlas labels (automatically handled)
- simple: One label per line (e.g., "ROI_1", "ROI_2", "ROI_3")
- power: Power atlas format (e.g., "Default_1", "DorsAttn_2")
- harvard_oxford: Harvard-Oxford format (e.g., "ROI_1_Frontal_Pole")
- custom: User-defined regex pattern for complex formats

OUTPUT FILES:
=============

- {subject}_{session}_task-rest_{atlas_name}_roiroi_matrix_avg.npy: Correlation matrix
- {subject}_{session}_task-rest_{atlas_name}_pairwise_fc_avg.csv: Pairwise FC values

REQUIREMENTS:
=============

- BIDS-formatted fMRI data
- Preprocessed fMRI images (fmriprep output)
- Atlas file (.nii.gz format) or Nilearn atlas name
- ROI labels file (text format) or automatic for Nilearn atlases
- Confounds file (motion parameters, aCompCor)
"""

import os
import glob
import re
import json
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import resample_to_img
from nilearn.datasets import (
    fetch_atlas_schaefer_2018, fetch_atlas_harvard_oxford,
    fetch_atlas_aal, fetch_atlas_talairach, fetch_atlas_power_2011,
    fetch_atlas_coords_power_2012, fetch_atlas_pauli_2017,
    fetch_atlas_yeo_2011
)
import argparse
import logging
from itertools import combinations
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
import warnings
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default configuration
DEFAULT_CONFIG = {
    'project_dir': '/project/6079231/dliang55/R01_AOCD/derivatives/fmriprep-1.4.1',
    'bids_dir': '/project/6079231/dliang55/R01_AOCD/derivatives/fmriprep-1.4.1',
    'scratch_dir': '/scratch/xxqian',
    'output_dir': '/project/6079231/dliang55/R01_AOCD',
    'work_dir': '/scratch/xxqian/work_flow',
    'roi_dir': '/scratch/xxqian/roi',
    'log_file': 'roi_to_roi_fc_analysis.log',
    'sessions': ['ses-baseline', 'ses-followup'],
    'tr': 2.0,
    'low_pass': 0.1,
    'high_pass': 0.01,
    'fd_threshold': 0.5,
    'min_timepoints': 10,
    'compcor_components': 5
}

# Nilearn atlas configurations
NILEARN_ATLASES = {
    'schaefer_2018': {
        'function': fetch_atlas_schaefer_2018,
        'default_params': {'n_rois': 400, 'yeo_networks': 7, 'resolution_mm': 2},
        'description': 'Schaefer 2018 parcellation (100-1000 ROIs, 7/17 networks)',
        'param_options': {
            'n_rois': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'yeo_networks': [7, 17],
            'resolution_mm': [1, 2]
        }
    },
    'harvard_oxford': {
        'function': fetch_atlas_harvard_oxford,
        'default_params': {'atlas_name': 'cort-maxprob-thr25-2mm'},
        'description': 'Harvard-Oxford cortical/subcortical atlases',
        'param_options': {
            'atlas_name': [
                'cort-maxprob-thr25-2mm', 'cort-maxprob-thr50-2mm',
                'sub-maxprob-thr25-2mm', 'sub-maxprob-thr50-2mm',
                'cort-prob-2mm', 'sub-prob-2mm'
            ]
        }
    },
    'aal': {
        'function': fetch_atlas_aal,
        'default_params': {},
        'description': 'Automated Anatomical Labeling atlas (116 ROIs)',
        'param_options': {}
    },
    'talairach': {
        'function': fetch_atlas_talairach,
        'default_params': {},
        'description': 'Talairach atlas (1107 ROIs)',
        'param_options': {}
    },
    'power_2011': {
        'function': fetch_atlas_power_2011,
        'default_params': {},
        'description': 'Power 2011 atlas (264 ROIs)',
        'param_options': {}
    },
    'coords_power_2012': {
        'function': fetch_atlas_coords_power_2012,
        'default_params': {},
        'description': 'Power 2012 coordinate-based atlas (264 ROIs)',
        'param_options': {}
    },
    'pauli_2017': {
        'function': fetch_atlas_pauli_2017,
        'default_params': {},
        'description': 'Pauli 2017 subcortical atlas (16 ROIs)',
        'param_options': {}
    },
    'yeo_2011': {
        'function': fetch_atlas_yeo_2011,
        'default_params': {'n_rois': 7},
        'description': 'Yeo 2011 network parcellation (7/17 networks)',
        'param_options': {
            'n_rois': [7, 17]
        }
    }
}

# =============================================================================
# USAGE AND HELP FUNCTIONS
# =============================================================================

def print_usage():
    """Print comprehensive usage information."""
    usage_text = """
ROI-to-ROI Functional Connectivity Analysis
==========================================

DESCRIPTION:
This script performs ROI-to-ROI functional connectivity analysis using any user-specified
atlas. It processes resting-state fMRI data and generates correlation matrices and
pairwise connectivity measures.

BASIC USAGE:
python ROI_1st.py --subject <SUBJECT_ID> --atlas <ATLAS_PATH> --labels <LABELS_PATH> --label-pattern <PATTERN>

REQUIRED ARGUMENTS:
------------------
--subject <SUBJECT_ID>     Subject ID (e.g., sub-AOCD001)
--atlas <ATLAS_PATH>       Path to atlas file (.nii.gz) or atlas name for predefined atlases
--labels <LABELS_PATH>     Path to ROI labels file (required for custom atlases, not needed for Nilearn atlases)
--label-pattern <PATTERN>  Pattern for reading ROI labels

OPTIONAL ARGUMENTS:
------------------
--atlas-params <JSON>      JSON string with atlas parameters (required for Nilearn atlases)
--custom-regex <REGEX>     Custom regex for label parsing (when pattern is 'custom')
--atlas-name <NAME>        Custom name for atlas (used in output filenames)
--expected-rois <N>        Expected number of ROIs (for validation)
--output-dir <PATH>        Output directory (overrides default)
--work-dir <PATH>          Working directory (overrides default)
--verbose                  Enable verbose logging
--help                     Show this help message
--usage                    Show detailed usage examples
--list-atlases            List available Nilearn atlases and their parameters

ATLAS TYPES:
------------
Built-in Nilearn Atlases:
- schaefer_2018: Schaefer 2018 parcellation (100-1000 ROIs, 7/17 networks)
- harvard_oxford: Harvard-Oxford cortical/subcortical atlases
- aal: Automated Anatomical Labeling atlas (116 ROIs)
- talairach: Talairach atlas (1107 ROIs)
- power_2011: Power 2011 atlas (264 ROIs)
- coords_power_2012: Power 2012 coordinate-based atlas (264 ROIs)
- pauli_2017: Pauli 2017 subcortical atlas (16 ROIs)
- yeo_2011: Yeo 2011 network parcellation (7/17 networks)

Custom Atlases:
- File-based: Provide path to .nii.gz file and labels file

LABEL PATTERN TYPES:
--------------------
nilearn          Built-in atlas labels (automatically handled)
simple           One label per line (e.g., "ROI_1", "ROI_2")
power            Power atlas format (e.g., "Default_1", "DorsAttn_2")
harvard_oxford   Harvard-Oxford format (e.g., "ROI_1_Frontal_Pole")
custom           User-defined regex pattern

EXAMPLES:
---------
1. Schaefer 2018 Atlas (400 ROIs, 7 networks):
   python ROI_1st.py \\
     --subject sub-AOCD001 \\
     --atlas schaefer_2018 \\
     --atlas-params '{"n_rois": 400, "yeo_networks": 7, "resolution_mm": 2}' \\
     --label-pattern nilearn

2. Harvard-Oxford Atlas:
   python ROI_1st.py \\
     --subject sub-AOCD001 \\
     --atlas harvard_oxford \\
     --atlas-params '{"atlas_name": "cort-maxprob-thr25-2mm"}' \\
     --label-pattern nilearn

3. Custom Atlas with Simple Labels:
   python ROI_1st.py \\
     --subject sub-AOCD001 \\
     --atlas /path/to/custom_atlas.nii.gz \\
     --labels /path/to/custom_labels.txt \\
     --label-pattern simple \\
     --atlas-name custom_atlas

OUTPUT FILES:
-------------
- {subject}_{session}_task-rest_{atlas_name}_roiroi_matrix_avg.npy: Correlation matrix
- {subject}_{session}_task-rest_{atlas_name}_pairwise_fc_avg.csv: Pairwise FC values

REQUIREMENTS:
-------------
- BIDS-formatted fMRI data
- Preprocessed fMRI images (fmriprep output)
- Atlas file (.nii.gz format) or Nilearn atlas name
- ROI labels file (text format) or automatic for Nilearn atlases
- Confounds file (motion parameters, aCompCor)

For more information, see the script docstring or run with --help.
"""
    print(usage_text)

def print_examples():
    """Print detailed usage examples."""
    examples_text = """
DETAILED USAGE EXAMPLES
=======================

1. SCHAEFER 2018 ATLAS (Built-in Nilearn)
   ----------------------------------------
   This is a popular parcellation with 100-1000 ROIs organized by functional networks.
   
   python ROI_1st.py \\
     --subject sub-AOCD001 \\
     --atlas schaefer_2018 \\
     --atlas-params '{"n_rois": 400, "yeo_networks": 7, "resolution_mm": 2}' \\
     --label-pattern nilearn
   
   Available parameters:
   - n_rois: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
   - yeo_networks: 7 or 17
   - resolution_mm: 1 or 2
   
   Example with 1000 ROIs and 17 networks:
   python ROI_1st.py \\
     --subject sub-AOCD001 \\
     --atlas schaefer_2018 \\
     --atlas-params '{"n_rois": 1000, "yeo_networks": 17, "resolution_mm": 1}' \\
     --label-pattern nilearn

2. HARVARD-OXFORD ATLAS (Built-in Nilearn)
   -----------------------------------------
   This atlas provides cortical and subcortical parcellations.
   
   python ROI_1st.py \\
     --subject sub-AOCD001 \\
     --atlas harvard_oxford \\
     --atlas-params '{"atlas_name": "cort-maxprob-thr25-2mm"}' \\
     --label-pattern nilearn
   
   Available atlas names:
   - cort-maxprob-thr25-2mm: Cortical, max probability, 25% threshold, 2mm
   - cort-maxprob-thr50-2mm: Cortical, max probability, 50% threshold, 2mm
   - sub-maxprob-thr25-2mm: Subcortical, max probability, 25% threshold, 2mm
   - sub-maxprob-thr50-2mm: Subcortical, max probability, 50% threshold, 2mm
   - cort-prob-2mm: Cortical, probabilistic, 2mm
   - sub-prob-2mm: Subcortical, probabilistic, 2mm

3. POWER 2011 ATLAS (Built-in Nilearn)
   ------------------------------------
   This atlas contains 264 ROIs organized by functional networks.
   
   python ROI_1st.py \\
     --subject sub-AOCD001 \\
     --atlas power_2011 \\
     --label-pattern nilearn
   
   No additional parameters needed.

4. YEO 2011 ATLAS (Built-in Nilearn)
   ----------------------------------
   This atlas provides 7 or 17 network parcellations.
   
   python ROI_1st.py \\
     --subject sub-AOCD001 \\
     --atlas yeo_2011 \\
     --atlas-params '{"n_rois": 17}' \\
     --label-pattern nilearn
   
   Available parameters:
   - n_rois: 7 or 17

5. CUSTOM ATLAS WITH SIMPLE LABELS
   --------------------------------
   For atlases with simple, one-per-line labels.
   
   python ROI_1st.py \\
     --subject sub-AOCD001 \\
     --atlas /path/to/custom_atlas.nii.gz \\
     --labels /path/to/custom_labels.txt \\
     --label-pattern simple \\
     --atlas-name custom_atlas
   
   Expected labels format:
   ROI_1
   ROI_2
   ROI_3
   ...

6. CUSTOM ATLAS WITH COMPLEX LABELS
   ---------------------------------
   For atlases with complex label formats that need regex parsing.
   
   python ROI_1st.py \\
     --subject sub-AOCD001 \\
     --atlas /path/to/custom_atlas.nii.gz \\
     --labels /path/to/custom_labels.txt \\
     --label-pattern custom \\
     --custom-regex "ROI_(\\d+)_(.+)" \\
     --atlas-name custom_atlas
   
   Expected labels format:
   ROI_1_Frontal_Pole
   ROI_2_Insular_Cortex
   ROI_3_Superior_Frontal_Gyrus
   ...
   
   The regex "ROI_(\\d+)_(.+)" will extract:
   - Group 1: The ROI number (1, 2, 3, ...)
   - Group 2: The region name (Frontal_Pole, Insular_Cortex, ...)
   
   Generated names will be: "1_Frontal_Pole", "2_Insular_Cortex", ...

7. WITH CUSTOM DIRECTORIES
   ------------------------
   Override default output and working directories.
   
   python ROI_1st.py \\
     --subject sub-AOCD001 \\
     --atlas schaefer_2018 \\
     --atlas-params '{"n_rois": 400}' \\
     --label-pattern nilearn \\
     --output-dir /custom/output/path \\
     --work-dir /custom/work/path

8. WITH VALIDATION
   ----------------
   Specify expected ROI count for validation.
   
   python ROI_1st.py \\
     --subject sub-AOCD001 \\
     --atlas schaefer_2018 \\
     --atlas-params '{"n_rois": 400}' \\
     --label-pattern nilearn \\
     --expected-rois 400

9. VERBOSE LOGGING
   ----------------
   Enable detailed logging for debugging.
   
   python ROI_1st.py \\
     --subject sub-AOCD001 \\
     --atlas schaefer_2018 \\
     --atlas-params '{"n_rois": 400}' \\
     --label-pattern nilearn \\
     --verbose

NILEARN ATLAS PARAMETERS:
-------------------------
Schaefer 2018:
  - n_rois: Number of ROIs (100-1000)
  - yeo_networks: Number of networks (7 or 17)
  - resolution_mm: Spatial resolution (1 or 2mm)

Harvard-Oxford:
  - atlas_name: Atlas variant name

Yeo 2011:
  - n_rois: Number of networks (7 or 17)

Power 2011, AAL, Talairach, Pauli 2017:
  - No additional parameters needed

OUTPUT FILE NAMING:
------------------
The script generates output files with names based on the atlas:
- Nilearn atlases: {atlas_name}_{parameters}_roiroi_matrix_avg.npy
- Custom atlases: {atlas_name}_roiroi_matrix_avg.npy

Examples:
- Schaefer 2018: schaefer_2018_400_7_2_roiroi_matrix_avg.npy
- Harvard-Oxford: harvard_oxford_cort_maxprob_thr25_2mm_roiroi_matrix_avg.npy
- Custom: custom_atlas_roiroi_matrix_avg.npy

TROUBLESHOOTING:
----------------
1. Check that atlas name is valid for Nilearn atlases
2. Verify atlas parameters are correct and in valid ranges
3. Ensure BIDS directory structure is correct
4. Check that confounds files are available
5. Verify output directory is writable
6. Use --verbose for detailed error messages
7. For Nilearn atlases, ensure internet connection for first download
"""
    print(examples_text)

def print_available_atlases():
    """Print a list of available Nilearn atlases and their parameters."""
    print("\nAvailable Nilearn Atlases:")
    print("-" * 50)
    for atlas_name, atlas_info in NILEARN_ATLASES.items():
        print(f"- {atlas_name}:")
        print(f"  Description: {atlas_info['description']}")
        print(f"  Default Parameters: {atlas_info['default_params']}")
        if atlas_info['param_options']:
            print(f"  Parameter Options: {atlas_info['param_options']}")
        print()

# =============================================================================
# ATLAS FETCHING FUNCTIONS
# =============================================================================

def fetch_nilearn_atlas(atlas_name: str, atlas_params: Dict[str, Any], logger: logging.Logger) -> Tuple[image.Nifti1Image, List[str]]:
    """Fetch a built-in Nilearn atlas and return the image and labels."""
    try:
        if atlas_name not in NILEARN_ATLASES:
            raise ValueError(f"Unknown Nilearn atlas: {atlas_name}. Available: {list(NILEARN_ATLASES.keys())}")
        
        atlas_info = NILEARN_ATLASES[atlas_name]
        fetch_func = atlas_info['function']
        
        # Merge default parameters with user parameters
        params = atlas_info['default_params'].copy()
        params.update(atlas_params)
        
        logger.info(f"Fetching {atlas_name} atlas with parameters: {params}")
        
        # Fetch the atlas
        atlas_data = fetch_func(**params)
        
        # Load the atlas image
        atlas_img = image.load_img(atlas_data['maps'])
        
        # Get labels
        if 'labels' in atlas_data:
            labels = atlas_data['labels']
        elif 'lut' in atlas_data and hasattr(atlas_data['lut'], 'columns'):
            # Handle lookup table format
            if 'name' in atlas_data['lut'].columns:
                labels = atlas_data['lut']['name'].tolist()
            else:
                labels = [f"ROI_{i}" for i in range(len(atlas_data['lut']))]
        else:
            # Fallback: generate generic labels
            n_rois = len(np.unique(atlas_img.get_fdata())) - 1  # Exclude background (0)
            labels = [f"ROI_{i+1}" for i in range(n_rois)]
        
        logger.info(f"Successfully fetched {atlas_name} atlas: {len(labels)} ROIs, shape={atlas_img.shape}")
        
        return atlas_img, labels
        
    except Exception as e:
        logger.error(f"Failed to fetch {atlas_name} atlas: {str(e)}")
        raise

def validate_atlas_params(atlas_name: str, atlas_params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate atlas parameters against available options."""
    if atlas_name not in NILEARN_ATLASES:
        raise ValueError(f"Unknown Nilearn atlas: {atlas_name}")
    
    atlas_info = NILEARN_ATLASES[atlas_name]
    param_options = atlas_info['param_options']
    
    # Start with default parameters
    validated_params = atlas_info['default_params'].copy()
    
    # Update with user parameters and validate
    for param, value in atlas_params.items():
        if param in param_options:
            if value not in param_options[param]:
                raise ValueError(f"Invalid value for {param}: {value}. Valid options: {param_options[param]}")
        validated_params[param] = value
    
    return validated_params

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_filename: str) -> logging.Logger:
    """Set up logging configuration with both file and console handlers."""
    log_file = os.path.join(DEFAULT_CONFIG['output_dir'], log_filename)
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create logger
    logger = logging.getLogger('ROI_FC_Analysis')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute ROI-to-ROI functional connectivity using customizable atlas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Schaefer 2018 Atlas (Built-in Nilearn)
  python ROI_1st.py --subject sub-AOCD001 \\
    --atlas schaefer_2018 \\
    --atlas-params '{"n_rois": 400, "yeo_networks": 7, "resolution_mm": 2}' \\
    --label-pattern nilearn

  # Harvard-Oxford Atlas (Built-in Nilearn)
  python ROI_1st.py --subject sub-AOCD001 \\
    --atlas harvard_oxford \\
    --atlas-params '{"atlas_name": "cort-maxprob-thr25-2mm"}' \\
    --label-pattern nilearn

  # Custom Atlas
  python ROI_1st.py --subject sub-AOCD001 \\
    --atlas /path/to/atlas.nii.gz \\
    --labels /path/to/labels.txt \\
    --label-pattern simple \\
    --atlas-name custom_atlas

Run with --usage for detailed examples or --help for full help.
        """
    )
    parser.add_argument(
        '--subject', 
        type=str, 
        required=True, 
        help='Subject ID (e.g., sub-AOCD001)'
    )
    parser.add_argument(
        '--atlas', 
        type=str, 
        required=True,
        help='Path to atlas file (.nii.gz) or atlas name for predefined atlases (e.g., schaefer_2018, harvard_oxford, power_2011)'
    )
    parser.add_argument(
        '--labels', 
        type=str, 
        help='Path to ROI labels file (required for custom atlases, not needed for Nilearn atlases)'
    )
    parser.add_argument(
        '--label-pattern',
        type=str,
        default='simple',
        choices=['nilearn', 'simple', 'power', 'harvard_oxford', 'custom'],
        help='Pattern for reading ROI labels: nilearn (built-in atlas), simple (one label per line), power (network_ROI format), harvard_oxford (ROI_Label format), custom (user-defined regex)'
    )
    parser.add_argument(
        '--atlas-params',
        type=str,
        help='JSON string with atlas parameters (required for Nilearn atlases, e.g., \'{"n_rois": 400, "yeo_networks": 7}\')'
    )
    parser.add_argument(
        '--custom-regex',
        type=str,
        help='Custom regex pattern for parsing labels when label-pattern is custom (e.g., r"ROI_(\\d+)_(.+)" for ROI_1_Label format)'
    )
    parser.add_argument(
        '--atlas-name',
        type=str,
        help='Name for the atlas (used in output filenames, defaults to atlas filename without extension)'
    )
    parser.add_argument(
        '--expected-rois',
        type=int,
        help='Expected number of ROIs (for validation, defaults to actual count)'
    )
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true', 
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results (overrides default)'
    )
    parser.add_argument(
        '--work-dir',
        type=str,
        help='Working directory for temporary files (overrides default)'
    )
    parser.add_argument(
        '--usage',
        action='store_true',
        help='Show detailed usage examples'
    )
    parser.add_argument(
        '--list-atlases',
        action='store_true',
        help='List available Nilearn atlases and their parameters'
    )
    
    return parser.parse_args()

# =============================================================================
# ATLAS AND LABELS LOADING
# =============================================================================

def load_atlas(atlas_path: str, logger: logging.Logger) -> image.Nifti1Image:
    """Load atlas image from file."""
    logger.info(f"Loading atlas: {atlas_path}")
    
    if not os.path.exists(atlas_path):
        raise FileNotFoundError(f"Atlas file not found: {atlas_path}")
    
    atlas = image.load_img(atlas_path)
    logger.info(f"Loaded atlas: {atlas_path}, shape: {atlas.shape}")
    
    return atlas

def parse_labels_simple(labels_path: str, logger: logging.Logger) -> List[str]:
    """Parse labels using simple one-label-per-line format."""
    logger.info(f"Parsing labels using simple format: {labels_path}")
    
    with open(labels_path, 'r') as f:
        roi_labels = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(roi_labels)} ROI labels using simple format")
    return roi_labels

def parse_labels_power(labels_path: str, logger: logging.Logger) -> List[str]:
    """Parse labels using Power atlas format (network_ROI)."""
    logger.info(f"Parsing labels using Power format: {labels_path}")
    
    with open(labels_path, 'r') as f:
        roi_labels = [line.strip() for line in f if line.strip()]
    
    # Parse Power format: network_ROI
    parsed_labels = []
    for label in roi_labels:
        if '_' in label:
            parts = label.rsplit('_', 1)
            if len(parts) == 2:
                network, roi_num = parts
                parsed_labels.append(f"{network}_ROI_{roi_num}")
            else:
                parsed_labels.append(label)
        else:
            parsed_labels.append(label)
    
    logger.info(f"Loaded {len(parsed_labels)} ROI labels using Power format")
    return parsed_labels

def parse_labels_harvard_oxford(labels_path: str, logger: logging.Logger) -> List[str]:
    """Parse labels using Harvard-Oxford format (ROI_Label)."""
    logger.info(f"Parsing labels using Harvard-Oxford format: {labels_path}")
    
    with open(labels_path, 'r') as f:
        roi_labels = [line.strip() for line in f if line.strip()]
    
    # Parse Harvard-Oxford format: ROI_Label
    parsed_labels = []
    for label in roi_labels:
        if label.startswith('ROI_'):
            # Extract the label part after ROI_
            label_part = label[4:]  # Remove 'ROI_' prefix
            parsed_labels.append(f"ROI_{label_part.replace(' ', '_').replace(',', '')}")
        else:
            parsed_labels.append(label)
    
    logger.info(f"Loaded {len(parsed_labels)} ROI labels using Harvard-Oxford format")
    return parsed_labels

def parse_labels_custom(labels_path: str, custom_regex: str, logger: logging.Logger) -> List[str]:
    """Parse labels using custom regex pattern."""
    logger.info(f"Parsing labels using custom regex: {custom_regex}")
    
    if not custom_regex:
        raise ValueError("Custom regex pattern must be provided when label-pattern is 'custom'")
    
    try:
        pattern = re.compile(custom_regex)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern '{custom_regex}': {e}")
    
    with open(labels_path, 'r') as f:
        roi_labels = [line.strip() for line in f if line.strip()]
    
    # Parse using custom regex
    parsed_labels = []
    for i, label in enumerate(roi_labels):
        match = pattern.match(label)
        if match:
            # Use the matched groups or the full match
            if match.groups():
                parsed_labels.append('_'.join(match.groups()))
            else:
                parsed_labels.append(match.group(0))
        else:
            logger.warning(f"Label '{label}' at line {i+1} does not match custom regex pattern")
            parsed_labels.append(f"ROI_{i+1}")
    
    logger.info(f"Loaded {len(parsed_labels)} ROI labels using custom regex pattern")
    return parsed_labels

def load_roi_labels(labels_path: str, label_pattern: str, custom_regex: str = None, logger: logging.Logger = None) -> List[str]:
    """Load and parse ROI labels based on the specified pattern."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Loading ROI labels from {labels_path} using pattern: {label_pattern}")
    
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"ROI labels file not found: {labels_path}")
    
    # Parse labels based on pattern
    if label_pattern == 'simple':
        roi_labels = parse_labels_simple(labels_path, logger)
    elif label_pattern == 'power':
        roi_labels = parse_labels_power(labels_path, logger)
    elif label_pattern == 'harvard_oxford':
        roi_labels = parse_labels_harvard_oxford(labels_path, logger)
    elif label_pattern == 'custom':
        roi_labels = parse_labels_custom(labels_path, custom_regex, logger)
    else:
        raise ValueError(f"Unknown label pattern: {label_pattern}")
    
    # Validate labels
    if not roi_labels:
        raise ValueError(f"No valid ROI labels found in {labels_path}")
    
    # Create ROI names
    roi_names = []
    for i, label in enumerate(roi_labels):
        if label_pattern == 'power':
            # For Power atlas, use the parsed format
            roi_names.append(label)
        elif label_pattern == 'harvard_oxford':
            # For Harvard-Oxford, use the parsed format
            roi_names.append(label)
        else:
            # For simple and custom, create standardized names
            clean_label = label.replace(' ', '_').replace(',', '').replace('-', '_')
            roi_names.append(f"ROI_{i+1}_{clean_label}")
    
    logger.info(f"Generated {len(roi_names)} ROI names")
    logger.info(f"Sample ROI names: {roi_names[:5]}...")  # Show first 5 names
    
    return roi_names

def load_atlas_and_labels(
    atlas_path: str, 
    labels_path: str, 
    label_pattern: str, 
    custom_regex: str = None,
    expected_rois: int = None,
    logger: logging.Logger = None
) -> Tuple[image.Nifti1Image, List[str]]:
    """Load atlas and ROI labels with validation."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Load atlas
    atlas = load_atlas(atlas_path, logger)
    
    # Load and parse ROI labels
    roi_names = load_roi_labels(labels_path, label_pattern, custom_regex, logger)
    
    # Validate ROI count if expected_rois is provided
    if expected_rois is not None and len(roi_names) != expected_rois:
        logger.warning(
            f"Expected {expected_rois} ROIs but found {len(roi_names)}. "
            f"Proceeding with actual count."
        )
    
    # Get unique ROI values from atlas
    atlas_data = atlas.get_fdata()
    unique_rois = np.unique(atlas_data)
    unique_rois = unique_rois[unique_rois > 0]  # Exclude background (0)
    
    logger.info(f"Atlas contains {len(unique_rois)} unique ROI values: {unique_rois}")
    logger.info(f"Generated {len(roi_names)} ROI names")
    
    # Warn if there's a mismatch
    if len(roi_names) != len(unique_rois):
        logger.warning(
            f"ROI count mismatch: {len(roi_names)} labels vs {len(unique_rois)} atlas ROIs. "
            f"This may cause issues in analysis."
        )
    
    return atlas, roi_names

# =============================================================================
# FILE VALIDATION AND PATH MANAGEMENT
# =============================================================================

def validate_paths(subject: str, session: str, bids_dir: str, logger: logging.Logger) -> Optional[Tuple[str, str]]:
    """Validate and return paths for fMRI and brain mask files."""
    # Define file patterns
    patterns = {
        'mask': f'{subject}_{session}_task-rest*_space-MNI152NLin6Asym_desc-brain_mask.nii.gz',
        'fmri': f'{subject}_{session}_task-rest*_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz'
    }
    
    # Search for files
    file_paths = {}
    for file_type, pattern in patterns.items():
        search_pattern = os.path.join(bids_dir, subject, session, 'func', pattern)
        files = glob.glob(search_pattern)
        
        if not files:
            logger.warning(f"No {file_type} file found for {subject} {session}")
            return None
        
        file_paths[file_type] = files[0]
    
    logger.info(f"Found files for {subject} {session}:")
    for file_type, path in file_paths.items():
        logger.info(f"  {file_type}: {path}")
    
    return file_paths['mask'], file_paths['fmri']

def find_confounds_file(subject: str, session: str, bids_dir: str, logger: logging.Logger) -> Optional[str]:
    """Find confounds file for the given subject and session."""
    confounds_pattern = os.path.join(
        bids_dir, subject, session, 'func',
        f'{subject}_{session}_task-rest*_desc-confounds_regressors.tsv'
    )
    
    confounds_files = sorted(glob.glob(confounds_pattern))
    if not confounds_files:
        logger.warning(f"No confounds file found for {subject} {session}")
        return None
    
    confounds_file = confounds_files[0]  # Use first match
    logger.info(f"Found confounds file: {confounds_file}")
    
    return confounds_file

def extract_run_id(fmri_file: str) -> str:
    """Extract run ID from fMRI filename."""
    run_match = re.search(r'_run-(\d+)_', fmri_file)
    if run_match:
        run_id = run_match.group(1).lstrip('0') or '1'
    else:
        run_id = '1'
    return run_id

# =============================================================================
# IMAGE PROCESSING
# =============================================================================

def resample_to_atlas_space(
    img: image.Nifti1Image,
    target_img: image.Nifti1Image,
    output_path: str,
    interpolation: str = 'continuous',
    logger: logging.Logger = None
) -> Optional[image.Nifti1Image]:
    """Resample an image to the space of the target image."""
    try:
        # Validate input image
        img_data = img.get_fdata()
        if np.any(np.isnan(img_data)) or np.any(np.isinf(img_data)):
            if logger:
                logger.error(f"Invalid data in image: contains NaN or Inf")
            return None
        
        # Perform resampling
        resampled_img = resample_to_img(img, target_img, interpolation=interpolation)
        resampled_img.to_filename(output_path)
        
        if logger:
            logger.info(f"Resampled image saved to {output_path}")
        
        return resampled_img
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to resample image: {str(e)}")
        return None

# =============================================================================
# CONFOUNDS PROCESSING
# =============================================================================

def process_confounds(confounds_file: str, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.Series]:
    """Process confounds file and return motion parameters and valid timepoints."""
    try:
        confounds_df = pd.read_csv(confounds_file, sep='\t')
        
        # Select aCompCor components (top N)
        compcor_cols = [
            col for col in confounds_df.columns 
            if 'a_comp_cor' in col
        ][:DEFAULT_CONFIG['compcor_components']]
        
        # Select motion parameters
        motion_cols = [
            'trans_x', 'trans_y', 'trans_z',
            'rot_x', 'rot_y', 'rot_z',
            'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
            'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1'
        ]
        available_motion = [col for col in motion_cols if col in confounds_df.columns]
        
        # Combine selected confounds
        selected_cols = compcor_cols + available_motion
        motion_params = confounds_df[selected_cols].fillna(0) if selected_cols else pd.DataFrame(index=confounds_df.index)
        
        # Create motion flags
        if 'framewise_displacement' in confounds_df.columns:
            fd_flags = confounds_df['framewise_displacement'].fillna(0) > DEFAULT_CONFIG['fd_threshold']
        else:
            logger.warning("No framewise_displacement column found, assuming no excessive motion")
            fd_flags = pd.Series([False] * len(confounds_df))
        
        valid_timepoints = ~fd_flags
        
        logger.info(f"Confounds processed: {len(compcor_cols)} aCompCor, {len(available_motion)} motion parameters")
        logger.info(f"Valid timepoints: {valid_timepoints.sum()}/{len(valid_timepoints)}")
        
        return motion_params, valid_timepoints
        
    except Exception as e:
        logger.error(f"Failed to process confounds: {str(e)}")
        raise

# =============================================================================
# FUNCTIONAL CONNECTIVITY COMPUTATION
# =============================================================================

def extract_time_series(
    fmri_img: image.Nifti1Image,
    atlas: image.Nifti1Image,
    brain_mask: image.Nifti1Image,
    motion_params: pd.DataFrame,
    valid_timepoints: pd.Series,
    work_dir: str,
    logger: logging.Logger
) -> Optional[np.ndarray]:
    """Extract ROI time series using NiftiLabelsMasker."""
    try:
        masker = NiftiLabelsMasker(
            labels_img=atlas,
            mask_img=brain_mask,
            standardize='zscore',
            memory=os.path.join(work_dir, 'nilearn_cache'),
            memory_level=1,
            detrend=True,
            low_pass=DEFAULT_CONFIG['low_pass'],
            high_pass=DEFAULT_CONFIG['high_pass'],
            t_r=DEFAULT_CONFIG['tr'],
            confounds=motion_params[valid_timepoints] if not motion_params.empty else None
        )
        
        time_series = masker.fit_transform(
            fmri_img,
            confounds=motion_params[valid_timepoints] if not motion_params.empty else None
        )
        
        # Apply motion censoring
        time_series = time_series[valid_timepoints]
        
        logger.info(f"Extracted time series with shape: {time_series.shape}")
        
        if time_series.shape[0] < DEFAULT_CONFIG['min_timepoints']:
            logger.error(f"Time series too short ({time_series.shape[0]} timepoints)")
            return None
        
        return time_series
        
    except Exception as e:
        logger.error(f"Failed to extract time series: {str(e)}")
        return None

def compute_connectivity_measures(
    time_series: np.ndarray,
    roi_names: List[str],
    output_prefix: str,
    logger: logging.Logger
) -> Tuple[str, str]:
    """Compute ROI-to-ROI connectivity measures and save results."""
    try:
        # Compute correlation matrix
        logger.info("Computing correlation matrix...")
        corr_matrix = np.corrcoef(time_series.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # Save correlation matrix
        output_matrix = f"{output_prefix}_roiroi_matrix.npy"
        np.save(output_matrix, corr_matrix)
        logger.info(f"Saved correlation matrix: {output_matrix}")
        
        # Compute pair-wise FC for each unique ROI pair
        logger.info("Computing pairwise functional connectivity...")
        pairwise_fc = []
        for i, j in combinations(range(len(roi_names)), 2):
            pairwise_fc.append({
                'ROI_1': roi_names[i],
                'ROI_2': roi_names[j],
                'FC': corr_matrix[i, j]
            })
        
        # Save pair-wise FC
        output_pairwise_csv = f"{output_prefix}_pairwise_fc.csv"
        pd.DataFrame(pairwise_fc).to_csv(output_pairwise_csv, index=False)
        logger.info(f"Saved pairwise FC: {output_pairwise_csv} with {len(pairwise_fc)} pairs")
        
        return output_matrix, output_pairwise_csv
        
    except Exception as e:
        logger.error(f"Failed to compute connectivity measures: {str(e)}")
        raise

# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_run(
    fmri_file: str,
    confounds_file: str,
    atlas: image.Nifti1Image,
    brain_mask: str,
    output_prefix: str,
    work_dir: str,
    logger: logging.Logger
) -> Optional[Tuple[str, str]]:
    """Process a single fMRI run to compute ROI-to-ROI functional connectivity."""
    try:
        logger.info(f"Processing run: {os.path.basename(fmri_file)}")
        
        # Validate input files
        for file_path in [fmri_file, brain_mask, confounds_file]:
            if not os.path.exists(file_path):
                logger.error(f"File does not exist: {file_path}")
                return None
        
        # Load images
        fmri_img = image.load_img(fmri_file)
        brain_mask_img = image.load_img(brain_mask)
        
        # Resample to atlas space
        logger.info("Resampling images to atlas space...")
        fmri_resampled_path = os.path.join(work_dir, f"{os.path.basename(output_prefix)}_resampled_fmri.nii.gz")
        mask_resampled_path = os.path.join(work_dir, f"{os.path.basename(output_prefix)}_resampled_mask.nii.gz")
        
        fmri_img = resample_to_atlas_space(fmri_img, atlas, fmri_resampled_path, 'continuous', logger)
        if fmri_img is None:
            return None
        
        brain_mask_img = resample_to_atlas_space(brain_mask_img, atlas, mask_resampled_path, 'nearest', logger)
        if fmri_img is None:
            return None
        
        # Process confounds
        logger.info("Processing confounds...")
        motion_params, valid_timepoints = process_confounds(confounds_file, logger)
        
        if valid_timepoints.sum() < DEFAULT_CONFIG['min_timepoints']:
            logger.error(f"Too few valid timepoints ({valid_timepoints.sum()})")
            return None
        
        # Extract time series
        logger.info("Extracting ROI time series...")
        time_series = extract_time_series(
            fmri_img, atlas, brain_mask_img, motion_params, valid_timepoints, work_dir, logger
        )
        
        if time_series is None:
            return None
        
        # Compute connectivity measures
        logger.info("Computing connectivity measures...")
        results = compute_connectivity_measures(time_series, roi_names, output_prefix, logger)
        
        logger.info(f"Run processing completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Failed to process run: {str(e)}")
        return None

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to process ROI-to-ROI functional connectivity analysis."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Handle usage and help requests
    if args.usage:
        print_examples()
        return
    
    if args.list_atlases:
        print_available_atlases()
        return
    
    # Validate arguments
    if args.label_pattern == 'nilearn':
        # For Nilearn atlases, labels are not needed
        if args.atlas not in NILEARN_ATLASES:
            print(f"Error: '{args.atlas}' is not a valid Nilearn atlas")
            print("Available Nilearn atlases:", list(NILEARN_ATLASES.keys()))
            print("\nRun with --list-atlases to see all available atlases and their parameters")
            return
        
        # Atlas parameters are required for Nilearn atlases
        if not args.atlas_params:
            print(f"Error: --atlas-params is required for Nilearn atlas '{args.atlas}'")
            print("\nExample:")
            print(f"  python ROI_1st.py --subject sub-AOCD001 --atlas {args.atlas} --atlas-params '{{\"n_rois\": 400}}' --label-pattern nilearn")
            return
    else:
        # For non-Nilearn atlases, labels are required
        if not args.labels:
            print("Error: --labels is required for non-Nilearn atlases")
            print("Use --label-pattern nilearn for built-in Nilearn atlases")
            print("\nExamples:")
            print("  # For Nilearn atlases:")
            print("  python ROI_1st.py --subject sub-AOCD001 --atlas schaefer_2018 --atlas-params '{\"n_rois\": 400}' --label-pattern nilearn")
            print("  # For custom atlases:")
            print("  python ROI_1st.py --subject sub-AOCD001 --atlas /path/to/atlas.nii.gz --labels /path/to/labels.txt --label-pattern simple")
            return
    
    # Override default configuration with command line arguments
    config = DEFAULT_CONFIG.copy()
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.work_dir:
        config['work_dir'] = args.work_dir
    
    # Setup logging
    logger = setup_logging(config['log_file'])
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("=" * 80)
    logger.info("Starting ROI-to-ROI Functional Connectivity Analysis")
    logger.info("=" * 80)
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Atlas: {args.atlas}")
    if args.atlas_params:
        logger.info(f"Atlas parameters: {args.atlas_params}")
    if args.labels:
        logger.info(f"Labels: {args.labels}")
    logger.info(f"Label pattern: {args.label_pattern}")
    if args.custom_regex:
        logger.info(f"Custom regex: {args.custom_regex}")
    logger.info(f"Output directory: {config['output_dir']}")
    
    try:
        # Create output directories
        os.makedirs(config['output_dir'], exist_ok=True)
        os.makedirs(config['work_dir'], exist_ok=True)
        
        # Load atlas and ROI labels
        logger.info("Setting up atlas and ROI labels...")
        
        if args.label_pattern == 'nilearn':
            # Fetch Nilearn atlas
            try:
                atlas_params = json.loads(args.atlas_params)
                atlas_name = args.atlas
                atlas_img, roi_names = fetch_nilearn_atlas(atlas_name, atlas_params, logger)
                
                # Generate atlas name for output files
                if args.atlas_name:
                    output_atlas_name = args.atlas_name
                else:
                    # Create descriptive name from parameters
                    param_str = "_".join([str(v) for v in atlas_params.values()])
                    output_atlas_name = f"{atlas_name}_{param_str}"
                
                logger.info(f"Fetched {atlas_name} atlas with {len(roi_names)} ROIs, shape={atlas_img.shape}")
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in --atlas-params: {str(e)}")
                return
            except Exception as e:
                logger.error(f"Failed to fetch Nilearn atlas: {str(e)}")
                return
        else:
            # Load custom atlas
            atlas_img, roi_names = load_atlas_and_labels(
                args.atlas, 
                args.labels, 
                args.label_pattern, 
                args.custom_regex,
                args.expected_rois,
                logger
            )
            
            # Determine atlas name for output files
            if args.atlas_name:
                output_atlas_name = args.atlas_name
            else:
                output_atlas_name = os.path.splitext(os.path.basename(args.atlas))[0]
        
        n_rois = len(roi_names)
        logger.info(f"Atlas loaded: shape={atlas_img.shape}, ROIs={n_rois}")
        
        # Process each session
        subjects = [args.subject]
        processed_any = False
        
        for subject in subjects:
            logger.info(f"Processing subject: {subject}")
            
            for session in config['sessions']:
                try:
                    logger.info(f"Processing session: {session}")
                    
                    # Validate paths
                    paths = validate_paths(subject, session, config['bids_dir'], logger)
                    if not paths:
                        logger.warning(f"Skipping session {session} for {subject}")
                        continue
                    
                    brain_mask_path, fmri_file = paths
                    
                    # Find confounds file
                    confounds_file = find_confounds_file(subject, session, config['bids_dir'], logger)
                    if not confounds_file:
                        logger.warning(f"No confounds file found for {subject} {session}")
                        continue
                    
                    # Extract run ID
                    run_id = extract_run_id(fmri_file)
                    logger.info(f"Run ID: {run_id}")
                    
                    # Define output prefix
                    output_prefix = os.path.join(
                        config['output_dir'],
                        f'{subject}_{session}_task-rest_run-{run_id}_{output_atlas_name}'
                    )
                    
                    # Process the run
                    result = process_run(
                        fmri_file, confounds_file, atlas_img, brain_mask_path,
                        output_prefix, config['work_dir'], logger
                    )
                    
                    if result:
                        # Rename output files to indicate they are from single runs
                        src_matrix, src_pairwise_csv = result
                        
                        # Move files to final locations
                        file_moves = [
                            (src_matrix, f'{subject}_{session}_task-rest_{output_atlas_name}_roiroi_matrix_avg.npy'),
                            (src_pairwise_csv, f'{subject}_{session}_task-rest_{output_atlas_name}_pairwise_fc_avg.csv')
                        ]
                        
                        for src, dst_name in file_moves:
                            dst_path = os.path.join(config['output_dir'], dst_name)
                            os.rename(src, dst_path)
                            logger.info(f"Moved {os.path.basename(src)} to {dst_name}")
                        
                        processed_any = True
                    else:
                        logger.warning(f"No results generated for {subject} {session}")
                
                except Exception as e:
                    logger.error(f"Failed to process {subject} {session}: {str(e)}")
                    continue
        
        if not processed_any:
            logger.error(f"No functional connectivity matrices generated for {args.subject}")
        else:
            logger.info(f"Analysis completed successfully for {args.subject} using atlas: {output_atlas_name}")
    
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise
    
    finally:
        logger.info("=" * 80)
        logger.info("ROI-to-ROI Functional Connectivity Analysis Completed")
        logger.info("=" * 80)

def print_quick_help():
    """Print quick help information."""
    quick_help = """
QUICK HELP - ROI-to-ROI Functional Connectivity Analysis
========================================================

BASIC USAGE:
  python ROI_1st.py --subject <SUBJECT_ID> --atlas <ATLAS_NAME> --label-pattern <PATTERN>

QUICK EXAMPLES:
  1. Nilearn Atlas (Schaefer 2018):
     python ROI_1st.py --subject sub-AOCD001 --atlas schaefer_2018 --atlas-params '{"n_rois": 400}' --label-pattern nilearn

  2. Custom Atlas with Simple Labels:
     python ROI_1st.py --subject sub-AOCD001 --atlas /path/to/atlas.nii.gz --labels /path/to/labels.txt --label-pattern simple

  3. Power 2011 Atlas:
     python ROI_1st.py --subject sub-AOCD001 --atlas /path/to/power_atlas.nii.gz --labels /path/to/power_labels.txt --label-pattern power

HELP OPTIONS:
  --help          Show full help with all arguments
  --usage         Show detailed usage examples
  --list-atlases  List available Nilearn atlases and their parameters

For more information, run with --usage or --help.
"""
    print(quick_help)

if __name__ == "__main__":
    try:
        # Check for help requests first
        if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
            print_quick_help()
            sys.exit(0)
        
        main()
    except Exception as e:
        logging.error("Main execution failed: %s", e)
        print(f"\nError: {e}")
        print("\nFor help, run: python ROI_1st.py --help")
        print("For usage examples, run: python ROI_1st.py --usage")
        print("For available atlases, run: python ROI_1st.py --list-atlases")
        raise