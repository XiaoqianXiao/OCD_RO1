#!/usr/bin/env python3
"""
ROI-to-ROI and ROI-to-Network Functional Connectivity Analysis using Customizable Atlas

This script computes functional connectivity matrices using any user-specified atlas
for resting-state fMRI data. It processes individual subjects and sessions,
generating various connectivity measures and network analyses.

The script is compatible with both custom atlases and Nilearn built-in atlases (e.g., Schaefer 2018,
Harvard-Oxford, Power 2011, etc.) and automatically handles network labeling for different atlas types.

Author: [Your Name]
Date: [Current Date]

USAGE EXAMPLES:
==============

1. Power 2011 Atlas (Default, Network-based):
   python NW_1st.py \
     --subject sub-AOCD001 \
     --atlas power_2011 \
     --label-pattern power

2. Schaefer 2018 Atlas (Network-based):
   python NW_1st.py \
     --subject sub-AOCD001 \
     --atlas schaefer_2018 \
     --atlas-params '{"n_rois": 400, "yeo_networks": 7, "resolution_mm": 2}' \
     --label-pattern nilearn

3. Harvard-Oxford Atlas (Cortical Regions):
   python NW_1st.py \
     --subject sub-AOCD001 \
     --atlas harvard_oxford \
     --atlas-params '{"atlas_name": "cort-maxprob-thr25-2mm"}' \
     --label-pattern nilearn

4. Custom Atlas with Network Labels:
   python NW_1st.py \
     --subject sub-AOCD001 \
     --atlas /path/to/custom_atlas.nii.gz \
     --labels /path/to/network_labels.txt \
     --label-pattern custom \
     --custom-regex "network_(\\d+)_(.+)" \
     --atlas-name custom_atlas

5. With Custom Output and Work Directories:
   python NW_1st.py \
     --subject sub-AOCD001 \
     --atlas power_2011 \
     --label-pattern power \
     --output-dir /custom/output/path \
     --work-dir /custom/work/path

ATLAS TYPES:
============

Built-in Nilearn Atlases:
- power_2011: Power 2011 atlas (264 ROIs, 13 networks) - DEFAULT
- schaefer_2018: Schaefer 2018 parcellation (100-1000 ROIs, 7/17 networks)
- harvard_oxford: Harvard-Oxford cortical/subcortical atlases
- aal: Automated Anatomical Labeling atlas (116 ROIs)
- talairach: Talairach atlas (1107 ROIs)
- coords_power_2012: Power 2012 coordinate-based atlas (264 ROIs)
- pauli_2017: Pauli 2017 subcortical atlas (16 ROIs)
- yeo_2011: Yeo 2011 7/17 network parcellation

Custom Atlases:
- File-based: Provide path to .nii.gz file and network labels file

LABEL PATTERN TYPES:
===================

- power: Power atlas format (e.g., "Default_1", "DorsAttn_2")
- nilearn: Built-in atlas labels (automatically handled)
- custom: User-defined regex pattern for complex formats

OUTPUT FILES:
=============

- {subject}_{session}_task-rest_{atlas_name}_roiroi_matrix_avg.npy: Correlation matrix
- {subject}_{session}_task-rest_{atlas_name}_roi_fc_avg.csv: ROI-level FC (within vs between network)
- {subject}_{session}_task-rest_{atlas_name}_roiroi_fc_avg.csv: ROI-to-ROI FC
- {subject}_{session}_task-rest_{atlas_name}_network_fc_avg.csv: ROI-to-network FC
- {subject}_{session}_task-rest_{atlas_name}_network_summary_avg.csv: Network summary statistics

REQUIREMENTS:
=============

- BIDS-formatted fMRI data
- Preprocessed fMRI images (fmriprep output)
- Atlas file (.nii.gz format) or Nilearn atlas name
- Network labels file (text format) or automatic for Nilearn atlases
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
# Nilearn atlas imports with version compatibility
try:
    from nilearn.datasets import (
        load_mni152_template,
        fetch_atlas_schaefer_2018, fetch_atlas_harvard_oxford,
        fetch_atlas_aal, fetch_atlas_talairach,
        fetch_atlas_coords_power_2012, fetch_atlas_pauli_2017,
        fetch_atlas_yeo_2011
    )
except ImportError as e:
    # Try alternative function names for older Nilearn versions
    try:
        from nilearn.datasets import (
            load_mni152_template,
            fetch_atlas_harvard_oxford, fetch_atlas_aal, 
            fetch_atlas_talairach, fetch_atlas_yeo_2011
        )
        # Try alternative names for Schaefer atlas
        try:
            from nilearn.datasets import fetch_atlas_schaefer_2018
        except ImportError:
            try:
                from nilearn.datasets import fetch_schaefer_2018 as fetch_atlas_schaefer_2018
            except ImportError:
                try:
                    from nilearn.datasets import fetch_atlas_schaefer as fetch_atlas_schaefer_2018
                except ImportError:
                    print("Warning: Schaefer 2018 atlas not available in this Nilearn version")
                    fetch_atlas_schaefer_2018 = None
        
        # Try alternative names for Power 2012 atlas
        try:
            from nilearn.datasets import fetch_atlas_coords_power_2012
        except ImportError:
            try:
                from nilearn.datasets import fetch_coords_power_2012 as fetch_atlas_coords_power_2012
            except ImportError:
                print("Warning: Power 2012 atlas not available in this Nilearn version")
                fetch_atlas_coords_power_2012 = None
        
        # Try alternative names for Pauli 2017 atlas
        try:
            from nilearn.datasets import fetch_atlas_pauli_2017
        except ImportError:
            try:
                from nilearn.datasets import fetch_pauli_2017 as fetch_atlas_pauli_2017
            except ImportError:
                print("Warning: Pauli 2017 atlas not available in this Nilearn version")
                fetch_atlas_pauli_2017 = None
                
    except ImportError as e2:
        print(f"Critical import error: {e2}")
        print("Please check your Nilearn installation")
        sys.exit(1)
from nilearn.image import resample_to_img
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

# Directory configuration
CONFIG = {
    'project_dir': '/project/6079231/dliang55/R01_AOCD',
    'bids_dir': '/project/6079231/dliang55/R01_AOCD/derivatives/fmriprep-1.4.1',
    'scratch_dir': '/scratch/xxqian',
    'output_dir': '/scratch/xxqian/OCD',
    'work_dir': '/scratch/xxqian/work_flow',
    'roi_dir': '/scratch/xxqian/roi',
    'log_file': '/scratch/xxqian/OCD/roi_to_roi_fc_analysis.log'
}

# Analysis parameters
ANALYSIS_PARAMS = {
    'sessions': ['ses-baseline', 'ses-followup'],
    'tr': 2.0,
    'low_pass': 0.1,
    'high_pass': 0.01,
    'fd_threshold': 0.5,
    'min_timepoints': 10,
    'sphere_radius': 3,
    'atlas_resolution': 2
}

# Nilearn atlas configurations (only include available atlases)
NILEARN_ATLASES = {}

# Add Schaefer 2018 atlas if available
if 'fetch_atlas_schaefer_2018' in globals() and fetch_atlas_schaefer_2018 is not None:
    NILEARN_ATLASES['schaefer_2018'] = {
        'function': fetch_atlas_schaefer_2018,
        'default_params': {'n_rois': 400, 'yeo_networks': 7, 'resolution_mm': 2},
        'description': 'Schaefer 2018 parcellation (100-1000 ROIs, 7/17 networks)',
        'param_options': {
            'n_rois': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'yeo_networks': [7, 17],
            'resolution_mm': [1, 2]
        },
        'network_based': True,
        'network_names': {
            7: ['Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention', 'Limbic', 'Frontoparietal', 'Default'],
            17: ['Visual1', 'Visual2', 'Somatomotor1', 'Somatomotor2', 'Dorsal Attention1', 'Dorsal Attention2', 
                 'Ventral Attention1', 'Ventral Attention2', 'Limbic1', 'Limbic2', 'Frontoparietal1', 'Frontoparietal2',
                 'Default1', 'Default2', 'Temporal Parietal', 'Orbital Frontal', 'Cingulo-opercular']
        }
    }

# Add Harvard-Oxford atlas if available
if 'fetch_atlas_harvard_oxford' in globals():
    NILEARN_ATLASES['harvard_oxford'] = {
        'function': fetch_atlas_harvard_oxford,
        'default_params': {'atlas_name': 'cort-maxprob-thr25-2mm'},
        'description': 'Harvard-Oxford cortical/subcortical atlases',
        'param_options': {
            'atlas_name': [
                'cort-maxprob-thr25-2mm', 'cort-maxprob-thr50-2mm',
                'sub-maxprob-thr25-2mm', 'sub-maxprob-thr50-2mm',
                'cort-prob-2mm', 'sub-prob-2mm'
            ]
        },
        'network_based': False
    }

# Add AAL atlas if available
if 'fetch_atlas_aal' in globals():
    NILEARN_ATLASES['aal'] = {
        'function': fetch_atlas_aal,
        'default_params': {},
        'description': 'Automated Anatomical Labeling atlas (116 ROIs)',
        'param_options': {},
        'network_based': False
    }

# Add Talairach atlas if available
if 'fetch_atlas_talairach' in globals():
    NILEARN_ATLASES['talairach'] = {
        'function': fetch_atlas_talairach,
        'default_params': {},
        'description': 'Talairach atlas (1107 ROIs)',
        'param_options': {},
        'network_based': False
    }

# Add Power 2012 atlas if available
if 'fetch_atlas_coords_power_2012' in globals() and fetch_atlas_coords_power_2012 is not None:
    NILEARN_ATLASES['coords_power_2012'] = {
        'function': fetch_atlas_coords_power_2012,
        'default_params': {},
        'description': 'Power 2012 atlas (264 ROIs, 13 networks)',
        'param_options': {},
        'network_based': True
    }

# Add Pauli 2017 atlas if available
if 'fetch_atlas_pauli_2017' in globals() and fetch_atlas_pauli_2017 is not None:
    NILEARN_ATLASES['pauli_2017'] = {
        'function': fetch_atlas_pauli_2017,
        'default_params': {},
        'description': 'Pauli 2017 atlas (83 ROIs, 7 networks)',
        'param_options': {},
        'network_based': True
    }

# Add Yeo 2011 atlas if available
if 'fetch_atlas_yeo_2011' in globals():
    NILEARN_ATLASES['yeo_2011'] = {
        'function': fetch_atlas_yeo_2011,
        'default_params': {'n_rois': 7},
        'description': 'Yeo 2011 network parcellation (7 or 17 networks)',
        'param_options': {
            'n_rois': [7, 17]
        },
        'network_based': True
    }

# =============================================================================
# USAGE AND HELP FUNCTIONS
# =============================================================================

def print_usage():
    """Print comprehensive usage information."""
    usage_text = """
ROI-to-ROI and ROI-to-Network Functional Connectivity Analysis
=============================================================

DESCRIPTION:
This script performs ROI-to-ROI and ROI-to-network functional connectivity analysis using any
user-specified atlas. It processes resting-state fMRI data and generates correlation matrices,
network-level connectivity measures, and summary statistics.

BASIC USAGE:
python NW_1st.py --subject <SUBJECT_ID> --atlas <ATLAS_NAME> --label-pattern <PATTERN>

REQUIRED ARGUMENTS:
------------------
--subject <SUBJECT_ID>     Subject ID (e.g., sub-AOCD001)
--atlas <ATLAS_NAME>       Atlas name or path to atlas file

OPTIONAL ARGUMENTS:
------------------
--atlas-params <JSON>      JSON string with atlas parameters (for Nilearn atlases)
--labels <LABELS_PATH>     Path to network labels file (for custom atlases)
--label-pattern <PATTERN>  Pattern for reading network labels
--custom-regex <REGEX>     Custom regex for label parsing (when pattern is 'custom')
--atlas-name <NAME>        Custom name for atlas (used in output filenames)
--output-dir <PATH>        Output directory (overrides default)
--work-dir <PATH>          Working directory (overrides default)
--verbose                  Enable verbose logging
--help                     Show this help message
--usage                    Show detailed usage examples
--list-atlases            List available Nilearn atlases and their parameters

ATLAS TYPES:
------------
Built-in Nilearn Atlases:
- power_2011: Power 2011 atlas (264 ROIs, 13 networks) - DEFAULT
- schaefer_2018: Schaefer 2018 parcellation (100-1000 ROIs, 7/17 networks)
- harvard_oxford: Harvard-Oxford cortical/subcortical atlases
- aal: Automated Anatomical Labeling atlas (116 ROIs)
- talairach: Talairach atlas (1107 ROIs)
- coords_power_2012: Power 2012 coordinate-based atlas (264 ROIs)
- pauli_2017: Pauli 2017 subcortical atlas (16 ROIs)
- yeo_2011: Yeo 2011 network parcellation (7/17 networks)

Custom Atlases:
- File-based: Provide path to .nii.gz file and network labels file

LABEL PATTERN TYPES:
--------------------
power            Power atlas format (e.g., "Default_1", "DorsAttn_2")
nilearn          Built-in atlas labels (automatically handled)
custom           User-defined regex pattern

EXAMPLES:
---------
1. Power 2011 Atlas (Default):
   python NW_1st.py \\
     --subject sub-AOCD001 \\
     --atlas power_2011 \\
     --label-pattern power

2. Schaefer 2018 Atlas:
   python NW_1st.py \\
     --subject sub-AOCD001 \\
     --atlas schaefer_2018 \\
     --atlas-params '{"n_rois": 400, "yeo_networks": 7, "resolution_mm": 2}' \\
     --label-pattern nilearn

3. Custom Atlas with Network Labels:
   python NW_1st.py \\
     --subject sub-AOCD001 \\
     --atlas /path/to/custom_atlas.nii.gz \\
     --labels /path/to/network_labels.txt \\
     --label-pattern custom \\
     --custom-regex "network_(\\d+)_(.+)" \\
     --atlas-name custom_atlas

OUTPUT FILES:
-------------
- {subject}_{session}_task-rest_{atlas_name}_roiroi_matrix_avg.npy: Correlation matrix
- {subject}_{session}_task-rest_{atlas_name}_roi_fc_avg.csv: ROI-level FC
- {subject}_{session}_task-rest_{atlas_name}_roiroi_fc_avg.csv: ROI-to-ROI FC
- {subject}_{session}_task-rest_{atlas_name}_network_fc_avg.csv: ROI-to-network FC
- {subject}_{session}_task-rest_{atlas_name}_network_summary_avg.csv: Network summary

REQUIREMENTS:
-------------
- BIDS-formatted fMRI data
- Preprocessed fMRI images (fmriprep output)
- Atlas file (.nii.gz format) or Nilearn atlas name
- Network labels file (text format) or automatic for Nilearn atlases
- Confounds file (motion parameters, aCompCor)

For more information, see the script docstring or run with --help.
"""
    print(usage_text)

def print_examples():
    """Print detailed usage examples."""
    examples_text = """
DETAILED USAGE EXAMPLES
=======================

1. POWER 2011 ATLAS (Default, Network-based)
   ------------------------------------------
   This is the original atlas used in the script. It contains 264 ROIs organized by 13 networks.
   
   python NW_1st.py \\
     --subject sub-AOCD001 \\
     --atlas power_2011 \\
     --label-pattern power
   
   No additional parameters needed.

2. SCHAEFER 2018 ATLAS (Built-in Nilearn)
   ---------------------------------------
   This is a popular parcellation with 100-1000 ROIs organized by functional networks.
   
   # Basic usage with default parameters (400 ROIs, 7 networks, 2mm resolution)
   python NW_1st.py \\
     --subject sub-AOCD001 \\
     --atlas schaefer_2018 \\
     --label-pattern nilearn
   
   # Custom parameters (200 ROIs, 7 networks, 2mm resolution)
   python NW_1st.py \\
     --subject sub-AOCD001 \\
     --atlas schaefer_2018 \\
     --atlas-params '{"n_rois": 200, "yeo_networks": 7, "resolution_mm": 2}' \\
     --label-pattern nilearn
   
   # High-resolution version (1000 ROIs, 17 networks, 1mm resolution)
   python NW_1st.py \\
     --subject sub-AOCD001 \\
     --atlas schaefer_2018 \\
     --atlas-params '{"n_rois": 1000, "yeo_networks": 17, "resolution_mm": 1}' \\
     --label-pattern nilearn
   
   Available parameters:
   - n_rois: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
   - yeo_networks: 7 or 17
   - resolution_mm: 1 or 2
   
   Network configurations:
   - 7 networks: Visual, Somatomotor, Dorsal Attention, Ventral Attention, Limbic, Frontoparietal, Default
   - 17 networks: Extended version with sub-networks (e.g., Visual1, Visual2, etc.)
   
   Note: The script automatically generates appropriate network labels based on your parameters.

3. HARVARD-OXFORD ATLAS (Built-in Nilearn)
   -----------------------------------------
   This atlas provides cortical and subcortical parcellations.
   
   python NW_1st.py \\
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

4. YEO 2011 ATLAS (Built-in Nilearn)
   ----------------------------------
   This atlas provides 7 or 17 network parcellations.
   
   python NW_1st.py \\
     --subject sub-AOCD001 \\
     --atlas yeo_2011 \\
     --atlas-params '{"n_rois": 17}' \\
     --label-pattern nilearn
   
   Available parameters:
   - n_rois: 7 or 17

5. CUSTOM ATLAS WITH NETWORK LABELS
   --------------------------------
   For atlases with custom network label formats.
   
   python NW_1st.py \\
     --subject sub-AOCD001 \\
     --atlas /path/to/custom_atlas.nii.gz \\
     --labels /path/to/network_labels.txt \\
     --label-pattern custom \\
     --custom-regex "network_(\\d+)_(.+)" \\
     --atlas-name custom_atlas
   
   Expected labels format:
   network_1_Default
   network_2_Default
   network_3_DorsAttn
   ...
   
   The regex "network_(\\d+)_(.+)" will extract:
   - Group 1: The ROI number (1, 2, 3, ...)
   - Group 2: The network name (Default, DorsAttn, ...)

6. WITH CUSTOM DIRECTORIES
   ------------------------
   Override default output and working directories.
   
   python NW_1st.py \\
     --subject sub-AOCD001 \\
     --atlas schaefer_2018 \\
     --atlas-params '{"n_rois": 400}' \\
     --label-pattern nilearn \\
     --output-dir /custom/output/path \\
     --work-dir /custom/work/path

7. VERBOSE LOGGING
   ----------------
   Enable detailed logging for debugging.
   
   python NW_1st.py \\
     --subject sub-AOCD001 \\
     --atlas power_2011 \\
     --label-pattern power \\
     --verbose

8. TESTING SCHAEFER 2018 CONFIGURATION
   ------------------------------------
   Test if the Schaefer 2018 atlas is properly configured.
   
   python NW_1st.py --test-schaefer
   
   This will verify:
   - Atlas function availability
   - Parameter validation
   - Network name generation
   - Configuration consistency

NILEARN ATLAS PARAMETERS:
-------------------------
Schaefer 2018:
  - n_rois: Number of ROIs (100-1000)
  - yeo_networks: Number of networks (7 or 17)
  - resolution_mm: Spatial resolution (1 or 2mm)
  
  Network names (automatically generated):
  - 7 networks: Visual, Somatomotor, Dorsal Attention, Ventral Attention, Limbic, Frontoparietal, Default
  - 17 networks: Extended version with sub-networks for more detailed analysis

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
- Power 2011: power_2011_roiroi_matrix_avg.npy
- Schaefer 2018: schaefer_2018_400_7_2_roiroi_matrix_avg.npy
- Custom: custom_atlas_roiroi_matrix_avg.npy

NETWORK ANALYSIS FEATURES:
-------------------------
- ROI-to-ROI functional connectivity
- ROI-to-network functional connectivity
- Within-network vs between-network connectivity
- Network summary statistics
- Automatic network label handling for Nilearn atlases

TROUBLESHOOTING:
----------------
1. Check that atlas name is valid for Nilearn atlases
2. Verify atlas parameters are correct and in valid ranges
3. Ensure BIDS directory structure is correct
4. Check that confounds files are available
5. Verify output directory is writable
6. Use --verbose for detailed error messages
7. For Nilearn atlases, ensure internet connection for first download

SCHAEFER 2018 SPECIFIC TROUBLESHOOTING:
----------------------------------------
1. Parameter validation errors:
   - Ensure n_rois is one of: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
   - Ensure yeo_networks is either 7 or 17
   - Ensure resolution_mm is either 1 or 2
   
2. Network label issues:
   - Use --test-schaefer to verify configuration
   - Check that n_rois is evenly divisible by yeo_networks for optimal results
   - For 17 networks, ensure you have enough ROIs (recommend at least 200)
   
3. Memory issues:
   - Higher ROI counts (800-1000) require more memory
   - Consider using lower resolution (2mm) for large ROI counts
   - Monitor memory usage with --verbose logging
   
4. First-time usage:
   - The atlas will be downloaded automatically on first use
   - Ensure stable internet connection
   - Check available disk space for atlas storage
"""
    print(examples_text)

def print_available_atlases():
    """Print information about available atlases for debugging."""
    print(f"Available atlases in this Nilearn version: {list(NILEARN_ATLASES.keys())}")
    print(f"Total atlases available: {len(NILEARN_ATLASES)}")
    
    if len(NILEARN_ATLASES) == 0:
        print("WARNING: No atlases are available!")
        print("This might indicate a Nilearn version compatibility issue.")
        return False
    
    # Print details for each available atlas
    for atlas_name, atlas_info in NILEARN_ATLASES.items():
        print(f"  - {atlas_name}: {atlas_info['description']}")
    
    return True

def test_schaefer_2018_configuration():
    """Test the Schaefer 2018 atlas configuration to ensure it's working correctly."""
    print("Testing Schaefer 2018 Atlas Configuration")
    print("=" * 50)
    
    if 'schaefer_2018' not in NILEARN_ATLASES:
        print("❌ Schaefer 2018 atlas is not available")
        return False
    
    atlas_info = NILEARN_ATLASES['schaefer_2018']
    print(f"✅ Atlas function: {atlas_info['function'].__name__}")
    print(f"✅ Default parameters: {atlas_info['default_params']}")
    print(f"✅ Available parameters: {atlas_info['param_options']}")
    print(f"✅ Network-based: {atlas_info['network_based']}")
    
    if 'network_names' in atlas_info:
        print(f"✅ Network names available for: {list(atlas_info['network_names'].keys())} networks")
        for n_networks, names in atlas_info['network_names'].items():
            print(f"   {n_networks} networks: {names}")
    else:
        print("❌ No predefined network names")
    
    # Test parameter validation
    try:
        test_params = {'n_rois': 400, 'yeo_networks': 7, 'resolution_mm': 2}
        validated = validate_atlas_params('schaefer_2018', test_params)
        print(f"✅ Parameter validation passed: {validated}")
    except Exception as e:
        print(f"❌ Parameter validation failed: {e}")
        return False
    
    print("\nSchaefer 2018 Atlas is properly configured! 🎉")
    return True

# =============================================================================
# USAGE AND HELP FUNCTIONS
# =============================================================================

# =============================================================================
# ATLAS FETCHING FUNCTIONS
# =============================================================================

def fetch_nilearn_atlas(atlas_name: str, atlas_params: Dict[str, Any], logger: logging.Logger) -> Tuple[image.Nifti1Image, Dict[int, str]]:
    """Fetch a built-in Nilearn atlas and return the image and network labels."""
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
        
        # Generate network labels
        network_labels = {}
        if atlas_info['network_based']:
            # For network-based atlases, generate network labels
            if atlas_name == 'schaefer_2018' and 'network_names' in atlas_info:
                # Use predefined network names for Schaefer 2018
                yeo_networks = params.get('yeo_networks', 7)
                if yeo_networks in atlas_info['network_names']:
                    network_names = atlas_info['network_names'][yeo_networks]
                    # Get the number of ROIs per network
                    n_rois = params.get('n_rois', 400)
                    rois_per_network = n_rois // yeo_networks
                    
                    roi_idx = 1
                    for network_idx, network_name in enumerate(network_names):
                        for _ in range(rois_per_network):
                            network_labels[roi_idx] = network_name
                            roi_idx += 1
                    
                    # Handle any remaining ROIs
                    while roi_idx <= n_rois:
                        network_labels[roi_idx] = network_names[roi_idx % yeo_networks]
                        roi_idx += 1
                    
                    logger.info(f"Generated Schaefer 2018 network labels: {yeo_networks} networks, {n_rois} ROIs")
                else:
                    logger.warning(f"Unknown number of networks for Schaefer 2018: {yeo_networks}")
                    # Fall back to generic approach
                    if 'labels' in atlas_data:
                        labels = atlas_data['labels']
                        for i, label in enumerate(labels):
                            if isinstance(label, str):
                                # Extract network name from label
                                if '_' in label:
                                    network_name = label.rsplit('_', 1)[0]
                                else:
                                    network_name = label
                                network_labels[i + 1] = network_name
                            else:
                                network_labels[i + 1] = f"Network_{i+1}"
                    else:
                        # Generate generic network labels
                        n_rois = len(np.unique(atlas_img.get_fdata())) - 1  # Exclude background (0)
                        for i in range(n_rois):
                            network_labels[i + 1] = f"Network_{i+1}"
            elif 'labels' in atlas_data:
                labels = atlas_data['labels']
                for i, label in enumerate(labels):
                    if isinstance(label, str):
                        # Extract network name from label
                        if '_' in label:
                            network_name = label.rsplit('_', 1)[0]
                        else:
                            network_name = label
                        network_labels[i + 1] = network_name
                    else:
                        network_labels[i + 1] = f"Network_{i+1}"
            else:
                # Generate generic network labels
                n_rois = len(np.unique(atlas_img.get_fdata())) - 1  # Exclude background (0)
                for i in range(n_rois):
                    network_labels[i + 1] = f"Network_{i+1}"
        else:
            # For non-network atlases, generate anatomical labels
            n_rois = len(np.unique(atlas_img.get_fdata())) - 1  # Exclude background (0)
            for i in range(n_rois):
                network_labels[i + 1] = f"Region_{i+1}"
        
        logger.info(f"Successfully fetched {atlas_name} atlas: {len(network_labels)} ROIs, shape={atlas_img.shape}")
        logger.info(f"Generated {len(set(network_labels.values()))} unique network labels")
        
        return atlas_img, network_labels
        
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
    
    # Special validation for Schaefer 2018
    if atlas_name == 'schaefer_2018':
        n_rois = validated_params.get('n_rois', 400)
        yeo_networks = validated_params.get('yeo_networks', 7)
        
        # Check if n_rois is divisible by yeo_networks
        if n_rois % yeo_networks != 0:
            logger.warning(f"Warning: n_rois ({n_rois}) is not evenly divisible by yeo_networks ({yeo_networks})")
            logger.warning(f"This may result in uneven distribution of ROIs across networks")
        
        # Validate that the combination is reasonable
        if n_rois < yeo_networks:
            raise ValueError(f"n_rois ({n_rois}) must be greater than or equal to yeo_networks ({yeo_networks})")
        
        # Check if network names are available for this configuration
        if 'network_names' in atlas_info and yeo_networks in atlas_info['network_names']:
            logger.info(f"Using predefined network names for Schaefer 2018: {yeo_networks} networks")
        else:
            logger.warning(f"No predefined network names for Schaefer 2018 with {yeo_networks} networks")
    
    return validated_params

def validate_atlas_availability(atlas_name: str) -> bool:
    """Validate that the requested atlas is available in the current Nilearn version."""
    if atlas_name not in NILEARN_ATLASES:
        available_atlases = list(NILEARN_ATLASES.keys())
        print(f"ERROR: Atlas '{atlas_name}' is not available in this Nilearn version.")
        print(f"Available atlases: {available_atlases}")
        print("\nThis might be due to:")
        print("1. Using an older version of Nilearn")
        print("2. Atlas function names have changed")
        print("3. Atlas is not installed")
        return False
    return True

def get_atlas_function(atlas_name: str):
    """Get the atlas function, with validation."""
    if not validate_atlas_availability(atlas_name):
        return None
    return NILEARN_ATLASES[atlas_name]['function']

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_file: str) -> logging.Logger:
    """Set up logging configuration with both file and console handlers."""
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create logger
    logger = logging.getLogger('FC_Analysis')
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
# ATLAS MANAGEMENT
# =============================================================================

def load_custom_network_labels(labels_path: str, label_pattern: str, custom_regex: str = None, logger: logging.Logger = None) -> Dict[int, str]:
    """Load network labels from custom file with different patterns."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        with open(labels_path, 'r') as f:
            labels_list = [line.strip() for line in f if line.strip()]
        
        network_labels = {}
        
        if label_pattern == 'power':
            # Power atlas format: "Network_ROI" (e.g., "Default_1", "DorsAttn_2")
            for i, label in enumerate(labels_list):
                parts = label.rsplit('_', 1)
                if len(parts) == 2:
                    network_name, roi_num_str = parts
                    try:
                        roi_num = int(roi_num_str)
                        if roi_num == i + 1:  # ROI numbers should be 1-indexed
                            network_labels[i + 1] = network_name
                        else:
                            logger.warning(f"ROI number mismatch at line {i+1}: {roi_num} != {i+1}")
                    except ValueError:
                        logger.warning(f"Invalid ROI number at line {i+1}: {roi_num_str}")
                        continue
                else:
                    logger.warning(f"Invalid label format at line {i+1}: {label}")
                    continue
        
        elif label_pattern == 'custom' and custom_regex:
            # Custom regex pattern
            try:
                pattern = re.compile(custom_regex)
                for i, label in enumerate(labels_list):
                    match = pattern.match(label)
                    if match:
                        if len(match.groups()) >= 2:
                            roi_num_str, network_name = match.groups()[:2]
                            try:
                                roi_num = int(roi_num_str)
                                if roi_num == i + 1:  # ROI numbers should be 1-indexed
                                    network_labels[i + 1] = network_name
                                else:
                                    logger.warning(f"ROI number mismatch at line {i+1}: {roi_num} != {i+1}")
                            except ValueError:
                                logger.warning(f"Invalid ROI number at line {i+1}: {roi_num_str}")
                                continue
                        else:
                            logger.warning(f"Custom regex must have at least 2 groups for ROI number and network name")
                            continue
                    else:
                        logger.warning(f"Label at line {i+1} does not match custom regex: {label}")
                        continue
            except re.error as e:
                logger.error(f"Invalid custom regex pattern: {e}")
                raise
        
        else:
            # Simple pattern: one label per line
            for i, label in enumerate(labels_list):
                network_labels[i + 1] = label
        
        logger.info(f"Loaded {len(network_labels)} network labels from {labels_path}")
        return network_labels
        
    except Exception as e:
        logger.error(f"Failed to load custom network labels: {str(e)}")
        raise

# =============================================================================
# FILE VALIDATION AND PATH MANAGEMENT
# =============================================================================

def validate_paths(subject: str, session: str, bids_dir: str, logger: logging.Logger) -> Optional[Tuple[str, str, str]]:
    """Validate and return paths for fMRI, brain mask, and confounds files."""
    # Define file patterns
    patterns = {
        'mask': f'{subject}_{session}_task-rest*_space-MNI152NLin6Asym_desc-brain_mask.nii.gz',
        'fmri': f'{subject}_{session}_task-rest*_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz',
        'confounds': f'{subject}_{session}_task-rest*_desc-confounds_regressors.tsv'
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
    
    return (
        file_paths['mask'],
        file_paths['fmri'],
        file_paths['confounds']
    )

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
        
        # Select aCompCor components (top 5)
        compcor_cols = [
            col for col in confounds_df.columns 
            if 'a_comp_cor' in col or 'aCompCor' in col
        ][:5]
        
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
            fd_flags = confounds_df['framewise_displacement'].fillna(0) > ANALYSIS_PARAMS['fd_threshold']
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
            low_pass=ANALYSIS_PARAMS['low_pass'],
            high_pass=ANALYSIS_PARAMS['high_pass'],
            t_r=ANALYSIS_PARAMS['tr'],
            confounds=motion_params[valid_timepoints] if not motion_params.empty else None
        )
        
        time_series = masker.fit_transform(
            fmri_img,
            confounds=motion_params[valid_timepoints] if not motion_params.empty else None
        )
        
        logger.info(f"Extracted time series with shape: {time_series.shape}")
        
        if time_series.shape[0] < ANALYSIS_PARAMS['min_timepoints']:
            logger.error(f"Time series too short ({time_series.shape[0]} timepoints)")
            return None
        
        return time_series
        
    except Exception as e:
        logger.error(f"Failed to extract time series: {str(e)}")
        return None

def compute_connectivity_measures(
    time_series: np.ndarray,
    network_labels: Dict[int, str],
    roi_names: List[str],
    unique_networks: List[str],
    output_prefix: str,
    logger: logging.Logger
) -> Tuple[str, str, str, str, str]:
    """Compute all connectivity measures and save results."""
    try:
        # Compute correlation matrix
        logger.info("Computing correlation matrix...")
        corr_matrix = np.corrcoef(time_series.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # Save correlation matrix
        output_matrix = f"{output_prefix}_roiroi_matrix.npy"
        np.save(output_matrix, corr_matrix)
        logger.info(f"Saved correlation matrix: {output_matrix}")
        
        # Compute ROI-to-ROI FC
        logger.info("Computing ROI-to-ROI FC...")
        roiroi_fc = []
        for i, j in combinations(range(len(roi_names)), 2):
            net_i = network_labels.get(i + 1, 'Unknown')
            net_j = network_labels.get(j + 1, 'Unknown')
            if net_i == 'Unknown' or net_j == 'Unknown':
                continue
            
            roiroi_fc.append({
                'ROI': f"{roi_names[i]}_{roi_names[j]}",
                'network1': net_i,
                'network2': net_j,
                'FC': corr_matrix[i, j]
            })
        
        # Save ROI-to-ROI FC
        output_roiroi_csv = f"{output_prefix}_roiroi_fc.csv"
        roiroi_df = pd.DataFrame(roiroi_fc)
        if not roiroi_df.empty:
            roiroi_df.to_csv(output_roiroi_csv, index=False)
            logger.info(f"Saved ROI-to-ROI FC: {output_roiroi_csv}")
        
        # Compute ROI-to-network FC
        logger.info("Computing ROI-to-network FC...")
        roi_network_fc = []
        for i in range(len(roi_names)):
            net_i = network_labels.get(i + 1, 'Unknown')
            if net_i == 'Unknown':
                continue
            
            for net_j in unique_networks:
                corrs = [
                    corr_matrix[i, j] for j in range(len(roi_names))
                    if network_labels.get(j + 1, 'Unknown') == net_j and i != j
                ]
                
                if corrs:
                    roi_network_fc.append({
                        'ROI': f"{roi_names[i]}_{net_j}",
                        'network1': net_i,
                        'network2': net_j,
                        'fc_value': np.mean(corrs)
                    })
        
        # Save ROI-to-network FC
        output_pairwise_csv = f"{output_prefix}_network_fc_avg.csv"
        pairwise_df = pd.DataFrame(roi_network_fc)
        if not pairwise_df.empty:
            pairwise_df.to_csv(output_pairwise_csv, index=False)
            logger.info(f"Saved ROI-to-network FC: {output_pairwise_csv}")
        
        # Compute ROI-level FC (within vs between network)
        logger.info("Computing ROI-level FC...")
        roi_fc = []
        for i in range(len(roi_names)):
            net_i = network_labels.get(i + 1, 'Unknown')
            if net_i == 'Unknown':
                continue
            
            within_corrs = []
            between_corrs = []
            
            for j in range(len(roi_names)):
                if i == j:
                    continue
                
                net_j = network_labels.get(j + 1, 'Unknown')
                if net_j == 'Unknown':
                    continue
                
                corr = corr_matrix[i, j]
                if net_i == net_j:
                    within_corrs.append(corr)
                else:
                    between_corrs.append(corr)
            
            roi_fc.append({
                'network_name': net_i,
                'roi_name': roi_names[i],
                'within_network_FC': np.mean(within_corrs) if within_corrs else np.nan,
                'between_network_FC': np.mean(between_corrs) if between_corrs else np.nan
            })
        
        # Save ROI-level FC
        output_roi_csv = f"{output_prefix}_roi_fc.csv"
        pd.DataFrame(roi_fc).to_csv(output_roi_csv, index=False)
        logger.info(f"Saved ROI-level FC: {output_roi_csv}")
        
        # Compute network summary
        logger.info("Computing network summary...")
        network_summary = []
        for net in unique_networks:
            within_corrs = [
                corr_matrix[i, j] for i in range(len(roi_names)) for j in range(len(roi_names))
                if i != j and network_labels.get(i + 1, 'Unknown') == net
                and network_labels.get(j + 1, 'Unknown') == net
            ]
            
            row = {
                'Network': net,
                'Within_Network_FC': np.mean(within_corrs) if within_corrs else np.nan
            }
            
            for other_net in unique_networks:
                if other_net == net:
                    continue
                
                between_corrs = [
                    corr_matrix[i, j] for i in range(len(roi_names)) for j in range(len(roi_names))
                    if i != j and network_labels.get(i + 1, 'Unknown') == net
                    and network_labels.get(j + 1, 'Unknown') == other_net
                ]
                
                row[f'Between_{other_net}_FC'] = np.mean(between_corrs) if between_corrs else np.nan
            
            network_summary.append(row)
        
        # Save network summary
        output_network_csv = f"{output_prefix}_network_summary.csv"
        pd.DataFrame(network_summary).to_csv(output_network_csv, index=False)
        logger.info(f"Saved network summary: {output_network_csv}")
        
        return output_matrix, output_roi_csv, output_roiroi_csv, output_pairwise_csv, output_network_csv
        
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
    network_labels: Dict[int, str],
    logger: logging.Logger
) -> Optional[Tuple[str, str, str, str, str]]:
    """Process a single fMRI run to compute functional connectivity."""
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
        if brain_mask_img is None:
            return None
        
        # Process confounds
        logger.info("Processing confounds...")
        motion_params, valid_timepoints = process_confounds(confounds_file, logger)
        
        if valid_timepoints.sum() < ANALYSIS_PARAMS['min_timepoints']:
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
        roi_names = [f"ROI_{i+1}" for i in range(time_series.shape[1])]
        unique_networks = sorted(set(network_labels.values()) - {'Unknown'})
        
        results = compute_connectivity_measures(
            time_series, network_labels, roi_names, unique_networks, output_prefix, logger
        )
        
        logger.info(f"Run processing completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Failed to process run: {str(e)}")
        return None

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute ROI-to-ROI and ROI-to-network functional connectivity using customizable atlas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Power 2011 Atlas (Default)
  python NW_1st.py --subject sub-AOCD001 \\
    --atlas power_2011 \\
    --label-pattern power

  # Schaefer 2018 Atlas
  python NW_1st.py --subject sub-AOCD001 \\
    --atlas schaefer_2018 \\
    --atlas-params '{"n_rois": 400, "yeo_networks": 7, "resolution_mm": 2}' \\
    --label-pattern nilearn

  # Custom Atlas
  python NW_1st.py --subject sub-AOCD001 \\
    --atlas /path/to/atlas.nii.gz \\
    --labels /path/to/labels.txt \\
    --label-pattern custom \\
    --custom-regex "network_(\\d+)_(.+)" \\
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
        default='aal',
        help='Atlas name or path (default: aal)')
    parser.add_argument(
        '--atlas-params',
        type=str,
        help='JSON string with atlas parameters (for Nilearn atlases, e.g., \'{"n_rois": 400, "yeo_networks": 7}\')'
    )
    parser.add_argument(
        '--labels', 
        type=str, 
        help='Path to network labels file (required for custom atlases, not needed for Nilearn atlases)'
    )
    parser.add_argument(
        '--label-pattern',
        type=str,
        default='power',
        choices=['power', 'nilearn', 'custom'],
        help='Pattern for reading network labels: power (Power atlas format), nilearn (built-in atlas), custom (user-defined regex)'
    )
    parser.add_argument(
        '--custom-regex',
        type=str,
        help='Custom regex pattern for parsing labels when label-pattern is custom (e.g., r"network_(\\d+)_(.+)" for network_1_Default format)'
    )
    parser.add_argument(
        '--atlas-name',
        type=str,
        help='Custom name for the atlas (used in output filenames, defaults to atlas name or filename)'
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
        '--verbose', '-v', 
        action='store_true', 
        help='Enable verbose logging'
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
    parser.add_argument(
        '--test-schaefer',
        action='store_true',
        help='Test the Schaefer 2018 atlas configuration'
    )
    
    return parser.parse_args()

# =============================================================================
# ATLAS LOADING FUNCTIONS
# =============================================================================

def load_atlas_and_labels(
    atlas_spec: str, 
    atlas_params: str = None,
    labels_path: str = None, 
    label_pattern: str = 'power',
    custom_regex: str = None,
    logger: logging.Logger = None
) -> Tuple[image.Nifti1Image, Dict[int, str], str]:
    """Load atlas and network labels, returning atlas image, network labels, and atlas name."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Check if this is a Nilearn atlas
    if atlas_spec in NILEARN_ATLASES:
        logger.info(f"Loading Nilearn atlas: {atlas_spec}")
        
        # Parse atlas parameters
        if atlas_params:
            try:
                params = json.loads(atlas_params)
                params = validate_atlas_params(atlas_spec, params)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in --atlas-params: {str(e)}")
                raise
        else:
            params = NILEARN_ATLASES[atlas_spec]['default_params']
        
        # Fetch atlas
        atlas_img, network_labels = fetch_nilearn_atlas(atlas_spec, params, logger)
        
        # Generate atlas name for output files
        if params:
            param_str = "_".join([str(v) for v in params.values()])
            atlas_name = f"{atlas_spec}_{param_str}"
        else:
            atlas_name = atlas_spec
        
        return atlas_img, network_labels, atlas_name
    
    else:
        # Custom atlas file
        logger.info(f"Loading custom atlas: {atlas_spec}")
        
        if not os.path.exists(atlas_spec):
            raise FileNotFoundError(f"Atlas file not found: {atlas_spec}")
        
        # Load atlas image
        atlas_img = image.load_img(atlas_spec)
        
        # Load network labels
        if labels_path is None:
            raise ValueError("--labels is required for custom atlases")
        
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
        network_labels = load_custom_network_labels(labels_path, label_pattern, custom_regex, logger)
        
        # Determine atlas name for output files
        if atlas_name:
            atlas_name = atlas_name
        else:
            atlas_name = os.path.splitext(os.path.basename(atlas_spec))[0]
        
        return atlas_img, network_labels, atlas_name

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to process functional connectivity analysis."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Handle usage and help requests
    if args.usage:
        print_examples()
        return
    
    if args.list_atlases:
        print_available_atlases()
        return
    
    if args.test_schaefer:
        test_schaefer_2018_configuration()
        return

    # Validate arguments
    if args.label_pattern == 'nilearn':
        # For Nilearn atlases, labels are not needed
        if args.atlas not in NILEARN_ATLASES:
            print(f"Error: '{args.atlas}' is not a valid Nilearn atlas")
            print("Available Nilearn atlases:", list(NILEARN_ATLASES.keys()))
            print("\nRun with --list-atlases to see all available atlases and their parameters")
            return
    elif args.label_pattern == 'custom':
        # For custom pattern, regex is required
        if not args.custom_regex:
            print("Error: --custom-regex is required when --label-pattern is custom")
            print("\nExample:")
            print("  python NW_1st.py --subject sub-AOCD001 \\")
            print("    --atlas /path/to/atlas.nii.gz \\")
            print("    --labels /path/to/labels.txt \\")
            print("    --label-pattern custom \\")
            print("    --custom-regex 'network_(\\d+)_(.+)'")
            return
    else:
        # For power pattern, labels are required for custom atlases
        if args.atlas not in NILEARN_ATLASES and not args.labels:
            print("Error: --labels is required for custom atlases")
            print("Use --label-pattern nilearn for built-in Nilearn atlases")
            print("\nExamples:")
            print("  # For Nilearn atlases:")
            print("  python NW_1st.py --subject sub-AOCD001 --atlas schaefer_2018 --label-pattern nilearn")
            print("  # For custom atlases:")
            print("  python NW_1st.py --subject sub-AOCD001 --atlas /path/to/atlas.nii.gz --labels /path/to/labels.txt --label-pattern power")
            return
    
    # Override default configuration with command line arguments
    config = CONFIG.copy()
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.work_dir:
        config['work_dir'] = args.work_dir
    
    # Setup logging
    logger = setup_logging(config['log_file'])
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Print available atlases for debugging
    logger.info("Checking available atlases...")
    print_available_atlases()
    
    logger.info("=" * 80)
    logger.info("Starting Functional Connectivity Analysis")
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
        
        # Load atlas and network labels
        logger.info("Setting up atlas and network labels...")
        atlas_img, network_labels, atlas_name = load_atlas_and_labels(
            args.atlas, args.atlas_params, args.labels, args.label_pattern, args.custom_regex, logger
        )
        
        logger.info(f"Atlas loaded: shape={atlas_img.shape}, ROIs={len(network_labels)}")
        logger.info(f"Network labels: {len(set(network_labels.values()))} unique networks")
        
        # Process each session
        subjects = [args.subject]
        processed_any = False
        
        for subject in subjects:
            logger.info(f"Processing subject: {subject}")
            
            for session in ANALYSIS_PARAMS['sessions']:
                try:
                    logger.info(f"Processing session: {session}")
                    
                    # Validate paths
                    paths = validate_paths(subject, session, config['bids_dir'], logger)
                    if not paths:
                        logger.warning(f"Skipping session {session} for {subject}")
                        continue
                    
                    brain_mask_path, fmri_file, confounds_file = paths
                    
                    # Extract run ID
                    run_id = extract_run_id(fmri_file)
                    logger.info(f"Run ID: {run_id}")
                    
                    # Define output prefix
                    output_prefix = os.path.join(
                        config['output_dir'],
                        f'{subject}_{session}_task-rest_run-{run_id}_{atlas_name}'
                    )
                    
                    # Process the run
                    result = process_run(
                        fmri_file, confounds_file, atlas_img, brain_mask_path,
                        output_prefix, config['work_dir'], network_labels, logger
                    )
                    
                    if result:
                        # Rename output files to indicate they are from single runs
                        src_matrix, src_roi_csv, src_roiroi_csv, src_pairwise_csv, src_network_csv = result
                        
                        # Move files to final locations
                        file_moves = [
                            (src_matrix, f'{subject}_{session}_task-rest_{atlas_name}_roiroi_matrix_avg.npy'),
                            (src_roi_csv, f'{subject}_{session}_task-rest_{atlas_name}_roi_fc_avg.csv'),
                            (src_roiroi_csv, f'{subject}_{session}_task-rest_{atlas_name}_roiroi_fc_avg.csv'),
                            (src_pairwise_csv, f'{subject}_{session}_task-rest_{atlas_name}_network_fc_avg.csv'),
                            (src_network_csv, f'{subject}_{session}_task-rest_{atlas_name}_network_summary_avg.csv')
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
            logger.info(f"Analysis completed successfully for {args.subject} using atlas: {atlas_name}")
    
    except Exception as e:
        logger.error(f"Main function failed: {str(e)}")
        raise
    
    finally:
        logger.info("=" * 80)
        logger.info("Functional Connectivity Analysis Completed")
        logger.info("=" * 80)

def print_quick_help():
    """Print quick help information."""
    quick_help = """
QUICK HELP - ROI-to-ROI and ROI-to-Network Functional Connectivity Analysis
==========================================================================

BASIC USAGE:
  python NW_1st.py --subject <SUBJECT_ID> --atlas <ATLAS_NAME> --label-pattern <PATTERN>

QUICK EXAMPLES:
  1. Power 2011 Atlas (Default):
     python NW_1st.py --subject sub-AOCD001 --atlas power_2011 --label-pattern power

  2. Schaefer 2018 Atlas:
     python NW_1st.py --subject sub-AOCD001 --atlas schaefer_2018 --label-pattern nilearn

  3. Custom Atlas:
     python NW_1st.py --subject sub-AOCD001 --atlas /path/to/atlas.nii.gz --labels /path/to/labels.txt --label-pattern power

HELP OPTIONS:
  --help          Show full help with all arguments
  --usage         Show detailed usage examples
  --list-atlases  List available Nilearn atlases and parameters

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
        print("\nFor help, run: python NW_1st.py --help")
        print("For usage examples, run: python NW_1st.py --usage")
        print("For available atlases, run: python NW_1st.py --list-atlases")
        raise