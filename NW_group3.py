#!/usr/bin/env python3
"""
ROI-to-ROI Functional Connectivity Group Analysis

This script performs group-level statistical analysis on ROI-to-ROI functional connectivity data.
It compares healthy controls (HC) vs. OCD patients and performs longitudinal analyses.

IMPORTANT: This script analyzes ROI-TO-ROI connectivity (individual ROI pairs).
It requires input files with "*_roiroi_fc_avg.csv" naming pattern.

SUITABLE ATLASES:
- ALL atlases that generate *_roiroi_fc_avg.csv files:
  * Network-based atlases: Power 2011, Schaefer 2018, YEO 2011
  * Anatomical atlases: Harvard-Oxford, AAL, Talairach
  * Custom atlases: Any atlas that generates ROI-to-ROI connectivity files

ANALYSIS TYPE:
- ROI-to-ROI connectivity analysis (individual ROI pairs)
- Group comparisons (HC vs OCD)
- Longitudinal analysis (baseline vs follow-up)
- Clinical correlation analysis

Author: [Your Name]
Date: [Current Date]

USAGE EXAMPLES:
==============

1. Power 2011 Atlas (Network-based):
   python NW_group3.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --atlas_name power_2011
kl
2. Schaefer 2018 Atlas (Network-based, 400 ROIs, 7 networks):
   python NW_group3.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --atlas_name schaefer_2018_400_7_2

3. YEO 2011 Atlas (Network-based, 7 networks):
   python NW_group3.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --atlas_name yeo_2011_7_thick

4. Harvard-Oxford Atlas (Anatomical):
   python NW_group3.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --atlas_name harvard_oxford_cort-maxprob-thr25-2mm

5. AAL Atlas (Anatomical):
   python NW_group3.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --atlas_name aal

6. Auto-detect Atlas:
   python NW_group3.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data

ATLAS NAMING CONVENTIONS:
========================

The script automatically detects atlas names from input FC files:
- Network-based atlases:
  * Power 2011: power_2011_roiroi_fc_avg.csv
  * Schaefer 2018: schaefer_2018_{n_rois}_{yeo_networks}_{resolution}_roiroi_fc_avg.csv
  * YEO 2011: yeo_2011_{n_networks}_{thickness}_roiroi_fc_avg.csv
- Anatomical atlases:
  * Harvard-Oxford: harvard_oxford_{atlas_name}_roiroi_fc_avg.csv
  * AAL: aal_roiroi_fc_avg.csv
  * Talairach: talairach_roiroi_fc_avg.csv
- Custom: custom_atlas_roiroi_fc_avg.csv

IMPORTANT: This script (NW_group3.py) analyzes ROI-to-ROI connectivity and requires
*_roiroi_fc_avg.csv input files. ALL atlases (both network-based and anatomical) 
generate these files when processed by NW_1st.py.

OUTPUT FILES:
============
- group_diff_baseline_{atlas_name}_roiroi_fc.csv: Group difference t-test results
- baselineFC_vs_deltaYBOCS_{atlas_name}_roiroi_fc.csv: Baseline FC vs symptom change
- deltaFC_vs_deltaYBOCS_{atlas_name}_roiroi_fc.csv: FC change vs symptom change
- group_diff_followup_{atlas_name}_roiroi_fc.csv: Follow-up group differences
- summary_{atlas_name}_roiroi_fc.csv: Summary statistics

DIFFERENCE FROM NW_group.py and NW_group2.py:
---------------------------------------------
This script (NW_group3.py) analyzes ROI-to-ROI pairwise functional connectivity 
(individual ROI pairs) and requires *_roiroi_fc_avg.csv input files.

NW_group.py analyzes ROI-to-network connectivity and works with both:
- *_network_fc_avg.csv files (for network-based atlases)
- *_roiroi_fc_avg.csv files (for anatomical atlases)

NW_group2.py analyzes network-level connectivity and works with:
- *_network_fc_avg.csv files (for network-based atlases only)

ATLAS COMPATIBILITY:
-------------------
- SUITABLE: Harvard-Oxford, AAL, Talairach (anatomical atlases)
- NOT SUITABLE: Power 2011, Schaefer 2018, YEO 2011 (network-based atlases)
"""

import os
import sys
import glob
import re
import json
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    'default_atlas_name': 'auto',  # Auto-detect atlas from available files
    'sessions': ['baseline', 'followup'],
    'groups': ['HC', 'OCD'],
    'min_subjects_per_group': 5,
    'significance_threshold': 0.05,
    'fdr_correction': True,
    'output_dir': './results',
    'log_file': './logs/roiroi_fc_analysis.log'
}

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_file: str) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ROI-to-ROI Functional Connectivity Group Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
QUICK HELP - ROI-to-ROI Functional Connectivity Group Analysis
=============================================================

BASIC USAGE:
  python NW_group3.py --subjects_csv <GROUP_CSV> --clinical_csv <CLINICAL_CSV> --input_dir <FC_DATA_DIR>

QUICK EXAMPLES:
  1. Default Atlas (Harvard-Oxford):
     python NW_group3.py \\
       --subjects_csv group.csv \\
       --clinical_csv clinical.csv \\
       --input_dir /path/to/fc/data

  2. Specific Atlas:
     python NW_group3.py \\
       --subjects_csv group.csv \\
       --clinical_csv clinical.csv \\
       --input_dir /path/to/fc/data \\
       --atlas_name harvard_oxford_cort-maxprob-thr25-2mm

  3. Auto-detect Atlas:
     python NW_group3.py \\
       --subjects_csv group.csv \\
       --clinical_csv clinical.csv \\
       --input_dir /path/to/fc/data

HELP OPTIONS:
  --help          Show this help message
  --usage         Show detailed usage examples
        """
    )
    
    parser.add_argument('--subjects_csv', type=str, required=False,
                       help='CSV file containing subject IDs and group labels')
    parser.add_argument('--clinical_csv', type=str, required=False,
                       help='CSV file containing clinical data including YBOCS scores')
    parser.add_argument('--input_dir', type=str, required=False,
                       help='Directory containing FC data files')
    parser.add_argument('--atlas_name', type=str, default=None,
                       help='Specific atlas name (auto-detected if not provided)')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_CONFIG['output_dir'],
                       help=f'Output directory for results (default: {DEFAULT_CONFIG["output_dir"]})')
    parser.add_argument('--min_subjects', type=int, default=DEFAULT_CONFIG['min_subjects_per_group'],
                       help=f'Minimum subjects per group (default: {DEFAULT_CONFIG["min_subjects_per_group"]})')
    parser.add_argument('--significance_threshold', type=float, default=DEFAULT_CONFIG['significance_threshold'],
                       help=f'Significance threshold (default: {DEFAULT_CONFIG["significance_threshold"]})')
    parser.add_argument('--no_fdr', action='store_true',
                       help='Disable FDR correction for multiple comparisons')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--usage', action='store_true',
                       help='Show detailed usage examples')
    
    return parser.parse_args()

def print_examples():
    """Print detailed usage examples."""
    print("""
DETAILED USAGE EXAMPLES:
========================

1. POWER 2011 ATLAS (Network-based)
   ---------------------------------
   python NW_group3.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --atlas_name power_2011
   
   Expected FC files: *_task-rest_power_2011_roiroi_fc_avg.csv

2. SCHAEFER 2018 ATLAS (Network-based)
   ------------------------------------
   python NW_group3.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --atlas_name schaefer_2018_400_7_2
   
   Expected FC files: *_task-rest_schaefer_2018_400_7_2_roiroi_fc_avg.csv

3. YEO 2011 ATLAS (Network-based)
   -------------------------------
   python NW_group3.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --atlas_name yeo_2011_7_thick
   
   Expected FC files: *_task-rest_yeo_2011_7_thick_roiroi_fc_avg.csv

4. HARVARD-OXFORD ATLAS (Anatomical)
   ----------------------------------
   python NW_group3.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --atlas_name harvard_oxford_cort-maxprob-thr25-2mm
   
   Expected FC files: *_task-rest_harvard_oxford_cort-maxprob-thr25-2mm_roiroi_fc_avg.csv

5. AAL ATLAS (Anatomical)
   -----------------------
   python NW_group3.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --atlas_name aal
   
   Expected FC files: *_task-rest_aal_roiroi_fc_avg.csv

6. AUTO-DETECT ATLAS
   -----------------
   python NW_group3.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data
   
   The script will automatically detect available atlases from the input directory.

REQUIRED FILES:
---------------
- group.csv: Contains subject IDs and group labels (HC/OCD)
- clinical.csv: Contains clinical data including YBOCS scores
- FC data files: Generated by NW_1st.py with naming pattern:
  *_{session}_task-rest_{atlas_name}_roiroi_fc_avg.csv
  (Only anatomical atlases generate these files)

OUTPUT FILES:
-------------
- group_diff_baseline_{atlas_name}_roiroi_fc.csv: Group difference t-test results
- baselineFC_vs_deltaYBOCS_{atlas_name}_roiroi_fc.csv: Baseline FC vs symptom change
- deltaFC_vs_deltaYBOCS_{atlas_name}_roiroi_fc.csv: FC change vs symptom change
- group_diff_followup_{atlas_name}_roiroi_fc.csv: Follow-up group differences
- summary_{atlas_name}_roiroi_fc.csv: Summary statistics

ATLAS NAMING CONVENTIONS:
-------------------------
The script automatically detects atlas names from input FC files:
- Harvard-Oxford: harvard_oxford_{atlas_name}_roiroi_fc_avg.csv
  Examples:
    - harvard_oxford_cort-maxprob-thr25-2mm_roiroi_fc_avg.csv
    - harvard_oxford_sub-maxprob-thr50-2mm_roiroi_fc_avg.csv
- AAL: aal_roiroi_fc_avg.csv
- Talairach: talairach_roiroi_fc_avg.csv
- Custom: custom_atlas_roiroi_fc_avg.csv (if anatomical)

IMPORTANT: This script (NW_group3.py) analyzes ROI-to-ROI connectivity and requires
*_roiroi_fc_avg.csv input files. Only anatomical atlases generate these files.

DIFFERENCE FROM NW_group.py and NW_group2.py:
---------------------------------------------
This script (NW_group3.py) analyzes ROI-to-ROI pairwise functional connectivity 
(individual ROI pairs) and requires *_roiroi_fc_avg.csv input files.

NW_group.py analyzes ROI-to-network connectivity and works with both:
- *_network_fc_avg.csv files (for network-based atlases)
- *_roiroi_fc_avg.csv files (for anatomical atlases)

NW_group2.py analyzes network-level connectivity and works with:
- *_network_fc_avg.csv files (for network-based atlases only)

ATLAS COMPATIBILITY:
-------------------
- SUITABLE: Harvard-Oxford, AAL, Talairach (anatomical atlases)
- NOT SUITABLE: Power 2011, Schaefer 2018, YEO 2011 (network-based atlases)
    """)

# =============================================================================
# ATLAS DETECTION AND VALIDATION
# =============================================================================

def detect_atlas_name(input_dir: str, logger: logging.Logger) -> str:
    """Auto-detect atlas name from FC files in input directory."""
    if not os.path.exists(input_dir):
        logger.warning("Input directory does not exist, using default atlas name: %s", DEFAULT_CONFIG['default_atlas_name'])
        return DEFAULT_CONFIG['default_atlas_name']
    
    # Look for FC files with roiroi pattern (all atlases generate these)
    fc_patterns = [
        '*_task-rest_*_roiroi_fc_avg.csv',  # All atlases (network-based and anatomical)
        '*_task-rest_*_roiroi_matrix_avg.npy'  # Matrix files
    ]
    
    detected_atlases = set()
    
    for pattern in fc_patterns:
        files = glob.glob(os.path.join(input_dir, pattern))
        for file_path in files:
            filename = os.path.basename(file_path)
            # Pattern: sub-XXX_ses-XXX_task-rest_ATLAS_NAME_roiroi_fc_avg.csv
            # or: sub-XXX_ses-XXX_task-rest_ATLAS_NAME_roiroi_matrix_avg.npy
            match = re.search(r'_task-rest_(.+?)_(?:roiroi_fc|roiroi_matrix)_avg', filename)
            if match:
                atlas_name = match.group(1)
                detected_atlases.add(atlas_name)
                logger.debug("Detected atlas name '%s' from file: %s", atlas_name, filename)
    
    if not detected_atlases:
        logger.warning("No FC files found, using default atlas name: %s", DEFAULT_CONFIG['default_atlas_name'])
        return DEFAULT_CONFIG['default_atlas_name']
    
    if len(detected_atlases) == 1:
        atlas_name = list(detected_atlases)[0]
        logger.info("Auto-detected atlas: %s", atlas_name)
        return atlas_name
    
    # Multiple atlases detected
    logger.info("Multiple atlases detected: %s", list(detected_atlases))
    logger.info("Using first atlas: %s", list(detected_atlases)[0])
    return list(detected_atlases)[0]

def validate_atlas_files(input_dir: str, atlas_name: str, logger: logging.Logger) -> bool:
    """Validate that ROI-to-ROI FC files exist for the specified atlas."""
    if not os.path.exists(input_dir):
        logger.error("Input directory does not exist: %s", input_dir)
        return False
    
    # Check for ROI-to-ROI FC files with this atlas name
    fc_pattern = os.path.join(input_dir, f'*_task-rest_{atlas_name}_roiroi_fc_avg.csv')
    fc_files = glob.glob(fc_pattern)
    
    if not fc_files:
        logger.error("No ROI-to-ROI FC files found for atlas '%s' in directory: %s", atlas_name, input_dir)
        logger.error("Expected pattern: *_task-rest_%s_roiroi_fc_avg.csv", atlas_name)
        logger.error("Make sure NW_1st.py has been run to generate ROI-to-ROI connectivity files")
        return False
    
    logger.info("Found %d ROI-to-ROI FC files for atlas '%s': %s", len(fc_files), atlas_name, 
                [os.path.basename(f) for f in fc_files[:5]])  # Show first 5 files
    
    return True

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_roiroi_fc_path(subject: str, session: str, input_dir: str, atlas_name: str) -> str:
    """Get path to ROI-to-ROI FC CSV file."""
    if not subject.startswith('sub-'):
        subject = f"sub-{subject}"
    
    path = os.path.join(
        input_dir, 
        f"{subject}_{session}_task-rest_{atlas_name}_roiroi_fc_avg.csv"
    )
    return path

def get_group(subject_id: str, metadata_df: pd.DataFrame) -> Optional[str]:
    """Get group label for a subject."""
    if subject_id.startswith('sub-'):
        subject_id = subject_id.replace('sub-', '')
    
    group_row = metadata_df[metadata_df['subject_id'] == subject_id]
    if group_row.empty:
        return None
    
    return group_row.iloc[0]['group']

def get_ybocs_scores(subject_id: str, clinical_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """Get YBOCS scores for baseline and follow-up sessions."""
    if subject_id.startswith('sub-'):
        subject_id = subject_id.replace('sub-', '')
    
    subject_data = clinical_df[clinical_df['subject_id'] == subject_id]
    if subject_data.empty:
        return None, None
    
    baseline_ybocs = subject_data[subject_data['session'] == 'baseline']['ybocs_total'].values
    followup_ybocs = subject_data[subject_data['session'] == 'followup']['ybocs_total'].values
    
    baseline_score = baseline_ybocs[0] if len(baseline_ybocs) > 0 else None
    followup_score = followup_ybocs[0] if len(followup_ybocs) > 0 else None
    
    return baseline_score, followup_score

def load_roiroi_fc_data(fc_dir: str, atlas_name: str, logger: logging.Logger) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load ROI-to-ROI FC data for all subjects and sessions."""
    if not os.path.exists(fc_dir):
        logger.error("Input directory %s does not exist", fc_dir)
        raise ValueError(f"Input directory {fc_dir} does not exist")
    
    # Find FC files with the specific atlas (only roiroi files for anatomical atlases)
    fc_files = glob.glob(os.path.join(fc_dir, f'*_task-rest_{atlas_name}_roiroi_fc_avg.csv'))
    logger.info("Found %d FC files for atlas %s", len(fc_files), atlas_name)
    
    if not fc_files:
        # Try to find any FC files to help with debugging
        all_fc_patterns = [
            os.path.join(fc_dir, '*_task-rest_*_roiroi_fc_avg.csv'),
            os.path.join(fc_dir, '*_task-rest_*_network_fc_avg.csv')
        ]
        all_fc_files = []
        for pattern in all_fc_patterns:
            all_fc_files.extend(glob.glob(pattern))
        if all_fc_files:
            detected_atlases = set()
            for f in all_fc_files:
                filename = os.path.basename(f)
                match = re.search(r'_task-rest_(.+?)_(?:roiroi_fc|network_fc)_avg', filename)
                if match:
                    detected_atlases.add(match.group(1))
            logger.error("No FC files found for atlas '%s'. Available atlases: %s", atlas_name, detected_atlases)
        else:
            dir_contents = os.listdir(fc_dir)
            logger.error("No FC files found in directory. Directory contents: %s", dir_contents[:10])
        raise ValueError(f"No FC files found for atlas {atlas_name}")
    
    subject_sessions = {}
    dropped_subjects = []
    
    for f in fc_files:
        filename = os.path.basename(f)
        if '_ses-' not in filename or f'_task-rest_{atlas_name}_roiroi_fc_avg.csv' not in filename:
            logger.debug("Skipping invalid FC file: %s", filename)
            continue
        
        parts = filename.split('_')
        if len(parts) < 2:
            logger.debug("Skipping file with invalid format: %s", filename)
            continue
        
        subject = parts[0]  # sub-XXX
        session = parts[1]  # ses-XXX
        
        try:
            fc_data = pd.read_csv(f)
            logger.debug("Loaded FC data for %s %s: %d ROI pairs", subject, session, len(fc_data))
            
            if subject not in subject_sessions:
                subject_sessions[subject] = {}
            subject_sessions[subject][session] = fc_data
            
        except Exception as e:
            logger.warning("Failed to load FC data from %s: %s", f, str(e))
            dropped_subjects.append(f)
            continue
    
    if dropped_subjects:
        logger.warning("Dropped %d subjects due to loading errors", len(dropped_subjects))
    
    logger.info("Successfully loaded FC data for %d subjects", len(subject_sessions))
    return subject_sessions

# =============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# =============================================================================

def perform_group_comparison(fc_data: Dict[str, Dict[str, pd.DataFrame]], 
                           metadata_df: pd.DataFrame, 
                           session: str,
                           atlas_name: str,
                           logger: logging.Logger) -> pd.DataFrame:
    """Perform group comparison (HC vs OCD) for ROI-to-ROI connectivity."""
    logger.info("Performing group comparison for session: %s", session)
    
    # Collect data for each group
    hc_data = []
    ocd_data = []
    
    for subject, sessions_data in fc_data.items():
        if session not in sessions_data:
            continue
        
        group = get_group(subject, metadata_df)
        if group is None:
            logger.warning("No group found for subject %s", subject)
            continue
        
        fc_df = sessions_data[session]
        
        if group == 'HC':
            hc_data.append(fc_df)
        elif group == 'OCD':
            ocd_data.append(fc_df)
    
    logger.info("Group sizes - HC: %d, OCD: %d", len(hc_data), len(ocd_data))
    
    if len(hc_data) < DEFAULT_CONFIG['min_subjects_per_group'] or len(ocd_data) < DEFAULT_CONFIG['min_subjects_per_group']:
        logger.warning("Insufficient subjects per group (min: %d)", DEFAULT_CONFIG['min_subjects_per_group'])
        logger.info("Returning empty results due to insufficient subjects")
        return pd.DataFrame()
    
    # Perform t-tests for each ROI pair
    results = []
    
    # Get all ROI pairs from the first subject
    if hc_data:
        roi_pairs = hc_data[0][['ROI1', 'ROI2']].copy()
    elif ocd_data:
        roi_pairs = ocd_data[0][['ROI1', 'ROI2']].copy()
    else:
        logger.error("No data available for analysis")
        return pd.DataFrame()
    
    for _, row in roi_pairs.iterrows():
        roi1, roi2 = row['ROI1'], row['ROI2']
        
        # Extract network information if available
        network1 = row.get('network1', 'Unknown') if 'network1' in row else 'Unknown'
        network2 = row.get('network2', 'Unknown') if 'network2' in row else 'Unknown'
        
        # Extract FC values for this ROI pair
        hc_values = []
        ocd_values = []
        
        for fc_df in hc_data:
            pair_data = fc_df[(fc_df['ROI1'] == roi1) & (fc_df['ROI2'] == roi2)]
            if not pair_data.empty:
                hc_values.append(pair_data['fc_value'].iloc[0])
                # Get network info from first available data
                if network1 == 'Unknown' and 'network1' in pair_data.columns:
                    network1 = pair_data['network1'].iloc[0]
                if network2 == 'Unknown' and 'network2' in pair_data.columns:
                    network2 = pair_data['network2'].iloc[0]
        
        for fc_df in ocd_data:
            pair_data = fc_df[(fc_df['ROI1'] == roi1) & (fc_df['ROI2'] == roi2)]
            if not pair_data.empty:
                ocd_values.append(pair_data['fc_value'].iloc[0])
                # Get network info from first available data
                if network1 == 'Unknown' and 'network1' in pair_data.columns:
                    network1 = pair_data['network1'].iloc[0]
                if network2 == 'Unknown' and 'network2' in pair_data.columns:
                    network2 = pair_data['network2'].iloc[0]
        
        if len(hc_values) > 0 and len(ocd_values) > 0:
            # Perform t-test
            t_stat, p_value = ttest_ind(hc_values, ocd_values)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(hc_values) - 1) * np.var(hc_values, ddof=1) + 
                                 (len(ocd_values) - 1) * np.var(ocd_values, ddof=1)) / 
                                (len(hc_values) + len(ocd_values) - 2))
            cohens_d = (np.mean(hc_values) - np.mean(ocd_values)) / pooled_std if pooled_std > 0 else 0
            
            results.append({
                'ROI1': roi1,
                'ROI2': roi2,
                'network1': network1,
                'network2': network2,
                'HC_mean': np.mean(hc_values),
                'OCD_mean': np.mean(ocd_values),
                'HC_std': np.std(hc_values, ddof=1),
                'OCD_std': np.std(ocd_values, ddof=1),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'HC_n': len(hc_values),
                'OCD_n': len(ocd_values)
            })
    
    results_df = pd.DataFrame(results)
    
    if not results_df.empty and DEFAULT_CONFIG['fdr_correction']:
        from statsmodels.stats.multitest import multipletests
        _, p_corrected, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
        results_df['p_corrected'] = p_corrected
        results_df['significant'] = p_corrected < DEFAULT_CONFIG['significance_threshold']
    else:
        results_df['p_corrected'] = results_df['p_value']
        results_df['significant'] = results_df['p_value'] < DEFAULT_CONFIG['significance_threshold']
    
    logger.info("Found %d significant ROI pairs (p < %.3f)", 
                results_df['significant'].sum(), DEFAULT_CONFIG['significance_threshold'])
    
    return results_df

def perform_longitudinal_analysis(fc_data: Dict[str, Dict[str, pd.DataFrame]], 
                                clinical_df: pd.DataFrame,
                                atlas_name: str,
                                logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform longitudinal analysis (baseline FC vs symptom change, FC change vs symptom change)."""
    logger.info("Performing longitudinal analysis")
    
    # Collect data for subjects with both sessions
    subjects_with_both_sessions = []
    for subject, sessions_data in fc_data.items():
        if 'baseline' in sessions_data and 'followup' in sessions_data:
            subjects_with_both_sessions.append(subject)
    
    logger.info("Found %d subjects with both baseline and follow-up data", len(subjects_with_both_sessions))
    
    if len(subjects_with_both_sessions) < DEFAULT_CONFIG['min_subjects_per_group']:
        logger.warning("Insufficient subjects for longitudinal analysis")
        logger.info("Returning empty results due to insufficient subjects")
        return pd.DataFrame(), pd.DataFrame()
    
    # Get all ROI pairs from the first subject
    first_subject = subjects_with_both_sessions[0]
    roi_pairs = fc_data[first_subject]['baseline'][['ROI1', 'ROI2']].copy()
    
    baseline_fc_results = []
    delta_fc_results = []
    
    for _, row in roi_pairs.iterrows():
        roi1, roi2 = row['ROI1'], row['ROI2']
        
        # Extract network information if available
        network1 = row.get('network1', 'Unknown') if 'network1' in row else 'Unknown'
        network2 = row.get('network2', 'Unknown') if 'network2' in row else 'Unknown'
        
        baseline_fc_values = []
        delta_ybocs_values = []
        delta_fc_values = []
        
        for subject in subjects_with_both_sessions:
            baseline_score, followup_score = get_ybocs_scores(subject, clinical_df)
            
            if baseline_score is not None and followup_score is not None:
                delta_ybocs = followup_score - baseline_score
                
                # Get baseline FC value
                baseline_fc_df = fc_data[subject]['baseline']
                baseline_pair = baseline_fc_df[(baseline_fc_df['ROI1'] == roi1) & (baseline_fc_df['ROI2'] == roi2)]
                
                if not baseline_pair.empty:
                    baseline_fc = baseline_pair['fc_value'].iloc[0]
                    baseline_fc_values.append(baseline_fc)
                    delta_ybocs_values.append(delta_ybocs)
                    # Get network info from first available data
                    if network1 == 'Unknown' and 'network1' in baseline_pair.columns:
                        network1 = baseline_pair['network1'].iloc[0]
                    if network2 == 'Unknown' and 'network2' in baseline_pair.columns:
                        network2 = baseline_pair['network2'].iloc[0]
                
                # Get FC change
                followup_fc_df = fc_data[subject]['followup']
                followup_pair = followup_fc_df[(followup_fc_df['ROI1'] == roi1) & (followup_fc_df['ROI2'] == roi2)]
                
                if not baseline_pair.empty and not followup_pair.empty:
                    baseline_fc = baseline_pair['fc_value'].iloc[0]
                    followup_fc = followup_pair['fc_value'].iloc[0]
                    delta_fc = followup_fc - baseline_fc
                    delta_fc_values.append(delta_fc)
        
        # Baseline FC vs Delta YBOCS
        if len(baseline_fc_values) > 2 and len(delta_ybocs_values) > 2:
            r, p = pearsonr(baseline_fc_values, delta_ybocs_values)
            baseline_fc_results.append({
                'ROI1': roi1,
                'ROI2': roi2,
                'network1': network1,
                'network2': network2,
                'correlation': r,
                'p_value': p,
                'n_subjects': len(baseline_fc_values)
            })
        
        # Delta FC vs Delta YBOCS
        if len(delta_fc_values) > 2 and len(delta_ybocs_values) > 2:
            r, p = pearsonr(delta_fc_values, delta_ybocs_values)
            delta_fc_results.append({
                'ROI1': roi1,
                'ROI2': roi2,
                'network1': network1,
                'network2': network2,
                'correlation': r,
                'p_value': p,
                'n_subjects': len(delta_fc_values)
            })
    
    baseline_df = pd.DataFrame(baseline_fc_results)
    delta_df = pd.DataFrame(delta_fc_results)
    
    # Apply FDR correction
    if not baseline_df.empty and DEFAULT_CONFIG['fdr_correction']:
        from statsmodels.stats.multitest import multipletests
        _, p_corrected, _, _ = multipletests(baseline_df['p_value'], method='fdr_bh')
        baseline_df['p_corrected'] = p_corrected
        baseline_df['significant'] = p_corrected < DEFAULT_CONFIG['significance_threshold']
    elif not baseline_df.empty:
        baseline_df['p_corrected'] = baseline_df['p_value']
        baseline_df['significant'] = baseline_df['p_value'] < DEFAULT_CONFIG['significance_threshold']
    
    if not delta_df.empty and DEFAULT_CONFIG['fdr_correction']:
        from statsmodels.stats.multitest import multipletests
        _, p_corrected, _, _ = multipletests(delta_df['p_value'], method='fdr_bh')
        delta_df['p_corrected'] = p_corrected
        delta_df['significant'] = p_corrected < DEFAULT_CONFIG['significance_threshold']
    elif not delta_df.empty:
        delta_df['p_corrected'] = delta_df['p_value']
        delta_df['significant'] = delta_df['p_value'] < DEFAULT_CONFIG['significance_threshold']
    
    logger.info("Longitudinal analysis complete - Baseline FC: %d pairs, Delta FC: %d pairs", 
                len(baseline_df), len(delta_df))
    
    return baseline_df, delta_df

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_analysis(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Run the complete ROI-to-ROI FC analysis."""
    logger.info("=" * 80)
    logger.info("Starting ROI-to-ROI Functional Connectivity Group Analysis")
    logger.info("=" * 80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metadata
    logger.info("Loading metadata files...")
    try:
        metadata_df = pd.read_csv(args.subjects_csv)
        clinical_df = pd.read_csv(args.clinical_csv)
        logger.info("Loaded metadata: %d subjects, %d clinical records", 
                   len(metadata_df), len(clinical_df))
    except Exception as e:
        logger.error("Failed to load metadata files: %s", str(e))
        return
    
    # Detect or validate atlas
    if args.atlas_name:
        atlas_name = args.atlas_name
        if not validate_atlas_files(args.input_dir, atlas_name, logger):
            return
    else:
        atlas_name = detect_atlas_name(args.input_dir, logger)
        if not validate_atlas_files(args.input_dir, atlas_name, logger):
            return
    
    logger.info("Using atlas: %s", atlas_name)
    
    # Update log file name to include atlas name
    atlas_log_file = os.path.join(args.output_dir, f'roiroi_fc_analysis_{atlas_name}.log')
    logger.info("Switching to atlas-specific log file: %s", atlas_log_file)
    
    # Create new logger with atlas-specific log file
    atlas_logger = setup_logging(atlas_log_file)
    if args.verbose:
        atlas_logger.setLevel(logging.DEBUG)
    
    # Load FC data
    atlas_logger.info("Loading FC data...")
    try:
        fc_data = load_roiroi_fc_data(args.input_dir, atlas_name, atlas_logger)
    except Exception as e:
        atlas_logger.error("Failed to load FC data: %s", str(e))
        return
    
    # Perform group comparisons
    atlas_logger.info("Performing group comparisons...")
    baseline_results = perform_group_comparison(fc_data, metadata_df, 'baseline', atlas_name, atlas_logger)
    followup_results = perform_group_comparison(fc_data, metadata_df, 'followup', atlas_name, atlas_logger)
    
    # Perform longitudinal analysis
    atlas_logger.info("Performing longitudinal analysis...")
    baseline_fc_long, delta_fc_long = perform_longitudinal_analysis(fc_data, clinical_df, atlas_name, atlas_logger)
    
    # Save results (always save, regardless of significance)
    atlas_logger.info("Saving results...")
    
    # Save baseline group comparison results
    if not baseline_results.empty:
        baseline_file = os.path.join(args.output_dir, f'group_diff_baseline_{atlas_name}_roiroi_fc.csv')
        baseline_results.to_csv(baseline_file, index=False)
        atlas_logger.info("Saved baseline group comparison: %s", baseline_file)
    else:
        atlas_logger.info("No baseline group comparison results to save")
    
    # Save follow-up group comparison results
    if not followup_results.empty:
        followup_file = os.path.join(args.output_dir, f'group_diff_followup_{atlas_name}_roiroi_fc.csv')
        followup_results.to_csv(followup_file, index=False)
        atlas_logger.info("Saved follow-up group comparison: %s", followup_file)
    else:
        atlas_logger.info("No follow-up group comparison results to save")
    
    # Save baseline FC vs delta YBOCS results
    if not baseline_fc_long.empty:
        baseline_long_file = os.path.join(args.output_dir, f'baselineFC_vs_deltaYBOCS_{atlas_name}_roiroi_fc.csv')
        baseline_fc_long.to_csv(baseline_long_file, index=False)
        atlas_logger.info("Saved baseline FC vs delta YBOCS: %s", baseline_long_file)
    else:
        atlas_logger.info("No baseline FC vs delta YBOCS results to save")
    
    # Save delta FC vs delta YBOCS results
    if not delta_fc_long.empty:
        delta_long_file = os.path.join(args.output_dir, f'deltaFC_vs_deltaYBOCS_{atlas_name}_roiroi_fc.csv')
        delta_fc_long.to_csv(delta_long_file, index=False)
        atlas_logger.info("Saved delta FC vs delta YBOCS: %s", delta_long_file)
    else:
        atlas_logger.info("No delta FC vs delta YBOCS results to save")
    
    # Create summary
    summary_data = {
        'atlas_name': [atlas_name],
        'total_subjects': [len(fc_data)],
        'baseline_significant_pairs': [baseline_results['significant'].sum() if not baseline_results.empty else 0],
        'followup_significant_pairs': [followup_results['significant'].sum() if not followup_results.empty else 0],
        'baseline_fc_significant_pairs': [baseline_fc_long['significant'].sum() if not baseline_fc_long.empty else 0],
        'delta_fc_significant_pairs': [delta_fc_long['significant'].sum() if not delta_fc_long.empty else 0],
        'significance_threshold': [DEFAULT_CONFIG['significance_threshold']],
        'fdr_correction': [DEFAULT_CONFIG['fdr_correction']]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(args.output_dir, f'summary_{atlas_name}_roiroi_fc.csv')
    summary_df.to_csv(summary_file, index=False)
    atlas_logger.info("Saved summary: %s", summary_file)
    
    atlas_logger.info("=" * 80)
    atlas_logger.info("Analysis complete!")
    atlas_logger.info("=" * 80)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function."""
    args = parse_arguments()
    
    if args.usage:
        print_examples()
        return
    
    # Validate required arguments
    if not args.subjects_csv or not args.clinical_csv or not args.input_dir:
        print("Error: --subjects_csv, --clinical_csv, and --input_dir are required")
        print("Use --usage for detailed examples")
        sys.exit(1)
    
    # Setup logging (will be updated with atlas name after detection)
    logger = setup_logging(args.output_dir + '/roiroi_fc_analysis.log')
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Update configuration
    if args.no_fdr:
        DEFAULT_CONFIG['fdr_correction'] = False
    if args.significance_threshold != DEFAULT_CONFIG['significance_threshold']:
        DEFAULT_CONFIG['significance_threshold'] = args.significance_threshold
    if args.min_subjects != DEFAULT_CONFIG['min_subjects_per_group']:
        DEFAULT_CONFIG['min_subjects_per_group'] = args.min_subjects
    
    # Run analysis
    try:
        run_analysis(args, logger)
    except Exception as e:
        logger.error("Analysis failed: %s", str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
