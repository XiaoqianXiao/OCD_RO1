#!/usr/bin/env python3
"""
ROI-to-ROI Functional Connectivity Group Analysis using Customizable Atlas

This script performs group-level statistical analysis on ROI-to-ROI functional connectivity data.
It compares healthy controls (HC) vs. OCD patients and performs longitudinal analyses.

The script is compatible with both custom atlases and Nilearn built-in atlases (e.g., Schaefer 2018,
Harvard-Oxford, Power 2011, etc.) and automatically handles atlas detection and validation.

Author: [Your Name]
Date: [Current Date]

USAGE EXAMPLES:
==============

1. Power 2011 Atlas (Default):
   python NW_group3.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data

2. Schaefer 2018 Atlas (400 ROIs, 7 networks):
   python NW_group3.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --atlas_name schaefer_2018_400_7_2

3. Schaefer 2018 Atlas (1000 ROIs, 17 networks):
   python NW_group3.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --atlas_name schaefer_2018_1000_17_1

4. With Specific Atlas Name:
   python NW_group3.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --atlas_name power_2011

5. YEO 2011 Atlas (7 networks):
   python NW_group3.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --atlas_name yeo_2011_7_thick

6. YEO 2011 Atlas (17 networks):
   python NW_group3.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --atlas_name yeo_2011_17_thick

7. Harvard-Oxford Atlas:
   python NW_group3.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --atlas_name harvard_oxford_cort-maxprob-thr25-2mm

8. Auto-detect Atlas from Input Files:
   python NW_group3.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --auto-detect-atlas

ATLAS NAMING CONVENTIONS:
========================

The script automatically detects atlas names from input FC files:
- Power 2011: power_2011_roiroi_fc_avg.csv
- Schaefer 2018: schaefer_2018_{n_rois}_{yeo_networks}_{resolution_mm}_roiroi_fc_avg.csv
  Examples:
    - schaefer_2018_400_7_2_roiroi_fc_avg.csv (400 ROIs, 7 networks, 2mm)
    - schaefer_2018_1000_17_1_roiroi_fc_avg.csv (1000 ROIs, 17 networks, 1mm)
- YEO 2011: yeo_2011_{n_networks}_{thickness}_roiroi_fc_avg.csv
  Examples:
    - yeo_2011_7_thick_roiroi_fc_avg.csv (7 networks, thick parcellation)
    - yeo_2011_17_thin_roiroi_fc_avg.csv (17 networks, thin parcellation)
- Harvard-Oxford: harvard_oxford_{atlas_name}_roiroi_fc_avg.csv
  Examples:
    - harvard_oxford_cort-maxprob-thr25-2mm_roiroi_fc_avg.csv
    - harvard_oxford_sub-maxprob-thr25-2mm_roiroi_fc_avg.csv
- AAL: aal_roiroi_fc_avg.csv
- Talairach: talairach_roiroi_fc_avg.csv
- Custom: custom_atlas_roiroi_fc_avg.csv

Note: For Schaefer 2018, the naming follows the pattern: schaefer_2018_{n_rois}_{yeo_networks}_{resolution_mm}
where n_rois can be 100, 200, 300, 400, 500, 600, 700, 800, 900, or 1000,
yeo_networks can be 7 or 17, and resolution_mm can be 1 or 2.
For YEO 2011, the naming follows: yeo_2011_{n_networks}_{thickness}
where n_networks can be 7 or 17, and thickness can be 'thick' or 'thin'.

OUTPUT FILES:
============

- group_diff_baseline_{atlas_name}_roiroi_fc.csv: Group difference t-test results
- group_diff_followup_{atlas_name}_roiroi_fc.csv: Group difference t-test results (followup)
- baselineFC_vs_deltaYBOCS_{atlas_name}_roiroi_fc.csv: Baseline FC vs symptom change
- deltaFC_vs_deltaYBOCS_{atlas_name}_roiroi_fc.csv: FC change vs symptom change
- summary_{atlas_name}_roiroi_fc.csv: Analysis summary

REQUIREMENTS:
============

- FC data files from NW_1st.py with atlas-specific naming
- group.csv and clinical.csv metadata files
- BIDS-formatted subject and session information
"""

import os
import glob
import re
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import warnings
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default configuration
DEFAULT_CONFIG = {
    'output_dir': '/project/6079231/dliang55/R01_AOCD/NW_group',
    'input_dir': '/project/6079231/dliang55/R01_AOCD',
    'log_file': 'roiroi_fc_analysis.log',
    'sessions': ['ses-baseline', 'ses-followup'],
    'min_subjects_per_group': 2,
    'fdr_alpha': 0.05,
    'default_atlas_name': 'power_2011'  # Default atlas name for backward compatibility
}

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir: str, log_filename: str) -> logging.Logger:
    """Set up logging configuration with both file and console handlers."""
    log_file = os.path.join(output_dir, log_filename)
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create logger
    logger = logging.getLogger('ROI_ROI_Analysis')
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
        description='ROI-to-ROI functional connectivity group analysis using customizable atlas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Power 2011 Atlas (Default)
  python NW_group3.py \\
    --subjects_csv group.csv \\
    --clinical_csv clinical.csv \\
    --input_dir /path/to/fc/data

  # Schaefer 2018 Atlas (400 ROIs, 7 networks)
  python NW_group3.py \\
    --subjects_csv group.csv \\
    --clinical_csv clinical.csv \\
    --input_dir /path/to/fc/data \\
    --atlas_name schaefer_2018_400_7_2

  # Schaefer 2018 Atlas (1000 ROIs, 17 networks)
  python NW_group3.py \\
    --subjects_csv group.csv \\
    --clinical_csv clinical.csv \\
    --input_dir /path/to/fc/data \\
    --atlas_name schaefer_2018_1000_17_1

  # With Specific Atlas Name
  python NW_group3.py \\
    --subjects_csv group.csv \\
    --clinical_csv clinical.csv \\
    --input_dir /path/to/fc/data \\
    --atlas_name power_2011

  # YEO 2011 Atlas (7 networks)
  python NW_group3.py \\
    --subjects_csv group.csv \\
    --clinical_csv clinical.csv \\
    --input_dir /path/to/fc/data \\
    --atlas_name yeo_2011_7_thick

  # YEO 2011 Atlas (17 networks)
  python NW_group3.py \\
    --subjects_csv group.csv \\
    --clinical_csv clinical.csv \\
    --input_dir /path/to/fc/data \\
    --atlas_name yeo_2011_17_thick

  # Harvard-Oxford Atlas
  python NW_group3.py \\
    --subjects_csv group.csv \\
    --clinical_csv clinical.csv \\
    --input_dir /path/to/fc/data \\
    --atlas_name harvard_oxford_cort-maxprob-thr25-2mm

  # Auto-detect Atlas from Input Files
  python NW_group3.py \\
    --subjects_csv group.csv \\
    --clinical_csv clinical.csv \\
    --input_dir /path/to/fc/data \\
    --auto-detect-atlas

Run with --help for full help.
        """
    )
    parser.add_argument(
        '--subjects_csv', 
        type=str, 
        required=True, 
        help='Path to group.csv file'
    )
    parser.add_argument(
        '--clinical_csv', 
        type=str, 
        required=True, 
        help='Path to clinical.csv file'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=DEFAULT_CONFIG['output_dir'],
        help='Output directory for results'
    )
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default=DEFAULT_CONFIG['input_dir'],
        help='Input directory for FC data'
    )
    parser.add_argument(
        '--atlas_name',
        type=str,
        help='Explicitly specify the atlas name (e.g., power_2011, schaefer_2018_400_7_2, schaefer_2018_1000_17_1)'
    )
    parser.add_argument(
        '--auto-detect-atlas',
        action='store_true',
        help='Auto-detect atlas name from available FC files in input directory'
    )
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true', 
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_group(subject_id: str, metadata_df: pd.DataFrame) -> Optional[str]:
    """Get group label for a subject."""
    if subject_id.startswith('sub-'):
        subject_id = subject_id.replace('sub-', '')
    
    group = metadata_df[metadata_df['subject_id'] == subject_id]['group']
    if group.empty:
        return None
    
    return group.iloc[0]

# Remove the get_ybocs_scores function - we'll handle this directly in the longitudinal analysis like NW_group.py

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_and_validate_metadata(subjects_csv: str, clinical_csv: str, logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate metadata CSVs including condition information."""
    logger.info("Loading metadata from %s and %s", subjects_csv, clinical_csv)
    
    try:
        df = pd.read_csv(subjects_csv)
        df['subject_id'] = df['sub'].astype(str)
        df = df[df['group'].isin(['HC', 'OCD'])]
        logger.info("Loaded %d subjects from %s", len(df), subjects_csv)
    except Exception as e:
        logger.error("Failed to load subjects CSV: %s", e)
        raise ValueError(f"Failed to load subjects CSV: {e}")

    try:
        df_clinical = pd.read_csv(clinical_csv)
        df_clinical['subject_id'] = df_clinical['sub'].astype(str)
        logger.info("Loaded %d clinical records from %s", len(df_clinical), clinical_csv)
    except Exception as e:
        logger.error("Failed to load clinical CSV: %s", e)
        raise ValueError(f"Failed to load clinical CSV: {e}")

    # Load condition information from shared_demographics.csv
    try:
        # The subjects_csv IS the shared_demographics.csv file, so condition info is already there
        if 'condition' in df.columns:
            # Fill missing conditions with 'unknown' for HC subjects, keep actual values for OCD subjects
            df.loc[df['group'] == 'HC', 'condition'] = 'unknown'
            df.loc[df['group'] == 'OCD', 'condition'] = df.loc[df['group'] == 'OCD', 'condition'].fillna('unknown')
            
            logger.info("Loaded condition information from subjects CSV")
        else:
            logger.warning("No condition column found in subjects CSV")
    except Exception as e:
        logger.warning("Failed to load condition information: %s", e)

    return df, df_clinical

def detect_atlas_name_from_files(input_dir: str, logger: logging.Logger) -> Optional[str]:
    """Auto-detect atlas name from available FC files."""
    logger.info("Auto-detecting atlas from files in %s", input_dir)
    
    # Look for ROI-to-ROI FC files
    roi_pattern = os.path.join(input_dir, '*_task-rest_*_roiroi_fc_avg.csv')
    roi_files = glob.glob(roi_pattern)
    
    if not roi_files:
        logger.error("No ROI-to-ROI FC files found in %s", input_dir)
        return None
    
    # Extract atlas names from filenames
    atlas_names = set()
    for file_path in roi_files:
        filename = os.path.basename(file_path)
        # Pattern: sub-XXX_ses-XXX_task-rest_ATLAS_roiroi_fc_avg.csv
        match = re.search(r'_task-rest_(.+?)_roiroi_fc_avg\.csv', filename)
        if match:
            atlas_names.add(match.group(1))
    
    if len(atlas_names) == 1:
        atlas_name = list(atlas_names)[0]
        logger.info("Detected atlas: %s", atlas_name)
        return atlas_name
    elif len(atlas_names) > 1:
        logger.warning("Multiple atlases detected: %s. Using the first one: %s", 
                      list(atlas_names), list(atlas_names)[0])
        return list(atlas_names)[0]
    else:
        logger.error("Could not detect atlas from files")
        return None

def validate_atlas_name(input_dir: str, atlas_name: str, logger: logging.Logger) -> bool:
    """Validate that FC files exist for the specified atlas."""
    pattern = os.path.join(input_dir, f'*_task-rest_{atlas_name}_roiroi_fc_avg.csv')
    files = glob.glob(pattern)
    
    if not files:
        logger.error("No FC files found for atlas '%s' in %s", atlas_name, input_dir)
        return False
    
    logger.info("Found %d FC files for atlas '%s'", len(files), atlas_name)
    return True

def validate_roiroi_fc_file(fc_path: str, logger: logging.Logger) -> bool:
    """Validate that ROI-to-ROI FC file has required columns."""
    required_columns = {'ROI', 'network1', 'network2', 'FC'}
    
    try:
        df = pd.read_csv(fc_path)
        missing_columns = required_columns - set(df.columns)
        
        if missing_columns:
            logger.warning("FC file %s missing required columns: %s", fc_path, missing_columns)
            return False
        
        if df.empty:
            logger.warning("FC file %s is empty", fc_path)
            return False
        
        return True
        
    except Exception as e:
        logger.error("Failed to validate FC file %s: %s", fc_path, e)
        return False

def load_roiroi_fc_data(
    subject_ids: List[str], 
    session: str, 
    input_dir: str, 
    atlas_name: str,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, str]]]:
    """Load ROI-to-ROI FC data for specified subjects and session."""
    logger.info("Loading ROI-to-ROI FC data for %d subjects, session: %s", len(subject_ids), session)
    
    fc_data = []
    dropped_subjects = []
    
    for sid in subject_ids:
        # Ensure subject ID has sub- prefix for file naming
        sid_with_sub = f"sub-{sid}" if not sid.startswith('sub-') else sid
        fc_path = os.path.join(input_dir, f'{sid_with_sub}_{session}_task-rest_{atlas_name}_roiroi_fc_avg.csv')
        
        if not os.path.exists(fc_path):
            logger.warning("FC file not found for subject %s: %s", sid, fc_path)
            dropped_subjects.append((sid, "file not found"))
            continue
        
        if not validate_roiroi_fc_file(fc_path, logger):
            dropped_subjects.append((sid, "invalid file format"))
            continue
        
        try:
            df = pd.read_csv(fc_path)
            df['subject_id'] = sid  # Use original subject ID without sub- prefix
            fc_data.append(df)
            logger.debug("Loaded FC data for %s %s: %d ROI pairs", sid, session, len(df))
            
        except Exception as e:
            logger.error("Failed to load FC data for subject %s: %s", sid, e)
            dropped_subjects.append((sid, f"loading error: {e}"))
            continue
    
    if dropped_subjects:
        logger.warning("Dropped %d subjects from FC data loading: %s", len(dropped_subjects), dropped_subjects)
    
    if not fc_data:
        logger.error("No FC data loaded for session %s", session)
        return pd.DataFrame(), {}
    
    # Combine all FC data
    combined_fc_data = pd.concat(fc_data, ignore_index=True)
    logger.info("Loaded ROI-to-ROI FC data for %d subjects, %d ROI pairs", len(fc_data), len(combined_fc_data))
    
    # Create feature info (ROI pairs with network information)
    feature_info = {}
    for _, row in combined_fc_data.iterrows():
        roi_pair = row['ROI']
        network1 = row.get('network1', 'Unknown')
        network2 = row.get('network2', 'Unknown')
        feature_info[roi_pair] = (network1, network2)
    
    logger.info("Created feature info for %d ROI pairs", len(feature_info))
    
    return combined_fc_data, feature_info

def validate_subjects(
    fc_dir: str, 
    metadata_df: pd.DataFrame, 
    atlas_name: str,
    logger: logging.Logger
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """Validate subjects and identify valid subjects for group and longitudinal analyses."""
    logger.info("Validating subjects for atlas: %s", atlas_name)
    
    sessions = DEFAULT_CONFIG['sessions']
    subject_sessions = {}
    
    # Find all available FC files
    fc_pattern = os.path.join(fc_dir, f'*_task-rest_{atlas_name}_roiroi_fc_avg.csv')
    fc_files = glob.glob(fc_pattern)
    
    if not fc_files:
        logger.error("No FC files found for atlas %s in %s", atlas_name, fc_dir)
        return [], [], {}
    
    logger.info("Found %d FC files for atlas '%s'", len(fc_files), atlas_name)
    
    # Parse subject sessions from filenames
    for fc_file in fc_files:
        filename = os.path.basename(fc_file)
        # Pattern: sub-XXX_ses-XXX_task-rest_ATLAS_roiroi_fc_avg.csv
        match = re.match(r'(sub-[^_]+)_(ses-[^_]+)_task-rest_.*_roiroi_fc_avg\.csv', filename)
        if match:
            subject = match.group(1)
            session = match.group(2)
            
            if subject not in subject_sessions:
                subject_sessions[subject] = []
            subject_sessions[subject].append(session)
    
    logger.info("Found FC data for %d subjects: %s", len(subject_sessions), list(subject_sessions.keys()))
    
    # Subjects valid for group analysis (need baseline)
    valid_group = []
    dropped_group = []
    for sid in metadata_df['subject_id']:
        # Check both with and without sub- prefix
        sid_with_sub = f"sub-{sid}" if not sid.startswith('sub-') else sid
        if sid_with_sub not in subject_sessions or 'ses-baseline' not in subject_sessions.get(sid_with_sub, []):
            logger.debug("Excluding subject %s from group analysis: no baseline FC file", sid)
            dropped_group.append((sid, "no baseline FC file"))
        else:
            valid_group.append(sid)
    
    # Subjects valid for longitudinal analysis (need both sessions)
    valid_longitudinal = []
    dropped_longitudinal = []
    for sid in metadata_df['subject_id']:
        # Check both with and without sub- prefix
        sid_with_sub = f"sub-{sid}" if not sid.startswith('sub-') else sid
        if sid_with_sub not in subject_sessions or not all(ses in subject_sessions.get(sid_with_sub, []) for ses in sessions):
            missing_sessions = [ses for ses in sessions if ses not in subject_sessions.get(sid_with_sub, [])]
            logger.debug(
                "Excluding subject %s from longitudinal analysis: missing session(s) %s",
                sid, missing_sessions
            )
            dropped_longitudinal.append((sid, f"missing session(s): {missing_sessions}"))
        else:
            valid_longitudinal.append(sid)
    
    # Log validation results
    logger.info("Valid subjects for group analysis: %d (%s)", len(valid_group), valid_group)
    if dropped_group:
        logger.info("Dropped %d subjects from group analysis: %s", len(dropped_group), dropped_group)
    
    logger.info("Valid subjects for longitudinal analysis: %d (%s)", len(valid_longitudinal), valid_longitudinal)
    if dropped_longitudinal:
        logger.info("Dropped %d subjects from longitudinal analysis: %s", len(dropped_longitudinal), dropped_longitudinal)
    
    return valid_group, valid_longitudinal, subject_sessions

# =============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# =============================================================================

def run_ttest(
    fc_data_hc: pd.DataFrame, 
    fc_data_ocd: pd.DataFrame, 
    feature_info: Dict[str, Tuple[str, str]],
    logger: logging.Logger
) -> pd.DataFrame:
    """Run two-sample t-tests with FDR correction for ROI-to-ROI FC."""
    logger.info(
        "Running t-tests with %d HC subjects and %d OCD subjects for %d ROI pairs",
        len(fc_data_hc), len(fc_data_ocd), len(feature_info)
    )
    
    results = []
    dropped_features = []
    
    for roi_pair, (net1, net2) in feature_info.items():
        hc_values = fc_data_hc[fc_data_hc['ROI'] == roi_pair]['FC'].dropna()
        ocd_values = fc_data_ocd[fc_data_ocd['ROI'] == roi_pair]['FC'].dropna()
        
        logger.debug(
            "ROI pair %s (%s_%s): HC n=%d, OCD n=%d", 
            roi_pair, net1, net2, len(hc_values), len(ocd_values)
        )
        
        if len(hc_values) < DEFAULT_CONFIG['min_subjects_per_group'] or \
           len(ocd_values) < DEFAULT_CONFIG['min_subjects_per_group']:
            logger.warning(
                "Skipping ROI pair %s (%s_%s) due to insufficient data (HC n=%d, OCD n=%d)",
                roi_pair, net1, net2, len(hc_values), len(ocd_values)
            )
            dropped_features.append((roi_pair, f"HC n={len(hc_values)}, OCD n={len(ocd_values)}"))
            continue
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(hc_values, ocd_values)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(hc_values) - 1) * np.var(hc_values, ddof=1) + 
                             (len(ocd_values) - 1) * np.var(ocd_values, ddof=1)) / 
                            (len(hc_values) + len(ocd_values) - 2))
        cohens_d = (np.mean(hc_values) - np.mean(ocd_values)) / pooled_std if pooled_std > 0 else 0
        
        results.append({
            'ROI': roi_pair,
            'network1': net1,
            'network2': net2,
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
    
    if not results:
        logger.warning("No valid ROI pairs for t-test analysis")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    # Apply FDR correction
    if len(results_df) > 1:
        p_vals = results_df['p_value'].values
        _, p_vals_corr = fdrcorrection(p_vals, alpha=DEFAULT_CONFIG['fdr_alpha'])
        results_df['p_corrected'] = p_vals_corr
        results_df['significant'] = p_vals_corr < DEFAULT_CONFIG['fdr_alpha']
    else:
        results_df['p_corrected'] = results_df['p_value']
        results_df['significant'] = results_df['p_value'] < DEFAULT_CONFIG['fdr_alpha']
    
    logger.info(
        "Generated t-test results for %d ROI pairs with %d HC and %d OCD subjects",
        len(results_df), len(fc_data_hc), len(fc_data_ocd)
    )
    
    if dropped_features:
        logger.warning("Dropped %d ROI pairs due to insufficient data", len(dropped_features))
    
    return results_df

def run_regression(
    fc_data: pd.DataFrame, 
    y_values: pd.Series, 
    feature_info: Dict[str, Tuple[str, str]],
    analysis_name: str,
    metadata_df: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """Run linear regression with condition confounder and FDR correction for ROI-to-ROI FC."""
    logger.info(
        "Running %s regression with condition confounder for %d subjects and %d ROI pairs",
        analysis_name, len(fc_data), len(feature_info)
    )
    
    results = []
    dropped_features = []
    
    # Ensure consistent index types
    fc_data.index = fc_data.index.astype(str)
    y_values.index = y_values.index.astype(str)
    
    # Find common subjects
    common_subjects = fc_data.index.intersection(y_values.index)
    logger.debug(
        "Common subjects for %s regression: %d (%s)", 
        analysis_name, len(common_subjects), list(common_subjects)
    )
    
    if not common_subjects.size:
        logger.warning(
            "No common subjects for %s regression. FC subjects: %s, YBOCS subjects: %s",
            analysis_name, list(fc_data.index), list(y_values.index)
        )
        return pd.DataFrame()
    
    # Filter data to common subjects
    fc_data = fc_data.loc[common_subjects]
    y_values = y_values.loc[common_subjects]
    
    # Add condition information for OCD subjects
    fc_data_with_condition = fc_data.reset_index()
    fc_data_with_condition = fc_data_with_condition.merge(
        metadata_df[['subject_id', 'condition']], 
        left_on='subject_id', 
        right_on='subject_id', 
        how='left'
    )
    
    # Fill missing conditions
    fc_data_with_condition['condition'] = fc_data_with_condition['condition'].fillna('unknown')
    
    logger.info("Condition distribution in %s regression: %s", 
                analysis_name, fc_data_with_condition['condition'].value_counts().to_dict())
    
    dropped_subjects = [sid for sid in fc_data.index if sid not in y_values.index]
    if dropped_subjects:
        logger.info(
            "Dropped %d subjects from %s regression due to missing YBOCS data: %s",
            len(dropped_subjects), analysis_name, dropped_subjects
        )

    # Run regression for each ROI pair
    for roi_pair, (net1, net2) in feature_info.items():
        roi_data = fc_data[fc_data['ROI'] == roi_pair]
        if roi_data.empty:
            logger.warning(
                "Skipping ROI pair %s (%s_%s) in %s regression due to empty data",
                roi_pair, net1, net2, analysis_name
            )
            dropped_features.append((roi_pair, "empty data"))
            continue
        
        # Get FC values for this ROI pair
        fc_values = roi_data['FC'].dropna()
        if fc_values.empty:
            dropped_features.append((roi_pair, "no FC data"))
            continue
        
        # Get corresponding YBOCS values
        y_vals = y_values.loc[fc_values.index].dropna()
        logger.debug(
            "ROI pair %s (%s_%s) in %s regression: n=%d", 
            roi_pair, net1, net2, analysis_name, len(y_vals)
        )
        
        if len(fc_values) < DEFAULT_CONFIG['min_subjects_per_group'] or \
           len(y_vals) < DEFAULT_CONFIG['min_subjects_per_group']:
            logger.warning(
                "Skipping ROI pair %s (%s_%s) in %s regression due to insufficient data (n=%d)",
                roi_pair, net1, net2, analysis_name, len(y_vals)
            )
            dropped_features.append((roi_pair, f"n={len(y_vals)}"))
            continue
        
        # Get condition data for this ROI pair
        roi_condition_data = fc_data_with_condition[
            fc_data_with_condition['subject_id'].isin(fc_values.index)
        ][['subject_id', 'FC', 'condition']].dropna()
        
        if len(roi_condition_data) < DEFAULT_CONFIG['min_subjects_per_group']:
            logger.warning(
                "Skipping ROI pair %s (%s_%s) in %s regression due to insufficient data (n=%d)",
                roi_pair, net1, net2, analysis_name, len(roi_condition_data)
            )
            dropped_features.append((roi_pair, f"n={len(roi_condition_data)}"))
            continue
        
        try:
            # Perform multiple linear regression with condition as confounder
            # Create formula for regression with condition confounder
            # Sanitize ROI pair name for formula (remove special characters)
            safe_roi = f"roi_{roi_pair.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}"
            
            # Prepare data for regression with sanitized column names
            regression_data = roi_condition_data.copy()
            regression_data['y_values'] = y_vals.loc[roi_condition_data['subject_id']].values
            regression_data[safe_roi] = regression_data['FC']
            
            # Create formula with sanitized names
            formula = f"{safe_roi} ~ y_values + condition"
            
            # Fit the model
            model = ols(formula, data=regression_data).fit()
            
            # Get FC effect (slope)
            fc_effect = model.params.get('y_values', 0)
            fc_pval = model.pvalues.get('y_values', 1.0)
            
            # Get condition effects
            condition_effects = {}
            condition_pvals = {}
            for cond in regression_data['condition'].unique():
                if cond != regression_data['condition'].iloc[0]:  # Reference condition
                    cond_param = f"condition[T.{cond}]"
                    condition_effects[cond] = model.params.get(cond_param, 0)
                    condition_pvals[cond] = model.pvalues.get(cond_param, 1.0)
            
            # Calculate R-squared
            r_squared = model.rsquared
            
            results.append({
                'ROI': roi_pair,
                'network1': net1,
                'network2': net2,
                'slope': fc_effect,
                'intercept': model.params.get('Intercept', 0),
                'r_value': np.sqrt(r_squared) if r_squared >= 0 else 0,
                'r_squared': r_squared,
                'p_value': fc_pval,
                'n': len(regression_data),
                'condition_effects': condition_effects,
                'condition_pvals': condition_pvals
            })
            
        except Exception as e:
            logger.warning("Failed to run regression for ROI pair %s: %s", roi_pair, e)
            dropped_features.append((roi_pair, f"regression error: {e}"))
            continue
    
    if not results:
        logger.warning("No valid ROI pairs for regression analysis")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    # Apply FDR correction
    if len(results_df) > 1:
        p_vals = results_df['p_value'].values
        _, p_vals_corr = fdrcorrection(p_vals, alpha=DEFAULT_CONFIG['fdr_alpha'])
        results_df['p_corrected'] = p_vals_corr
        results_df['significant'] = p_vals_corr < DEFAULT_CONFIG['fdr_alpha']
    else:
        results_df['p_corrected'] = results_df['p_value']
        results_df['significant'] = results_df['p_value'] < DEFAULT_CONFIG['fdr_alpha']
    
    logger.info("Generated regression results for %d ROI pairs", len(results_df))
    
    if dropped_features:
        logger.warning("Dropped %d ROI pairs from regression analysis", len(dropped_features))
    
    return results_df

def perform_group_analysis(
    baseline_fc_data: pd.DataFrame,
    metadata_df: pd.DataFrame,
    feature_info: Dict[str, Tuple[str, str]],
    output_dir: str,
    atlas_name: str,
    logger: logging.Logger
) -> bool:
    """Perform group difference analysis at baseline."""
    logger.info("Performing group difference analysis at baseline for atlas: %s", atlas_name)
    
    # Separate HC and OCD data
    hc_data = baseline_fc_data[
        baseline_fc_data['subject_id'].isin(metadata_df[metadata_df['group'] == 'HC']['subject_id'])
    ]
    ocd_data = baseline_fc_data[
        baseline_fc_data['subject_id'].isin(metadata_df[metadata_df['group'] == 'OCD']['subject_id'])
    ]
    
    logger.info("Group t-test analysis: %d HC subjects, %d OCD subjects", len(hc_data), len(ocd_data))
    
    if hc_data.empty or ocd_data.empty:
        logger.warning(
            "Insufficient data for group t-test analysis (HC empty: %s, OCD empty: %s)",
            hc_data.empty, ocd_data.empty
        )
        return False
    
    # Run t-tests
    ttest_results = run_ttest(hc_data, ocd_data, feature_info, logger)
    
    if not ttest_results.empty:
        output_path = os.path.join(output_dir, f'group_diff_baseline_{atlas_name}_roiroi_fc.csv')
        ttest_results.to_csv(output_path, index=False)
        logger.info(
            "Saved t-test results to %s with columns: %s", 
            output_path, list(ttest_results.columns)
        )
        return True
    else:
        logger.info("No significant t-test results to save")
        return False

def perform_longitudinal_analysis(
    baseline_fc_data: pd.DataFrame,
    metadata_df: pd.DataFrame,
    df_clinical: pd.DataFrame,
    valid_longitudinal: List[str],
    feature_info: Dict[str, Tuple[str, str]],
    input_dir: str,
    output_dir: str,
    atlas_name: str,
    logger: logging.Logger
) -> None:
    """Perform longitudinal analysis for OCD subjects."""
    logger.info("Performing longitudinal analysis for atlas: %s", atlas_name)
    
    # Prepare OCD clinical data
    ocd_df = df_clinical[
        df_clinical['subject_id'].isin(metadata_df[metadata_df['group'] == 'OCD']['subject_id'])
    ].copy()
    ocd_df['delta_ybocs'] = ocd_df['ybocs_baseline'] - ocd_df['ybocs_followup']
    
    logger.info(
        "Longitudinal analysis: %d OCD subjects with YBOCS data: %s",
        len(ocd_df), list(ocd_df['subject_id'])
    )

    # 1. Baseline FC vs symptom change
    baseline_fc_ocd = baseline_fc_data[
        baseline_fc_data['subject_id'].isin(ocd_df['subject_id'])
    ]
    
    logger.info(
        "Baseline FC vs delta YBOCS regression: %d OCD subjects with FC data: %s",
        len(baseline_fc_ocd), list(baseline_fc_ocd['subject_id'])
    )
    
    if not baseline_fc_ocd.empty:
        regression_results = run_regression(
            baseline_fc_ocd.set_index('subject_id'),
            ocd_df.set_index('subject_id')['delta_ybocs'],
            feature_info,
            "baseline FC vs delta YBOCS",
            metadata_df,
            logger
        )
        
        if not regression_results.empty:
            output_path = os.path.join(output_dir, f'baselineFC_vs_deltaYBOCS_{atlas_name}_roiroi_fc.csv')
            regression_results.to_csv(output_path, index=False)
            logger.info(
                "Saved baseline FC regression results to %s with columns: %s", 
                output_path, list(regression_results.columns)
            )
        else:
            logger.info("No significant baseline FC regression results to save")
    else:
        logger.warning("No baseline FC data for OCD subjects in longitudinal analysis")

    # 2. FC change vs symptom change
    logger.info("Analyzing FC change vs symptom change")
    fc_change_data = []
    dropped_longitudinal_subjects = []
    
    for sid in valid_longitudinal:
        # Ensure subject ID has sub- prefix for file naming
        sid_with_sub = f"sub-{sid}" if not sid.startswith('sub-') else sid
        base_path = os.path.join(input_dir, f'{sid_with_sub}_ses-baseline_task-rest_{atlas_name}_roiroi_fc_avg.csv')
        follow_path = os.path.join(input_dir, f'{sid_with_sub}_ses-followup_task-rest_{atlas_name}_roiroi_fc_avg.csv')
        
        if not (os.path.exists(base_path) and os.path.exists(follow_path)):
            logger.warning(
                "Missing ROI-to-ROI FC files for subject %s (baseline: %s, followup: %s)",
                sid, os.path.exists(base_path), os.path.exists(follow_path)
            )
            dropped_longitudinal_subjects.append(
                (sid, f"missing files: baseline={os.path.exists(base_path)}, followup={os.path.exists(follow_path)}")
            )
            continue
        
        if not (validate_roiroi_fc_file(base_path, logger) and validate_roiroi_fc_file(follow_path, logger)):
            dropped_longitudinal_subjects.append((sid, "invalid FC file format"))
            continue
        
        try:
            # Load baseline and followup data
            base_fc = pd.read_csv(base_path)
            follow_fc = pd.read_csv(follow_path)
            
            logger.debug(
                "Loaded baseline ROI-to-ROI FC (%d rows) and followup FC (%d rows) for %s",
                len(base_fc), len(follow_fc), sid
            )
            
            # Calculate FC change
            change_data = base_fc.copy()
            change_data['FC'] = follow_fc['FC'] - base_fc['FC']
            change_data['subject_id'] = sid  # Use original subject ID without sub- prefix
            fc_change_data.append(change_data)
            
        except Exception as e:
            logger.error(
                "Failed to process longitudinal ROI-to-ROI FC for subject %s: %s", 
                sid, e
            )
            dropped_longitudinal_subjects.append((sid, f"processing error: {e}"))
            continue

    if dropped_longitudinal_subjects:
        logger.info(
            "Dropped %d subjects from FC change analysis: %s",
            len(dropped_longitudinal_subjects), dropped_longitudinal_subjects
        )
    
    if fc_change_data:
        fc_change_data = pd.concat(fc_change_data, ignore_index=True)
        fc_change_data['subject_id'] = fc_change_data['subject_id'].astype(str)
        
        logger.info(
            "Loaded ROI-to-ROI FC change data for %d subjects, %d ROI pairs: %s",
            len(fc_change_data['subject_id'].unique()), len(fc_change_data),
            list(fc_change_data['subject_id'].unique())
        )
        
        # Run regression analysis
        regression_results = run_regression(
            fc_change_data.set_index('subject_id'),
            ocd_df.set_index('subject_id')['delta_ybocs'],
            feature_info,
            "delta FC vs delta YBOCS",
            metadata_df,
            logger
        )
        
        if not regression_results.empty:
            output_path = os.path.join(output_dir, f'deltaFC_vs_deltaYBOCS_{atlas_name}_roiroi_fc.csv')
            regression_results.to_csv(output_path, index=False)
            logger.info(
                "Saved FC change regression results to %s with columns: %s", 
                output_path, list(regression_results.columns)
            )
        else:
            logger.info("No significant FC change regression results to save")
    else:
        logger.warning("No ROI-to-ROI FC change data loaded for longitudinal analysis")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run ROI-to-ROI FC group analysis."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging (will be updated with atlas name after detection)
    logger = setup_logging(args.output_dir, DEFAULT_CONFIG['log_file'])
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("=" * 80)
    logger.info("Starting ROI-to-ROI Functional Connectivity Group Analysis")
    logger.info("=" * 80)
    logger.info("Arguments: %s", vars(args))
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load metadata
        df, df_clinical = load_and_validate_metadata(args.subjects_csv, args.clinical_csv, logger)

        # Normalize subject IDs
        df['subject_id'] = df['subject_id'].str.replace('sub-', '')
        df_clinical['subject_id'] = df_clinical['subject_id'].str.replace('sub-', '')
        logger.debug("Normalized subject IDs in metadata")

        # Determine atlas name
        if args.atlas_name:
            atlas_name = args.atlas_name
            logger.info("Using explicitly specified atlas name: %s", atlas_name)
        elif args.auto_detect_atlas:
            atlas_name = detect_atlas_name_from_files(args.input_dir, logger)
            logger.info("Auto-detected atlas name: %s", atlas_name)
        else:
            atlas_name = DEFAULT_CONFIG['default_atlas_name']
            logger.info("Using default atlas name: %s", atlas_name)
        
        # Validate atlas name
        if not validate_atlas_name(args.input_dir, atlas_name, logger):
            raise ValueError(f"Invalid atlas name: {atlas_name}")
        
        logger.info("Using atlas: %s", atlas_name)
        
        # Update log file name to include atlas name
        atlas_log_file = f'roiroi_fc_analysis_{atlas_name}.log'
        logger.info("Switching to atlas-specific log file: %s", atlas_log_file)
        
        # Create new logger with atlas-specific log file
        atlas_logger = setup_logging(args.output_dir, atlas_log_file)
        if args.verbose:
            atlas_logger.setLevel(logging.DEBUG)
        
        # Validate subjects
        valid_group, valid_longitudinal, subject_sessions = validate_subjects(
            args.input_dir, df, atlas_name, atlas_logger
        )
        
        if not valid_group and not valid_longitudinal:
            dir_contents = os.listdir(args.input_dir) if os.path.exists(args.input_dir) else []
            atlas_logger.error(
                "No valid subjects found for any analysis. Check input directory %s (contents: %s) and FC file generation from NW_1st.py.",
                args.input_dir, dir_contents
            )
            raise ValueError("No valid subjects found for any analysis. Check input directory and FC file generation.")

        # Load baseline ROI-to-ROI FC data
        baseline_fc_data, feature_info = load_roiroi_fc_data(
            valid_group, 'ses-baseline', args.input_dir, atlas_name, atlas_logger
        )
        
        if baseline_fc_data.empty:
            atlas_logger.warning("No baseline ROI-to-ROI FC data loaded. Skipping group and longitudinal analyses.")
            return

        # 1. Group difference at baseline
        if valid_group:
            perform_group_analysis(baseline_fc_data, df, feature_info, args.output_dir, atlas_name, atlas_logger)

        # 2. Longitudinal analyses
        if valid_longitudinal:
            perform_longitudinal_analysis(
                baseline_fc_data, df, df_clinical, valid_longitudinal, 
                feature_info, args.input_dir, args.output_dir, atlas_name, atlas_logger
            )

        atlas_logger.info("Main ROI-to-ROI FC analysis completed successfully for atlas: %s", atlas_name)
        atlas_logger.info("Analyses completed: Group comparison, Longitudinal analysis")
    
    except Exception as e:
        logger.error("Main execution failed: %s", e)
        raise
    
    finally:
        logger.info("=" * 80)
        logger.info("Analysis complete!")
        logger.info("=" * 80)

if __name__ == "__main__":
    main()
    