#!/usr/bin/env python3
"""
ROI-to-Network Functional Connectivity Group Analysis using Customizable Atlas

This script performs group-level statistical analysis on ROI-to-network functional connectivity data.
It compares healthy controls (HC) vs. OCD patients and performs longitudinal analyses.

The script is compatible with both custom atlases and Nilearn built-in atlases (e.g., Schaefer 2018,
Harvard-Oxford, Power 2011, etc.) and automatically handles atlas detection and validation.

Author: [Your Name]
Date: [Current Date]

USAGE EXAMPLES:
==============

1. Power 2011 Atlas (Default):
   python NW_group.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data

2. With Specific Atlas Name:
   python NW_group.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --atlas_name power_2011

3. Auto-detect Atlas from Input Files:
   python NW_group.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --auto-detect-atlas

ATLAS NAMING CONVENTIONS:
========================

The script automatically detects atlas names from input FC files:
- Power 2011: power_2011_network_fc_avg.csv
- Schaefer 2018: schaefer_2018_400_7_2_network_fc_avg.csv
- Custom: custom_atlas_network_fc_avg.csv

OUTPUT FILES:
============

- group_diff_baseline_{atlas_name}_roi_network_fc.csv: Group difference t-test results
- baselineFC_vs_deltaYBOCS_{atlas_name}_roi_network_fc.csv: Baseline FC vs symptom change
- deltaFC_vs_deltaYBOCS_{atlas_name}_roi_network_fc.csv: FC change vs symptom change

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
    'output_dir': '/scratch/xxqian/OCD/NW_group',
    'input_dir': '/scratch/xxqian/OCD',
    'log_file': 'roi_network_group_analysis.log',
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
    logger = logging.getLogger('ROI_Network_Analysis')
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
        description='ROI-to-network functional connectivity group analysis using customizable atlas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Power 2011 Atlas (Default)
  python NW_group.py \\
    --subjects_csv group.csv \\
    --clinical_csv clinical.csv \\
    --input_dir /path/to/fc/data

  # With Specific Atlas Name
  python NW_group.py \\
    --subjects_csv group.csv \\
    --clinical_csv clinical.csv \\
    --input_dir /path/to/fc/data \\
    --atlas_name power_2011

  # Auto-detect Atlas from Input Files
  python NW_group.py \\
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
        help='Explicitly specify the atlas name (e.g., power_2011, schaefer_2018_400_7_2)'
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
# ATLAS DETECTION AND VALIDATION
# =============================================================================

def detect_atlas_name_from_files(input_dir: str, logger: logging.Logger) -> str:
    """Auto-detect atlas name from available FC files."""
    logger.info("Auto-detecting atlas name from input directory: %s", input_dir)
    
    if not os.path.exists(input_dir):
        logger.warning("Input directory does not exist, using default atlas name: %s", DEFAULT_CONFIG['default_atlas_name'])
        return DEFAULT_CONFIG['default_atlas_name']
    
    # Look for FC files with different atlas patterns
    fc_patterns = [
        '*_task-rest_*_network_fc_avg.csv',  # General pattern
        '*_task-rest_*_roiroi_matrix_avg.npy'  # Matrix files
    ]
    
    detected_atlases = set()
    
    for pattern in fc_patterns:
        files = glob.glob(os.path.join(input_dir, pattern))
        for file_path in files:
            filename = os.path.basename(file_path)
            # Pattern: sub-XXX_ses-XXX_task-rest_ATLAS_NAME_network_fc_avg.csv
            # or: sub-XXX_ses-XXX_task-rest_ATLAS_NAME_roiroi_matrix_avg.npy
            match = re.search(r'_task-rest_(.+?)_(?:network_fc|roiroi_matrix)_avg', filename)
            if match:
                atlas_name = match.group(1)
                detected_atlases.add(atlas_name)
                logger.debug("Detected atlas name '%s' from file: %s", atlas_name, filename)
    
    if not detected_atlases:
        logger.warning("No atlas names detected, using default: %s", DEFAULT_CONFIG['default_atlas_name'])
        return DEFAULT_CONFIG['default_atlas_name']
    
    if len(detected_atlases) > 1:
        logger.warning("Multiple atlas names detected: %s. Using first one: %s", detected_atlases, list(detected_atlases)[0])
        return list(detected_atlases)[0]
    
    detected_atlas = list(detected_atlases)[0]
    logger.info("Auto-detected atlas name: %s", detected_atlas)
    return detected_atlas

def validate_atlas_name(atlas_name: str, input_dir: str, logger: logging.Logger) -> bool:
    """Validate that the detected or specified atlas name actually corresponds to existing files."""
    logger.info("Validating atlas name '%s' in input directory: %s", atlas_name, input_dir)
    
    if not os.path.exists(input_dir):
        logger.error("Input directory does not exist: %s", input_dir)
        return False
    
    # Check for FC files with this atlas name
    fc_pattern = os.path.join(input_dir, f'*_task-rest_{atlas_name}_network_fc_avg.csv')
    fc_files = glob.glob(fc_pattern)
    
    if not fc_files:
        logger.error("No FC files found for atlas '%s' in directory: %s", atlas_name, input_dir)
        logger.error("Expected pattern: *_task-rest_%s_network_fc_avg.csv", atlas_name)
        return False
    
    logger.info("Found %d FC files for atlas '%s': %s", len(fc_files), atlas_name, 
                [os.path.basename(f) for f in fc_files[:5]])  # Show first 5 files
    
    return True

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_network_fc_path(subject: str, session: str, input_dir: str, atlas_name: str) -> str:
    """Get path to ROI-to-network FC CSV file."""
    if not subject.startswith('sub-'):
        subject = f"sub-{subject}"
    
    path = os.path.join(
        input_dir, 
        f"{subject}_{session}_task-rest_{atlas_name}_network_fc_avg.csv"
    )
    return path

def get_group(subject_id: str, metadata_df: pd.DataFrame) -> Optional[str]:
    """Get group label for a subject."""
    if subject_id.startswith('sub-'):
        subject_id = subject_id.replace('sub-', '')
    
    group = metadata_df[metadata_df['subject_id'] == subject_id]['group']
    if group.empty:
        return None
    
    return group.iloc[0]

# =============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# =============================================================================

def run_ttest(
    fc_data_hc: pd.DataFrame, 
    fc_data_ocd: pd.DataFrame, 
    feature_info: Dict[str, Tuple[str, str]],
    logger: logging.Logger
) -> pd.DataFrame:
    """Run two-sample t-tests with FDR correction for ROI-to-network FC."""
    logger.info(
        "Running t-tests with %d HC subjects and %d OCD subjects for %d ROI-to-network features",
        len(fc_data_hc), len(fc_data_ocd), len(feature_info)
    )
    
    results = []
    dropped_features = []
    
    for feature, (net1, net2) in feature_info.items():
        hc_values = fc_data_hc[feature].dropna()
        ocd_values = fc_data_ocd[feature].dropna()
        
        logger.debug(
            "Feature %s (%s_%s): HC n=%d, OCD n=%d", 
            feature, net1, net2, len(hc_values), len(ocd_values)
        )
        
        if len(hc_values) < DEFAULT_CONFIG['min_subjects_per_group'] or \
           len(ocd_values) < DEFAULT_CONFIG['min_subjects_per_group']:
            logger.warning(
                "Skipping feature %s (%s_%s) due to insufficient data (HC n=%d, OCD n=%d)",
                feature, net1, net2, len(hc_values), len(ocd_values)
            )
            dropped_features.append((feature, f"HC n={len(hc_values)}, OCD n={len(ocd_values)}"))
            continue
        
        # Perform t-test
        t_stat, p_val = stats.ttest_ind(ocd_values, hc_values, equal_var=False)
        
        results.append({
            'ROI': feature,
            'network1': net1,
            'network2': net2,
            't_statistic': t_stat,
            'p_value': p_val,
            'OCD_mean': np.mean(ocd_values),
            'HC_mean': np.mean(hc_values),
            'OCD_n': len(ocd_values),
            'HC_n': len(ocd_values)
        })
    
    if dropped_features:
        logger.info(
            "Dropped %d features due to insufficient data: %s", 
            len(dropped_features), dropped_features
        )
    
    if not results:
        logger.info("No t-test results generated (no valid features)")
        return pd.DataFrame()
    
    # Create results DataFrame and apply FDR correction
    results_df = pd.DataFrame(results)
    p_vals = results_df['p_value'].values
    _, p_vals_corr = fdrcorrection(p_vals, alpha=DEFAULT_CONFIG['fdr_alpha'])
    results_df['p_value_fdr'] = p_vals_corr
    
    logger.info(
        "Generated t-test results for %d features with %d HC and %d OCD subjects",
        len(results_df), len(fc_data_hc), len(fc_data_ocd)
    )
    
    return results_df

def run_regression(
    fc_data: pd.DataFrame, 
    y_values: pd.Series, 
    feature_info: Dict[str, Tuple[str, str]], 
    analysis_name: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """Run linear regression with FDR correction for ROI-to-network FC."""
    logger.info(
        "Running %s regression for %d subjects and %d ROI-to-network features",
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
    
    dropped_subjects = [sid for sid in fc_data.index if sid not in y_values.index]
    if dropped_subjects:
        logger.info(
            "Dropped %d subjects from %s regression due to missing YBOCS data: %s",
            len(dropped_subjects), analysis_name, dropped_subjects
        )

    # Run regression for each feature
    for feature, (net1, net2) in feature_info.items():
        x = fc_data[feature].dropna()
        if x.empty:
            logger.warning(
                "Skipping feature %s (%s_%s) in %s regression due to empty data",
                feature, net1, net2, analysis_name
            )
            dropped_features.append((feature, "empty data"))
            continue
        
        y = y_values.loc[x.index].dropna()
        logger.debug(
            "Feature %s (%s_%s) in %s regression: n=%d", 
            feature, net1, net2, analysis_name, len(y)
        )
        
        if len(x) < DEFAULT_CONFIG['min_subjects_per_group'] or \
           len(y) < DEFAULT_CONFIG['min_subjects_per_group']:
            logger.warning(
                "Skipping feature %s (%s_%s) in %s regression due to insufficient data (n=%d)",
                feature, net1, net2, analysis_name, len(y)
            )
            dropped_features.append((feature, f"n={len(y)}"))
            continue
        
        # Perform linear regression
        x = x.values.reshape(-1, 1)
        y = y.values
        slope, intercept, r_value, p_val, _ = stats.linregress(x.flatten(), y)
        
        results.append({
            'ROI': feature,
            'network1': net1,
            'network2': net2,
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'p_value': p_val,
            'n': len(y)
        })
    
    if dropped_features:
        logger.info(
            "Dropped %d features in %s regression due to insufficient data: %s",
            len(dropped_features), analysis_name, dropped_features
        )
    
    if not results:
        logger.info("No %s regression results generated (no valid features)", analysis_name)
        return pd.DataFrame()
    
    # Create results DataFrame and apply FDR correction
    results_df = pd.DataFrame(results)
    p_vals = results_df['p_value'].values
    _, p_vals_corr = fdrcorrection(p_vals, alpha=DEFAULT_CONFIG['fdr_alpha'])
    results_df['p_value_fdr'] = p_vals_corr
    
    logger.info(
        "Generated %s regression results for %d features with %d subjects",
        analysis_name, len(results_df), len(common_subjects)
    )
    
    return results_df

# =============================================================================
# DATA LOADING AND VALIDATION
# =============================================================================

def load_and_validate_metadata(
    subjects_csv: str, 
    clinical_csv: str, 
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate metadata CSVs."""
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

    return df, df_clinical

def validate_subjects(
    fc_dir: str, 
    metadata_df: pd.DataFrame, 
    atlas_name: str,
    logger: logging.Logger
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """Validate subjects based on available ROI-to-network FC files."""
    logger.info("Validating subjects in FC directory %s with atlas %s", fc_dir, atlas_name)
    
    if not os.path.exists(fc_dir):
        logger.error("Input directory %s does not exist", fc_dir)
        raise ValueError(f"Input directory {fc_dir} does not exist")
    
    # Find FC files with the specific atlas
    fc_files = glob.glob(os.path.join(fc_dir, f'*_task-rest_{atlas_name}_network_fc_avg.csv'))
    logger.info("Found %d FC files for atlas %s", len(fc_files), atlas_name)
    
    if not fc_files:
        # Try to find any FC files to help with debugging
        all_fc_files = glob.glob(os.path.join(fc_dir, '*_task-rest_*_network_fc_avg.csv'))
        if all_fc_files:
            detected_atlases = set()
            for f in all_fc_files:
                filename = os.path.basename(f)
                match = re.search(r'_task-rest_(.+?)_network_fc_avg', filename)
                if match:
                    detected_atlases.add(match.group(1))
            logger.error("No FC files found for atlas '%s'. Available atlases: %s", atlas_name, detected_atlases)
        else:
            dir_contents = os.listdir(fc_dir)
            logger.error("No FC files found in %s. Directory contents: %s", fc_dir, dir_contents)
        raise ValueError(f"No FC files found for atlas '{atlas_name}' in directory {fc_dir}")
    
    # Parse subject and session information
    subject_sessions = {}
    dropped_subjects = []
    
    for f in fc_files:
        filename = os.path.basename(f)
        if '_ses-' not in filename or f'_task-rest_{atlas_name}_network_fc_avg.csv' not in filename:
            logger.debug("Skipping invalid FC file: %s", filename)
            continue
        
        parts = filename.split('_')
        if len(parts) < 2:
            logger.debug("Skipping invalid FC filename: %s", filename)
            continue
        
        subject = parts[0].replace('sub-', '')
        session = parts[1]
        subject_sessions.setdefault(subject, []).append(session)
        logger.debug("Found subject %s, session %s in file %s", subject, session, filename)

    # Validate against metadata
    csv_subjects = set(metadata_df['subject_id'])
    file_subjects = set(subject_sessions.keys())
    unmatched = file_subjects - csv_subjects
    
    if unmatched:
        logger.warning(
            "Found FC files for %d subjects not in subjects_csv: %s", 
            len(unmatched), unmatched
        )
        dropped_subjects.extend([(sid, "not in subjects_csv") for sid in unmatched])
    
    logger.info(
        "CSV subjects: %d, FC file subjects: %d, overlap: %d",
        len(csv_subjects), len(file_subjects), len(csv_subjects & file_subjects)
    )
    
    # Determine valid subjects for different analyses
    sessions = DEFAULT_CONFIG['sessions']
    
    # Subjects valid for group analysis (need baseline)
    valid_group = []
    dropped_group = []
    for sid in metadata_df['subject_id']:
        sid_clean = sid.replace('sub-', '')
        if sid_clean not in subject_sessions or 'ses-baseline' not in subject_sessions.get(sid_clean, []):
            logger.debug("Excluding subject %s from group analysis: no baseline FC file", sid)
            dropped_group.append((sid, "no baseline FC file"))
        else:
            valid_group.append(sid)
    
    # Subjects valid for longitudinal analysis (need both sessions)
    valid_longitudinal = []
    dropped_longitudinal = []
    for sid in metadata_df['subject_id']:
        sid_clean = sid.replace('sub-', '')
        if sid_clean not in subject_sessions or not all(ses in subject_sessions.get(sid_clean, []) for ses in sessions):
            missing_sessions = [ses for ses in sessions if ses not in subject_sessions.get(sid_clean, [])]
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

def validate_network_fc_file(fc_path: str, logger: logging.Logger) -> bool:
    """Validate that ROI-to-network FC file has required columns."""
    required_columns = {'ROI', 'network1', 'network2', 'fc_value'}
    
    try:
        df = pd.read_csv(fc_path)
        missing_columns = required_columns - set(df.columns)
        
        if missing_columns:
            logger.error(
                "ROI-to-network FC file %s missing required columns: %s. Found columns: %s",
                fc_path, missing_columns, list(df.columns)
            )
            return False
        
        logger.debug("ROI-to-network FC file %s validated successfully", fc_path)
        return True
        
    except Exception as e:
        logger.error("Failed to validate ROI-to-network FC file %s: %s", fc_path, e)
        return False

def load_network_fc_data(
    subject_ids: List[str], 
    session: str, 
    input_dir: str, 
    atlas_name: str,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, Optional[Dict[str, Tuple[str, str]]]]:
    """Load ROI-to-network FC data for given subjects and session."""
    logger.info(
        "Loading ROI-to-network FC data for %d subjects, session %s, atlas %s: %s",
        len(subject_ids), session, atlas_name, subject_ids
    )
    
    fc_data = []
    feature_info = None
    valid_subjects = []
    dropped_subjects = []
    
    for sid in subject_ids:
        sid_no_prefix = sid.replace('sub-', '')
        fc_path = get_network_fc_path(sid_no_prefix, session, input_dir, atlas_name)
        
        if not os.path.exists(fc_path):
            logger.warning("ROI-to-network FC file not found for subject %s: %s", sid, fc_path)
            dropped_subjects.append((sid, f"missing FC file: {fc_path}"))
            continue
        
        if not validate_network_fc_file(fc_path, logger):
            dropped_subjects.append((sid, f"invalid FC file format: {fc_path}"))
            continue
        
        try:
            fc_df = pd.read_csv(fc_path)
            logger.debug("Loaded ROI-to-network FC file %s with %d rows", fc_path, len(fc_df))
            
            # Create feature identifier and map networks
            fc_df['feature_id'] = fc_df['ROI']
            
            if feature_info is None:
                feature_info = {
                    row['ROI']: (row['network1'], row['network2'])
                    for _, row in fc_df[['ROI', 'network1', 'network2']].drop_duplicates().iterrows()
                }
                logger.debug(
                    "Identified %d ROI-to-network feature columns with network mappings", 
                    len(feature_info)
                )
            
            # Pivot to make features as columns
            fc_pivot = fc_df.pivot_table(
                index=None,
                columns='feature_id',
                values='fc_value'
            ).reset_index(drop=True)
            fc_pivot['subject_id'] = sid_no_prefix
            fc_data.append(fc_pivot)
            valid_subjects.append(sid)
            
        except Exception as e:
            logger.error(
                "Failed to process ROI-to-network FC file %s for subject %s: %s", 
                fc_path, sid, e
            )
            dropped_subjects.append((sid, f"processing error: {e}"))
            continue
    
    if dropped_subjects:
        logger.info(
            "Dropped %d subjects from %s FC data loading: %s", 
            len(dropped_subjects), session, dropped_subjects
        )
    
    if not fc_data:
        logger.warning("No valid ROI-to-network FC data loaded for session %s", session)
        return pd.DataFrame(), feature_info
    
    fc_data_df = pd.concat(fc_data, ignore_index=True)
    logger.info(
        "Loaded ROI-to-network FC data for %d subjects, %d features: %s",
        len(valid_subjects), len(feature_info), valid_subjects
    )
    
    return fc_data_df, feature_info

# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

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
        output_path = os.path.join(output_dir, f'group_diff_baseline_{atlas_name}_roi_network_fc.csv')
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
            logger
        )
        
        if not regression_results.empty:
            output_path = os.path.join(output_dir, f'baselineFC_vs_deltaYBOCS_{atlas_name}_roi_network_fc.csv')
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
        sid_clean = sid.replace('sub-', '')
        base_path = get_network_fc_path(sid_clean, 'ses-baseline', input_dir, atlas_name)
        follow_path = get_network_fc_path(sid_clean, 'ses-followup', input_dir, atlas_name)
        
        if not (os.path.exists(base_path) and os.path.exists(follow_path)):
            logger.warning(
                "Missing ROI-to-network FC files for subject %s (baseline: %s, followup: %s)",
                sid, os.path.exists(base_path), os.path.exists(follow_path)
            )
            dropped_longitudinal_subjects.append(
                (sid, f"missing files: baseline={os.path.exists(base_path)}, followup={os.path.exists(follow_path)}")
            )
            continue
        
        if not (validate_network_fc_file(base_path, logger) and validate_network_fc_file(follow_path, logger)):
            dropped_longitudinal_subjects.append((sid, "invalid FC file format"))
            continue
        
        try:
            # Load baseline and followup data
            base_fc = pd.read_csv(base_path)
            follow_fc = pd.read_csv(follow_path)
            
            logger.debug(
                "Loaded baseline ROI-to-network FC (%d rows) and followup FC (%d rows) for %s",
                len(base_fc), len(follow_fc), sid
            )
            
            # Create feature identifiers
            base_fc['feature_id'] = base_fc['ROI']
            follow_fc['feature_id'] = follow_fc['ROI']
            
            # Pivot and compute change
            base_pivot = base_fc.pivot_table(
                index=None, columns='feature_id', values='fc_value'
            ).reset_index(drop=True)
            follow_pivot = follow_fc.pivot_table(
                index=None, columns='feature_id', values='fc_value'
            ).reset_index(drop=True)
            
            change_pivot = follow_pivot - base_pivot
            change_pivot['subject_id'] = sid_clean
            fc_change_data.append(change_pivot)
            
        except Exception as e:
            logger.error(
                "Failed to process longitudinal ROI-to-network FC for subject %s: %s", 
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
        feature_columns = [col for col in fc_change_data.columns if col != 'subject_id']
        
        logger.info(
            "Loaded ROI-to-network FC change data for %d subjects, %d features: %s",
            len(fc_change_data), len(feature_columns), list(fc_change_data['subject_id'])
        )
        
        # Run regression analysis
        regression_results = run_regression(
            fc_change_data.set_index('subject_id'),
            ocd_df.set_index('subject_id')['delta_ybocs'],
            feature_info,
            "delta FC vs delta YBOCS",
            logger
        )
        
        if not regression_results.empty:
            output_path = os.path.join(output_dir, f'deltaFC_vs_deltaYBOCS_{atlas_name}_roi_network_fc.csv')
            regression_results.to_csv(output_path, index=False)
            logger.info(
                "Saved FC change regression results to %s with columns: %s",
                output_path, list(regression_results.columns)
            )
        else:
            logger.info("No significant FC change regression results to save")
    else:
        logger.warning("No ROI-to-network FC change data loaded for longitudinal analysis")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run ROI-to-network FC group analysis."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.output_dir, DEFAULT_CONFIG['log_file'])
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("=" * 80)
    logger.info("Starting ROI-to-Network FC Group Analysis")
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
        if not validate_atlas_name(atlas_name, args.input_dir, logger):
            raise ValueError(f"Invalid atlas name: {atlas_name}")
        
        logger.info("Using atlas: %s", atlas_name)
        
        # Validate subjects
        valid_group, valid_longitudinal, subject_sessions = validate_subjects(
            args.input_dir, df, atlas_name, logger
        )
        
        if not valid_group and not valid_longitudinal:
            dir_contents = os.listdir(args.input_dir) if os.path.exists(args.input_dir) else []
            logger.error(
                "No valid subjects found for any analysis. Check input directory %s (contents: %s) and FC file generation from NW_1st.py.",
                args.input_dir, dir_contents
            )
            raise ValueError("No valid subjects found for any analysis. Check input directory and FC file generation.")

        # Load baseline ROI-to-network FC data
        baseline_fc_data, feature_info = load_network_fc_data(
            valid_group, 'ses-baseline', args.input_dir, atlas_name, logger
        )
        
        if baseline_fc_data.empty:
            logger.warning("No baseline ROI-to-network FC data loaded. Skipping group and longitudinal analyses.")
            return

        # 1. Group difference at baseline
        if valid_group:
            perform_group_analysis(baseline_fc_data, df, feature_info, args.output_dir, atlas_name, logger)

        # 2. Longitudinal analyses
        if valid_longitudinal:
            perform_longitudinal_analysis(
                baseline_fc_data, df, df_clinical, valid_longitudinal, 
                feature_info, args.input_dir, args.output_dir, atlas_name, logger
            )

        logger.info("Main ROI-to-network FC analysis completed successfully for atlas: %s", atlas_name)
    
    except Exception as e:
        logger.error("Main execution failed: %s", e)
        raise
    
    finally:
        logger.info("=" * 80)
        logger.info("ROI-to-Network FC Group Analysis Completed")
        logger.info("=" * 80)

def print_quick_help():
    """Print quick help information."""
    quick_help = """
QUICK HELP - ROI-to-Network Functional Connectivity Group Analysis
=================================================================

BASIC USAGE:
  python NW_group.py --subjects_csv <GROUP_CSV> --clinical_csv <CLINICAL_CSV> --input_dir <FC_DATA_DIR>

QUICK EXAMPLES:
  1. Default Atlas (Power 2011):
     python NW_group.py \\
       --subjects_csv group.csv \\
       --clinical_csv clinical.csv \\
       --input_dir /path/to/fc/data

  2. Specific Atlas:
     python NW_group.py \\
       --subjects_csv group.csv \\
       --clinical_csv clinical.csv \\
       --input_dir /path/to/fc/data \\
       --atlas_name schaefer_2018_400_7_2

  3. Auto-detect Atlas:
     python NW_group.py \\
       --subjects_csv group.csv \\
       --clinical_csv clinical.csv \\
       --input_dir /path/to/fc/data \\
       --auto-detect-atlas

HELP OPTIONS:
  --help          Show full help with all arguments
  --usage         Show detailed usage examples

For more information, run with --usage or --help.
"""
    print(quick_help)

def print_examples():
    """Print detailed usage examples."""
    examples_text = """
DETAILED USAGE EXAMPLES
=======================

1. DEFAULT ATLAS (Power 2011)
   ---------------------------
   This is the simplest usage, using the default Power 2011 atlas.
   
   python NW_group.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data
   
   Expected FC files: *_task-rest_power_2011_network_fc_avg.csv

2. SPECIFIC ATLAS NAME
   --------------------
   Use when you know the exact atlas name from NW_1st.py output.
   
   python NW_group.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --atlas_name schaefer_2018_400_7_2
   
   Expected FC files: *_task-rest_schaefer_2018_400_7_2_network_fc_avg.csv

3. AUTO-DETECT ATLAS
   ------------------
   Automatically detect atlas name from available FC files.
   
   python NW_group.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --auto-detect-atlas
   
   This will scan the input directory and find the atlas name automatically.

4. WITH CUSTOM OUTPUT DIRECTORY
   -----------------------------
   Override the default output directory.
   
   python NW_group.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --output_dir /custom/output/path \\
     --auto-detect-atlas

5. VERBOSE LOGGING
   ----------------
   Enable detailed logging for debugging.
   
   python NW_group.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --verbose

REQUIRED FILES:
---------------
- group.csv: Contains subject IDs and group labels (HC/OCD)
- clinical.csv: Contains clinical data including YBOCS scores
- FC data files: Generated by NW_1st.py with naming pattern:
  *_{session}_task-rest_{atlas_name}_network_fc_avg.csv

OUTPUT FILES:
-------------
- group_diff_baseline_{atlas_name}_roi_network_fc.csv: Group difference t-test results
- baselineFC_vs_deltaYBOCS_{atlas_name}_roi_network_fc.csv: Baseline FC vs symptom change
- deltaFC_vs_deltaYBOCS_{atlas_name}_roi_network_fc.csv: FC change vs symptom change

ATLAS NAMING CONVENTIONS:
-------------------------
The script automatically detects atlas names from input FC files:
- Power 2011: power_2011_network_fc_avg.csv
- Schaefer 2018: schaefer_2018_400_7_2_network_fc_avg.csv
- Custom: custom_atlas_network_fc_avg.csv

TROUBLESHOOTING:
----------------
1. Check that FC data files exist in the input directory
2. Verify atlas naming matches between NW_1st.py and NW_group.py
3. Ensure group.csv and clinical.csv have correct subject IDs
4. Use --verbose for detailed logging
5. Use --auto-detect-atlas to automatically find the correct atlas name

For more information, see the script docstring or run with --help.
"""
    print(examples_text)

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
        print("\nFor help, run: python NW_group.py --help")
        print("For usage examples, run: python NW_group.py --usage")
        raise