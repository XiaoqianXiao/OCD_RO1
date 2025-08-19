#!/usr/bin/env python3
"""
Network-Level Pairwise Functional Connectivity Analysis using Customizable Atlas

This script performs group-level statistical analysis on network-level pairwise functional connectivity data.
It compares healthy controls (HC) vs. OCD patients and performs longitudinal analyses.

The script is compatible with both custom atlases and Nilearn built-in atlases (e.g., Schaefer 2018,
Harvard-Oxford, Power 2011, etc.) and automatically handles atlas detection and validation.

Author: [Your Name]
Date: [Current Date]

USAGE EXAMPLES:
==============

1. Power 2011 Atlas (Default):
   python NW_group2.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data

2. Schaefer 2018 Atlas (400 ROIs, 7 networks):
   python NW_group2.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --atlas_name schaefer_2018_400_7_2

3. Schaefer 2018 Atlas (1000 ROIs, 17 networks):
   python NW_group2.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --atlas_name schaefer_2018_1000_17_1

4. With Specific Atlas Name:
   python NW_group2.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --atlas_name power_2011

5. Auto-detect Atlas from Input Files:
   python NW_group2.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --auto-detect-atlas

ATLAS NAMING CONVENTIONS:
========================

The script automatically detects atlas names from input FC files:
- Power 2011: power_2011_network_fc_avg.csv
- Schaefer 2018: schaefer_2018_{n_rois}_{yeo_networks}_{resolution_mm}_network_fc_avg.csv
  Examples:
    - schaefer_2018_400_7_2_network_fc_avg.csv (400 ROIs, 7 networks, 2mm)
    - schaefer_2018_1000_17_1_network_fc_avg.csv (1000 ROIs, 17 networks, 1mm)
- Custom: custom_atlas_network_fc_avg.csv

Note: For Schaefer 2018, the naming follows the pattern: schaefer_2018_{n_rois}_{yeo_networks}_{resolution_mm}
where n_rois can be 100, 200, 300, 400, 500, 600, 700, 800, 900, or 1000,
yeo_networks can be 7 or 17, and resolution_mm can be 1 or 2.

OUTPUT FILES:
============

- group_diff_baseline_{atlas_name}_network_fc.csv: Group difference t-test results
- baselineFC_vs_deltaYBOCS_{atlas_name}_network_fc.csv: Baseline FC vs symptom change
- deltaFC_vs_deltaYBOCS_{atlas_name}_network_fc.csv: FC change vs symptom change

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
    'log_file': 'network_fc_analysis.log',
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
    logger = logging.getLogger('Network_FC_Analysis')
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
        description='Network-level pairwise functional connectivity analysis using customizable atlas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Power 2011 Atlas (Default)
  python NW_group2.py \\
    --subjects_csv group.csv \\
    --clinical_csv clinical.csv \\
    --input_dir /path/to/fc/data

  # Schaefer 2018 Atlas (400 ROIs, 7 networks)
  python NW_group2.py \\
    --subjects_csv group.csv \\
    --clinical_csv clinical.csv \\
    --input_dir /path/to/fc/data \\
    --atlas_name schaefer_2018_400_7_2

  # Schaefer 2018 Atlas (1000 ROIs, 17 networks)
  python NW_group2.py \\
    --subjects_csv group.csv \\
    --clinical_csv clinical.csv \\
    --input_dir /path/to/fc/data \\
    --atlas_name schaefer_2018_1000_17_1

  # With Specific Atlas Name
  python NW_group2.py \\
    --subjects_csv group.csv \\
    --clinical_csv clinical.csv \\
    --input_dir /path/to/fc/data \\
    --atlas_name power_2011

  # Auto-detect Atlas from Input Files
  python NW_group2.py \\
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
    """Get path to network FC CSV file."""
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
    feature_columns: List[str],
    logger: logging.Logger
) -> pd.DataFrame:
    """Run two-sample t-tests with FDR correction for network-level FC."""
    logger.info("Running t-tests for %d features", len(feature_columns))
    
    results = []
    dropped_features = []
    
    for col in feature_columns:
        hc_values = fc_data_hc[col].dropna()
        ocd_values = fc_data_ocd[col].dropna()
        
        logger.debug("Feature %s: HC n=%d, OCD n=%d", col, len(hc_values), len(ocd_values))
        
        if len(hc_values) < DEFAULT_CONFIG['min_subjects_per_group'] or \
           len(ocd_values) < DEFAULT_CONFIG['min_subjects_per_group']:
            logger.warning(
                "Skipping feature %s due to insufficient data (HC n=%d, OCD n=%d)",
                col, len(hc_values), len(ocd_values)
            )
            dropped_features.append((col, f"HC n={len(hc_values)}, OCD n={len(ocd_values)}"))
            continue
        
        # Perform t-test
        t_stat, p_val = stats.ttest_ind(ocd_values, hc_values, equal_var=False)
        
        results.append({
            'Feature': col,
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
    
    logger.info("Generated t-test results for %d features", len(results_df))
    
    return results_df

def run_regression(
    fc_data: pd.DataFrame, 
    y_values: pd.Series, 
    feature_columns: List[str],
    analysis_name: str,
    metadata_df: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """Run linear regression with condition confounder and FDR correction for network-level FC."""
    logger.info(
        "Running %s regression with condition confounder for %d subjects and %d network-level features",
        analysis_name, len(fc_data), len(feature_columns)
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

    # Run regression for each feature
    for col in feature_columns:
        x = fc_data[col].dropna()
        if x.empty:
            logger.warning(
                "Skipping feature %s in %s regression due to empty data",
                col, analysis_name
            )
            dropped_features.append((col, "empty data"))
            continue
        
        y = y_values.loc[x.index].dropna()
        logger.debug(
            "Feature %s in %s regression: n=%d", 
            col, analysis_name, len(y)
        )
        
        if len(x) < DEFAULT_CONFIG['min_subjects_per_group'] or \
           len(y) < DEFAULT_CONFIG['min_subjects_per_group']:
            logger.warning(
                "Skipping feature %s in %s regression due to insufficient data (n=%d)",
                col, analysis_name, len(y)
            )
            dropped_features.append((col, f"n={len(y)}"))
            continue
        
        # Get condition data for this feature
        feature_data = fc_data_with_condition[['subject_id', col, 'condition']].dropna()
        
        if len(feature_data) < DEFAULT_CONFIG['min_subjects_per_group']:
            logger.warning(
                "Skipping feature %s in %s regression due to insufficient data (n=%d)",
                col, analysis_name, len(feature_data)
            )
            dropped_features.append((col, f"n={len(feature_data)}"))
            continue
        
        try:
            # Perform multiple linear regression with condition as confounder
            # Create formula for regression with condition confounder
            # Sanitize feature name for formula (remove special characters)
            safe_feature = f"feature_{col.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}"
            
            # Prepare data for regression with sanitized column names
            regression_data = feature_data.copy()
            regression_data['y_values'] = y_values.loc[feature_data['subject_id']].values
            regression_data[safe_feature] = regression_data[col]
            
            # Create formula with sanitized names
            formula = f"{safe_feature} ~ y_values + condition"
            
            # Fit the model
            model = ols(formula, data=regression_data).fit()
            
            # Get FC effect (slope)
            fc_effect = model.params.get('y_values', 0)
            fc_pval = model.pvalues.get('y_values', 1.0)
            
            # Get condition effects
            condition_effects = {}
            condition_pvals = {}
            for cond in feature_data['condition'].unique():
                if cond != feature_data['condition'].iloc[0]:  # Reference condition
                    cond_param = f"condition[T.{cond}]"
                    condition_effects[cond] = model.params.get(cond_param, 0)
                    condition_pvals[cond] = model.pvalues.get(cond_param, 1.0)
            
            # Calculate correlation for backward compatibility
            x = feature_data[col].values
            y_vals = regression_data['y_values'].values
            r_value = np.corrcoef(x, y_vals)[0, 1] if len(x) > 1 else 0
            
            results.append({
                'Feature': col,
                'fc_effect': fc_effect,  # Effect of FC on outcome
                'fc_p_value': fc_pval,
                'r_value': r_value,
                'n': len(feature_data),
                'condition_effects': condition_effects,
                'condition_p_values': condition_pvals,
                'model_r_squared': model.rsquared,
                'model_adj_r_squared': model.rsquared_adj
            })
            
        except Exception as e:
            logger.warning(
                "Failed to run regression with condition confounder for feature %s: %s",
                col, e
            )
            # Fallback to simple regression
            x = feature_data[col].values.reshape(-1, 1)
            y_vals = y_values.loc[feature_data['subject_id']].values
            slope, intercept, r_value, p_val, _ = stats.linregress(x.flatten(), y_vals)
            
            results.append({
                'Feature': col,
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'p_value': p_val,
                'n': len(y_vals)
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
    
    # Handle different p-value fields based on analysis type
    if 'fc_p_value' in results_df.columns:
        # New format with condition confounder
        p_vals = results_df['fc_p_value'].values
        _, p_vals_corr = fdrcorrection(p_vals, alpha=DEFAULT_CONFIG['fdr_alpha'])
        results_df['fc_p_value_fdr'] = p_vals_corr
    else:
        # Fallback format
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
            logger.info("Condition distribution: %s", df['condition'].value_counts().to_dict())
        else:
            logger.warning("No 'condition' column found in subjects CSV. Adding default condition column.")
            df['condition'] = 'unknown'
            
    except Exception as e:
        logger.warning("Failed to load condition information: %s. Adding default condition column.", e)
        df['condition'] = 'unknown'

    return df, df_clinical

def validate_subjects(
    fc_dir: str, 
    metadata_df: pd.DataFrame, 
    atlas_name: str,
    logger: logging.Logger
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """Validate subjects based on available network FC files."""
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
    
    logger.info(
        "CSV subjects: %d, FC file subjects: %d, overlap: %d",
        len(csv_subjects), len(file_subjects), len(csv_subjects & file_subjects)
    )
    
    # Determine valid subjects for different analyses
    sessions = DEFAULT_CONFIG['sessions']
    
    # Subjects valid for group analysis (need baseline)
    valid_group = [
        sid for sid in metadata_df['subject_id'] 
        if sid.replace('sub-', '') in subject_sessions and 
        'ses-baseline' in subject_sessions[sid.replace('sub-', '')]
    ]
    
    # Subjects valid for longitudinal analysis (need both sessions)
    valid_longitudinal = [
        sid for sid in metadata_df['subject_id'] 
        if sid.replace('sub-', '') in subject_sessions and 
        all(ses in subject_sessions[sid.replace('sub-', '')] for ses in sessions)
    ]
    
    logger.info("Valid subjects for group analysis: %d", len(valid_group))
    logger.info("Valid subjects for longitudinal analysis: %d", len(valid_longitudinal))
    
    return valid_group, valid_longitudinal, subject_sessions

def load_network_fc_data(
    subject_ids: List[str], 
    session: str, 
    input_dir: str, 
    atlas_name: str,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, Optional[List[str]]]:
    """Load network-level pairwise FC data for given subjects and session."""
    logger.info("Loading FC data for %d subjects, session %s, atlas %s", len(subject_ids), session, atlas_name)
    
    fc_data = []
    feature_columns = None
    
    for sid in subject_ids:
        sid_no_prefix = sid.replace('sub-', '')
        fc_path = get_network_fc_path(sid_no_prefix, session, input_dir, atlas_name)
        
        if not os.path.exists(fc_path):
            logger.warning("FC file not found: %s", fc_path)
            continue
        
        try:
            fc_df = pd.read_csv(fc_path)
            logger.debug("Loaded FC file %s with %d rows", fc_path, len(fc_df))
            
            # Create unique feature identifier (sorted to avoid duplicates like A_B vs B_A)
            fc_df['feature_id'] = fc_df.apply(
                lambda x: '_'.join(sorted([x['network1'], x['network2']])) + '_fc', 
                axis=1
            )
            
            if feature_columns is None:
                feature_columns = fc_df['feature_id'].unique().tolist()
                logger.debug("Identified %d feature columns", len(feature_columns))
            
            # Pivot to make features as columns
            fc_pivot = fc_df.pivot_table(
                index=None,
                columns='feature_id',
                values='fc_value'
            ).reset_index(drop=True)
            fc_pivot['subject_id'] = sid_no_prefix
            fc_data.append(fc_pivot)
            
        except Exception as e:
            logger.error("Failed to process FC file %s: %s", fc_path, e)
            continue
    
    if not fc_data:
        logger.warning("No FC data loaded for session %s", session)
        return pd.DataFrame(), feature_columns
    
    fc_data_df = pd.concat(fc_data, ignore_index=True)
    logger.info("Loaded FC data for %d subjects, %d features", len(fc_data_df), len(feature_columns))
    
    return fc_data_df, feature_columns

# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def perform_group_analysis(
    baseline_fc_data: pd.DataFrame,
    metadata_df: pd.DataFrame,
    feature_columns: List[str],
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
    
    logger.info("Group analysis: HC n=%d, OCD n=%d", len(hc_data), len(ocd_data))
    
    if hc_data.empty or ocd_data.empty:
        logger.warning(
            "Insufficient data for group analysis (HC empty: %s, OCD empty: %s)",
            hc_data.empty, ocd_data.empty
        )
        return False
    
    # Run t-tests
    ttest_results = run_ttest(hc_data, ocd_data, feature_columns, logger)
    
    if not ttest_results.empty:
        output_path = os.path.join(output_dir, f'group_diff_baseline_{atlas_name}_network_fc.csv')
        ttest_results.to_csv(output_path, index=False)
        logger.info("Saved t-test results to %s", output_path)
        return True
    else:
        logger.info("No significant t-test results to save")
        return False

def perform_longitudinal_analysis(
    baseline_fc_data: pd.DataFrame,
    metadata_df: pd.DataFrame,
    df_clinical: pd.DataFrame,
    valid_longitudinal: List[str],
    feature_columns: List[str],
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
    
    logger.info("Longitudinal analysis: OCD subjects n=%d", len(ocd_df))

    # 1. Baseline FC vs symptom change
    baseline_fc_ocd = baseline_fc_data[
        baseline_fc_data['subject_id'].isin(ocd_df['subject_id'])
    ]
    
    logger.info("Baseline FC for OCD: n=%d", len(baseline_fc_ocd))
    
    if not baseline_fc_ocd.empty:
        regression_results = run_regression(
            baseline_fc_ocd.set_index('subject_id'),
            ocd_df.set_index('subject_id')['delta_ybocs'],
            feature_columns,
            "baseline FC vs delta YBOCS",
            metadata_df,
            logger
        )
        
        if not regression_results.empty:
            output_path = os.path.join(output_dir, f'baselineFC_vs_deltaYBOCS_{atlas_name}_network_fc.csv')
            regression_results.to_csv(output_path, index=False)
            logger.info("Saved baseline FC regression results to %s", output_path)
        else:
            logger.info("No significant baseline FC regression results to save")
    else:
        logger.warning("No baseline FC data for OCD subjects in longitudinal analysis")

    # 2. FC change vs symptom change
    logger.info("Analyzing FC change vs symptom change")
    fc_change_data = []
    
    for sid in valid_longitudinal:
        sid_clean = sid.replace('sub-', '')
        base_path = get_network_fc_path(sid_clean, 'ses-baseline', input_dir, atlas_name)
        follow_path = get_network_fc_path(sid_clean, 'ses-followup', input_dir, atlas_name)
        
        if not (os.path.exists(base_path) and os.path.exists(follow_path)):
            logger.warning(
                "Missing FC files for subject %s (baseline: %s, followup: %s)",
                sid, os.path.exists(base_path), os.path.exists(follow_path)
            )
            continue
        
        try:
            # Load baseline and followup data
            base_fc = pd.read_csv(base_path)
            follow_fc = pd.read_csv(follow_path)
            
            logger.debug(
                "Loaded baseline FC (%d rows) and followup FC (%d rows) for %s",
                len(base_fc), len(follow_fc), sid
            )
            
            # Create feature identifiers
            base_fc['feature_id'] = base_fc.apply(
                lambda x: '_'.join(sorted([x['network1'], x['network2']])) + '_fc', 
                axis=1
            )
            follow_fc['feature_id'] = follow_fc.apply(
                lambda x: '_'.join(sorted([x['network1'], x['network2']])) + '_fc', 
                axis=1
            )
            
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
            logger.error("Failed to process longitudinal FC for subject %s: %s", sid, e)
            continue

    if fc_change_data:
        fc_change_data = pd.concat(fc_change_data, ignore_index=True)
        fc_change_data['subject_id'] = fc_change_data['subject_id'].astype(str)
        feature_columns = [col for col in fc_change_data.columns if col != 'subject_id']
        
        logger.info(
            "Loaded FC change data for %d subjects, %d features",
            len(fc_change_data), len(feature_columns)
        )
        
        # Run regression analysis
        regression_results = run_regression(
            fc_change_data.set_index('subject_id'),
            ocd_df.set_index('subject_id')['delta_ybocs'],
            feature_columns,
            "delta FC vs delta YBOCS",
            metadata_df,
            logger
        )
        
        if not regression_results.empty:
            output_path = os.path.join(output_dir, f'deltaFC_vs_deltaYBOCS_{atlas_name}_network_fc.csv')
            regression_results.to_csv(output_path, index=False)
            logger.info("Saved FC change regression results to %s", output_path)
        else:
            logger.info("No significant FC change regression results to save")
    else:
        logger.warning("No FC change data loaded for longitudinal analysis")

# =============================================================================
# CONDITION-BASED ANALYSIS FUNCTIONS
# =============================================================================

def perform_condition_analysis(
    baseline_fc_data: pd.DataFrame,
    metadata_df: pd.DataFrame,
    valid_longitudinal: List[str],
    input_dir: str,
    output_dir: str,
    atlas_name: str,
    logger: logging.Logger
) -> None:
    """Perform condition-based analysis for OCD subjects only."""
    logger.info("Performing condition-based analysis for atlas: %s", atlas_name)
    
    # Filter for OCD subjects only
    ocd_baseline_fc = baseline_fc_data[
        baseline_fc_data['subject_id'].isin(metadata_df[metadata_df['group'] == 'OCD']['subject_id'])
    ]
    
    if ocd_baseline_fc.empty:
        logger.warning("No baseline FC data for OCD subjects in condition analysis")
        return
    
    # Get unique conditions (excluding 'unknown')
    conditions = metadata_df[metadata_df['group'] == 'OCD']['condition'].unique()
    conditions = [c for c in conditions if c != 'unknown']
    
    if len(conditions) < 2:
        logger.warning("Need at least 2 conditions for comparison. Found: %s", conditions)
        return
    
    logger.info("Analyzing conditions: %s", conditions)
    
    # 1. Baseline FC differences between conditions
    logger.info("1. Analyzing baseline FC differences between conditions")
    baseline_condition_results = run_condition_ttest(
        ocd_baseline_fc, conditions, metadata_df, "baseline FC", logger
    )
    
    if not baseline_condition_results.empty:
        output_path = os.path.join(output_dir, f'condition_baseline_FC_{atlas_name}_network_fc.csv')
        baseline_condition_results.to_csv(output_path, index=False)
        logger.info("Saved baseline condition analysis to %s", output_path)
    
    # 2. Followup FC differences between conditions
    logger.info("2. Analyzing followup FC differences between conditions")
    followup_condition_results = analyze_followup_by_condition(
        valid_longitudinal, input_dir, atlas_name, conditions, metadata_df, logger
    )
    
    if not followup_condition_results.empty:
        output_path = os.path.join(output_dir, f'condition_followup_FC_{atlas_name}_network_fc.csv')
        followup_condition_results.to_csv(output_path, index=False)
        logger.info("Saved followup condition analysis to %s", output_path)
    
    # 3. FC change differences between conditions
    logger.info("3. Analyzing FC change differences between conditions")
    change_condition_results = analyze_fc_change_by_condition(
        valid_longitudinal, input_dir, atlas_name, conditions, metadata_df, logger
    )
    
    if not change_condition_results.empty:
        output_path = os.path.join(output_dir, f'condition_FC_change_{atlas_name}_network_fc.csv')
        change_condition_results.to_csv(output_path, index=False)
        logger.info("Saved FC change condition analysis to %s", output_path)
    
    logger.info("Condition-based analysis completed for atlas: %s", atlas_name)

def run_condition_ttest(
    fc_data: pd.DataFrame,
    conditions: List[str],
    metadata_df: pd.DataFrame,
    analysis_type: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """Run t-tests between conditions for a given FC dataset."""
    logger.info("Running condition t-tests for %s", analysis_type)
    
    results = []
    reference_condition = conditions[0]
    
    for col in fc_data.columns:
        if col == 'subject_id':
            continue
            
        # Get data for each condition
        condition_data = {}
        for cond in conditions:
            cond_subjects = metadata_df[
                (metadata_df['group'] == 'OCD') & 
                (metadata_df['condition'] == cond)
            ]['subject_id']
            cond_fc = fc_data[fc_data['subject_id'].isin(cond_subjects)][col].dropna()
            condition_data[cond] = cond_fc
        
        # Skip if insufficient data
        if any(len(data) < DEFAULT_CONFIG['min_subjects_per_group'] for data in condition_data.values()):
            continue
        
        # Perform t-tests between reference condition and others
        for cond in conditions[1:]:
            try:
                t_stat, p_val = stats.ttest_ind(
                    condition_data[cond], 
                    condition_data[reference_condition], 
                    equal_var=False
                )
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(
                    ((len(condition_data[cond]) - 1) * condition_data[cond].var() + 
                     (len(condition_data[reference_condition]) - 1) * condition_data[reference_condition].var()) /
                    (len(condition_data[cond]) + len(condition_data[reference_condition]) - 2)
                )
                effect_size = (condition_data[cond].mean() - condition_data[reference_condition].mean()) / pooled_std
                
                results.append({
                    'Feature': col,
                    'condition1': reference_condition,
                    'condition2': cond,
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'condition1_mean': condition_data[reference_condition].mean(),
                    'condition2_mean': condition_data[cond].mean(),
                    'condition1_n': len(condition_data[reference_condition]),
                    'condition2_n': len(condition_data[cond]),
                    'effect_size': effect_size
                })
                
            except Exception as e:
                logger.warning("Failed to run t-test for feature %s, conditions %s vs %s: %s", 
                             col, reference_condition, cond, e)
                continue
    
    if not results:
        logger.warning("No condition t-test results generated")
        return pd.DataFrame()
    
    # Create results DataFrame and apply FDR correction
    results_df = pd.DataFrame(results)
    p_vals = results_df['p_value'].values
    _, p_vals_corr = fdrcorrection(p_vals, alpha=DEFAULT_CONFIG['fdr_alpha'])
    results_df['p_value_fdr'] = p_vals_corr
    
    logger.info("Generated condition t-test results for %d comparisons", len(results_df))
    return results_df

def analyze_followup_by_condition(
    valid_longitudinal: List[str],
    input_dir: str,
    atlas_name: str,
    conditions: List[str],
    metadata_df: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """Load followup FC data for OCD subjects and analyze by condition."""
    logger.info("Analyzing followup FC by condition")
    
    fc_data = []
    for sid in valid_longitudinal:
        sid_clean = sid.replace('sub-', '')
        fc_path = get_network_fc_path(sid_clean, 'ses-followup', input_dir, atlas_name)
        
        if not os.path.exists(fc_path):
            continue
            
        try:
            fc_df = pd.read_csv(fc_path)
            fc_df['feature_id'] = fc_df.apply(
                lambda x: '_'.join(sorted([x['network1'], x['network2']])) + '_fc', 
                axis=1
            )
            
            fc_pivot = fc_df.pivot_table(
                index=None, columns='feature_id', values='fc_value'
            ).reset_index(drop=True)
            fc_pivot['subject_id'] = sid_clean
            fc_data.append(fc_pivot)
            
        except Exception as e:
            logger.error("Failed to process followup FC for subject %s: %s", sid, e)
            continue
    
    if not fc_data:
        logger.warning("No followup FC data loaded for condition analysis")
        return pd.DataFrame()
    
    fc_data_df = pd.concat(fc_data, ignore_index=True)
    return run_condition_ttest(fc_data_df, conditions, metadata_df, "followup FC", logger)

def analyze_fc_change_by_condition(
    valid_longitudinal: List[str],
    input_dir: str,
    atlas_name: str,
    conditions: List[str],
    metadata_df: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """Load baseline and followup FC data, compute change, and analyze by condition."""
    logger.info("Analyzing FC change by condition")
    
    fc_change_data = []
    for sid in valid_longitudinal:
        sid_clean = sid.replace('sub-', '')
        base_path = get_network_fc_path(sid_clean, 'ses-baseline', input_dir, atlas_name)
        follow_path = get_network_fc_path(sid_clean, 'ses-followup', input_dir, atlas_name)
        
        if not (os.path.exists(base_path) and os.path.exists(follow_path)):
            continue
            
        try:
            base_fc = pd.read_csv(base_path)
            follow_fc = pd.read_csv(follow_path)
            
            base_fc['feature_id'] = base_fc.apply(
                lambda x: '_'.join(sorted([x['network1'], x['network2']])) + '_fc', 
                axis=1
            )
            follow_fc['feature_id'] = follow_fc.apply(
                lambda x: '_'.join(sorted([x['network1'], x['network2']])) + '_fc', 
                axis=1
            )
            
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
            logger.error("Failed to process FC change for subject %s: %s", sid, e)
            continue
    
    if not fc_change_data:
        logger.warning("No FC change data loaded for condition analysis")
        return pd.DataFrame()
    
    fc_change_df = pd.concat(fc_change_data, ignore_index=True)
    return run_condition_ttest(fc_change_df, conditions, metadata_df, "FC change", logger)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run network-level FC analysis."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.output_dir, DEFAULT_CONFIG['log_file'])
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("=" * 80)
    logger.info("Starting Network-Level FC Analysis")
    logger.info("=" * 80)
    logger.info("Arguments: %s", vars(args))
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load metadata
        metadata_df, df_clinical = load_and_validate_metadata(args.subjects_csv, args.clinical_csv, logger)

        # Normalize subject IDs
        metadata_df['subject_id'] = metadata_df['subject_id'].str.replace('sub-', '')
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
            args.input_dir, metadata_df, atlas_name, logger
        )
        
        if not valid_group and not valid_longitudinal:
            logger.error("No valid subjects found for any analysis")
            raise ValueError("No valid subjects found for any analysis.")

        # Load baseline network FC data
        baseline_fc_data, feature_columns = load_network_fc_data(
            valid_group, 'ses-baseline', args.input_dir, atlas_name, logger
        )
        
        if baseline_fc_data.empty:
            logger.warning("No baseline FC data loaded. Exiting.")
            return

        # 1. Group difference at baseline
        if valid_group:
            perform_group_analysis(baseline_fc_data, metadata_df, feature_columns, args.output_dir, atlas_name, logger)

        # 2. Longitudinal analyses
        if valid_longitudinal:
            perform_longitudinal_analysis(
                baseline_fc_data, metadata_df, df_clinical, valid_longitudinal, 
                feature_columns, args.input_dir, args.output_dir, atlas_name, logger
            )

        # 3. Condition-based analysis (Test 4)
        if valid_group:
            perform_condition_analysis(
                baseline_fc_data, metadata_df, valid_longitudinal, 
                args.input_dir, args.output_dir, atlas_name, logger
            )

        logger.info("Main analysis completed successfully for atlas: %s", atlas_name)
        logger.info("Analyses completed: Group comparison, Longitudinal analysis, Condition-based analysis")
    
    except Exception as e:
        logger.error("Main execution failed: %s", e)
        raise
    
    finally:
        logger.info("=" * 80)
        logger.info("Network-Level FC Analysis Completed")
        logger.info("=" * 80)

def print_quick_help():
    """Print quick help information."""
    quick_help = """
QUICK HELP - Network-Level Pairwise Functional Connectivity Analysis
===================================================================

BASIC USAGE:
  python NW_group2.py --subjects_csv <GROUP_CSV> --clinical_csv <CLINICAL_CSV> --input_dir <FC_DATA_DIR>

QUICK EXAMPLES:
  1. Default Atlas (Power 2011):
     python NW_group2.py \\
       --subjects_csv group.csv \\
       --clinical_csv clinical.csv \\
       --input_dir /path/to/fc/data

  2. Specific Atlas:
     python NW_group2.py \\
       --subjects_csv group.csv \\
       --clinical_csv clinical.csv \\
       --input_dir /path/to/fc/data \\
       --atlas_name schaefer_2018_400_7_2

  3. Auto-detect Atlas:
     python NW_group2.py \\
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
   
   python NW_group2.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data
   
   Expected FC files: *_task-rest_power_2011_network_fc_avg.csv

2. SPECIFIC ATLAS NAME
   --------------------
   Use when you know the exact atlas name from NW_1st.py output.
   
   python NW_group2.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --atlas_name schaefer_2018_400_7_2
   
   Expected FC files: *_task-rest_schaefer_2018_400_7_2_network_fc_avg.csv

3. AUTO-DETECT ATLAS
   ------------------
   Automatically detect atlas name from available FC files.
   
   python NW_group2.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --auto-detect-atlas
   
   This will scan the input directory and find the atlas name automatically.

4. WITH CUSTOM OUTPUT DIRECTORY
   -----------------------------
   Override the default output directory.
   
   python NW_group2.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --output_dir /custom/output/path \\
     --auto-detect-atlas

5. VERBOSE LOGGING
   ----------------
   Enable detailed logging for debugging.
   
   python NW_group2.py \\
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
- group_diff_baseline_{atlas_name}_network_fc.csv: Group difference t-test results
- baselineFC_vs_deltaYBOCS_{atlas_name}_network_fc.csv: Baseline FC vs symptom change
- deltaFC_vs_deltaYBOCS_{atlas_name}_network_fc.csv: FC change vs symptom change
- condition_baseline_FC_{atlas_name}_network_fc.csv: Condition differences in baseline FC
- condition_followup_FC_{atlas_name}_network_fc.csv: Condition differences in followup FC
- condition_FC_change_{atlas_name}_network_fc.csv: Condition differences in FC change

ATLAS NAMING CONVENTIONS:
-------------------------
The script automatically detects atlas names from input FC files:
- Power 2011: power_2011_network_fc_avg.csv
- Schaefer 2018: schaefer_2018_{n_rois}_{yeo_networks}_{resolution_mm}_network_fc_avg.csv
  Examples:
    - schaefer_2018_400_7_2_network_fc_avg.csv (400 ROIs, 7 networks, 2mm)
    - schaefer_2018_1000_17_1_network_fc_avg.csv (1000 ROIs, 17 networks, 1mm)
- Custom: custom_atlas_network_fc_avg.csv

Note: For Schaefer 2018, the naming follows the pattern: schaefer_2018_{n_rois}_{yeo_networks}_{resolution_mm}
where n_rois can be 100, 200, 300, 400, 500, 600, 700, 800, 900, or 1000,
yeo_networks can be 7 or 17, and resolution_mm can be 1 or 2.

DIFFERENCE FROM NW_group.py:
----------------------------
This script analyzes network-level pairwise functional connectivity (network-to-network),
while NW_group.py analyzes ROI-to-network connectivity. Both scripts work with the same
input data but provide different levels of network analysis.

TROUBLESHOOTING:
----------------
1. Check that FC data files exist in the input directory
2. Verify atlas naming matches between NW_1st.py and NW_group2.py
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
        print("\nFor help, run: python NW_group2.py --help")
        print("For usage examples, run: python NW_group2.py --usage")
        raise