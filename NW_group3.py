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
- Custom: custom_atlas_roiroi_fc_avg.csv (if network-based) or custom_atlas_roiroi_fc_avg.csv (if anatomical)

Note: For Schaefer 2018, the naming follows the pattern: schaefer_2018_{n_rois}_{yeo_networks}_{resolution_mm}
where n_rois can be 100, 200, 300, 400, 500, 600, 700, 800, 900, or 1000,
yeo_networks can be 7 or 17, and resolution_mm can be 1 or 2.
For YEO 2011, the naming follows: yeo_2011_{n_networks}_{thickness}
where n_networks can be 7 or 17, and thickness can be 'thick' or 'thin'.

IMPORTANT: This script (NW_group3.py) can handle BOTH file types:
- All atlases (Power 2011, Schaefer 2018, YEO 2011) generate *_roiroi_fc_avg.csv files
- Anatomical atlases (Harvard-Oxford, AAL, Talairach) generate *_roiroi_fc_avg.csv files
Both file types are automatically detected and processed appropriately.

OUTPUT FILES:
============

- group_diff_baseline_{atlas_name}_roiroi_fc.csv: Group difference t-test results
- baselineFC_vs_deltaYBOCS_{atlas_name}_roiroi_fc.csv: Baseline FC vs symptom change
- deltaFC_vs_deltaYBOCS_{atlas_name}_roiroi_fc.csv: FC change vs symptom change
- condition_baseline_FC_{atlas_name}_roiroi_fc.csv: Condition differences in baseline FC
- condition_followup_FC_{atlas_name}_roiroi_fc.csv: Condition differences in followup FC
- condition_FC_change_{atlas_name}_roiroi_fc.csv: Condition differences in FC change

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
import gc
# Memory monitoring disabled for compatibility
import warnings
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Memory monitoring function removed for compatibility

def log_dataframe_info(logger: logging.Logger, df: pd.DataFrame, name: str):
    """Log DataFrame information for debugging."""
    try:
        memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"DataFrame [{name}]: {df.shape[0]} rows × {df.shape[1]} columns, {memory_usage:.1f} MB")
    except Exception as e:
        logger.warning(f"Could not get DataFrame info for {name}: {e}")

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
        '*_task-rest_*_roiroi_fc_avg.csv',   # ROI-to-ROI atlases
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
    
    # Check for FC files with this atlas name (roiroi files)
    fc_patterns = [
        os.path.join(input_dir, f'*_task-rest_{atlas_name}_roiroi_fc_avg.csv')
    ]
    
    fc_files = []
    for pattern in fc_patterns:
        fc_files.extend(glob.glob(pattern))
    
    if not fc_files:
        logger.error("No FC files found for atlas '%s' in directory: %s", atlas_name, input_dir)
        logger.error("Expected patterns: *_task-rest_%s_roiroi_fc_avg.csv", atlas_name)
        return False
    
    logger.info("Found %d FC files for atlas '%s': %s", len(fc_files), atlas_name, 
                [os.path.basename(f) for f in fc_files[:5]])  # Show first 5 files
    
    return True

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_roiroi_fc_path(subject: str, session: str, input_dir: str, atlas_name: str) -> str:
    """Get path to ROI-to-ROI FC CSV file."""
    if not subject.startswith('sub-'):
        subject = f"sub-{subject}"
    
    # ROI-to-ROI file path
    roiroi_path = os.path.join(
        input_dir, 
        f"{subject}_{session}_task-rest_{atlas_name}_roiroi_fc_avg.csv"
    )
    
    return roiroi_path

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
    """Run two-sample t-tests with FDR correction for ROI-to-ROI FC."""
    logger.info(
        "Running t-tests with %d HC subjects and %d OCD subjects for %d ROI-to-ROI features",
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
            'OCD_std': np.std(ocd_values),
            'HC_std': np.std(hc_values),
            'OCD_n': len(ocd_values),
            'HC_n': len(hc_values),
            'cohens_d': (np.mean(ocd_values) - np.mean(hc_values)) / np.sqrt((np.var(ocd_values) + np.var(hc_values)) / 2)
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
    results_df['p_corrected'] = p_vals_corr
    results_df['significant'] = p_vals_corr < DEFAULT_CONFIG['fdr_alpha']
    
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
    metadata_df: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """Run linear regression with condition confounder and FDR correction for ROI-to-ROI FC."""
    logger.info(
        "Running %s regression with condition confounder for %d subjects and %d ROI-to-ROI features",
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
        
        # Get condition data for this feature
        feature_data = fc_data_with_condition[['subject_id', feature, 'condition']].dropna()
        
        if len(feature_data) < DEFAULT_CONFIG['min_subjects_per_group']:
            logger.warning(
                "Skipping feature %s (%s_%s) in %s regression due to insufficient data (n=%d)",
                feature, net1, net2, analysis_name, len(feature_data)
            )
            dropped_features.append((feature, f"n={len(feature_data)}"))
            continue
        
        try:
            # Perform multiple linear regression with condition as confounder
            # Create formula for regression with condition confounder
            # Sanitize feature name for formula (remove special characters)
            safe_feature = f"feature_{feature.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}"
            
            # Prepare data for regression with sanitized column names
            regression_data = feature_data.copy()
            regression_data['y_values'] = y_values.loc[feature_data['subject_id']].values
            regression_data[safe_feature] = regression_data[feature]
            
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
            x = feature_data[feature].values
            y_vals = regression_data['y_values'].values
            r_value = np.corrcoef(x, y_vals)[0, 1] if len(x) > 1 else 0
            
            results.append({
                'ROI': feature,
                'network1': net1,
                'network2': net2,
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
                "Failed to run regression with condition confounder for feature %s (%s_%s): %s",
                feature, net1, net2, e
            )
            # Fallback to simple regression
            x = feature_data[feature].values.reshape(-1, 1)
            y_vals = y_values.loc[feature_data['subject_id']].values
            slope, intercept, r_value, p_val, _ = stats.linregress(x.flatten(), y_vals)
            
            results.append({
                'ROI': feature,
                'network1': net1,
                'network2': net2,
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
    """Validate subjects based on available ROI-to-ROI FC files."""
    logger.info("Validating subjects in FC directory %s with atlas %s", fc_dir, atlas_name)
    
    if not os.path.exists(fc_dir):
        logger.error("Input directory %s does not exist", fc_dir)
        raise ValueError(f"Input directory {fc_dir} does not exist")
    
    # Find FC files with the specific atlas (roiroi files)
    fc_patterns = [
        os.path.join(fc_dir, f'*_task-rest_{atlas_name}_roiroi_fc_avg.csv')
    ]
    
    fc_files = []
    for pattern in fc_patterns:
        fc_files.extend(glob.glob(pattern))
    
    logger.info("Found %d FC files for atlas %s", len(fc_files), atlas_name)
    
    if not fc_files:
        # Try to find any FC files to help with debugging
        all_fc_patterns = [
            os.path.join(fc_dir, '*_task-rest_*_roiroi_fc_avg.csv')
        ]
        all_fc_files = []
        for pattern in all_fc_patterns:
            all_fc_files.extend(glob.glob(pattern))
        if all_fc_files:
            detected_atlases = set()
            for f in all_fc_files:
                filename = os.path.basename(f)
                match = re.search(r'_task-rest_(.+?)_roiroi_fc_avg', filename)
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
        if '_ses-' not in filename or f'_task-rest_{atlas_name}_roiroi_fc_avg.csv' not in filename:
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

def validate_roiroi_fc_file(fc_path: str, logger: logging.Logger) -> bool:
    """Validate that ROI-to-ROI FC file has required columns."""
    required_columns = {'ROI', 'FC'}
    
    try:
        df = pd.read_csv(fc_path)
        missing_columns = required_columns - set(df.columns)
        
        if missing_columns:
            logger.error(
                "ROI-to-ROI FC file %s missing required columns: %s. Found columns: %s",
                fc_path, missing_columns, list(df.columns)
            )
            return False
        
        logger.debug("ROI-to-ROI FC file %s validated successfully", fc_path)
        return True
        
    except Exception as e:
        logger.error("Failed to validate ROI-to-ROI FC file %s: %s", fc_path, e)
        return False

def load_roiroi_fc_data(
    subject_ids: List[str], 
    session: str, 
    input_dir: str, 
    atlas_name: str,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, Optional[Dict[str, Tuple[str, str]]]]:
    """Load ROI-to-ROI FC data for given subjects and session."""
    logger.info(
        "Loading ROI-to-ROI FC data for %d subjects, session %s, atlas %s: %s",
        len(subject_ids), session, atlas_name, subject_ids
    )
    
    # Memory monitoring disabled
    
    fc_data = []
    feature_info = None
    valid_subjects = []
    dropped_subjects = []
    
    for sid in subject_ids:
        sid_no_prefix = sid.replace('sub-', '')
        fc_path = get_roiroi_fc_path(sid_no_prefix, session, input_dir, atlas_name)
        
        if not os.path.exists(fc_path):
            logger.warning("ROI-to-ROI FC file not found for subject %s: %s", sid, fc_path)
            dropped_subjects.append((sid, f"missing FC file: {fc_path}"))
            continue
        
        if not validate_roiroi_fc_file(fc_path, logger):
            dropped_subjects.append((sid, f"invalid FC file format: {fc_path}"))
            continue
        
        try:
            logger.debug("Loading FC file: %s", fc_path)
            fc_df = pd.read_csv(fc_path)
            logger.debug("Loaded ROI-to-ROI FC file %s with %d rows", fc_path, len(fc_df))
            log_dataframe_info(logger, fc_df, f"raw_data_{sid}")
            
            # Create feature identifier and map networks (same as NW_group3.py)
            fc_df['feature_id'] = fc_df['ROI']
            if feature_info is None:
                feature_info = {}
                for _, row in fc_df.iterrows():
                    roi_pair = row['ROI']
                    # Use actual network information from the data
                    network1 = row.get('network1', 'Unknown')
                    network2 = row.get('network2', 'Unknown')
                    feature_info[roi_pair] = (network1, network2)
                logger.debug(
                    "Identified %d ROI-to-ROI feature columns with network mappings", 
                    len(feature_info)
                )
            
            # Pivot to make features as columns (same as NW_group3.py)
            logger.debug("Starting pivot_table operation for subject %s", sid)
            # Memory monitoring disabled
            
            fc_pivot = fc_df.pivot_table(
                index=None,
                columns='feature_id',
                values='FC'
            ).reset_index(drop=True)
            
            # Memory monitoring disabled
            log_dataframe_info(logger, fc_pivot, f"pivoted_data_{sid}")
            
            fc_pivot['subject_id'] = sid_no_prefix
            fc_data.append(fc_pivot)
            valid_subjects.append(sid)
            
            # Log progress every 5 subjects and concatenate in batches
            if len(valid_subjects) % 5 == 0:
                logger.info("Processed %d/%d subjects", len(valid_subjects), len(subject_ids))
                
                # Concatenate in batches to prevent memory accumulation
                if len(fc_data) >= 5:
                    logger.info("Concatenating batch of %d DataFrames", len(fc_data))
                    try:
                        batch_df = pd.concat(fc_data, ignore_index=True)
                        fc_data = [batch_df]  # Replace list with single concatenated DataFrame
                        logger.info("Batch concatenated successfully")
                        gc.collect()  # Force garbage collection
                    except Exception as e:
                        logger.error("Failed to concatenate batch: %s", e)
                        raise
            
        except Exception as e:
            logger.error(
                "Failed to process ROI-to-ROI FC file %s for subject %s: %s", 
                fc_path, sid, e
            )
            # Memory monitoring disabled
            dropped_subjects.append((sid, f"processing error: {e}"))
            continue
    
    if dropped_subjects:
        logger.info(
            "Dropped %d subjects from %s FC data loading: %s", 
            len(dropped_subjects), session, dropped_subjects
        )
    
    if not fc_data:
        logger.warning("No valid ROI-to-ROI FC data loaded for session %s", session)
        return pd.DataFrame(), feature_info
    
    # Memory monitoring disabled
    logger.info("About to concatenate %d DataFrames", len(fc_data))
    
    try:
        fc_data_df = pd.concat(fc_data, ignore_index=True)
        # Memory monitoring disabled
        log_dataframe_info(logger, fc_data_df, "final_combined_data")
        
        logger.info(
            "Loaded ROI-to-ROI FC data for %d subjects, %d ROI pairs: %s",
            len(valid_subjects), len(feature_info), valid_subjects
        )
    except Exception as e:
        logger.error("Failed to concatenate DataFrames: %s", e)
        # Memory monitoring disabled
        raise
    
    return fc_data_df, feature_info

# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def perform_condition_analysis(
    baseline_fc_data: pd.DataFrame,
    metadata_df: pd.DataFrame,
    feature_info: Dict[str, Tuple[str, str]],
    input_dir: str,
    output_dir: str,
    atlas_name: str,
    logger: logging.Logger
) -> None:
    """Perform condition-based analysis for OCD subjects only."""
    logger.info("Performing condition-based analysis for atlas: %s", atlas_name)
    
    # Filter for OCD subjects only
    ocd_subjects = metadata_df[metadata_df['group'] == 'OCD']['subject_id'].tolist()
    ocd_baseline_fc = baseline_fc_data[
        baseline_fc_data['subject_id'].isin(ocd_subjects)
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
        ocd_baseline_fc, metadata_df, feature_info, "baseline FC", logger
    )
    
    if not baseline_condition_results.empty:
        output_path = os.path.join(output_dir, f'condition_baseline_FC_{atlas_name}_roiroi_fc.csv')
        baseline_condition_results.to_csv(output_path, index=False)
        logger.info("Saved baseline condition analysis to %s", output_path)
    
    # 2. Followup FC differences between conditions
    logger.info("2. Analyzing followup FC differences between conditions")
    followup_condition_results = analyze_followup_by_condition(
        ocd_subjects, metadata_df, feature_info, input_dir, atlas_name, logger
    )
    
    if not followup_condition_results.empty:
        output_path = os.path.join(output_dir, f'condition_followup_FC_{atlas_name}_roiroi_fc.csv')
        followup_condition_results.to_csv(output_path, index=False)
        logger.info("Saved followup condition analysis to %s", output_path)
    
    # 3. FC change differences between conditions
    logger.info("3. Analyzing FC change differences between conditions")
    change_condition_results = analyze_fc_change_by_condition(
        ocd_subjects, metadata_df, feature_info, input_dir, atlas_name, logger
    )
    
    if not change_condition_results.empty:
        output_path = os.path.join(output_dir, f'condition_FC_change_{atlas_name}_roiroi_fc.csv')
        change_condition_results.to_csv(output_path, index=False)
        logger.info("Saved FC change condition analysis to %s", output_path)
    
    logger.info("Condition-based analysis completed for atlas: %s", atlas_name)

def run_condition_ttest(
    fc_data: pd.DataFrame,
    metadata_df: pd.DataFrame,
    feature_info: Dict[str, Tuple[str, str]],
    analysis_name: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """Run t-tests between conditions for a given FC dataset."""
    logger.info("Running condition t-tests for %s", analysis_name)
    
    # Get unique conditions (excluding 'unknown')
    conditions = metadata_df[metadata_df['group'] == 'OCD']['condition'].unique()
    conditions = [c for c in conditions if c != 'unknown']
    
    if len(conditions) < 2:
        logger.warning("Need at least 2 conditions for comparison. Found: %s", conditions)
        return pd.DataFrame()
    
    # Use first condition as reference
    ref_condition = conditions[0]
    other_conditions = conditions[1:]
    
    results = []
    
    for feature, (net1, net2) in feature_info.items():
        # Get data for each condition
        condition_data = {}
        for cond in conditions:
            cond_subjects = metadata_df[
                (metadata_df['group'] == 'OCD') & (metadata_df['condition'] == cond)
            ]['subject_id'].tolist()
            cond_fc = fc_data[fc_data['subject_id'].isin(cond_subjects)][feature].dropna()
            condition_data[cond] = cond_fc
        
        # Check if we have enough data for each condition
        if any(len(data) < DEFAULT_CONFIG['min_subjects_per_group'] for data in condition_data.values()):
            continue
        
        # Run t-tests comparing each condition to reference
        for cond in other_conditions:
            ref_values = condition_data[ref_condition]
            cond_values = condition_data[cond]
            
            # Perform t-test
            t_stat, p_val = stats.ttest_ind(cond_values, ref_values, equal_var=False)
            
            # Calculate means
            ref_mean = np.mean(ref_values)
            cond_mean = np.mean(cond_values)
            
            results.append({
                'ROI': feature,
                'network1': net1,
                'network2': net2,
                'reference_condition': ref_condition,
                'comparison_condition': cond,
                't_statistic': t_stat,
                'p_value': p_val,
                'reference_mean': ref_mean,
                'comparison_mean': cond_mean,
                'reference_n': len(ref_values),
                'comparison_n': len(cond_values),
                'effect_size': (cond_mean - ref_mean) / np.sqrt((np.var(ref_values) + np.var(cond_values)) / 2)
            })
    
    if not results:
        logger.info("No condition t-test results generated")
        return pd.DataFrame()
    
    # Create results DataFrame and apply FDR correction
    results_df = pd.DataFrame(results)
    p_vals = results_df['p_value'].values
    _, p_vals_corr = fdrcorrection(p_vals, alpha=DEFAULT_CONFIG['fdr_alpha'])
    results_df['p_value_fdr'] = p_vals_corr
    
    logger.info("Generated condition t-test results for %d comparisons", len(results_df))
    return results_df

def analyze_followup_by_condition(
    ocd_subjects: List[str],
    metadata_df: pd.DataFrame,
    feature_info: Dict[str, Tuple[str, str]],
    input_dir: str,
    atlas_name: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """Analyze followup FC differences between conditions."""
    logger.info("Analyzing followup FC by condition")
    
    # Load followup FC data for OCD subjects
    followup_fc_data = []
    
    for sid in ocd_subjects:
        sid_clean = sid.replace('sub-', '')
        follow_path = get_roiroi_fc_path(sid_clean, 'ses-followup', input_dir, atlas_name)
        
        if not os.path.exists(follow_path):
            continue
            
        if not validate_roiroi_fc_file(follow_path, logger):
            continue
        
        try:
            follow_fc = pd.read_csv(follow_path)
            follow_fc['feature_id'] = follow_fc['ROI']
            
            # Pivot to make features as columns
            follow_pivot = follow_fc.pivot_table(
                index=None, columns='feature_id', values='FC'
            ).reset_index(drop=True)
            follow_pivot['subject_id'] = sid_clean
            followup_fc_data.append(follow_pivot)
            
        except Exception as e:
            logger.warning("Failed to load followup FC for subject %s: %s", sid, e)
            continue
    
    if not followup_fc_data:
        logger.warning("No followup FC data loaded for condition analysis")
        return pd.DataFrame()
    
    # Combine followup data
    followup_df = pd.concat(followup_fc_data, ignore_index=True)
    
    # Run condition t-tests on followup data
    return run_condition_ttest(followup_df, metadata_df, feature_info, "followup FC", logger)

def analyze_fc_change_by_condition(
    ocd_subjects: List[str],
    metadata_df: pd.DataFrame,
    feature_info: Dict[str, Tuple[str, str]],
    input_dir: str,
    atlas_name: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """Analyze FC change differences between conditions."""
    logger.info("Analyzing FC change by condition")
    
    # Load baseline and followup FC data for OCD subjects
    fc_change_data = []
    
    for sid in ocd_subjects:
        sid_clean = sid.replace('sub-', '')
        base_path = get_roiroi_fc_path(sid_clean, 'ses-baseline', input_dir, atlas_name)
        follow_path = get_roiroi_fc_path(sid_clean, 'ses-followup', input_dir, atlas_name)
        
        if not (os.path.exists(base_path) and os.path.exists(follow_path)):
            continue
            
        if not (validate_roiroi_fc_file(base_path, logger) and validate_roiroi_fc_file(follow_path, logger)):
            continue
        
        try:
            # Load baseline and followup data
            base_fc = pd.read_csv(base_path)
            follow_fc = pd.read_csv(follow_path)
            
            # Create feature identifiers
            base_fc['feature_id'] = base_fc['ROI']
            follow_fc['feature_id'] = follow_fc['ROI']
            
            # Pivot and compute change
            base_pivot = base_fc.pivot_table(
                index=None, columns='feature_id', values='FC'
            ).reset_index(drop=True)
            follow_pivot = follow_fc.pivot_table(
                index=None, columns='feature_id', values='FC'
            ).reset_index(drop=True)
            
            change_pivot = follow_pivot - base_pivot
            change_pivot['subject_id'] = sid_clean
            fc_change_data.append(change_pivot)
            
        except Exception as e:
            logger.warning("Failed to process FC change for subject %s: %s", sid, e)
            continue
    
    if not fc_change_data:
        logger.warning("No FC change data loaded for condition analysis")
        return pd.DataFrame()
    
    # Combine change data
    change_df = pd.concat(fc_change_data, ignore_index=True)
    
    # Run condition t-tests on FC change data
    return run_condition_ttest(change_df, metadata_df, feature_info, "FC change", logger)

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
        sid_clean = sid.replace('sub-', '')
        base_path = get_roiroi_fc_path(sid_clean, 'ses-baseline', input_dir, atlas_name)
        follow_path = get_roiroi_fc_path(sid_clean, 'ses-followup', input_dir, atlas_name)
        
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
            
            # Create feature identifiers
            base_fc['feature_id'] = base_fc['ROI']
            follow_fc['feature_id'] = follow_fc['ROI']
            
            # Pivot and compute change
            base_pivot = base_fc.pivot_table(
                index=None, columns='feature_id', values='FC'
            ).reset_index(drop=True)
            follow_pivot = follow_fc.pivot_table(
                index=None, columns='feature_id', values='FC'
            ).reset_index(drop=True)
            
            change_pivot = follow_pivot - base_pivot
            change_pivot['subject_id'] = sid_clean
            fc_change_data.append(change_pivot)
            
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
        feature_columns = [col for col in fc_change_data.columns if col != 'subject_id']
        
        logger.info(
            "Loaded ROI-to-ROI FC change data for %d subjects, %d features: %s",
            len(fc_change_data), len(feature_columns), list(fc_change_data['subject_id'])
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
    logger.info("Starting ROI-to-ROI FC Group Analysis")
    logger.info("=" * 80)
    # Memory monitoring disabled
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
        
        # Update log file name to include atlas name
        atlas_log_file = f'roiroi_group_analysis_{atlas_name}.log'
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
        atlas_logger.info("Loading baseline ROI-to-ROI FC data...")
        # Memory monitoring disabled
        baseline_fc_data, feature_info = load_roiroi_fc_data(
            valid_group, 'ses-baseline', args.input_dir, atlas_name, atlas_logger
        )
        # Memory monitoring disabled
        
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
        
        # 3. Condition-based analysis (OCD only)
        if valid_group:  # Need at least some subjects for condition analysis
            perform_condition_analysis(
                baseline_fc_data, df, feature_info, 
                args.input_dir, args.output_dir, atlas_name, atlas_logger
            )

        atlas_logger.info("Main ROI-to-ROI FC analysis completed successfully for atlas: %s", atlas_name)
        atlas_logger.info("Analyses completed: Group comparison, Longitudinal analysis, Condition-based analysis")
    
    except Exception as e:
        logger.error("Main execution failed: %s", e)
        raise
    
    finally:
        logger.info("=" * 80)
        logger.info("ROI-to-ROI FC Group Analysis Completed")
        logger.info("=" * 80)

def print_quick_help():
    """Print quick help information."""
    quick_help = """
QUICK HELP - ROI-to-ROI Functional Connectivity Group Analysis
=================================================================

BASIC USAGE:
  python NW_group3.py --subjects_csv <GROUP_CSV> --clinical_csv <CLINICAL_CSV> --input_dir <FC_DATA_DIR>

QUICK EXAMPLES:
  1. Default Atlas (Power 2011):
     python NW_group3.py \\
       --subjects_csv group.csv \\
       --clinical_csv clinical.csv \\
       --input_dir /path/to/fc/data

  2. Specific Atlas:
     python NW_group3.py \\
       --subjects_csv group.csv \\
       --clinical_csv clinical.csv \\
       --input_dir /path/to/fc/data \\
       --atlas_name schaefer_2018_400_7_2

  3. YEO 2011 Atlas:
     python NW_group3.py \\
       --subjects_csv group.csv \\
       --clinical_csv clinical.csv \\
       --input_dir /path/to/fc/data \\
       --atlas_name yeo_2011_7_thick

  4. Auto-detect Atlas:
     python NW_group3.py \\
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
   
   python NW_group3.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data
   
   Expected FC files: *_task-rest_power_2011_roiroi_fc_avg.csv

2. SPECIFIC ATLAS NAME
   --------------------
   Use when you know the exact atlas name from NW_1st.py output.
   
   python NW_group3.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --atlas_name schaefer_2018_400_7_2
   
   Expected FC files: *_task-rest_schaefer_2018_400_7_2_roiroi_fc_avg.csv

3. AUTO-DETECT ATLAS
   ------------------
   Automatically detect atlas name from available FC files.
   
   python NW_group3.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --auto-detect-atlas
   
   This will scan the input directory and find the atlas name automatically.

4. WITH CUSTOM OUTPUT DIRECTORY
   -----------------------------
   Override the default output directory.
   
   python NW_group3.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --output_dir /custom/output/path \\
     --auto-detect-atlas

5. VERBOSE LOGGING
   ----------------
   Enable detailed logging for debugging.
   
   python NW_group3.py \\
     --subjects_csv group.csv \\
     --clinical_csv clinical.csv \\
     --input_dir /path/to/fc/data \\
     --verbose

REQUIRED FILES:
---------------
- group.csv: Contains subject IDs and group labels (HC/OCD)
- clinical.csv: Contains clinical data including YBOCS scores
- FC data files: Generated by NW_1st.py with naming patterns:
  *_{session}_task-rest_{atlas_name}_roiroi_fc_avg.csv (for network-based atlases)
  *_{session}_task-rest_{atlas_name}_roiroi_fc_avg.csv (for anatomical atlases)

OUTPUT FILES:
-------------
- group_diff_baseline_{atlas_name}_roiroi_fc.csv: Group difference t-test results
- baselineFC_vs_deltaYBOCS_{atlas_name}_roiroi_fc.csv: Baseline FC vs symptom change
- deltaFC_vs_deltaYBOCS_{atlas_name}_roiroi_fc.csv: FC change vs symptom change

ATLAS NAMING CONVENTIONS:
-------------------------
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
- Custom: custom_atlas_roiroi_fc_avg.csv (if network-based) or custom_atlas_roiroi_fc_avg.csv (if anatomical)

Note: For Schaefer 2018, the naming follows the pattern: schaefer_2018_{n_rois}_{yeo_networks}_{resolution_mm}
where n_rois can be 100, 200, 300, 400, 500, 600, 700, 800, 900, or 1000,
yeo_networks can be 7 or 17, and resolution_mm can be 1 or 2.
For YEO 2011, the naming follows: yeo_2011_{n_networks}_{thickness}
where n_networks can be 7 or 17, and thickness can be 'thick' or 'thin'.

IMPORTANT: This script (NW_group3.py) can handle BOTH file types:
- All atlases (Power 2011, Schaefer 2018, YEO 2011) generate *_roiroi_fc_avg.csv files
- Anatomical atlases (Harvard-Oxford, AAL, Talairach) generate *_roiroi_fc_avg.csv files
Both file types are automatically detected and processed appropriately.

TROUBLESHOOTING:
----------------
1. Check that FC data files exist in the input directory
2. Verify atlas naming matches between NW_1st.py and NW_group3.py
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
        # Memory monitoring disabled for compatibility
        print("\nFor help, run: python NW_group3.py --help")
        print("For usage examples, run: python NW_group3.py --usage")
        raise