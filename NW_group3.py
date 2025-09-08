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

def get_ybocs_scores(subject_id: str, clinical_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """Get YBOCS scores for baseline and follow-up sessions."""
    if subject_id.startswith('sub-'):
        subject_id = subject_id.replace('sub-', '')
    
    subject_data = clinical_df[clinical_df['subject_id'] == subject_id]
    if subject_data.empty:
        return None, None
    
    baseline_score = None
    followup_score = None
    
    for _, row in subject_data.iterrows():
        if row['session'] == 'baseline':
            baseline_score = row['ybocs_total']
        elif row['session'] == 'followup':
            followup_score = row['ybocs_total']
    
    return baseline_score, followup_score

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_metadata(subjects_csv: str, clinical_csv: str, logger: logging.Logger
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

def detect_atlas_from_files(input_dir: str, logger: logging.Logger) -> Optional[str]:
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

def validate_atlas_files(input_dir: str, atlas_name: str, logger: logging.Logger) -> bool:
    """Validate that FC files exist for the specified atlas."""
    pattern = os.path.join(input_dir, f'*_task-rest_{atlas_name}_roiroi_fc_avg.csv')
    files = glob.glob(pattern)
    
    if not files:
        logger.error("No FC files found for atlas '%s' in %s", atlas_name, input_dir)
        return False
    
    logger.info("Found %d FC files for atlas '%s'", len(files), atlas_name)
    return True

def load_roiroi_fc_data(input_dir: str, atlas_name: str, logger: logging.Logger) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load ROI-to-ROI FC data for all subjects and sessions."""
    logger.info("Loading FC data...")
    
    # Find FC files with the specific atlas
    fc_files = glob.glob(os.path.join(input_dir, f'*_task-rest_{atlas_name}_roiroi_fc_avg.csv'))
    logger.info("Found %d FC files for atlas %s", len(fc_files), atlas_name)
    
    if not fc_files:
        logger.error("No FC files found for atlas %s", atlas_name)
        return {}
    
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

def run_ttest(
    fc_data_hc: List[pd.DataFrame], 
    fc_data_ocd: List[pd.DataFrame], 
    logger: logging.Logger
) -> pd.DataFrame:
    """Run two-sample t-tests with FDR correction for ROI-to-ROI FC."""
    logger.info(
        "Running t-tests with %d HC subjects and %d OCD subjects",
        len(fc_data_hc), len(fc_data_ocd)
    )
    
    # Get all ROI pairs from the first subject
    if fc_data_hc:
        roi_pairs = fc_data_hc[0][['ROI1', 'ROI2']].copy()
    elif fc_data_ocd:
        roi_pairs = fc_data_ocd[0][['ROI1', 'ROI2']].copy()
    else:
        logger.error("No data available for analysis")
        return pd.DataFrame()
    
    results = []
    dropped_features = []
    
    for _, row in roi_pairs.iterrows():
        roi1, roi2 = row['ROI1'], row['ROI2']
        
        # Extract network information if available
        network1 = row.get('network1', 'Unknown') if 'network1' in row else 'Unknown'
        network2 = row.get('network2', 'Unknown') if 'network2' in row else 'Unknown'
        
        # Extract FC values for this ROI pair
        hc_values = []
        ocd_values = []
        
        for fc_df in fc_data_hc:
            pair_data = fc_df[(fc_df['ROI1'] == roi1) & (fc_df['ROI2'] == roi2)]
            if not pair_data.empty:
                hc_values.append(pair_data['fc_value'].iloc[0])
                # Get network info from first available data
                if network1 == 'Unknown' and 'network1' in pair_data.columns:
                    network1 = pair_data['network1'].iloc[0]
                if network2 == 'Unknown' and 'network2' in pair_data.columns:
                    network2 = pair_data['network2'].iloc[0]
        
        for fc_df in fc_data_ocd:
            pair_data = fc_df[(fc_df['ROI1'] == roi1) & (fc_df['ROI2'] == roi2)]
            if not pair_data.empty:
                ocd_values.append(pair_data['fc_value'].iloc[0])
                # Get network info from first available data
                if network1 == 'Unknown' and 'network1' in pair_data.columns:
                    network1 = pair_data['network1'].iloc[0]
                if network2 == 'Unknown' and 'network2' in pair_data.columns:
                    network2 = pair_data['network2'].iloc[0]
        
        if len(hc_values) < DEFAULT_CONFIG['min_subjects_per_group'] or \
           len(ocd_values) < DEFAULT_CONFIG['min_subjects_per_group']:
            logger.warning(
                "Skipping ROI pair %s-%s due to insufficient data (HC n=%d, OCD n=%d)",
                roi1, roi2, len(hc_values), len(ocd_values)
            )
            dropped_features.append((f"{roi1}-{roi2}", f"HC n={len(hc_values)}, OCD n={len(ocd_values)}"))
            continue
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(hc_values, ocd_values)
        
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

def run_longitudinal_analysis(
    fc_data: Dict[str, Dict[str, pd.DataFrame]], 
    clinical_df: pd.DataFrame, 
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run longitudinal analysis for ROI-to-ROI FC."""
    logger.info("Performing longitudinal analysis")
    
    # Find subjects with both baseline and follow-up data
    subjects_with_both_sessions = []
    for subject, sessions_data in fc_data.items():
        if 'ses-baseline' in sessions_data and 'ses-followup' in sessions_data:
            subjects_with_both_sessions.append(subject)
    
    logger.info("Found %d subjects with both baseline and follow-up data", len(subjects_with_both_sessions))
    
    if len(subjects_with_both_sessions) < DEFAULT_CONFIG['min_subjects_per_group']:
        logger.warning("Insufficient subjects for longitudinal analysis")
        return pd.DataFrame(), pd.DataFrame()
    
    # Get all ROI pairs from the first subject
    first_subject = subjects_with_both_sessions[0]
    roi_pairs = fc_data[first_subject]['ses-baseline'][['ROI1', 'ROI2']].copy()
    
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
                baseline_fc_df = fc_data[subject]['ses-baseline']
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
                followup_fc_df = fc_data[subject]['ses-followup']
                followup_pair = followup_fc_df[(followup_fc_df['ROI1'] == roi1) & (followup_fc_df['ROI2'] == roi2)]
                
                if not baseline_pair.empty and not followup_pair.empty:
                    baseline_fc = baseline_pair['fc_value'].iloc[0]
                    followup_fc = followup_pair['fc_value'].iloc[0]
                    delta_fc = followup_fc - baseline_fc
                    delta_fc_values.append(delta_fc)
        
        # Baseline FC vs Delta YBOCS
        if len(baseline_fc_values) > 2 and len(delta_ybocs_values) > 2:
            r, p = stats.pearsonr(baseline_fc_values, delta_ybocs_values)
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
            r, p = stats.pearsonr(delta_fc_values, delta_ybocs_values)
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
    if len(baseline_df) > 1:
        _, p_vals_corr = fdrcorrection(baseline_df['p_value'].values, alpha=DEFAULT_CONFIG['fdr_alpha'])
        baseline_df['p_corrected'] = p_vals_corr
        baseline_df['significant'] = p_vals_corr < DEFAULT_CONFIG['fdr_alpha']
    elif len(baseline_df) == 1:
        baseline_df['p_corrected'] = baseline_df['p_value']
        baseline_df['significant'] = baseline_df['p_value'] < DEFAULT_CONFIG['fdr_alpha']
    
    if len(delta_df) > 1:
        _, p_vals_corr = fdrcorrection(delta_df['p_value'].values, alpha=DEFAULT_CONFIG['fdr_alpha'])
        delta_df['p_corrected'] = p_vals_corr
        delta_df['significant'] = p_vals_corr < DEFAULT_CONFIG['fdr_alpha']
    elif len(delta_df) == 1:
        delta_df['p_corrected'] = delta_df['p_value']
        delta_df['significant'] = delta_df['p_value'] < DEFAULT_CONFIG['fdr_alpha']
    
    logger.info("Longitudinal analysis complete: %d baseline FC correlations, %d delta FC correlations", 
                len(baseline_df), len(delta_df))
    
    return baseline_df, delta_df

# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def perform_group_difference_analysis(
    fc_data: Dict[str, Dict[str, pd.DataFrame]], 
    metadata_df: pd.DataFrame, 
    atlas_name: str, 
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform group difference analysis at baseline and follow-up."""
    logger.info("Performing group difference analysis for atlas: %s", atlas_name)
    
    # Separate HC and OCD data for baseline
    hc_baseline_data = []
    ocd_baseline_data = []
    
    for subject, sessions_data in fc_data.items():
        if 'ses-baseline' not in sessions_data:
            continue
        
        group = get_group(subject, metadata_df)
        if group == 'HC':
            hc_baseline_data.append(sessions_data['ses-baseline'])
        elif group == 'OCD':
            ocd_baseline_data.append(sessions_data['ses-baseline'])
    
    logger.info("Baseline group t-test analysis: %d HC subjects, %d OCD subjects", 
                len(hc_baseline_data), len(ocd_baseline_data))
    
    baseline_results = pd.DataFrame()
    if hc_baseline_data and ocd_baseline_data:
        baseline_results = run_ttest(hc_baseline_data, ocd_baseline_data, logger)
    
    # Separate HC and OCD data for follow-up
    hc_followup_data = []
    ocd_followup_data = []
    
    for subject, sessions_data in fc_data.items():
        if 'ses-followup' not in sessions_data:
            continue
        
        group = get_group(subject, metadata_df)
        if group == 'HC':
            hc_followup_data.append(sessions_data['ses-followup'])
        elif group == 'OCD':
            ocd_followup_data.append(sessions_data['ses-followup'])
    
    logger.info("Follow-up group t-test analysis: %d HC subjects, %d OCD subjects", 
                len(hc_followup_data), len(ocd_followup_data))
    
    followup_results = pd.DataFrame()
    if hc_followup_data and ocd_followup_data:
        followup_results = run_ttest(hc_followup_data, ocd_followup_data, logger)
    
    return baseline_results, followup_results

def save_results(
    baseline_results: pd.DataFrame, 
    followup_results: pd.DataFrame, 
    baseline_fc_results: pd.DataFrame, 
    delta_fc_results: pd.DataFrame, 
    atlas_name: str, 
    output_dir: str, 
    logger: logging.Logger
) -> None:
    """Save all results to CSV files."""
    logger.info("Saving results...")
    
    # Save baseline group comparison results
    if not baseline_results.empty:
        baseline_file = os.path.join(output_dir, f'group_diff_baseline_{atlas_name}_roiroi_fc.csv')
        baseline_results.to_csv(baseline_file, index=False)
        logger.info("Saved baseline group comparison results: %s", baseline_file)
    else:
        logger.info("No baseline group comparison results to save")
    
    # Save follow-up group comparison results
    if not followup_results.empty:
        followup_file = os.path.join(output_dir, f'group_diff_followup_{atlas_name}_roiroi_fc.csv')
        followup_results.to_csv(followup_file, index=False)
        logger.info("Saved follow-up group comparison results: %s", followup_file)
    else:
        logger.info("No follow-up group comparison results to save")
    
    # Save baseline FC vs delta YBOCS results
    if not baseline_fc_results.empty:
        baseline_fc_file = os.path.join(output_dir, f'baselineFC_vs_deltaYBOCS_{atlas_name}_roiroi_fc.csv')
        baseline_fc_results.to_csv(baseline_fc_file, index=False)
        logger.info("Saved baseline FC vs delta YBOCS results: %s", baseline_fc_file)
    else:
        logger.info("No baseline FC vs delta YBOCS results to save")
    
    # Save delta FC vs delta YBOCS results
    if not delta_fc_results.empty:
        delta_fc_file = os.path.join(output_dir, f'deltaFC_vs_deltaYBOCS_{atlas_name}_roiroi_fc.csv')
        delta_fc_results.to_csv(delta_fc_file, index=False)
        logger.info("Saved delta FC vs delta YBOCS results: %s", delta_fc_file)
    else:
        logger.info("No delta FC vs delta YBOCS results to save")
    
    # Create summary
    summary_data = {
        'Analysis': ['Baseline Group Comparison', 'Follow-up Group Comparison', 
                    'Baseline FC vs Delta YBOCS', 'Delta FC vs Delta YBOCS'],
        'ROI_Pairs': [len(baseline_results), len(followup_results), 
                     len(baseline_fc_results), len(delta_fc_results)],
        'Significant_Pairs': [
            baseline_results['significant'].sum() if not baseline_results.empty else 0,
            followup_results['significant'].sum() if not followup_results.empty else 0,
            baseline_fc_results['significant'].sum() if not baseline_fc_results.empty else 0,
            delta_fc_results['significant'].sum() if not delta_fc_results.empty else 0
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, f'summary_{atlas_name}_roiroi_fc.csv')
    summary_df.to_csv(summary_file, index=False)
    logger.info("Saved summary: %s", summary_file)

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function."""
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging(args.output_dir, DEFAULT_CONFIG['log_file'])
    
    logger.info("=" * 80)
    logger.info("Starting ROI-to-ROI Functional Connectivity Group Analysis")
    logger.info("=" * 80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metadata
    logger.info("Loading metadata files...")
    try:
        metadata_df, clinical_df = load_metadata(args.subjects_csv, args.clinical_csv, logger)
    except Exception as e:
        logger.error("Failed to load metadata: %s", e)
        return
    
    # Detect or validate atlas
    if args.atlas_name:
        atlas_name = args.atlas_name
        if not validate_atlas_files(args.input_dir, atlas_name, logger):
            logger.error("Atlas validation failed")
            return
    elif args.auto_detect_atlas:
        atlas_name = detect_atlas_from_files(args.input_dir, logger)
        if not atlas_name:
            logger.error("Atlas auto-detection failed")
            return
    else:
        atlas_name = DEFAULT_CONFIG['default_atlas_name']
        if not validate_atlas_files(args.input_dir, atlas_name, logger):
            logger.error("Default atlas validation failed")
            return
    
    logger.info("Using atlas: %s", atlas_name)
    
    # Load FC data
    fc_data = load_roiroi_fc_data(args.input_dir, atlas_name, logger)
    if not fc_data:
        logger.error("Failed to load FC data")
        return
    
    # Perform group difference analysis
    baseline_results, followup_results = perform_group_difference_analysis(
        fc_data, metadata_df, atlas_name, logger
    )
    
    # Perform longitudinal analysis
    baseline_fc_results, delta_fc_results = run_longitudinal_analysis(
        fc_data, clinical_df, logger
    )
    
    # Save results
    save_results(
        baseline_results, followup_results, 
        baseline_fc_results, delta_fc_results, 
        atlas_name, args.output_dir, logger
    )
    
    logger.info("=" * 80)
    logger.info("Analysis complete!")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
    