#!/usr/bin/env python3
"""
Extract ROI-to-ROI Functional Connectivity Results for Each Subject

This script extracts individual subject ROI-to-ROI FC results from CSV files
and saves them in organized CSV format, similar to how NW_roi_roi.py processes the data.

Based on the structure and patterns from NW_roi_roi.py.

Author: [Your Name]
Date: [Current Date]
"""

import os
import glob
import pandas as pd
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default configuration
DEFAULT_CONFIG = {
    'output_dir': './roi_results',
    'log_file': 'extract_roi_results.log',
    'sessions': ['ses-baseline', 'ses-followup'],
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
    logger = logging.getLogger('Extract_ROI_Results')
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
        description='Extract ROI-to-ROI FC results for each subject and save to CSV'
    )
    parser.add_argument(
        '--subjects_csv', 
        type=str, 
        required=True, 
        help='Path to subjects CSV file (group.csv)'
    )
    parser.add_argument(
        '--clinical_csv', 
        type=str, 
        required=True,
        help='Path to clinical CSV file with YBOCS scores'
    )
    parser.add_argument(
        '--input_dir', 
        type=str, 
        required=True,
        help='Input directory containing FC CSV files'
    )
    parser.add_argument(
        '--atlas_name', 
        type=str, 
        default='power_2011',
        help='Atlas name (default: power_2011)'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=DEFAULT_CONFIG['output_dir'],
        help='Output directory for results'
    )
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true', 
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_roiroi_fc_path(subject: str, session: str, input_dir: str, atlas_name: str) -> str:
    """Get path to ROI-to-ROI FC CSV file (same as NW_roi_roi.py)."""
    if not subject.startswith('sub-'):
        subject = f"sub-{subject}"
    
    # ROI-to-ROI file path
    roiroi_path = os.path.join(
        input_dir, 
        f"{subject}_{session}_task-rest_{atlas_name}_roiroi_fc_avg.csv"
    )
    
    return roiroi_path

def validate_roiroi_fc_file(fc_path: str, logger: logging.Logger) -> bool:
    """Validate that ROI-to-ROI FC file has required columns (same as NW_roi_roi.py)."""
    required_columns = {'ROI1', 'ROI2', 'network1', 'network2', 'FC'}
    
    try:
        df = pd.read_csv(fc_path)
        missing_columns = required_columns - set(df.columns)
        
        if missing_columns:
            logger.error(
                "ROI-to-ROI FC file %s missing required columns: %s. Found columns: %s",
                fc_path, missing_columns, list(df.columns)
            )
            return False
        
        if df.empty:
            logger.error("ROI-to-ROI FC file %s is empty", fc_path)
            return False
        
        logger.debug("Validated ROI-to-ROI FC file: %s", fc_path)
        return True
        
    except Exception as e:
        logger.error("Failed to validate ROI-to-ROI FC file %s: %s", fc_path, e)
        return False

def auto_detect_atlas_from_files(input_dir: str, logger: logging.Logger) -> Optional[str]:
    """Auto-detect atlas name from existing FC files."""
    logger.info("Auto-detecting atlas name from FC files in %s", input_dir)
    
    # Look for FC files
    fc_files = glob.glob(os.path.join(input_dir, '*_roiroi_fc_avg.csv'))
    
    if not fc_files:
        logger.warning("No ROI-to-ROI FC files found in %s", input_dir)
        return None
    
    # Extract atlas names from filenames
    atlas_names = set()
    for fc_file in fc_files:
        filename = os.path.basename(fc_file)
        # Pattern: sub-XXX_ses-XXX_task-rest_ATLAS_roiroi_fc_avg.csv
        parts = filename.split('_')
        if len(parts) >= 5 and parts[-2] == 'roiroi' and parts[-1] == 'fc_avg.csv':
            # Extract atlas name (everything between task-rest and roiroi)
            atlas_part = '_'.join(parts[3:-2])
            atlas_names.add(atlas_part)
    
    if len(atlas_names) == 1:
        atlas_name = list(atlas_names)[0]
        logger.info("Auto-detected atlas name: %s", atlas_name)
        return atlas_name
    elif len(atlas_names) > 1:
        logger.warning("Multiple atlas names found: %s. Using the first one.", atlas_names)
        return list(atlas_names)[0]
    else:
        logger.error("Could not auto-detect atlas name from files")
        return None

def load_subjects_metadata(subjects_csv: str, logger: logging.Logger) -> pd.DataFrame:
    """Load subjects metadata from CSV file."""
    logger.info("Loading subjects metadata from %s", subjects_csv)
    
    try:
        df = pd.read_csv(subjects_csv)
        df['subject_id'] = df['sub'].astype(str)
        df = df[df['group'].isin(['HC', 'OCD'])]
        logger.info("Loaded %d subjects from %s", len(df), subjects_csv)
        return df
    except Exception as e:
        logger.error("Failed to load subjects CSV: %s", e)
        raise ValueError(f"Failed to load subjects CSV: {e}")

def load_clinical_data(clinical_csv: str, logger: logging.Logger) -> pd.DataFrame:
    """Load clinical data with YBOCS scores from CSV file."""
    logger.info("Loading clinical data from %s", clinical_csv)
    
    try:
        df_clinical = pd.read_csv(clinical_csv)
        df_clinical['subject_id'] = df_clinical['sub'].astype(str)
        logger.info("Loaded %d clinical records from %s", len(df_clinical), clinical_csv)
        
        # Check for required YBOCS columns
        required_columns = ['ybocs_baseline', 'ybocs_followup']
        missing_columns = [col for col in required_columns if col not in df_clinical.columns]
        
        if missing_columns:
            logger.error("Clinical CSV missing required YBOCS columns: %s", missing_columns)
            raise ValueError(f"Clinical CSV missing required YBOCS columns: {missing_columns}")
        
        logger.info("Clinical data validation passed - YBOCS columns found")
        return df_clinical
        
    except Exception as e:
        logger.error("Failed to load clinical CSV: %s", e)
        raise ValueError(f"Failed to load clinical CSV: {e}")

# =============================================================================
# MAIN EXTRACTION FUNCTIONS
# =============================================================================

def extract_subject_roi_data(
    subject_id: str, 
    session: str, 
    input_dir: str, 
    atlas_name: str,
    logger: logging.Logger
) -> Optional[pd.DataFrame]:
    """Extract ROI-to-ROI FC data for a single subject and session."""
    fc_path = get_roiroi_fc_path(subject_id, session, input_dir, atlas_name)
    
    if not os.path.exists(fc_path):
        logger.warning("ROI-to-ROI FC file not found for subject %s: %s", subject_id, fc_path)
        return None
    
    if not validate_roiroi_fc_file(fc_path, logger):
        logger.error("Invalid ROI-to-ROI FC file for subject %s: %s", subject_id, fc_path)
        return None
    
    try:
        logger.debug("Loading FC file: %s", fc_path)
        fc_df = pd.read_csv(fc_path)
        logger.debug("Loaded ROI-to-ROI FC file %s with %d rows", fc_path, len(fc_df))
        
        # Add subject and session information
        fc_df['subject_id'] = subject_id
        fc_df['session'] = session
        
        # Create feature identifier (same as NW_roi_roi.py)
        fc_df['feature_id'] = fc_df['ROI1'] + '_' + fc_df['ROI2']
        
        # Ensure all string values are regular strings (not byte strings)
        for col in ['ROI1', 'ROI2', 'network1', 'network2', 'feature_id']:
            if col in fc_df.columns:
                fc_df[col] = fc_df[col].astype(str)
        
        logger.info("Successfully extracted ROI data for %s %s: %d connections", 
                   subject_id, session, len(fc_df))
        
        return fc_df
        
    except Exception as e:
        logger.error("Failed to extract ROI data for subject %s %s: %s", subject_id, session, e)
        return None

def extract_all_subjects_roi_data(
    subjects_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    input_dir: str,
    atlas_name: str,
    output_dir: str,
    logger: logging.Logger
) -> None:
    """Extract ROI-to-ROI FC data for all subjects and save to CSV files."""
    logger.info("Starting ROI data extraction for %d subjects", len(subjects_df))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics
    total_subjects = len(subjects_df)
    successful_extractions = 0
    failed_extractions = 0
    
    # Extract data for each subject and session
    for _, row in subjects_df.iterrows():
        subject_id = row['subject_id']
        group = row['group']
        
        logger.info("Processing subject %s (group: %s)", subject_id, group)
        
        subject_data = []
        
        # Process both sessions
        for session in DEFAULT_CONFIG['sessions']:
            logger.debug("Extracting data for %s %s", subject_id, session)
            
            roi_data = extract_subject_roi_data(
                subject_id, session, input_dir, atlas_name, logger
            )
            
            if roi_data is not None:
                subject_data.append(roi_data)
                successful_extractions += 1
            else:
                failed_extractions += 1
        
        # Save individual subject data if any sessions were successful
        if subject_data:
            # Combine all sessions for this subject
            combined_data = pd.concat(subject_data, ignore_index=True)
            
            # Save to individual subject CSV
            subject_output_file = os.path.join(
                output_dir, 
                f"{subject_id}_roi_fc_data.csv"
            )
            
            try:
                combined_data.to_csv(subject_output_file, index=False)
                logger.info("Saved ROI data for %s to %s (%d connections)", 
                           subject_id, subject_output_file, len(combined_data))
            except Exception as e:
                logger.error("Failed to save ROI data for %s: %s", subject_id, e)
    
    # Create summary CSV with all subjects
    logger.info("Creating summary CSV with all subjects...")
    create_summary_csv(subjects_df, clinical_df, input_dir, atlas_name, output_dir, logger)
    
    logger.info("ROI data extraction completed")
    logger.info("Total subjects: %d, Successful: %d, Failed: %d", 
               total_subjects, successful_extractions, failed_extractions)

def create_summary_csv(
    subjects_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    input_dir: str,
    atlas_name: str,
    output_dir: str,
    logger: logging.Logger
) -> None:
    """Create a summary CSV with all subjects' ROI data."""
    logger.info("Creating summary CSV with all subjects' ROI data")
    
    # Collect data from all subjects and sessions
    subject_session_data = {}
    successful_subjects = []
    
    for _, row in subjects_df.iterrows():
        subject_id = row['subject_id']
        group = row['group']
        
        subject_session_data[subject_id] = {
            'group': group,
            'baseline': None,
            'followup': None
        }
        
        # Collect data from both sessions for this subject
        for session in DEFAULT_CONFIG['sessions']:
            roi_data = extract_subject_roi_data(
                subject_id, session, input_dir, atlas_name, logger
            )
            
            if roi_data is not None:
                subject_session_data[subject_id][session.replace('ses-', '')] = roi_data
        
        # Check if both sessions are available
        if (subject_session_data[subject_id]['baseline'] is not None and 
            subject_session_data[subject_id]['followup'] is not None):
            successful_subjects.append(subject_id)
    
    if successful_subjects:
        logger.info("Creating Power atlas summary with %d subjects having both sessions", 
                   len(successful_subjects))
        
        # Create Power atlas specific format
        if atlas_name == 'power_2011':
                create_power_atlas_summary(subject_session_data, successful_subjects, clinical_df, subjects_df, output_dir, logger)
        else:
            create_generic_summary(subject_session_data, successful_subjects, output_dir, logger)
        
        # Save subject list
        subjects_file = os.path.join(output_dir, 'successful_subjects.csv')
        successful_df = pd.DataFrame({
            'subject_id': successful_subjects,
            'group': [subject_session_data[sid]['group'] for sid in successful_subjects]
        })
        successful_df.to_csv(subjects_file, index=False)
        
        logger.info("Saved successful subjects list: %s", subjects_file)
    else:
        logger.warning("No subjects found with both baseline and followup data")

def create_power_atlas_summary(
    subject_session_data: Dict,
    successful_subjects: List[str],
    clinical_df: pd.DataFrame,
    subjects_df: pd.DataFrame,
    output_dir: str,
    logger: logging.Logger
) -> None:
    """Create Power atlas specific summary with subjectID, group, condition, ROI1, ROI2, fc_baseline, fc_followup, ybocs_baseline, ybocs_followup."""
    logger.info("Creating Power atlas specific summary format")
    
    # Define the 4 specific ROIs we want
    target_rois = ['Default_mode_91', 'Default_mode_92', 'Dorsal_attention_259', 'Dorsal_attention_263']
    logger.info("Filtering for specific ROIs: %s", target_rois)
    
    # Create clinical data mapping for YBOCS scores
    clinical_mapping = clinical_df.set_index('subject_id')[['ybocs_baseline', 'ybocs_followup']].to_dict('index')
    
    # Create subjects data mapping for group and condition
    subjects_mapping = subjects_df.set_index('subject_id')[['group', 'condition']].to_dict('index')
    
    # Collect all data efficiently using pandas operations
    all_data = []
    
    for subject_id in successful_subjects:
        group = subject_session_data[subject_id]['group']
        baseline_data = subject_session_data[subject_id]['baseline']
        followup_data = subject_session_data[subject_id]['followup']
        
        # Filter for only the target ROIs in both baseline and followup data
        baseline_filtered = baseline_data[
            (baseline_data['ROI1'].isin(target_rois)) & 
            (baseline_data['ROI2'].isin(target_rois))
        ][['ROI1', 'ROI2', 'FC']].copy()
        
        followup_filtered = followup_data[
            (followup_data['ROI1'].isin(target_rois)) & 
            (followup_data['ROI2'].isin(target_rois))
        ][['ROI1', 'ROI2', 'FC']].copy()
        
        # Merge baseline and followup data on ROI pairs
        merged_data = baseline_filtered.merge(
            followup_filtered, 
            on=['ROI1', 'ROI2'], 
            suffixes=('_baseline', '_followup')
        )
        
        if len(merged_data) == 0:
            logger.warning("No matching ROI pairs found for subject %s", subject_id)
            continue
        
        # Get YBOCS scores from clinical data
        ybocs_baseline = None
        ybocs_followup = None
        
        if subject_id in clinical_mapping:
            ybocs_baseline = clinical_mapping[subject_id]['ybocs_baseline']
            ybocs_followup = clinical_mapping[subject_id]['ybocs_followup']
        else:
            logger.warning("No clinical data found for subject %s", subject_id)
        
        # Get group and condition from subjects data
        group_from_subjects = None
        condition = None
        
        if subject_id in subjects_mapping:
            group_from_subjects = subjects_mapping[subject_id]['group']
            condition = subjects_mapping[subject_id]['condition']
        else:
            logger.warning("No subject metadata found for subject %s", subject_id)
        
        # Add subject information to each row
        merged_data['subject_ID'] = subject_id
        merged_data['group'] = group_from_subjects if group_from_subjects else group
        merged_data['condition'] = condition
        merged_data['ybocs_baseline'] = ybocs_baseline
        merged_data['ybocs_followup'] = ybocs_followup
        
        all_data.append(merged_data)
    
    # Combine all subjects' data
    if all_data:
        power_df = pd.concat(all_data, ignore_index=True)
        
        # Rename columns to match requested format
        power_df = power_df.rename(columns={
            'FC_baseline': 'FC_baseline',
            'FC_followup': 'FC_followup'
        })
        
        # Reorder columns as requested
        column_order = ['subject_ID', 'group', 'condition', 'ybocs_baseline', 'ybocs_followup', 'ROI1', 'ROI2', 'FC_baseline', 'FC_followup']
        power_df = power_df[column_order]
        
        # Save Power atlas summary
        power_file = os.path.join(output_dir, 'power_atlas_roi_fc_summary.csv')
        power_df.to_csv(power_file, index=False)
        
        logger.info("Saved Power atlas summary: %s (%d rows from %d subjects)", 
                   power_file, len(power_df), len(successful_subjects))
        logger.info("ROI pairs included: %d unique combinations", power_df[['ROI1', 'ROI2']].drop_duplicates().shape[0])
    else:
        logger.warning("No data to create Power atlas summary")
    
    # Also save the traditional format for compatibility
    create_generic_summary(subject_session_data, successful_subjects, output_dir, logger)

def create_generic_summary(
    subject_session_data: Dict,
    successful_subjects: List[str],
    output_dir: str,
    logger: logging.Logger
) -> None:
    """Create generic summary format for non-Power atlases."""
    logger.info("Creating generic summary format")
    
    all_data = []
    
    for subject_id in successful_subjects:
        group = subject_session_data[subject_id]['group']
        
        # Add baseline data
        baseline_data = subject_session_data[subject_id]['baseline'].copy()
        baseline_data['subject_id'] = subject_id
        baseline_data['session'] = 'ses-baseline'
        baseline_data['group'] = group
        all_data.append(baseline_data)
        
        # Add followup data
        followup_data = subject_session_data[subject_id]['followup'].copy()
        followup_data['subject_id'] = subject_id
        followup_data['session'] = 'ses-followup'
        followup_data['group'] = group
        all_data.append(followup_data)
    
    # Combine all data
    summary_data = pd.concat(all_data, ignore_index=True)
    
    # Reorder columns for better readability
    column_order = ['subject_id', 'session', 'group', 'ROI1', 'ROI2', 
                   'network1', 'network2', 'feature_id', 'FC']
    summary_data = summary_data[column_order]
    
    # Save generic summary CSV
    summary_file = os.path.join(output_dir, 'all_subjects_roi_fc_data.csv')
    summary_data.to_csv(summary_file, index=False)
    
    logger.info("Saved generic summary CSV: %s (%d total connections from %d subjects)", 
               summary_file, len(summary_data), len(successful_subjects))

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to extract ROI-to-ROI FC results for each subject."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.output_dir, DEFAULT_CONFIG['log_file'])
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("=" * 80)
    logger.info("Starting ROI-to-ROI FC Results Extraction")
    logger.info("=" * 80)
    logger.info("Arguments: %s", vars(args))
    
    try:
        # Create output directories
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load subjects metadata
        subjects_df = load_subjects_metadata(args.subjects_csv, logger)
        
        # Load clinical data
        clinical_df = load_clinical_data(args.clinical_csv, logger)
        
        # Auto-detect atlas if not specified
        atlas_name = args.atlas_name
        if not atlas_name or atlas_name == 'auto':
            detected_atlas = auto_detect_atlas_from_files(args.input_dir, logger)
            if detected_atlas:
                atlas_name = detected_atlas
            else:
                logger.error("Could not auto-detect atlas name and none specified")
                raise ValueError("Atlas name must be specified or auto-detectable")
        
        logger.info("Using atlas: %s", atlas_name)
        
        # Extract ROI data for all subjects
        extract_all_subjects_roi_data(
            subjects_df, clinical_df, args.input_dir, atlas_name, args.output_dir, logger
        )
        
        logger.info("ROI results extraction completed successfully")
    
    except Exception as e:
        logger.error("Main execution failed: %s", e)
        raise
    
    finally:
        logger.info("=" * 80)
        logger.info("ROI-to-ROI FC Results Extraction Completed")
        logger.info("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("Main execution failed: %s", e)
        raise
