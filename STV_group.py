#!/usr/bin/env python3
"""
Seed-to-Voxel Functional Connectivity Group Analysis

This script performs group-level statistical analysis on seed-to-voxel functional connectivity maps.
It compares healthy controls (HC) vs. OCD patients and performs longitudinal analyses using
voxel-wise statistical testing with FSL Randomise.

Author: [Your Name]
Date: [Current Date]
"""

import os
import glob
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.input_data import NiftiMasker
from nipype.interfaces.fsl import Randomise
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default configuration
DEFAULT_CONFIG = {
    'output_dir': '/project/6079231/dliang55/R01_AOCD/STV_group',
    'work_dir': '/scratch/xxqian/work_flow',
    'log_file': 'seed_to_voxel_group_analysis.log',
    'sessions': ['ses-baseline', 'ses-followup'],
    'group_mask_file': '/scratch/xxqian/roi/tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz',
    'num_permutations': 5000,
    'use_tfce': True
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
    logger = logging.getLogger('Seed_to_Voxel_Group_Analysis')
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
        description='Seed-to-voxel functional connectivity group analysis using FSL Randomise'
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
        '--work_dir', 
        type=str, 
        default=DEFAULT_CONFIG['work_dir'],
        help='Working directory for temporary files'
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

def get_fc_path(subject: str, session: str, output_dir: str) -> str:
    """Get path to seed-to-voxel FC map file."""
    if not subject.startswith('sub-'):
        subject = f"sub-{subject}"
    
    return os.path.join(
        output_dir, 
        f"{subject}_{session}_task-rest_seed-PCC_fcmap_avg.nii.gz"
    )

def get_group(subject: str, metadata_df: pd.DataFrame) -> Optional[str]:
    """Get group label for a subject."""
    if subject.startswith('sub-'):
        subject = subject.replace('sub-', '')
    
    group = metadata_df[metadata_df['subject_id'] == subject]['group']
    if group.empty:
        return None
    
    return group.iloc[0]

# =============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# =============================================================================

def run_voxelwise_regression(
    input_imgs: List[str], 
    y_values: List[float], 
    prefix: str,
    work_dir: str,
    output_dir: str,
    group_mask_file: str,
    logger: logging.Logger
) -> bool:
    """Run voxel-wise regression analysis using FSL Randomise."""
    logger.info("Running voxel-wise regression for %s with %d subjects", prefix, len(input_imgs))
    
    try:
        # Create design matrix
        design = np.array(y_values).reshape(-1, 1)
        mat_file = os.path.join(work_dir, f'{prefix}.mat')
        con_file = os.path.join(work_dir, f'{prefix}.con')
        
        # Save design matrix
        np.savetxt(
            mat_file, design, fmt='%0.4f', 
            header=f'/NumWaves 1\n/NumPoints {len(y_values)}\n/Matrix', 
            comments=''
        )
        
        # Save contrast file
        with open(con_file, 'w') as f:
            f.write('/NumWaves 1\n/NumContrasts 1\n/Matrix\n1\n')
        
        logger.debug("Created design matrix: %s", mat_file)
        logger.debug("Created contrast file: %s", con_file)
        
        # Merge input images
        merged = os.path.join(output_dir, f'{prefix}_4d.nii.gz')
        logger.info("Merging %d input images to %s", len(input_imgs), merged)
        
        imgs = [image.load_img(p) for p in input_imgs]
        merged_img = image.concat_imgs(imgs)
        merged_img.to_filename(merged)
        logger.info("Successfully merged files to: %s", merged)
        
        # Apply group mask
        masker = NiftiMasker(mask_img=group_mask_file, memory=work_dir)
        masked = masker.fit_transform(image.load_img(merged))
        masked_img = masker.inverse_transform(masked)
        masked_path = os.path.join(output_dir, f'{prefix}_masked.nii.gz')
        masked_img.to_filename(masked_path)
        logger.info("Applied group mask and saved to: %s", masked_path)
        
        # Run FSL Randomise
        logger.info("Running FSL Randomise with %d permutations", DEFAULT_CONFIG['num_permutations'])
        rand = Randomise()
        rand.inputs.in_file = masked_path
        rand.inputs.mask = group_mask_file
        rand.inputs.design_mat = mat_file
        rand.inputs.tcon = con_file
        rand.inputs.num_perm = DEFAULT_CONFIG['num_permutations']
        rand.inputs.tfce = DEFAULT_CONFIG['use_tfce']
        rand.inputs.base_name = os.path.join(output_dir, prefix)
        
        rand.run()
        logger.info("FSL Randomise completed successfully for %s", prefix)
        return True
        
    except Exception as e:
        logger.error("Failed to run voxel-wise regression for %s: %s", prefix, str(e))
        return False

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
    logger: logging.Logger
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """Validate subjects based on available seed-to-voxel FC files."""
    logger.info("Validating subjects in FC directory %s", fc_dir)
    
    if not os.path.exists(fc_dir):
        logger.error("Input directory %s does not exist", fc_dir)
        raise ValueError(f"Input directory {fc_dir} does not exist")
    
    # Find FC files
    fc_files = glob.glob(os.path.join(fc_dir, '*fcmap_avg.nii.gz'))
    logger.info("Found %d FC files", len(fc_files))
    
    if not fc_files:
        dir_contents = os.listdir(fc_dir)
        logger.warning("No FC files found in %s. Directory contents: %s", fc_dir, dir_contents)
    
    # Parse subject and session information
    subject_sessions = {}
    
    for f in fc_files:
        filename = os.path.basename(f)
        if '_ses-' in filename and '_task-rest_seed-PCC_fcmap_avg.nii.gz' in filename:
            parts = filename.split('_')
            if len(parts) >= 2:
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

def validate_input_files(
    file_paths: List[str], 
    group_name: str, 
    logger: logging.Logger
) -> List[str]:
    """Validate input files and return list of existing files."""
    logger.info("Validating %s input files", group_name)
    
    valid_files = []
    for path in file_paths:
        if not os.path.exists(path):
            logger.error("Missing file: %s", path)
        else:
            logger.debug("Found: %s", path)
            valid_files.append(path)
    
    if not valid_files:
        logger.warning("No valid %s input files found", group_name)
    else:
        logger.info("Found %d valid %s files", len(valid_files), group_name)
    
    return valid_files

# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def perform_group_analysis(
    valid_group: List[str],
    metadata_df: pd.DataFrame,
    output_dir: str,
    work_dir: str,
    group_mask_file: str,
    logger: logging.Logger
) -> bool:
    """Perform group difference analysis at baseline."""
    logger.info("Performing group difference analysis at baseline")
    
    # Get HC and OCD paths
    hc_paths = [
        get_fc_path(s, 'ses-baseline', output_dir) 
        for s in metadata_df[metadata_df['group'] == 'HC']['subject_id']
        if s in valid_group
    ]
    ocd_paths = [
        get_fc_path(s, 'ses-baseline', output_dir) 
        for s in metadata_df[metadata_df['group'] == 'OCD']['subject_id']
        if s in valid_group
    ]
    
    logger.info("Group analysis: HC n=%d, OCD n=%d", len(hc_paths), len(ocd_paths))
    
    if not hc_paths or not ocd_paths:
        logger.warning(
            "Insufficient data for group analysis (HC files: %d, OCD files: %d)",
            len(hc_paths), len(ocd_paths)
        )
        return False
    
    # Validate input files
    hc_valid = validate_input_files(hc_paths, "HC", logger)
    ocd_valid = validate_input_files(ocd_paths, "OCD", logger)
    
    if not hc_valid or not ocd_valid:
        logger.warning("Cannot proceed with group analysis due to missing files")
        return False
    
    # Verify directories
    if not os.path.exists(output_dir):
        logger.error("Output directory does not exist: %s", output_dir)
        return False
    
    if not os.access(output_dir, os.W_OK):
        logger.error("Cannot write to output directory: %s", output_dir)
        return False
    
    if not os.path.exists(work_dir):
        logger.error("Work directory does not exist: %s", work_dir)
        return False
    
    if not os.access(work_dir, os.W_OK):
        logger.error("Cannot write to work directory: %s", work_dir)
        return False
    
    logger.info("Directory permissions verified")
    
    # Merge HC files
    if hc_valid:
        hc_output = os.path.join(output_dir, 'hc_baseline_4d.nii.gz')
        logger.info("Merging HC files to: %s", hc_output)
        try:
            hc_imgs = [image.load_img(p) for p in hc_valid]
            merged_img = image.concat_imgs(hc_imgs)
            merged_img.to_filename(hc_output)
            logger.info("Successfully merged HC files to: %s", hc_output)
        except Exception as e:
            logger.error("Error during nilearn merge for HC files: %s", e)
            return False
    
    # Merge OCD files
    if ocd_valid:
        ocd_output = os.path.join(output_dir, 'ocd_baseline_4d.nii.gz')
        logger.info("Merging OCD files to: %s", ocd_output)
        try:
            ocd_imgs = [image.load_img(p) for p in ocd_valid]
            merged_img = image.concat_imgs(ocd_imgs)
            merged_img.to_filename(ocd_output)
            logger.info("Successfully merged OCD files to: %s", ocd_output)
        except Exception as e:
            logger.error("Error during nilearn merge for OCD files: %s", e)
            return False
    
    # Create design matrix for group difference
    if hc_valid and ocd_valid:
        logger.info("Creating design matrix for group difference analysis")
        
        # Create mat file
        mat_file = os.path.join(work_dir, 'group.mat')
        with open(mat_file, 'w') as f:
            f.write(f"/NumWaves 1\n/NumPoints {len(ocd_valid) + len(hc_valid)}\n/Matrix\n")
            f.writelines(['1\n'] * len(ocd_valid) + ['-1\n'] * len(hc_valid))
        
        # Create con file
        con_file = os.path.join(work_dir, 'group.con')
        with open(con_file, 'w') as f:
            f.write("/NumWaves 1\n/NumContrasts 1\n/Matrix\n1\n")
        
        # Merge all files for group analysis
        concat_output = os.path.join(output_dir, 'group_baseline_concat.nii.gz')
        logger.info("Merging all files to: %s", concat_output)
        
        try:
            concat_imgs = [image.load_img(p) for p in ocd_valid + hc_valid]
            merged_img = image.concat_imgs(concat_imgs)
            merged_img.to_filename(concat_output)
            logger.info("Successfully merged all files to: %s", concat_output)
        except Exception as e:
            logger.error("Error during nilearn merge for group concat: %s", e)
            return False
        
        # Run FSL Randomise for group difference
        logger.info("Running FSL Randomise for group difference analysis")
        try:
            rand = Randomise()
            rand.inputs.in_file = concat_output
            rand.inputs.mask = group_mask_file
            rand.inputs.design_mat = mat_file
            rand.inputs.tcon = con_file
            rand.inputs.num_perm = DEFAULT_CONFIG['num_permutations']
            rand.inputs.tfce = DEFAULT_CONFIG['use_tfce']
            rand.inputs.base_name = os.path.join(output_dir, 'group_diff_baseline')
            
            rand.run()
            logger.info("Group difference analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error("Failed to run group difference analysis: %s", e)
            return False
    
    return False

def perform_longitudinal_analysis(
    valid_longitudinal: List[str],
    metadata_df: pd.DataFrame,
    df_clinical: pd.DataFrame,
    output_dir: str,
    work_dir: str,
    group_mask_file: str,
    logger: logging.Logger
) -> None:
    """Perform longitudinal analysis for OCD subjects."""
    logger.info("Performing longitudinal analysis")
    
    # Prepare OCD clinical data
    ocd_df = df_clinical[
        df_clinical['subject_id'].isin(metadata_df[metadata_df['group'] == 'OCD']['subject_id'])
    ].copy()
    ocd_df['delta_ybocs'] = ocd_df['ybocs_baseline'] - ocd_df['ybocs_followup']
    
    logger.info("Longitudinal analysis: OCD subjects n=%d", len(ocd_df))
    
    if ocd_df.empty:
        logger.warning("No OCD subjects found for longitudinal analysis")
        return
    
    # 1. Baseline FC vs symptom change
    baseline_fc = [
        get_fc_path(s, 'ses-baseline', output_dir) 
        for s in ocd_df['subject_id']
    ]
    
    if baseline_fc:
        logger.info("Running baseline FC vs delta YBOCS regression")
        success = run_voxelwise_regression(
            baseline_fc, ocd_df['delta_ybocs'].tolist(), 
            'baselineFC_vs_deltaYBOCS', work_dir, output_dir, group_mask_file, logger
        )
        
        if success:
            logger.info("Baseline FC regression completed successfully")
        else:
            logger.error("Baseline FC regression failed")
    
    # 2. FC change vs symptom change
    logger.info("Analyzing FC change vs symptom change")
    change_maps = []
    deltas = []
    
    for _, row in ocd_df.iterrows():
        try:
            base_path = get_fc_path(row['subject_id'], 'ses-baseline', output_dir)
            follow_path = get_fc_path(row['subject_id'], 'ses-followup', output_dir)
            
            if not (os.path.exists(base_path) and os.path.exists(follow_path)):
                logger.warning(
                    "Missing FC files for subject %s (baseline: %s, followup: %s)",
                    row['subject_id'], os.path.exists(base_path), os.path.exists(follow_path)
                )
                continue
            
            # Load and compute difference
            base = image.load_img(base_path)
            follow = image.load_img(follow_path)
            diff = image.math_img('img2 - img1', img1=base, img2=follow)
            
            # Save difference map
            out_path = os.path.join(work_dir, f"{row['subject_id']}_fc_change.nii.gz")
            diff.to_filename(out_path)
            change_maps.append(out_path)
            deltas.append(row['delta_ybocs'])
            
            logger.debug("Created FC change map for subject %s: %s", row['subject_id'], out_path)
            
        except Exception as e:
            logger.error("Failed to process FC change for subject %s: %s", row['subject_id'], e)
            continue
    
    if change_maps:
        logger.info("Running FC change vs delta YBOCS regression with %d subjects", len(change_maps))
        success = run_voxelwise_regression(
            change_maps, deltas, 'deltaFC_vs_deltaYBOCS', 
            work_dir, output_dir, group_mask_file, logger
        )
        
        if success:
            logger.info("FC change regression completed successfully")
        else:
            logger.error("FC change regression failed")
    else:
        logger.warning("No FC change maps created for longitudinal analysis")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run seed-to-voxel functional connectivity group analysis."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.output_dir, DEFAULT_CONFIG['log_file'])
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("=" * 80)
    logger.info("Starting Seed-to-Voxel Functional Connectivity Group Analysis")
    logger.info("=" * 80)
    logger.info("Arguments: %s", vars(args))
    
    try:
        # Create output directories
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.work_dir, exist_ok=True)
        
        # Load metadata
        df, df_clinical = load_and_validate_metadata(args.subjects_csv, args.clinical_csv, logger)

        # Normalize subject IDs in metadata
        df['subject_id'] = df['subject_id'].str.replace('sub-', '')
        df_clinical['subject_id'] = df_clinical['subject_id'].str.replace('sub-', '')
        logger.debug("Normalized subject IDs in metadata")

        # Validate subjects
        valid_group, valid_longitudinal, _ = validate_subjects(args.output_dir, df, logger)
        
        if not valid_group and not valid_longitudinal:
            logger.error("No valid subjects found for any analysis")
            raise ValueError("No valid subjects found for any analysis. Check FC files in output directory.")

        # 1. Group difference at baseline
        if valid_group:
            perform_group_analysis(
                valid_group, df, args.output_dir, args.work_dir, 
                DEFAULT_CONFIG['group_mask_file'], logger
            )

        # 2. Longitudinal analyses
        if valid_longitudinal:
            perform_longitudinal_analysis(
                valid_longitudinal, df, df_clinical, args.output_dir, 
                args.work_dir, DEFAULT_CONFIG['group_mask_file'], logger
            )

        logger.info("Main analysis completed successfully")
    
    except Exception as e:
        logger.error("Main execution failed: %s", e)
        raise
    
    finally:
        logger.info("=" * 80)
        logger.info("Seed-to-Voxel Functional Connectivity Group Analysis Completed")
        logger.info("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("Main execution failed: %s", e)
        raise