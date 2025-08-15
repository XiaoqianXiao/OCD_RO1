#!/usr/bin/env python3
"""
Seed-to-Voxel Functional Connectivity Analysis

This script computes seed-based functional connectivity maps using a predefined seed ROI
(PCC - Posterior Cingulate Cortex) for resting-state fMRI data. It processes individual
subjects and sessions, generating connectivity maps and averaged results.

Author: [Your Name]
Date: [Current Date]
"""

import os
import glob
import re
import numpy as np
import shutil
from nilearn import image
from nilearn.input_data import NiftiMasker
import pandas as pd
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
    'project_dir': '/project/6079231/dliang55/R01_AOCD',
    'bids_dir': '/project/6079231/dliang55/R01_AOCD/derivatives/fmriprep-1.4.1',
    'scratch_dir': '/scratch/xxqian',
    'output_dir': '/project/6079231/dliang55/R01_AOCD',
    'work_dir': '/scratch/xxqian/work_flow',
    'roi_dir': '/scratch/xxqian/roi',
    'log_file': 'seed_to_voxel_fc_analysis.log',
    'sessions': ['ses-baseline', 'ses-followup'],
    'seed_roi_path': '/home/xxqian/scratch/roi/pcc_resampled.nii.gz',
    'tr': 2.0,
    'low_pass': 0.1,
    'high_pass': 0.01,
    'fd_threshold': 0.5,
    'gs_threshold': 3.0,
    'min_timepoints': 10
}

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
    logger = logging.getLogger('Seed_to_Voxel_FC')
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
        description='Compute seed-based functional connectivity maps using PCC seed ROI.'
    )
    parser.add_argument('--subject', type=str, required=True, help='Subject ID (e.g., sub-AOCD001)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--seed_roi', type=str, default=DEFAULT_CONFIG['seed_roi_path'], 
                       help='Path to seed ROI file')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_CONFIG['output_dir'],
                       help='Output directory for results')
    parser.add_argument('--work_dir', type=str, default=DEFAULT_CONFIG['work_dir'],
                       help='Working directory for temporary files')
    
    return parser.parse_args()

# =============================================================================
# FILE VALIDATION AND PATH MANAGEMENT
# =============================================================================

def validate_paths(subject: str, session: str, bids_dir: str, logger: logging.Logger) -> Optional[Tuple[str, str, str]]:
    """Validate and return paths for brain mask, fMRI, and confounds files."""
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
        
        file_paths[file_type] = sorted(files)
    
    # Validate that we have matching numbers of files
    if len(file_paths['fmri']) != len(file_paths['confounds']):
        logger.warning(f"Mismatched fMRI ({len(file_paths['fmri'])}) and confounds ({len(file_paths['confounds'])}) files for {subject} {session}")
        return None
    
    logger.info(f"Found files for {subject} {session}:")
    for file_type, paths in file_paths.items():
        logger.info(f"  {file_type}: {len(paths)} files")
    
    return (
        file_paths['mask'][0],  # Use first mask file
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
# CONFOUNDS PROCESSING
# =============================================================================

def process_confounds(confounds_file: str, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.Series]:
    """Process confounds file and return motion parameters and valid timepoints."""
    try:
        confounds_df = pd.read_csv(confounds_file, sep='\t')
        
        # Select motion parameters
        motion_cols = [
            'trans_x', 'trans_y', 'trans_z',
            'rot_x', 'rot_y', 'rot_z',
            'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
            'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1'
        ]
        available_motion = [col for col in motion_cols if col in confounds_df.columns]
        
        if not available_motion:
            logger.warning(f"No motion parameters found in {confounds_file}")
            motion_params = pd.DataFrame(index=confounds_df.index)
        else:
            motion_params = confounds_df[available_motion].fillna(0)
        
        # Create motion flags
        if 'framewise_displacement' in confounds_df.columns:
            fd_flags = confounds_df['framewise_displacement'].fillna(0) > DEFAULT_CONFIG['fd_threshold']
        else:
            logger.warning("No framewise_displacement column found, assuming no excessive motion")
            fd_flags = pd.Series([False] * len(confounds_df))
        
        # Create global signal flags
        gs_flags = pd.Series([False] * len(confounds_df))
        if 'global_signal' in confounds_df.columns:
            global_signal = confounds_df['global_signal'].fillna(confounds_df['global_signal'].mean())
            gs_z_scores = np.abs((global_signal - global_signal.mean()) / global_signal.std())
            gs_flags = gs_z_scores > DEFAULT_CONFIG['gs_threshold']
            logger.debug(f"Global signal flags: {gs_flags.sum()}/{len(gs_flags)}")
        else:
            logger.warning("Global signal not found in confounds; using FD flags only")
        
        # Combine artifact flags
        art_flags = fd_flags | gs_flags
        valid_timepoints = ~art_flags
        
        logger.info(f"Confounds processed: {len(available_motion)} motion parameters")
        logger.info(f"Valid timepoints: {valid_timepoints.sum()}/{len(valid_timepoints)}")
        
        return motion_params, valid_timepoints
        
    except Exception as e:
        logger.error(f"Failed to process confounds: {str(e)}")
        raise

# =============================================================================
# SEED-TO-VOXEL FUNCTIONAL CONNECTIVITY COMPUTATION
# =============================================================================

def extract_seed_time_series(
    fmri_img: image.Nifti1Image,
    seed_roi: image.Nifti1Image,
    motion_params: pd.DataFrame,
    valid_timepoints: pd.Series,
    work_dir: str,
    logger: logging.Logger
) -> Optional[np.ndarray]:
    """Extract seed ROI time series using NiftiMasker."""
    try:
        seed_masker = NiftiMasker(
            mask_img=seed_roi,
            standardize='zscore',
            memory=os.path.join(work_dir, 'nilearn_cache'),
            memory_level=1,
            detrend=True,
            low_pass=DEFAULT_CONFIG['low_pass'],
            high_pass=DEFAULT_CONFIG['high_pass'],
            t_r=DEFAULT_CONFIG['tr'],
            confounds=motion_params[valid_timepoints] if not motion_params.empty else None
        )
        
        seed_time_series = seed_masker.fit_transform(fmri_img)[valid_timepoints]
        seed_time_series = np.mean(seed_time_series, axis=1)  # Average across seed voxels
        
        logger.info(f"Extracted seed time series with shape: {seed_time_series.shape}")
        
        if seed_time_series.shape[0] < DEFAULT_CONFIG['min_timepoints']:
            logger.error(f"Seed time series too short ({seed_time_series.shape[0]} timepoints)")
            return None
        
        return seed_time_series
        
    except Exception as e:
        logger.error(f"Failed to extract seed time series: {str(e)}")
        return None

def extract_brain_time_series(
    fmri_img: image.Nifti1Image,
    brain_mask: image.Nifti1Image,
    motion_params: pd.DataFrame,
    valid_timepoints: pd.Series,
    work_dir: str,
    logger: logging.Logger
) -> Optional[np.ndarray]:
    """Extract brain voxel time series using NiftiMasker."""
    try:
        brain_masker = NiftiMasker(
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
        
        brain_time_series = brain_masker.fit_transform(fmri_img)[valid_timepoints]
        
        logger.info(f"Extracted brain time series with shape: {brain_time_series.shape}")
        
        if brain_time_series.shape[0] < DEFAULT_CONFIG['min_timepoints']:
            logger.error(f"Brain time series too short ({brain_time_series.shape[0]} timepoints)")
            return None
        
        return brain_time_series, brain_masker
        
    except Exception as e:
        logger.error(f"Failed to extract brain time series: {str(e)}")
        return None

def compute_seed_to_voxel_fc(
    seed_time_series: np.ndarray,
    brain_time_series: np.ndarray,
    brain_masker: NiftiMasker,
    output_path: str,
    logger: logging.Logger
) -> Optional[str]:
    """Compute seed-to-voxel functional connectivity map."""
    try:
        T = seed_time_series.shape[0]  # Number of valid timepoints
        
        # Ensure time series are standardized
        brain_time_series_std = brain_time_series - brain_time_series.mean(axis=0)
        seed_time_series_std = seed_time_series - seed_time_series.mean()
        
        brain_time_series_std /= brain_time_series.std(axis=0)
        seed_time_series_std /= seed_time_series.std()
        
        # Compute Pearson correlation using dot product
        logger.info("Computing seed-to-voxel functional connectivity...")
        correlations = np.dot(brain_time_series_std.T, seed_time_series_std) / (T - 1)
        
        # Create FC map image
        fc_img = brain_masker.inverse_transform(correlations)
        fc_img.to_filename(output_path)
        
        logger.info(f"Saved FC map: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to compute seed-to-voxel FC: {str(e)}")
        return None

# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_run(
    fmri_file: str,
    confounds_file: str,
    seed_roi: image.Nifti1Image,
    brain_mask: image.Nifti1Image,
    output_path: str,
    work_dir: str,
    logger: logging.Logger
) -> Optional[str]:
    """Process a single fMRI run to compute seed-based functional connectivity."""
    try:
        logger.info(f"Processing run: {os.path.basename(fmri_file)}")
        
        # Validate input files
        for file_path in [fmri_file, confounds_file]:
            if not os.path.exists(file_path):
                logger.error(f"File does not exist: {file_path}")
                return None
        
        # Load fMRI image
        fmri_img = image.load_img(fmri_file)
        
        # Process confounds
        logger.info("Processing confounds...")
        motion_params, valid_timepoints = process_confounds(confounds_file, logger)
        
        if valid_timepoints.sum() < DEFAULT_CONFIG['min_timepoints']:
            logger.error(f"Too few valid timepoints ({valid_timepoints.sum()})")
            return None
        
        # Extract seed time series
        logger.info("Extracting seed time series...")
        seed_time_series = extract_seed_time_series(
            fmri_img, seed_roi, motion_params, valid_timepoints, work_dir, logger
        )
        
        if seed_time_series is None:
            return None
        
        # Extract brain time series
        logger.info("Extracting brain time series...")
        brain_result = extract_brain_time_series(
            fmri_img, brain_mask, motion_params, valid_timepoints, work_dir, logger
        )
        
        if brain_result is None:
            return None
        
        brain_time_series, brain_masker = brain_result
        
        # Compute functional connectivity
        logger.info("Computing seed-to-voxel functional connectivity...")
        result = compute_seed_to_voxel_fc(
            seed_time_series, brain_time_series, brain_masker, output_path, logger
        )
        
        logger.info(f"Run processing completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Failed to process run: {str(e)}")
        return None

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to process seed-to-voxel functional connectivity analysis."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(os.path.join(args.output_dir, DEFAULT_CONFIG['log_file']))
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("=" * 80)
    logger.info("Starting Seed-to-Voxel Functional Connectivity Analysis")
    logger.info("=" * 80)
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Seed ROI: {args.seed_roi}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Create output directories
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.work_dir, exist_ok=True)
        
        # Load seed ROI
        logger.info("Loading seed ROI...")
        if not os.path.exists(args.seed_roi):
            logger.error(f"Seed ROI file not found: {args.seed_roi}")
            raise FileNotFoundError(f"Seed ROI file not found: {args.seed_roi}")
        
        seed_roi = image.load_img(args.seed_roi)
        logger.info(f"Seed ROI loaded: shape={seed_roi.shape}")
        
        # Process each session
        subjects = [args.subject]
        processed_any = False
        
        for subject in subjects:
            logger.info(f"Processing subject: {subject}")
            
            for session in DEFAULT_CONFIG['sessions']:
                try:
                    logger.info(f"Processing session: {session}")
                    
                    # Validate paths
                    paths = validate_paths(subject, session, DEFAULT_CONFIG['bids_dir'], logger)
                    if not paths:
                        logger.warning(f"Skipping session {session} for {subject}")
                        continue
                    
                    brain_mask_path, fmri_files, confounds_files = paths
                    brain_mask = image.load_img(brain_mask_path)
                    
                    # Process each run
                    fc_maps = []
                    for fmri_file, confounds_file in zip(fmri_files, confounds_files):
                        try:
                            # Extract run ID
                            run_id = extract_run_id(fmri_file)
                            logger.info(f"Run ID: {run_id}")
                            
                            # Define output path
                            output_path = os.path.join(
                                args.output_dir,
                                f'{subject}_{session}_task-rest_run-{run_id}_seed-PCC_fcmap.nii.gz'
                            )
                            
                            # Process the run
                            result = process_run(
                                fmri_file, confounds_file, seed_roi, brain_mask,
                                output_path, args.work_dir, logger
                            )
                            
                            if result:
                                fc_maps.append(result)
                            
                        except Exception as e:
                            logger.error(f"Failed to process run {fmri_file}: {str(e)}")
                            continue
                    
                    # Average FC maps across runs
                    if fc_maps:
                        avg_output_path = os.path.join(
                            args.output_dir,
                            f'{subject}_{session}_task-rest_seed-PCC_fcmap_avg.nii.gz'
                        )
                        
                        if len(fc_maps) > 1:
                            logger.info(f"Averaging {len(fc_maps)} FC maps...")
                            fc_imgs = [image.load_img(fc_map) for fc_map in fc_maps]
                            avg_fc_img = image.mean_img(fc_imgs)
                            avg_fc_img.to_filename(avg_output_path)
                            logger.info(f"Saved averaged FC map: {avg_output_path}")
                        else:
                            # Move the single FC map to the avg_output_path
                            shutil.move(fc_maps[0], avg_output_path)
                            logger.info(f"Moved single FC map to: {avg_output_path}")
                        
                        processed_any = True
                    else:
                        logger.warning(f"No FC maps generated for {subject} {session}")
                
                except Exception as e:
                    logger.error(f"Failed to process {subject} {session}: {str(e)}")
                    continue
        
        if not processed_any:
            logger.error(f"No functional connectivity maps generated for {args.subject}")
        else:
            logger.info(f"Analysis completed successfully for {args.subject}")
    
    except Exception as e:
        logger.error(f"Main function failed: {str(e)}")
        raise
    
    finally:
        logger.info("=" * 80)
        logger.info("Seed-to-Voxel Functional Connectivity Analysis Completed")
        logger.info("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("Main execution failed: %s", e)
        raise