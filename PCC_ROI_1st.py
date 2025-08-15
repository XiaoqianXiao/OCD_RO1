#!/usr/bin/env python3
"""
PCC-Enhanced Power 2011 Atlas ROI-to-ROI Functional Connectivity Analysis

This script computes ROI-to-ROI functional connectivity using the Power 2011 atlas
enhanced with a PCC (Posterior Cingulate Cortex) seed region. It processes individual
subjects and sessions, generating connectivity matrices and network analyses.

Author: [Your Name]
Date: [Current Date]
"""

import os
import glob
import re
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.input_data import NiftiLabelsMasker
from nilearn.datasets import fetch_coords_power_2011, load_mni152_template
from nilearn.image import resample_to_img
import argparse
import logging
from itertools import combinations
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
    'project_dir': '/project/6079231/dliang55/R01_AOCD/derivatives/fmriprep-1.4.1',
    'bids_dir': '/project/6079231/dliang55/R01_AOCD/derivatives/fmriprep-1.4.1',
    'scratch_dir': '/scratch/xxqian',
    'output_dir': '/project/6079231/dliang55/R01_AOCD',
    'work_dir': '/scratch/xxqian/work_flow',
    'roi_dir': '/scratch/xxqian/roi',
    'log_file': 'pcc_power2011_fc_analysis.log',
    'sessions': ['ses-baseline', 'ses-followup'],
    'pcc_coord': [0, -52, 26],  # MNI coordinates for PCC
    'sphere_radius': 3,
    'atlas_resolution': 2,
    'tr': 2.0,
    'low_pass': 0.1,
    'high_pass': 0.01,
    'fd_threshold': 0.5,
    'min_timepoints': 10,
    'compcor_components': 5
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
    logger = logging.getLogger('PCC_Power2011_FC')
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
        description='Compute PCC-enhanced Power 2011 atlas ROI-to-ROI functional connectivity.'
    )
    parser.add_argument('--subject', type=str, required=True, help='Subject ID (e.g., sub-AOCD001)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_CONFIG['output_dir'],
                       help='Output directory for results')
    parser.add_argument('--work_dir', type=str, default=DEFAULT_CONFIG['work_dir'],
                       help='Working directory for temporary files')
    
    return parser.parse_args()

# =============================================================================
# ATLAS MANAGEMENT
# =============================================================================

def generate_pcc_enhanced_atlas(roi_dir: str, logger: logging.Logger) -> str:
    """Generate PCC-enhanced Power 2011 atlas if it doesn't exist."""
    power_atlas_path = os.path.join(roi_dir, 'power_2011_pcc_atlas.nii.gz')
    
    if os.path.exists(power_atlas_path):
        logger.info(f"PCC-enhanced Power 2011 atlas already exists: {power_atlas_path}")
        return power_atlas_path
    
    logger.info("Generating PCC-enhanced Power 2011 atlas...")
    
    try:
        # Fetch Power 2011 coordinates
        power = fetch_coords_power_2011()
        required_fields = ['roi', 'x', 'y', 'z']
        
        if not all(field in power.rois.dtype.names for field in required_fields):
            raise ValueError(f"Power.rois missing required fields: {required_fields}")
        
        coords = np.vstack((power.rois['x'], power.rois['y'], power.rois['z'])).T
        logger.info(f"Power 2011 coordinates shape: {coords.shape} (expected: (264, 3))")
        
        # Add PCC coordinate
        pcc_coord = np.array(DEFAULT_CONFIG['pcc_coord'])
        coords = np.vstack([coords, pcc_coord])
        logger.info(f"Added PCC coordinate: {pcc_coord}")
        logger.info(f"Total coordinates: {coords.shape[0]} (264 Power + 1 PCC)")
        
        # Load MNI template
        template = load_mni152_template(resolution=DEFAULT_CONFIG['atlas_resolution'])
        atlas_data = np.zeros(template.shape, dtype=np.int32)
        
        # Create spherical ROIs
        radius = DEFAULT_CONFIG['sphere_radius']
        xx, yy, zz = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
        sphere = (xx**2 + yy**2 + zz**2 <= radius**2).astype(int)
        
        for idx, coord in enumerate(coords):
            try:
                x, y, z = coord
                voxel_coords = np.round(
                    np.linalg.inv(template.affine).dot([x, y, z, 1])[:3]
                ).astype(int)
                
                # Apply spherical mask
                start_idx = voxel_coords - radius
                end_idx = voxel_coords + radius + 1
                
                # Ensure indices are within bounds
                start_idx = np.maximum(start_idx, 0)
                end_idx = np.minimum(end_idx, template.shape)
                
                atlas_data[
                    start_idx[0]:end_idx[0],
                    start_idx[1]:end_idx[1],
                    start_idx[2]:end_idx[2]
                ] = np.maximum(
                    atlas_data[
                        start_idx[0]:end_idx[0],
                        start_idx[1]:end_idx[1],
                        start_idx[2]:end_idx[2]
                    ],
                    (idx + 1) * sphere[
                        :end_idx[0]-start_idx[0],
                        :end_idx[1]-start_idx[1],
                        :end_idx[2]-start_idx[2]
                    ]
                )
                
            except (ValueError, IndexError) as e:
                logger.warning(f"Skipping invalid coordinate at index {idx}: {coord}, error: {str(e)}")
                continue
        
        if np.all(atlas_data == 0):
            raise ValueError("Atlas generation produced empty data")
        
        # Create and save atlas
        power_atlas = image.new_img_like(template, atlas_data)
        power_atlas.to_filename(power_atlas_path)
        logger.info(f"Generated PCC-enhanced Power 2011 atlas saved to {power_atlas_path}")
        
        return power_atlas_path
        
    except Exception as e:
        logger.error(f"Failed to generate PCC-enhanced Power 2011 atlas: {str(e)}")
        raise

def load_network_labels(roi_dir: str, n_rois: int, logger: logging.Logger) -> Dict[int, str]:
    """Load network labels for Power 2011 atlas plus PCC."""
    network_labels_path = os.path.join(roi_dir, 'power264', 'power264NodeNames.txt')
    
    if not os.path.exists(network_labels_path):
        raise FileNotFoundError(f"Network labels file not found: {network_labels_path}")
    
    try:
        with open(network_labels_path, 'r') as f:
            network_labels_list = [line.strip() for line in f if line.strip()]
        
        if len(network_labels_list) != 264:
            raise ValueError(
                f"Network labels file has {len(network_labels_list)} entries, expected 264"
            )
        
        # Parse network labels for Power 2011 ROIs
        network_labels = {}
        for i, label in enumerate(network_labels_list):
            parts = label.rsplit('_', 1)
            if len(parts) != 2:
                logger.warning(f"Invalid label format at line {i+1}: {label}")
                continue
            
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
        
        # Add PCC label (ROI 265)
        network_labels[n_rois] = 'Default_mode'  # PCC assigned to Default network
        logger.info(f"Added PCC (ROI {n_rois}) to Default_mode network")
        
        logger.info(f"Loaded {len(network_labels)} network labels (264 Power + 1 PCC)")
        return network_labels
        
    except Exception as e:
        logger.error(f"Failed to load network labels: {str(e)}")
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
        ][:DEFAULT_CONFIG['compcor_components']]
        
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
            fd_flags = confounds_df['framewise_displacement'].fillna(0) > DEFAULT_CONFIG['fd_threshold']
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
            low_pass=DEFAULT_CONFIG['low_pass'],
            high_pass=DEFAULT_CONFIG['high_pass'],
            t_r=DEFAULT_CONFIG['tr'],
            confounds=motion_params[valid_timepoints] if not motion_params.empty else None
        )
        
        time_series = masker.fit_transform(
            fmri_img,
            confounds=motion_params[valid_timepoints] if not motion_params.empty else None
        )
        
        logger.info(f"Extracted time series with shape: {time_series.shape}")
        
        if time_series.shape[0] < DEFAULT_CONFIG['min_timepoints']:
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
    output_prefix: str,
    logger: logging.Logger
) -> str:
    """Compute ROI-to-ROI connectivity measures and save results."""
    try:
        # Compute correlation matrix
        logger.info("Computing correlation matrix...")
        corr_matrix = np.corrcoef(time_series.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # Save correlation matrix
        output_matrix = f"{output_prefix}_correlation_matrix.npy"
        np.save(output_matrix, corr_matrix)
        logger.info(f"Saved correlation matrix: {output_matrix}")
        
        # Create DataFrame for ROI-to-ROI FC
        logger.info("Computing ROI-to-ROI functional connectivity...")
        fc_data = []
        for i, j in combinations(range(len(roi_names)), 2):
            fc_data.append({
                'ROI1': roi_names[i],
                'ROI2': roi_names[j],
                'network1': network_labels.get(i + 1, 'Unknown'),
                'network2': network_labels.get(j + 1, 'Unknown'),
                'FC': corr_matrix[i, j]
            })
        
        # Save ROI-to-ROI FC
        output_csv = f"{output_prefix}_roi_to_roi_fc.csv"
        fc_df = pd.DataFrame(fc_data)
        fc_df.to_csv(output_csv, index=False)
        logger.info(f"Saved ROI-to-ROI FC: {output_csv} with {len(fc_data)} pairs")
        
        return output_csv
        
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
    logger: logging.Logger
) -> Optional[str]:
    """Process a single fMRI run to compute PCC-enhanced functional connectivity."""
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
        
        if valid_timepoints.sum() < DEFAULT_CONFIG['min_timepoints']:
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
        network_labels = load_network_labels(DEFAULT_CONFIG['roi_dir'], time_series.shape[1], logger)
        roi_names = [f"ROI_{i+1}" for i in range(time_series.shape[1] - 1)] + ['PCC']
        
        result = compute_connectivity_measures(
            time_series, network_labels, roi_names, output_prefix, logger
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
    """Main function to process PCC-enhanced functional connectivity analysis."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(os.path.join(args.output_dir, DEFAULT_CONFIG['log_file']))
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("=" * 80)
    logger.info("Starting PCC-Enhanced Power 2011 Atlas Functional Connectivity Analysis")
    logger.info("=" * 80)
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Create output directories
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.work_dir, exist_ok=True)
        
        # Generate or load PCC-enhanced Power 2011 atlas
        logger.info("Setting up PCC-enhanced Power 2011 atlas...")
        atlas_path = generate_pcc_enhanced_atlas(DEFAULT_CONFIG['roi_dir'], logger)
        atlas = image.load_img(atlas_path)
        logger.info(f"Atlas loaded: shape={atlas.shape}")
        
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
                    
                    # Process each run
                    for fmri_file, confounds_file in zip(fmri_files, confounds_files):
                        try:
                            # Extract run ID
                            run_id = extract_run_id(fmri_file)
                            logger.info(f"Run ID: {run_id}")
                            
                            # Define output prefix
                            output_prefix = os.path.join(
                                args.output_dir,
                                f'{subject}_{session}_task-rest_run-{run_id}_power2011_pcc'
                            )
                            
                            # Process the run
                            result = process_run(
                                fmri_file, confounds_file, atlas, brain_mask_path,
                                output_prefix, args.work_dir, logger
                            )
                            
                            if result:
                                # Rename output file to indicate it's from single runs
                                final_output = os.path.join(
                                    args.output_dir,
                                    f'{subject}_{session}_task-rest_power2011_pcc_roi_to_roi_fc.csv'
                                )
                                os.rename(result, final_output)
                                logger.info(f"Renamed output to: {final_output}")
                                processed_any = True
                            else:
                                logger.warning(f"No results generated for {subject} {session} run {run_id}")
                        
                        except Exception as e:
                            logger.error(f"Failed to process run {fmri_file}: {str(e)}")
                            continue
                
                except Exception as e:
                    logger.error(f"Failed to process {subject} {session}: {str(e)}")
                    continue
        
        if not processed_any:
            logger.error(f"No functional connectivity matrices generated for {args.subject}")
        else:
            logger.info(f"Analysis completed successfully for {args.subject}")
    
    except Exception as e:
        logger.error(f"Main function failed: {str(e)}")
        raise
    
    finally:
        logger.info("=" * 80)
        logger.info("PCC-Enhanced Power 2011 Atlas Functional Connectivity Analysis Completed")
        logger.info("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("Main execution failed: %s", e)
        raise