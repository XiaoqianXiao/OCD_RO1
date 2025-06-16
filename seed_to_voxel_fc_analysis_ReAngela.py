import os
import glob
import re
import numpy as np
from nilearn import image
from nilearn.input_data import NiftiMasker
import pandas as pd
import argparse
import logging

# Set up logging
log_file = '/scratch/xxqian/OCD/seed_to_voxel_fc_analysis.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Compute seed-based functional connectivity maps.')
parser.add_argument('--subject', type=str, required=True, help='Subject ID (e.g., sub-AOCD001)')
args = parser.parse_args()
subjects = [args.subject]

# Define directories
project_dir = '/project/6079231/dliang55/R01_AOCD'
bids_dir = os.path.join(project_dir, 'derivatives/fmriprep-1.4.1')
scratch_dir = '/scratch/xxqian'
output_dir = os.path.join(scratch_dir, 'OCD')
work_dir = os.path.join(scratch_dir, 'work_flow')
roi_dir = os.path.join(scratch_dir, 'roi')

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(work_dir, exist_ok=True)

# Define sessions
sessions = ['ses-baseline', 'ses-followup']

def validate_paths(subject, session):
    """Validate input paths for brain mask in MNI152NLin6Asym space."""
    mask_pattern = os.path.join(
        bids_dir, subject, session, 'func',
        f'{subject}_{session}_task-rest*_space-MNI152NLin6Asym_desc-brain_mask.nii.gz'
    )
    mask_files = glob.glob(mask_pattern)
    if not mask_files:
        logger.warning(f"No mask file found for {subject} {session} with pattern: {mask_pattern}")
        return None
    mask_path = mask_files[0]  # Use first match
    logger.info(f"Found mask: {mask_path}")
    return mask_path

def process_run(fmri_file, confounds_file, seed_roi, brain_mask, output_path):
    """Process a single fMRI run to compute seed-based functional connectivity."""
    try:
        logger.info(f"Processing run: fMRI={fmri_file}, confounds={confounds_file}, output={output_path}")
        fmri_img = image.load_img(fmri_file)
        confounds_df = pd.read_csv(confounds_file, sep='\t')
        motion_params = confounds_df[[
            'trans_x', 'trans_y', 'trans_z',
            'rot_x', 'rot_y', 'rot_z',
            'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
            'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1'
        ]].fillna(0)
        fd_flags = confounds_df['framewise_displacement'].fillna(0) > 0.5
        if 'global_signal' in confounds_df.columns:
            global_signal = confounds_df['global_signal'].fillna(confounds_df['global_signal'].mean())
            gs_z_scores = np.abs((global_signal - global_signal.mean()) / global_signal.std())
            gs_flags = gs_z_scores > 3
            art_flags = fd_flags | gs_flags
        else:
            art_flags = fd_flags
            logger.warning("Global signal not found in confounds; using FD flags only")
        valid_timepoints = ~art_flags
        logger.info(f"Valid timepoints: {valid_timepoints.sum()}/{len(valid_timepoints)}")
        if valid_timepoints.sum() < 10:
            logger.warning(f"Too few valid timepoints ({valid_timepoints.sum()}) for {fmri_file}")
            return None
        motion_params = motion_params[valid_timepoints]
        seed_masker = masking.NiftiMasker(
            mask_img=seed_roi,
            standardize='zscore',
            memory=os.path.join(work_dir, 'nilearn_cache'),
            memory_level=2,
            detrend=True,
            confounds=motion_params
        )
        seed_time_series = seed_masker.fit_transform(fmri_img)[valid_timepoints]
        seed_time_series = np.mean(seed_time_series, axis=1)
        brain_masker = masking.NiftiMasker(
            mask_img=brain_mask,
            standardize='zscore',
            memory=os.path.join(work_dir, 'nilearn_cache'),
            memory_level=2,
            detrend=True,
            confounds=motion_params
        )
        brain_time_series = brain_masker.fit_transform(fmri_img)[valid_timepoints]
        corr_matrix = np.corrcoef(seed_time_series, brain_time_series.T)
        fc_map = corr_matrix[0, 1:]
        fc_map = np.arctanh(fc_map)
        fc_img = brain_masker.inverse_transform(fc_map)
        fc_img.to_filename(output_path)
        logger.info(f"Saved FC map: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to process run {fmri_file}: {str(e)}")
        return None

def main():
    """Main function to process functional connectivity for all subjects and sessions."""
    try:
        seed_roi_path = "/home/xxqian/scratch/roi/pcc_resampled.nii.gz"
        if not os.path.exists(seed_roi_path):
            logger.error(f"Seed ROI file not found: {seed_roi_path}")
            raise FileNotFoundError(f"Seed ROI file not found: {seed_roi_path}")
        seed_roi = image.load_img(seed_roi_path)
        logger.info(f"Loaded seed ROI: {seed_roi_path}")

        for subject in subjects:
            logger.info(f"Processing subject: {subject}")
            for session in sessions:
                try:
                    logger.info(f"Processing session: {session}")
                    brain_mask_path = validate_paths(subject, session)
                    if not brain_mask_path:
                        continue
                    brain_mask = image.load_img(brain_mask_path)
                    fmri_pattern = os.path.join(
                        bids_dir, subject, session, 'func',
                        f'{subject}_{session}_task-rest*_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz'
                    )
                    confounds_pattern = os.path.join(
                        bids_dir, subject, session, 'func',
                        f'{subject}_{session}_task-rest*_desc-confounds_regressors.tsv'
                    )
                    fmri_files = sorted(glob.glob(fmri_pattern))
                    confounds_files = sorted(glob.glob(confounds_pattern))
                    logger.info(f"Found fMRI files: {fmri_files}")
                    logger.info(f"Found confounds files: {confounds_files}")
                    if not fmri_files or not confounds_files or len(fmri_files) != len(confounds_files):
                        logger.warning(f"Skipping {subject} {session}: No or mismatched fMRI/confounds files")
                        continue
                    fc_maps = []
                    for fmri_file, confounds_file in zip(fmri_files, confounds_files):
                        try:
                            # Extract run_id using regex
                            run_match = re.search(r'_run-(\d+)_', fmri_file)
                            run_id = run_match.group(1).lstrip('0') or '1' if run_match else '1'
                            logger.info(f"Parsed run_id={run_id} for fMRI file: {fmri_file}")
                            output_path = os.path.join(
                                output_dir,
                                f'{subject}_{session}_task-rest_run-{run_id}_seed-PCC_fcmap.nii.gz'
                            )
                            result = process_run(fmri_file, confounds_file, seed_roi, brain_mask, output_path)
                            if result:
                                fc_maps.append(result)
                        except Exception as e:
                            logger.error(f"Failed to parse or process fMRI file {fmri_file}: {str(e)}")
                            continue
                    if fc_maps:
                        if len(fc_maps) > 1:
                            fc_imgs = [image.load_img(fc_map) for fc_map in fc_maps]
                            avg_fc_img = image.mean_img(fc_imgs)
                            avg_output_path = os.path.join(
                                output_dir,
                                f'{subject}_{session}_task-rest_seed-PCC_fcmap_avg.nii.gz'
                            )
                            avg_fc_img.to_filename(avg_output_path)
                            logger.info(f"Saved averaged FC map: {avg_output_path}")
                        else:
                            logger.info(f"Single FC map for {subject} {session}: {fc_maps[0]}")
                    else:
                        logger.warning(f"No FC maps generated for {subject} {session}")
                except Exception as e:
                    logger.error(f"Failed to process {subject} {session}: {str(e)}")
                    continue
            if not any(os.path.exists(os.path.join(output_dir, f'{subject}_{session}_task-rest_seed-PCC_fcmap_avg.nii.gz'))
                       for session in sessions):
                logger.error(f"No FC maps generated for {subject}")
    except Exception as e:
        logger.error(f"Main function failed for {subject}: {str(e)}")
        raise

if __name__ == "__main__":
    main()