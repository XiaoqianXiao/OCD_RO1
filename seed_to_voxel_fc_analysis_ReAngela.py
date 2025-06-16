import os
import glob
import numpy as np
from nilearn import image, masking
import pandas as pd
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/scratch/xxqian/OCD/seed_to_voxel_fc_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Compute seed-based functional connectivity maps.')
parser.add_argument('--subject', type=str, required=True, help='Subject ID')
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

# Define path to CONN networks file
networks_file = os.path.join(roi_dir, 'networks.nii')
sessions = ['ses-baseline', 'ses-followup']

def validate_paths(subject, session):
    """Validate input paths exist for subject-specific mask and networks file."""
    mask_pattern = os.path.join(
        bids_dir, subject, session, 'func',
        f'{subject}_{session}*_task-rest_space-MNI152NLin6Asym_desc-brain_mask.nii.gz'
    )
    mask_files = glob.glob(mask_pattern)

    # Validate mask path
    if not mask_files:
        logger.error(f"No mask file found for {subject} {session} at: {mask_files[0]}")
        raise FileNotFoundError(f"No mask file found at: {mask_files[0]}")
    mask_path = mask_files[0]  # Use the first match, assuming a single mask per subject/session

    # Validate networks file
    if not os.path.exists(networks_file):
        logger.error(f"Networks file does not exist for {subject} {session}: {networks_file}")
        raise FileNotFoundError(f"Networks file not found: {networks_file}")

    # Log successful validation
    logger.info(f"Validated paths for {subject} {session}: mask={mask_path}, networks={networks_file}")
    return mask_path

def extract_pcc_roi(networks_file):
    """Extract PCC ROI from networks.nii (label 4) and resample to MNI152NLin6Asym space using FLIRT."""
    from nilearn import image
    from nipype.interfaces.fsl import FLIRT
    try:
        group_mask_file = os.path.join(roi_dir, 'tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz')
        pcc_temp_file = os.path.join(work_dir, 'pcc_temp.nii.gz')
        output_file = os.path.join(work_dir, 'pcc_resampled.nii.gz')
        pcc_mask = image.index_img(networks_file, 3)
        pcc_mask.to_filename(pcc_temp_file)
        # Resample PCC ROI with FLIRT
        flirt = FLIRT()
        flirt.inputs.in_file = pcc_temp_file
        flirt.inputs.reference = group_mask_file
        flirt.inputs.out_file = output_file
        flirt.inputs.apply_isoxfm = 2
        flirt.inputs.interp = 'nearestneighbour'
        flirt.run()
        logger.info(f"Resampled PCC ROI to {output_file}")
        pcc_mask = image.load_img(output_file)
        return pcc_mask
    except Exception as e:
        logger.error(f"Failed to extract PCC ROI: {str(e)}")
        raise

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
        seed_roi = "/home/xxqian/scratch/roi/pcc_resampled.nii.gz"
        for subject in subjects:
            logger.info(f"Processing subject: {subject}")
            for session in sessions:
                try:
                    logger.info(f"Processing session: {session}")
                    brain_mask_path = validate_paths(subject, session)
                    fmri_files = sorted(glob.glob(os.path.join(
                        bids_dir, subject, session, 'func',
                        f'{subject}_{session}_task-rest*_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz'
                    )))
                    confounds_files = sorted(glob.glob(os.path.join(
                        bids_dir, subject, session, 'func',
                        f'{subject}_{session}_task-rest*_desc-confounds_timeseries.tsv'
                    )))
                    if not fmri_files or not confounds_files:
                        logger.warning(f"No fMRI or confounds files found for {subject} {session}")
                        continue
                    logger.info(f"Found {len(fmri_files)} fMRI files and {len(confounds_files)} confounds files")
                    fc_maps = []
                    for fmri_file, confounds_file in zip(fmri_files, confounds_files):
                        run_id = fmri_file.split('_run-')[1].split('_')[0] if '_run-' in fmri_file else '1'
                        output_path = os.path.join(
                            output_dir,
                            f'{subject}_{session}_task-rest_run-{run_id}_seed-PCC_fcmap.nii.gz'
                        )
                        result = process_run(fmri_file, confounds_file, seed_roi, brain_mask_path, output_path)
                        if result:
                            fc_maps.append(result)
                    if len(fc_maps) > 1:
                        fc_imgs = [image.load_img(fc_map) for fc_map in fc_maps]
                        avg_fc_img = image.mean_img(fc_imgs)
                        avg_output_path = os.path.join(
                            output_dir,
                            f'{subject}_{session}_task-rest_seed-PCC_fcmap_avg.nii.gz'
                        )
                        avg_fc_img.to_filename(avg_output_path)
                        logger.info(f"Saved averaged FC map: {avg_output_path}")
                except Exception as e:
                    logger.error(f"Failed to process {subject} {session}: {str(e)}")
                    continue
    except Exception as e:
        logger.error(f"Main function failed: {str(e)}")

if __name__ == "__main__":
    main()