import os
import glob
import numpy as np
from nilearn import image, masking
import pandas as pd
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Compute seed-based functional connectivity maps.')
parser.add_argument('--subject', type=str, required=True, help='Subject ID')
args = parser.parse_args()
subjects = [args.subject]

# Define directories
project_dir = '/project/6079231/dliang55/R01_AOCD'
bids_dir = os.path.join(project_dir, 'derivatives/fmriprep-1.4.1')
scratch_dir = '/scratch/xxqian'
output_dir = os.path.join(project_dir, 'OCD')
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
    mask_path = os.path.join(
        bids_dir, subject, session, 'func',
        f'{subject}_{session}_task-rest_space-MNI152NLin6Asym_desc-brain_mask.nii.gz'
    )
    for path in [networks_file, mask_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")
    return mask_path


def extract_pcc_roi(networks_file):
    """Extract PCC ROI from networks.nii (label 4) and resample to MNI152NLin6Asym space
    using FLIRT."""
    from nilearn import image
    from nipype.interfaces.fsl import FLIRT
    import os
    from templateflow.api import get as tpl_get, templates as get_tpl_list
    group_mask_file = str(tpl_get('MNI152NLin6Asym', resolution=2, desc='brain', suffix='mask'))
    group_mask = image.load_img(group_mask_file)
    # Load the networks image
    networks_img = image.load_img(networks_file)
    # Define output file path for resampled image
    output_file = 'networks_resampled.nii.gz'
    # Run FLIRT to resample networks image to 2mm isotropic resolution in group mask space
    flirt = FLIRT()
    flirt.inputs.in_file = networks_file
    flirt.inputs.reference = group_mask
    flirt.inputs.out_file = output_file
    flirt.inputs.apply_isoxfm = 2  # Resample to 2mm isotropic resolution
    flirt.inputs.interp = 'nearestneighbour'  # Use nearest neighbor for label data
    flirt.run()
    # Load resampled image
    networks_resampled = image.load_img(output_file)
    # PCC is 4th volume (index 3)
    pcc_mask = image.index_img(networks_resampled, 3)
    # Clean up temporary file
    os.remove(output_file)
    return pcc_mask


def process_run(fmri_file, confounds_file, seed_roi, brain_mask, output_path):
    """Process a single fMRI run to compute seed-based functional connectivity."""
    try:
        # Load fMRI data
        fmri_img = image.load_img(fmri_file)

        # Load confounds
        confounds_df = pd.read_csv(confounds_file, sep='\t')
        motion_params = confounds_df[[
            'trans_x', 'trans_y', 'trans_z',
            'rot_x', 'rot_y', 'rot_z',
            'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
            'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1'
        ]].fillna(0)

        # Compute artifact flags
        fd_flags = confounds_df['framewise_displacement'].fillna(0) > 0.5
        # Compute global signal Z-scores
        if 'global_signal' in confounds_df.columns:
            global_signal = confounds_df['global_signal'].fillna(confounds_df['global_signal'].mean())
            gs_z_scores = np.abs((global_signal - global_signal.mean()) / global_signal.std())
            gs_flags = gs_z_scores > 3
            art_flags = fd_flags | gs_flags  # Combine FD and GS flags
        else:
            art_flags = fd_flags

        # Remove artifactual volumes
        valid_timepoints = ~art_flags
        if valid_timepoints.sum() < 10:  # Minimum number of valid timepoints
            return None

        # Subset confounds for valid timepoints
        motion_params = motion_params[valid_timepoints]

        # Extract seed time series with confound regression
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

        # Extract brain time series using subject-specific mask with confound regression
        brain_masker = masking.NiftiMasker(
            mask_img=brain_mask,
            standardize='zscore',
            memory=os.path.join(work_dir, 'nilearn_cache'),
            memory_level=2,
            detrend=True,
            confounds=motion_params
        )
        brain_time_series = brain_masker.fit_transform(fmri_img)[valid_timepoints]

        # Compute seed-to-voxel correlations
        corr_matrix = np.corrcoef(seed_time_series, brain_time_series.T)
        fc_map = corr_matrix[0, 1:]
        fc_map = np.arctanh(fc_map)
        fc_img = brain_masker.inverse_transform(fc_map)
        fc_img.to_filename(output_path)

        return output_path

    except Exception:
        return None


def main():
    """Main function to process functional connectivity for all subjects and sessions."""
    # Extract PCC ROI from networks.nii (label 5 for DefaultMode.PCC)
    seed_roi = extract_pcc_roi(networks_file)

    for subject in subjects:
        for session in sessions:
            try:
                # Validate paths and get subject-specific mask
                brain_mask_path = validate_paths(subject, session)

                # Find fMRI and confounds files
                fmri_files = sorted(glob.glob(os.path.join(
                    bids_dir, subject, session, 'func',
                    f'{subject}_{session}_task-rest*_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz'
                )))
                confounds_files = sorted(glob.glob(os.path.join(
                    bids_dir, subject, session, 'func',
                    f'{subject}_{session}_task-rest*_desc-confounds_timeseries.tsv'
                )))

                if not fmri_files or not confounds_files:
                    continue

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

                # Average FC maps across runs
                if len(fc_maps) > 1:
                    fc_imgs = [image.load_img(fc_map) for fc_map in fc_maps]
                    avg_fc_img = image.mean_img(fc_imgs)
                    avg_output_path = os.path.join(
                        output_dir,
                        f'{subject}_{session}_task-rest_seed-PCC_fcmap_avg.nii.gz'
                    )
                    avg_fc_img.to_filename(avg_output_path)

            except Exception:
                continue


if __name__ == "__main__":
    main()
