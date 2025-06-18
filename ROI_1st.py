import os
import glob
import re
import numpy as np
import pandas as pd
from nilearn import image, datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import resample_to_img
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Compute ROI-to-ROI functional connectivity using Harvard-Oxford Atlas.')
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

# Fetch Harvard-Oxford atlas for label information
atlas_labels = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
roi_labels = atlas_labels.labels[1:]  # Exclude background (index 0)
n_rois = len(roi_labels)
roi_names = [f"ROI_{label.replace(' ', '_')}" for label in roi_labels]

# Load local Harvard-Oxford atlas
harvard_oxford_atlas_path = os.path.join(roi_dir, 'HarvardOxford-cort-maxprob-thr25-2mm.nii.gz')
if not os.path.exists(harvard_oxford_atlas_path):
    raise FileNotFoundError(f"Missing {harvard_oxford_atlas_path}")
atlas = image.load_img(harvard_oxford_atlas_path)

def validate_paths(subject, session):
    """Validate and return paths for fMRI and brain mask."""
    mask_pattern = os.path.join(
        bids_dir, subject, session, 'func',
        f'{subject}_{session}_task-rest*_space-MNI152NLin6Asym_desc-brain_mask.nii.gz'
    )
    fmri_pattern = os.path.join(
        bids_dir, subject, session, 'func',
        f'{subject}_{session}_task-rest*_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz'
    )
    mask_files = glob.glob(mask_pattern)
    fmri_files = glob.glob(fmri_pattern)
    if not mask_files or not fmri_files:
        return None, None
    mask_path = mask_files[0]
    fmri_path = fmri_files[0]  # Use first match
    return mask_path, fmri_path

def resample_to_atlas_space(img, target_img, output_path, interpolation='continuous'):
    """Resample an image to the space of the target image."""
    resampled_img = resample_to_img(img, target_img, interpolation=interpolation)
    resampled_img.to_filename(output_path)
    return resampled_img

def process_run(fmri_file, confounds_file, atlas, brain_mask, output_prefix):
    """Process a single fMRI run to compute ROI-to-ROI functional connectivity."""
    try:
        # Resample fMRI and mask to atlas space
        fmri_img = image.load_img(fmri_file)
        brain_mask_img = image.load_img(brain_mask)
        fmri_resampled_path = os.path.join(work_dir, f"{output_prefix}_resampled_fmri.nii.gz")
        mask_resampled_path = os.path.join(work_dir, f"{output_prefix}_resampled_mask.nii.gz")
        fmri_img = resample_to_atlas_space(fmri_img, atlas, fmri_resampled_path, interpolation='continuous')
        brain_mask_img = resample_to_atlas_space(brain_mask_img, atlas, mask_resampled_path, interpolation='nearest')

        # Load confounds
        confounds_df = pd.read_csv(confounds_file, sep='\t')
        compcor_cols = [col for col in confounds_df.columns if 'a_compcor' in col][:5]  # Top 5 aCompCor
        motion_params = confounds_df[compcor_cols + [
            'trans_x', 'trans_y', 'trans_z',
            'rot_x', 'rot_y', 'rot_z',
            'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
            'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1'
        ]].fillna(0)
        fd_flags = confounds_df['framewise_displacement'].fillna(0) > 0.5
        art_flags = fd_flags  # Simplified; add global signal if needed
        valid_timepoints = ~art_flags
        if valid_timepoints.sum() < 10:
            return None

        # Extract ROI time series
        masker = NiftiLabelsMasker(
            labels_img=atlas,
            mask_img=brain_mask_img,
            standardize='zscore',
            memory=os.path.join(work_dir, 'nilearn_cache'),
            memory_level=1,
            detrend=True,
            low_pass=0.1,
            high_pass=0.01,
            t_r=2.0,
            confounds=motion_params[valid_timepoints]
        )
        time_series = masker.fit_transform(fmri_img)[valid_timepoints]

        # Compute Pearson correlation matrix
        corr_matrix = np.corrcoef(time_series.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Save correlation matrix
        output_matrix = f"{output_prefix}_roiroi_matrix.npy"
        np.save(output_matrix, corr_matrix)

        # Compute pair-wise FC for each unique ROI pair
        pairwise_fc = []
        for i, j in combinations(range(n_rois), 2):
            pairwise_fc.append({
                'ROI_1': roi_names[i],
                'ROI_2': roi_names[j],
                'FC': corr_matrix[i, j]
            })

        # Save pair-wise FC
        output_pairwise_csv = f"{output_prefix}_pairwise_fc.csv"
        pd.DataFrame(pairwise_fc).to_csv(output_pairwise_csv, index=False)

        return output_matrix, output_pairwise_csv
    except Exception:
        return None

def main():
    """Main function to process ROI-to-ROI functional connectivity for all subjects and sessions."""
    try:
        for subject in subjects:
            for session in sessions:
                brain_mask_path, fmri_file = validate_paths(subject, session)
                if not brain_mask_path or not fmri_file:
                    continue
                results = []
                confounds_pattern = os.path.join(
                    bids_dir, subject, session, 'func',
                    f'{subject}_{session}_task-rest*_desc-confounds_regressors.tsv'
                )
                confounds_files = sorted(glob.glob(confounds_pattern))
                if not confounds_files:
                    continue
                confounds_file = confounds_files[0]  # Use first match
                run_match = re.search(r'_run-(\d+)_', fmri_file)
                run_id = run_match.group(1).lstrip('0') or '1' if run_match else '1'
                output_prefix = os.path.join(
                    output_dir,
                    f'{subject}_{session}_task-rest_run-{run_id}_harvardoxford'
                )
                result = process_run(fmri_file, confounds_file, atlas, brain_mask_path, output_prefix)
                if result:
                    results.append(result)
                if results:
                    src_matrix, src_pairwise_csv = results[0]
                    avg_matrix_output = os.path.join(
                        output_dir,
                        f'{subject}_{session}_task-rest_harvardoxford_roiroi_matrix_avg.npy'
                    )
                    os.rename(src_matrix, avg_matrix_output)
                    avg_pairwise_csv = os.path.join(
                        output_dir,
                        f'{subject}_{session}_task-rest_harvardoxford_pairwise_fc_avg.csv'
                    )
                    os.rename(src_pairwise_csv, avg_pairwise_csv)
    except Exception:
        raise

if __name__ == "__main__":
    main()