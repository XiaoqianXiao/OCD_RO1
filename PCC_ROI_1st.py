import os
import glob
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.input_data import NiftiLabelsMasker
from nilearn.datasets import fetch_coords_power_2011, load_mni152_template
from nilearn.image import resample_to_img
import argparse
from itertools import combinations

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Compute ROI-to-ROI functional connectivity with Power 2011 atlas and PCC.')
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

# Load or generate Power 2011 atlas with PCC
power_atlas_path = os.path.join(roi_dir, 'power_2011_pcc_atlas.nii.gz')
if not os.path.exists(power_atlas_path):
    power = fetch_coords_power_2011()
    coords = np.vstack((power.rois['x'], power.rois['y'], power.rois['z'])).T
    # Add PCC coordinate (MNI: [0, -52, 26])
    pcc_coord = np.array([[0, -52, 26]])
    coords = np.vstack([coords, pcc_coord])
    template = load_mni152_template(resolution=2)
    atlas_data = np.zeros(template.shape, dtype=np.int32)
    for idx, coord in enumerate(coords):
        voxel_coords = np.round(np.linalg.inv(template.affine).dot([*coord, 1])[:3]).astype(int)
        xx, yy, zz = np.ogrid[-3:4, -3:4, -3:4]
        sphere = (xx**2 + yy**2 + zz**2 <= 3**2).astype(int)
        atlas_data[
            voxel_coords[0]-3:voxel_coords[0]+4,
            voxel_coords[1]-3:voxel_coords[1]+4,
            voxel_coords[2]-3:voxel_coords[2]+4
        ] = np.maximum(atlas_data[
            voxel_coords[0]-3:voxel_coords[0]+4,
            voxel_coords[1]-3:voxel_coords[1]+4,
            voxel_coords[2]-3:voxel_coords[2]+4
        ], (idx + 1) * sphere)
    power_atlas = image.new_img_like(template, atlas_data)
    power_atlas.to_filename(power_atlas_path)

# Load atlas
atlas = image.load_img(power_atlas_path)

# Load network labels and coordinates
power = fetch_coords_power_2011()
n_rois = len(power.rois) + 1  # 264 + PCC
network_labels_path = os.path.join(roi_dir, 'power264', 'power264NodeNames.txt')
with open(network_labels_path, 'r') as f:
    network_labels_list = [line.strip() for line in f if line.strip()]
network_labels = {}
for i, label in enumerate(network_labels_list):
    parts = label.rsplit('_', 1)
    network_name, _ = parts
    network_labels[i + 1] = network_name
network_labels[n_rois] = 'Default_mode'  # PCC assigned to Default network
coords = np.vstack((power.rois['x'], power.rois['y'], power.rois['z'])).T
coords = np.vstack([coords, pcc_coord])
roi_names = [f"ROI_{i+1}" for i in range(n_rois - 1)] + ['PCC']

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
    return mask_files[0], fmri_files[0]

def resample_to_atlas_space(img, target_img, output_path, interpolation='continuous'):
    """Resample an image to the space of the target image."""
    resampled_img = resample_to_img(img, target_img, interpolation=interpolation)
    resampled_img.to_filename(output_path)
    return resampled_img

def process_run(fmri_file, confounds_file, atlas, brain_mask, output_prefix):
    """Process a single fMRI run to compute ROI-to-ROI functional connectivity."""
    # Resample fMRI and mask to atlas space
    fmri_img = image.load_img(fmri_file)
    brain_mask_img = image.load_img(brain_mask)
    fmri_resampled_path = os.path.join(work_dir, f"{output_prefix}_resampled_fmri.nii.gz")
    mask_resampled_path = os.path.join(work_dir, f"{output_prefix}_resampled_mask.nii.gz")
    fmri_img = resample_to_atlas_space(fmri_img, atlas, fmri_resampled_path, interpolation='continuous')
    brain_mask_img = resample_to_atlas_space(brain_mask_img, atlas, mask_resampled_path, interpolation='nearest')

    # Load confounds
    confounds_df = pd.read_csv(confounds_file, sep='\t')
    compcor_cols = [col for col in confounds_df.columns if 'a_compcor' in col][:5]
    motion_params = confounds_df[compcor_cols + [
        'trans_x', 'trans_y', 'trans_z',
        'rot_x', 'rot_y', 'rot_z',
        'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
        'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1'
    ]].fillna(0)
    valid_timepoints = ~(confounds_df['framewise_displacement'].fillna(0) > 0.5)
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

    # Create DataFrame for ROI-to-ROI FC
    fc_data = []
    for i, j in combinations(range(n_rois), 2):
        fc_data.append({
            'ROI1': roi_names[i],
            'ROI2': roi_names[j],
            'network1': network_labels[i + 1],
            'network2': network_labels[j + 1],
            'FC': corr_matrix[i, j]
        })
    fc_df = pd.DataFrame(fc_data)

    # Save DataFrame
    output_csv = f"{output_prefix}_roi_to_roi_fc.csv"
    fc_df.to_csv(output_csv, index=False)

    return output_csv

def main():
    """Main function to process ROI-to-ROI functional connectivity."""
    for subject in subjects:
        for session in sessions:
            brain_mask_path, fmri_file = validate_paths(subject, session)
            if not brain_mask_path or not fmri_file:
                continue
            confounds_pattern = os.path.join(
                bids_dir, subject, session, 'func',
                f'{subject}_{session}_task-rest*_desc-confounds_regressors.tsv'
            )
            confounds_files = sorted(glob.glob(confounds_pattern))
            if not confounds_files:
                continue
            confounds_file = confounds_files[0]
            run_match = re.search(r'_run-(\d+)_', fmri_file)
            run_id = run_match.group(1).lstrip('0') or '1' if run_match else '1'
            output_prefix = os.path.join(
                output_dir,
                f'{subject}_{session}_task-rest_run-{run_id}_power2011_pcc'
            )
            result = process_run(fmri_file, confounds_file, atlas, brain_mask_path, output_prefix)
            if result:
                os.rename(result, os.path.join(
                    output_dir,
                    f'{subject}_{session}_task-rest_power2011_pcc_roi_to_roi_fc.csv'
                ))

if __name__ == "__main__":
    main()