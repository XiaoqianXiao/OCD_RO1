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

# Set up logging
log_file = '/scratch/xxqian/OCD/roi_to_roi_fc_analysis.log'
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
parser = argparse.ArgumentParser(description='Compute ROI-to-ROI functional connectivity using Power 2011 atlas.')
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

# Load or generate Power 2011 atlas
power_atlas_path = os.path.join(roi_dir, 'power_2011_atlas.nii.gz')
if not os.path.exists(power_atlas_path):
    logger.info("Generating Power 2011 atlas from coordinates")
    power = fetch_coords_power_2011()
    required_fields = ['roi', 'x', 'y', 'z']
    if not all(field in power.rois.dtype.names for field in required_fields):
        logger.error(f"Power.rois missing required fields {required_fields}: {power.rois.dtype.names}")
        raise ValueError("Invalid Power 2011 atlas data structure")
    coords = np.vstack((power.rois['x'], power.rois['y'], power.rois['z'])).T
    logger.info(f"Power 2011 coordinates shape: {coords.shape} (expected: (264, 3))")
    template = load_mni152_template(resolution=2)
    atlas_data = np.zeros(template.shape, dtype=np.int32)
    for idx, coord in enumerate(coords):
        try:
            x, y, z = coord
            logger.debug(f"Processing ROI {idx + 1}: ({x}, {y}, {z})")
            voxel_coords = np.round(np.linalg.inv(template.affine).dot([x, y, z, 1])[:3]).astype(int)
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
        except (ValueError, IndexError) as e:
            logger.error(f"Skipping invalid coordinate at index {idx}: {coord}, error: {str(e)}")
            continue
    if np.all(atlas_data == 0):
        logger.error("Failed to generate atlas: all data is zero")
        raise ValueError("Atlas generation produced empty data")
    power_atlas = image.new_img_like(template, atlas_data)
    power_atlas.to_filename(power_atlas_path)
    logger.info(f"Generated Power 2011 atlas saved to {power_atlas_path}")

# Load atlas and verify
atlas = image.load_img(power_atlas_path)
logger.info(f"Loaded Power 2011 atlas: {power_atlas_path}, shape: {atlas.shape}, affine: {atlas.affine.tolist()}")

# Load network labels and coordinates
power = fetch_coords_power_2011()
n_rois = len(power.rois)  # 264 ROIs
required_fields = ['roi', 'x', 'y', 'z']
if not all(field in power.rois.dtype.names for field in required_fields):
    logger.error(f"Power.rois missing required fields {required_fields}: {power.rois.dtype.names}")
    raise ValueError("Invalid Power 2011 atlas data structure")
if not np.array_equal(power.rois['roi'], np.arange(1, n_rois + 1)):
    logger.error("Power.rois['roi'] does not contain 1–264 in order")
    raise ValueError("Invalid ROI numbers in power.rois")
network_labels_path = os.path.join(roi_dir, 'power264', 'power264NodeNames.txt')
if not os.path.exists(network_labels_path):
    logger.error(f"Network labels file not found: {network_labels_path}")
    raise FileNotFoundError(f"Missing {network_labels_path}")
try:
    with open(network_labels_path, 'r') as f:
        network_labels_list = [line.strip() for line in f if line.strip()]
    if len(network_labels_list) != n_rois:
        logger.error(f"Network labels file has {len(network_labels_list)} entries, expected {n_rois}")
        raise ValueError(f"Invalid number of network labels in {network_labels_path}")
    network_labels = {}
    for i, (label, roi_num) in enumerate(zip(network_labels_list, power.rois['roi'])):
        parts = label.rsplit('_', 1)
        if len(parts) != 2:
            logger.error(f"Invalid label format in {network_labels_path} at line {i+1}: {label}")
            raise ValueError(f"Invalid label format: {label}")
        network_name, txt_roi_num = parts
        try:
            txt_roi_num = int(txt_roi_num)
        except ValueError:
            logger.error(f"Invalid ROI number in {network_labels_path} at line {i+1}: {txt_roi_num}")
            raise ValueError(f"Invalid ROI number: {txt_roi_num}")
        if txt_roi_num != roi_num:
            logger.error(f"ROI number mismatch in {network_labels_path} at line {i+1}: {txt_roi_num} != {roi_num}")
            raise ValueError(f"ROI number mismatch: {txt_roi_num} != {roi_num}")
        network_labels[i + 1] = network_name
    logger.info(f"Loaded {len(network_labels)} network labels from {network_labels_path}")
except Exception as e:
    logger.error(f"Failed to load network labels from {network_labels_path}: {str(e)}")
    raise
coords = np.vstack((power.rois['x'], power.rois['y'], power.rois['z'])).T
roi_names = [f"ROI_{i+1}" for i in range(n_rois)]

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
    if not mask_files:
        logger.warning(f"No mask file found for {subject} {session} with pattern: {mask_pattern}")
        return None, None
    if not fmri_files:
        logger.warning(f"No fMRI file found for {subject} {session} with pattern: {fmri_pattern}")
        return None, None
    mask_path = mask_files[0]
    fmri_path = fmri_files[0]  # Use first match
    logger.info(f"Found mask: {mask_path}, fMRI: {fmri_path}")
    return mask_path, fmri_path

def resample_to_atlas_space(img, target_img, output_path, interpolation='continuous'):
    """Resample an image to the space of the target image."""
    try:
        resampled_img = resample_to_img(img, target_img, interpolation=interpolation)
        resampled_img.to_filename(output_path)
        logger.info(f"Resampled image to {output_path}, shape: {resampled_img.shape}, affine: {resampled_img.affine.tolist()}")
        return resampled_img
    except Exception as e:
        logger.error(f"Failed to resample image to {output_path}: {str(e)}")
        raise

def process_run(fmri_file, confounds_file, atlas, brain_mask, output_prefix):
    """Process a single fMRI run to compute ROI-to-ROI functional connectivity."""
    try:
        logger.info(f"Processing run: fMRI={fmri_file}, confounds={confounds_file}, output={output_prefix}")
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
        logger.info(f"Valid timepoints: {valid_timepoints.sum()}/{len(valid_timepoints)}")
        if valid_timepoints.sum() < 10:
            logger.warning(f"Too few valid timepoints ({valid_timepoints.sum()}) for {fmri_file}")
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
        logger.info(f"Saved correlation matrix: {output_matrix}")

        # Compute within- and between-network FC for each ROI
        roi_fc = []
        for i in range(n_rois):
            net_i = network_labels.get(i + 1, 'Unknown')
            if net_i == 'Unknown':
                continue
            within_corrs = []
            between_corrs = []
            for j in range(n_rois):
                if i == j:
                    continue
                net_j = network_labels.get(j + 1, 'Unknown')
                if net_j == 'Unknown':
                    continue
                corr = corr_matrix[i, j]
                if net_i == net_j:
                    within_corrs.append(corr)
                else:
                    between_corrs.append(corr)
            within_mean = np.mean(within_corrs) if within_corrs else np.nan
            between_mean = np.mean(between_corrs) if between_corrs else np.nan
            roi_fc.append({
                'network_name': net_i,
                'roi_name': roi_names[i],
                'within_network_FC': within_mean,
                'between_network_FC': between_mean
            })

        # Save ROI-level FC
        output_roi_csv = f"{output_prefix}_roi_fc.csv"
        pd.DataFrame(roi_fc).to_csv(output_roi_csv, index=False)
        logger.info(f"Saved ROI-level FC: {output_roi_csv}")

        # Compute network-level FC
        unique_networks = sorted(set(network_labels.values()) - {'Unknown'})
        network_fc = {net: {'within': [], 'between': {net2: [] for net2 in unique_networks if net2 != net}}
                      for net in unique_networks}
        for i, j in combinations(range(n_rois), 2):
            net_i = network_labels.get(i + 1, 'Unknown')
            net_j = network_labels.get(j + 1, 'Unknown')
            if net_i == 'Unknown' or net_j == 'Unknown':
                continue
            corr = corr_matrix[i, j]
            if net_i == net_j:
                network_fc[net_i]['within'].append(corr)
            else:
                network_fc[net_i]['between'][net_j].append(corr)
                network_fc[net_j]['between'][net_i].append(corr)

        # Save network-level pairwise FC in required format
        pairwise_network_fc = []
        for net1, net2 in combinations(unique_networks, 2):
            corrs = network_fc[net1]['between'][net2]
            if corrs:
                mean_fc = np.mean(corrs)
                pairwise_network_fc.append({
                    'ROI': f"{net1}_{net2}",
                    'network1': net1,
                    'network2': net2,
                    'fc_value': mean_fc
                })
        for net in unique_networks:
            corrs = network_fc[net]['within']
            if corrs:
                mean_fc = np.mean(corrs)
                pairwise_network_fc.append({
                    'ROI': f"{net}_{net}",
                    'network1': net,
                    'network2': net,
                    'fc_value': mean_fc
                })

        # Save pairwise network FC
        output_pairwise_csv = f"{output_prefix}_network_fc_avg.csv"
        pairwise_df = pd.DataFrame(pairwise_network_fc)
        if not pairwise_df.empty:
            pairwise_df.to_csv(output_pairwise_csv, index=False)
            logger.info(f"Saved pairwise network FC: {output_pairwise_csv} with {len(pairwise_df)} entries")
        else:
            logger.warning(f"No pairwise network FC data to save for {output_pairwise_csv}")

        # Save existing network summary
        network_summary = []
        for net in unique_networks:
            within_mean = np.mean(network_fc[net]['within']) if network_fc[net]['within'] else np.nan
            row = {'Network': net, 'Within_Network_FC': within_mean}
            for other_net in network_fc[net]['between']:
                between_mean = np.mean(network_fc[net]['between'][other_net]) if network_fc[net]['between'][other_net] else np.nan
                row[f'Between_{other_net}_FC'] = between_mean
            network_summary.append(row)

        output_network_csv = f"{output_prefix}_network_summary.csv"
        pd.DataFrame(network_summary).to_csv(output_network_csv, index=False)
        logger.info(f"Saved network FC summary: {output_network_csv}")

        return output_matrix, output_roi_csv, output_pairwise_csv, output_network_csv
    except Exception as e:
        logger.error(f"Failed to process run {fmri_file}: {str(e)}")
        return None

def main():
    """Main function to process ROI-to-ROI functional connectivity for all subjects and sessions."""
    try:
        logger.info(f"Loaded Power 2011 atlas: {power_atlas_path}")

        for subject in subjects:
            logger.info(f"Processing subject: {subject}")
            for session in sessions:
                try:
                    logger.info(f"Processing session: {session}")
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
                        logger.warning(f"No confounds file found for {subject} {session}")
                        continue
                    confounds_file = confounds_files[0]  # Use first match
                    logger.info(f"Found confounds file: {confounds_file}")
                    run_match = re.search(r'_run-(\d+)_', fmri_file)
                    run_id = run_match.group(1).lstrip('0') or '1' if run_match else '1'
                    logger.info(f"Parsed run_id={run_id} for fMRI file: {fmri_file}")
                    output_prefix = os.path.join(
                        output_dir,
                        f'{subject}_{session}_task-rest_run-{run_id}_power2011'
                    )
                    result = process_run(fmri_file, confounds_file, atlas, brain_mask_path, output_prefix)
                    if result:
                        results.append(result)
                    if results:
                        src_matrix, src_roi_csv, src_pairwise_csv, src_network_csv = results[0]
                        avg_matrix_output = os.path.join(
                            output_dir,
                            f'{subject}_{session}_task-rest_power2011_roiroi_matrix_avg.npy'
                        )
                        os.rename(src_matrix, avg_matrix_output)
                        logger.info(f"Moved single correlation matrix to: {avg_matrix_output}")
                        avg_roi_csv = os.path.join(
                            output_dir,
                            f'{subject}_{session}_task-rest_power2011_roi_fc_avg.csv'
                        )
                        os.rename(src_roi_csv, avg_roi_csv)
                        logger.info(f"Moved single ROI-level FC to: {avg_roi_csv}")
                        avg_pairwise_csv = os.path.join(
                            output_dir,
                            f'{subject}_{session}_task-rest_power2011_network_fc_avg.csv'
                        )
                        os.rename(src_pairwise_csv, avg_pairwise_csv)
                        logger.info(f"Moved single pairwise network FC to: {avg_pairwise_csv}")
                        avg_network_csv = os.path.join(
                            output_dir,
                            f'{subject}_{session}_task-rest_power2011_network_summary_avg.csv'
                        )
                        os.rename(src_network_csv, avg_network_csv)
                        logger.info(f"Moved single network summary to: {avg_network_csv}")
                except Exception as e:
                    logger.error(f"Failed to process {subject} {session}: {str(e)}")
                    continue
            if not any(os.path.exists(os.path.join(output_dir, f'{subject}_{session}_task-rest_power2011_roiroi_matrix_avg.npy'))
                       for session in sessions):
                logger.error(f"No FC matrices generated for {subject}")
    except Exception as e:
        logger.error(f"Main function failed for {subject}: {str(e)}")
        raise

if __name__ == "__main__":
    main()