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
parser = argparse.ArgumentParser(description='Compute ROI-to-ROI and ROI-to-network functional connectivity using Power 2011 atlas.')
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
unique_networks = sorted(set(network_labels.values()) - {'Unknown'})

def validate_paths(subject, session):
    """Validate and return paths for fMRI, brain mask, and confounds."""
    mask_pattern = os.path.join(
        bids_dir, subject, session, 'func',
        f'{subject}_{session}_task-rest*_space-MNI152NLin6Asym_desc-brain_mask.nii.gz'
    )
    fmri_pattern = os.path.join(
        bids_dir, subject, session, 'func',
        f'{subject}_{session}_task-rest*_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz'
    )
    confounds_pattern = os.path.join(
        bids_dir, subject, session, 'func',
        f'{subject}_{session}_task-rest*_desc-confounds_regressors.tsv'
    )
    mask_files = glob.glob(mask_pattern)
    fmri_files = glob.glob(fmri_pattern)
    confounds_files = glob.glob(confounds_pattern)
    if not mask_files:
        logger.warning(f"No mask file found for {subject} {session} with pattern: {mask_pattern}")
        return None, None, None
    if not fmri_files:
        logger.warning(f"No fMRI file found for {subject} {session} with pattern: {fmri_pattern}")
        return None, None, None
    if not confounds_files:
        logger.warning(f"No confounds file found for {subject} {session} with pattern: {confounds_pattern}")
        return None, None, None
    mask_path = mask_files[0]
    fmri_path = fmri_files[0]
    confounds_path = confounds_files[0]
    logger.info(f"Found mask: {mask_path}, fMRI: {fmri_path}, confounds: {confounds_path}")
    return mask_path, fmri_path, confounds_path

def validate_confounds(confounds_file):
    """Validate confounds file for required columns."""
    try:
        confounds_df = pd.read_csv(confounds_file, sep='\t')
        required_cols = ['framewise_displacement']
        compcor_cols = [col for col in confounds_df.columns if 'a_compcor' in col]
        motion_cols = [
            'trans_x', 'trans_y', 'trans_z',
            'rot_x', 'rot_y', 'rot_z',
            'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
            'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1'
        ]
        missing_cols = [col for col in required_cols if col not in confounds_df.columns]
        available_motion = [col for col in motion_cols if col in confounds_df.columns]
        logger.debug(f"Confounds file {confounds_file}: {len(compcor_cols)} aCompCor cols, {len(available_motion)} motion cols")
        if missing_cols or len(compcor_cols) < 5 or len(available_motion) < 6:
            logger.error(f"Confounds file {confounds_file} missing required columns: {missing_cols}, "
                        f"found {len(compcor_cols)} aCompCor, {len(available_motion)} motion cols")
            return None
        return confounds_df
    except Exception as e:
        logger.error(f"Failed to validate confounds file {confounds_file}: {str(e)}")
        return None

def resample_to_atlas_space(img, target_img, output_path, interpolation='continuous'):
    """Resample an image to the space of the target image."""
    try:
        img_data = img.get_fdata()
        if np.any(np.isnan(img_data)) or np.any(np.isinf(img_data)):
            logger.error(f"Invalid data in image {img}: contains NaN or Inf")
            return None
        resampled_img = resample_to_img(img, target_img, interpolation=interpolation)
        resampled_img.to_filename(output_path)
        logger.info(f"Resampled image to {output_path}, shape: {resampled_img.shape}, affine: {resampled_img.affine.tolist()}")
        return resampled_img
    except Exception as e:
        logger.error(f"Failed to resample image to {output_path}: {str(e)}")
        return None

def process_run(fmri_file, confounds_file, atlas, brain_mask, output_prefix):
    """Process a single fMRI run to compute ROI-to-ROI and ROI-to-network functional connectivity."""
    try:
        logger.info(f"Processing run: fMRI={fmri_file}, confounds={confounds_file}, output={output_prefix}")
        # Validate inputs
        fmri_img = image.load_img(fmri_file)
        brain_mask_img = image.load_img(brain_mask)
        logger.debug(f"fMRI shape: {fmri_img.shape}, mask shape: {brain_mask_img.shape}")
        confounds_df = validate_confounds(confounds_file)
        if confounds_df is None:
            logger.error(f"Invalid confounds file {confounds_file}, skipping run")
            return None

        # Resample fMRI and mask to atlas space
        logger.debug("Starting resampling")
        fmri_resampled_path = os.path.join(work_dir, f"{output_prefix}_resampled_fmri.nii.gz")
        mask_resampled_path = os.path.join(work_dir, f"{output_prefix}_resampled_mask.nii.gz")
        fmri_img = resample_to_atlas_space(fmri_img, atlas, fmri_resampled_path, interpolation='continuous')
        if fmri_img is None:
            logger.error("fMRI resampling failed, skipping run")
            return None
        brain_mask_img = resample_to_atlas_space(brain_mask_img, atlas, mask_resampled_path, interpolation='nearest')
        if brain_mask_img is None:
            logger.error("Mask resampling failed, skipping run")
            return None
        logger.debug("Resampling completed")

        # Load confounds
        logger.debug("Processing confounds")
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
        logger.debug("Confounds processed")

        # Extract ROI time series
        logger.debug("Extracting time series")
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
        time_series = masker.fit_transform(fmri_img, confounds=motion_params[valid_timepoints])
        logger.info(f"Extracted time series with shape: {time_series.shape}")
        if time_series.shape[0] < 10:
            logger.error(f"Time series too short ({time_series.shape[0]} timepoints), skipping run")
            return None
        logger.debug("Time series extraction completed")

        # Compute Pearson correlation matrix
        logger.debug("Computing correlation matrix")
        corr_matrix = np.corrcoef(time_series.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        logger.info(f"Computed correlation matrix with shape: {corr_matrix.shape}")

        # Save correlation matrix
        output_matrix = f"{output_prefix}_roiroi_matrix.npy"
        np.save(output_matrix, corr_matrix)
        logger.info(f"Saved correlation matrix: {output_matrix}")

        # Compute and save ROI-to-ROI FC
        logger.debug("Computing ROI-to-ROI FC")
        roiroi_fc = []
        for i, j in combinations(range(n_rois), 2):
            net_i = network_labels.get(i + 1, 'Unknown')
            net_j = network_labels.get(j + 1, 'Unknown')
            if net_i == 'Unknown' or net_j == 'Unknown':
                continue
            corr = corr_matrix[i, j]
            roiroi_fc.append({
                'ROI': f"{roi_names[i]}_{roi_names[j]}",
                'network1': net_i,
                'network2': net_j,
                'FC': corr
            })
        output_roiroi_csv = f"{output_prefix}_roiroi_fc.csv"
        roiroi_df = pd.DataFrame(roiroi_fc)
        if not roiroi_df.empty:
            roiroi_df.to_csv(output_roiroi_csv, index=False)
            logger.info(f"Saved ROI-to-ROI FC: {output_roiroi_csv} with {len(roiroi_df)} entries")
        else:
            logger.warning(f"No ROI-to-ROI FC data to save for {output_roiroi_csv}")

        # Compute ROI-to-network FC (average FC of each ROI to each network)
        logger.debug("Computing ROI-to-network FC")
        roi_network_fc = []
        for i in range(n_rois):
            net_i = network_labels.get(i + 1, 'Unknown')
            if net_i == 'Unknown':
                continue
            for net_j in unique_networks:
                # Collect correlations between ROI i and all ROIs in network net_j
                corrs = [corr_matrix[i, j] for j in range(n_rois)
                         if network_labels.get(j + 1, 'Unknown') == net_j and i != j]
                if corrs:
                    mean_fc = np.mean(corrs)
                    roi_network_fc.append({
                        'ROI': f"{roi_names[i]}_{net_j}",
                        'network1': net_i,
                        'network2': net_j,
                        'fc_value': mean_fc
                    })
                else:
                    logger.debug(f"No correlations for ROI {roi_names[i]} to network {net_j}")

        # Save ROI-to-network FC
        output_pairwise_csv = f"{output_prefix}_network_fc_avg.csv"
        pairwise_df = pd.DataFrame(roi_network_fc)
        if not pairwise_df.empty:
            pairwise_df.to_csv(output_pairwise_csv, index=False)
            logger.info(f"Saved ROI-to-network FC: {output_pairwise_csv} with {len(pairwise_df)} entries")
        else:
            logger.warning(f"No ROI-to-network FC data to save for {output_pairwise_csv}")

        # Compute within- and between-network FC for each ROI
        logger.debug("Computing ROI-level FC")
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

        # Save network summary
        logger.debug("Saving network summary")
        network_summary = []
        for net in unique_networks:
            within_corrs = [corr_matrix[i, j] for i in range(n_rois) for j in range(n_rois)
                            if i != j and network_labels.get(i + 1, 'Unknown') == net
                            and network_labels.get(j + 1, 'Unknown') == net]
            within_mean = np.mean(within_corrs) if within_corrs else np.nan
            row = {'Network': net, 'Within_Network_FC': within_mean}
            for other_net in unique_networks:
                if other_net == net:
                    continue
                between_corrs = [corr_matrix[i, j] for i in range(n_rois) for j in range(n_rois)
                                 if network_labels.get(i + 1, 'Unknown') == net
                                 and network_labels.get(j + 1, 'Unknown') == other_net]
                between_mean = np.mean(between_corrs) if between_corrs else np.nan
                row[f'Between_{other_net}_FC'] = between_mean
            network_summary.append(row)

        output_network_csv = f"{output_prefix}_network_summary.csv"
        pd.DataFrame(network_summary).to_csv(output_network_csv, index=False)
        logger.info(f"Saved network FC summary: {output_network_csv}")

        logger.info(f"Run processing completed successfully for {output_prefix}")
        return output_matrix, output_roi_csv, output_roiroi_csv, output_pairwise_csv, output_network_csv
    except Exception as e:
        logger.error(f"Failed to process run {fmri_file}: {str(e)}")
        return None

def main():
    """Main function to process ROI-to-ROI and ROI-to-network functional connectivity for all subjects and sessions."""
    try:
        logger.info(f"Loaded Power 2011 atlas: {power_atlas_path}")
        for subject in subjects:
            logger.info(f"Processing subject: {subject}")
            for session in sessions:
                try:
                    logger.info(f"Processing session: {session}")
                    brain_mask_path, fmri_file, confounds_file = validate_paths(subject, session)
                    if not brain_mask_path or not fmri_file or not confounds_file:
                        continue
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
                        src_matrix, src_roi_csv, src_roiroi_csv, src_pairwise_csv, src_network_csv = result
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
                        avg_roiroi_csv = os.path.join(
                            output_dir,
                            f'{subject}_{session}_task-rest_power2011_roiroi_fc_avg.csv'
                        )
                        os.rename(src_roiroi_csv, avg_roiroi_csv)
                        logger.info(f"Moved single ROI-to-ROI FC to: {avg_roiroi_csv}")
                        avg_pairwise_csv = os.path.join(
                            output_dir,
                            f'{subject}_{session}_task-rest_power2011_network_fc_avg.csv'
                        )
                        os.rename(src_pairwise_csv, avg_pairwise_csv)
                        logger.info(f"Moved single ROI-to-network FC to: {avg_pairwise_csv}")
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
        logger.error(f"Main function failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()