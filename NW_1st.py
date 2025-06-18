import os
import glob
import re
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.input_data import NiftiLabelsMasker
from nilearn.datasets import fetch_coords_power_2011, load_mni152_template
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

# Load Power 2011 atlas
power_atlas_path = os.path.join(roi_dir, 'power_2011_atlas.nii.gz')
if not os.path.exists(power_atlas_path):
    logger.info("Generating Power 2011 atlas from coordinates")
    power = fetch_coords_power_2011()
    # Verify recarray fields
    required_fields = ['x', 'y', 'z']
    if not all(field in power.rois.dtype.names for field in required_fields):
        logger.error(f"Power 2011 rois recarray missing required fields: {required_fields}")
        raise ValueError("Invalid Power 2011 atlas data structure")
    # Extract x, y, z coordinates as a NumPy array
    coords = np.vstack((power.rois['x'], power.rois['y'], power.rois['z'])).T
    logger.info(f"Power 2011 coordinates shape: {coords.shape} (expected: (264, 3))")
    logger.info(f"Sample coordinates: {coords[:5].tolist()}")
    template = load_mni152_template(resolution=2)
    atlas_data = np.zeros(template.shape, dtype=np.int32)
    for idx, coord in enumerate(coords):
        try:
            x, y, z = coord  # Unpack exactly three values
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

# Load network labels and coordinates
power = fetch_coords_power_2011()
# Check if 'network' field exists in rois recarray
if 'network' not in power.rois.dtype.names:
    logger.error("Power 2011 rois recarray missing 'network' field")
    raise ValueError("Cannot generate network labels without 'network' field")
network_labels = {i + 1: net for i, net in enumerate(power.rois['network'])}  # 1-based indexing
coords = np.vstack((power.rois['x'], power.rois['y'], power.rois['z'])).T  # Consistent coordinate extraction
n_rois = len(power.rois)  # 264 ROIs
roi_names = [f"ROI_{i+1}" for i in range(n_rois)]  # Simple ROI names

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
    mask_path = mask_files[0]
    logger.info(f"Found mask: {mask_path}")
    return mask_path

def process_run(fmri_file, confounds_file, atlas, brain_mask, output_prefix):
    """Process a single fMRI run to compute ROI-to-ROI functional connectivity."""
    try:
        logger.info(f"Processing run: fMRI={fmri_file}, confounds={confounds_file}, output={output_prefix}")
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

        # Extract ROI time series
        masker = NiftiLabelsMasker(
            labels_img=atlas,
            mask_img=brain_mask,
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

        # Compute network-level FC (previous functionality)
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

        network_summary = []
        for net in unique_networks:
            within_mean = np.mean(network_fc[net]['within']) if network_fc[net]['within'] else np.nan
            row = {'Network': net, 'Within_Network_FC': within_mean}
            for other_net in network_fc[net]['between']:
                between_mean = np.mean(network_fc[net]['between'][other_net]) if network_fc[net]['between'][other_net] else np.nan
                row[f'Between_{other_net}_ doFC'] = between_mean
            network_summary.append(row)

        output_network_csv = f"{output_prefix}_network_fc.csv"
        pd.DataFrame(network_summary).to_csv(output_network_csv, index=False)
        logger.info(f"Saved network FC summary: {output_network_csv}")

        return output_matrix, output_roi_csv, output_network_csv
    except Exception as e:
        logger.error(f"Failed to process run {fmri_file}: {str(e)}")
        return None

def main():
    """Main function to process ROI-to-ROI functional connectivity for all subjects and sessions."""
    try:
        atlas = image.load_img(power_atlas_path)
        logger.info(f"Loaded Power 2011 atlas: {power_atlas_path}")

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
                    results = []
                    for fmri_file, confounds_file in zip(fmri_files, confounds_files):
                        try:
                            run_match = re.search(r'_run-(\d+)_', fmri_file)
                            run_id = run_match.group(1).lstrip('0') or '1' if run_match else '1'
                            logger.info(f"Parsed run_id={run_id} for fMRI file: {fmri_file}")
                            output_prefix = os.path.join(
                                output_dir,
                                f'{subject}_{session}_task-rest_run-{run_id}_power2011'
                            )
                            result = process_run(fmri_file, confounds_file, atlas, brain_mask, output_prefix)
                            if result:
                                results.append(result)
                        except Exception as e:
                            logger.error(f"Failed to parse or process fMRI file {fmri_file}: {str(e)}")
                            continue
                    if results and len(results) > 1:
                        # Average correlation matrices
                        matrices = [np.load(result[0]) for result in results]
                        avg_matrix = np.mean(matrices, axis=0)
                        avg_matrix_output = os.path.join(
                            output_dir,
                            f'{subject}_{session}_task-rest_power2011_roiroi_matrix_avg.npy'
                        )
                        np.save(avg_matrix_output, avg_matrix)
                        logger.info(f"Saved averaged correlation matrix: {avg_matrix_output}")

                        # Average ROI-level FC
                        roi_dfs = [pd.read_csv(result[1]) for result in results]
                        avg_roi_df = pd.DataFrame({
                            'network_name': roi_dfs[0]['network_name'],
                            'roi_name': roi_dfs[0]['roi_name'],
                            'within_network_FC': np.mean([df['within_network_FC'] for df in roi_dfs], axis=0),
                            'between_network_FC': np.mean([df['between_network_FC'] for df in roi_dfs], axis=0)
                        })
                        avg_roi_csv = os.path.join(
                            output_dir,
                            f'{subject}_{session}_task-rest_power2011_roi_fc_avg.csv'
                        )
                        avg_roi_df.to_csv(avg_roi_csv, index=False)
                        logger.info(f"Saved averaged ROI-level FC: {avg_roi_csv}")

                        # Average network-level FC
                        network_dfs = [pd.read_csv(result[2]) for result in results]
                        avg_network_df = pd.DataFrame({
                            col: np.mean([df[col] for df in network_dfs], axis=0) if col != 'Network' else network_dfs[0]['Network']
                            for col in network_dfs[0].columns
                        })
                        avg_network_csv = os.path.join(
                            output_dir,
                            f'{subject}_{session}_task-rest_power2011_network_fc_avg.csv'
                        )
                        avg_network_df.to_csv(avg_network_csv, index=False)
                        logger.info(f"Saved averaged network FC: {avg_network_csv}")
                    elif results:
                        # Move single run outputs to avg paths
                        src_matrix, src_roi_csv, src_network_csv = results[0]
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
                        avg_network_csv = os.path.join(
                            output_dir,
                            f'{subject}_{session}_task-rest_power2011_network_fc_avg.csv'
                        )
                        os.rename(src_network_csv, avg_network_csv)
                        logger.info(f"Moved single network FC to: {avg_network_csv}")
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