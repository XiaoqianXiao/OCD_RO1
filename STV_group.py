import os
import glob
import numpy as np
import pandas as pd
from nilearn import image, masking, plotting
from nipype.interfaces.fsl import Randomise, Cluster, Merge
from nipype.interfaces.fsl.utils import ImageStats
from templateflow.api import get as tpl_get
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Perform 2x2 mixed ANOVA for group and session effects on FC maps.')
parser.add_argument('--subjects_csv', type=str,
                    default='/project/6079231/dliang55/R01_AOCD/metadata/shared_demographics.csv',
                    help='CSV file with subject IDs and group labels')
parser.add_argument('--output_dir', type=str, default='/project/6079231/dliang55/R01_AOCD/OCD',
                    help='Output directory for FC maps and group results')
parser.add_argument('--work_dir', type=str, default='/scratch/xxqian/work_flow',
                    help='Working directory for intermediate files')
args = parser.parse_args()

# Define directories
output_dir = args.output_dir
work_dir = os.path.join(args.work_dir, 'group_analysis_mixed_anova')
group_output_dir = os.path.join(output_dir, 'group_analysis', 'mixed_anova')
os.makedirs(work_dir, exist_ok=True)
os.makedirs(group_output_dir, exist_ok=True)

# Load group brain mask (MNI152NLin6Asym, 2mm resolution)
group_mask_file = str(tpl_get('MNI152NLin6Asym', resolution=2, desc='brain', suffix='mask'))
group_mask = image.load_img(group_mask_file)

# Load subject metadata
subjects_df = pd.read_csv(args.subjects_csv)
subjects_df['subject_id'] = subjects_df['subject_id'].astype(str)
subjects_df = subjects_df[subjects_df['group'].isin(['HC', 'OCD'])]

# Collect FC maps for both sessions
sessions = ['ses-baseline', 'ses-followup']
fc_maps = {session: [] for session in sessions}
valid_subjects = []
for subject in subjects_df['subject_id']:
    all_sessions_found = True
    for session in sessions:
        fc_map_path = os.path.join(
            output_dir,
            f'{subject}_{session}_task-rest_seed-PCC_fcmap_avg.nii.gz'
        )
        if not os.path.exists(fc_map_path):
            fc_map_path = os.path.join(
                output_dir,
                f'{subject}_{session}_task-rest_run-1_seed-PCC_fcmap.nii.gz'
            )
        if os.path.exists(fc_map_path):
            fc_maps[session].append(fc_map_path)
        else:
            all_sessions_found = False
            print(f"FC map not found for {subject} in {session}")
            break
    if all_sessions_found:
        valid_subjects.append(subject)

if len(valid_subjects) < 5:
    raise ValueError(f"Too few subjects ({len(valid_subjects)}) with FC maps for both sessions")

# Update subjects_df to include only valid subjects
subjects_df = subjects_df[subjects_df['subject_id'].isin(valid_subjects)]

# Create design matrix for 2x2 mixed ANOVA
# Group: HC=0, OCD=1; Session: baseline=-1, followup=1
design_data = []
for subject in valid_subjects:
    group = 1 if subjects_df[subjects_df['subject_id'] == subject]['group'].iloc[0] == 'OCD' else 0
    for session, session_val in zip(sessions, [-1, 1]):
        design_data.append({
            'subject': subject,
            'group': group,
            'session': session_val,
            'group_x_session': group * session_val
        })
design_df = pd.DataFrame(design_data)

# Create design matrix file
design_matrix = design_df[['group', 'session', 'group_x_session']].values
design_matrix_file = os.path.join(work_dir, 'design.mat')
np.savetxt(design_matrix_file, design_matrix, fmt='%d', header=f'/NumWaves 3\n/NumPoints {len(design_data)}\n/Matrix',
           comments='')

# Create contrast file
contrasts = [
    [1, 0, 0],  # Group effect: OCD > HC
    [-1, 0, 0],  # Group effect: HC > OCD
    [0, 1, 0],  # Session effect: followup > baseline
    [0, -1, 0],  # Session effect: baseline > followup
    [0, 0, 1],  # Interaction: (OCD_followup - OCD_baseline) > (HC_followup - HC_baseline)
    [0, 0, -1]  # Interaction: (HC_followup - HC_baseline) > (OCD_followup - OCD_baseline)
]
contrast_file = os.path.join(work_dir, 'design.con')
with open(contrast_file, 'w') as f:
    f.write('/NumWaves 3\n/NumContrasts 6\n/Matrix\n')
    for contrast in contrasts:
        f.write(' '.join(map(str, contrast)) + '\n')

# Concatenate FC maps across sessions and subjects
all_fc_maps = []
subject_session_order = []
for subject in valid_subjects:
    for session in sessions:
        fc_map_path = os.path.join(
            output_dir,
            f'{subject}_{session}_task-rest_seed-PCC_fcmap_avg.nii.gz'
        )
        if not os.path.exists(fc_map_path):
            fc_map_path = os.path.join(
                output_dir,
                f'{subject}_{session}_task-rest_run-1_seed-PCC_fcmap.nii.gz'
            )
        all_fc_maps.append(fc_map_path)
        subject_session_order.append(f'{subject}_{session}')

# Merge FC maps into a 4D image
merger = Merge()
merger.inputs.in_files = all_fc_maps
merger.inputs.dimension = 't'
fc_4d_output = os.path.join(group_output_dir, 'group_mixed_anova_seed-PCC_fcmap_4d.nii.gz')
merger.inputs.merged_file = fc_4d_output
merger.run()

# Apply group mask
masker = masking.NiftiMasker(mask_img=group_mask, memory=os.path.join(work_dir, 'nilearn_cache'))
fc_4d_img = image.load_img(fc_4d_output)
fc_masked = masker.fit_transform(fc_4d_img)
fc_4d_masked_img = masker.inverse_transform(fc_masked)
fc_4d_masked_output = os.path.join(group_output_dir, 'group_mixed_anova_seed-PCC_fcmap_4d_masked.nii.gz')
fc_4d_masked_img.to_filename(fc_4d_masked_output)

# Perform 2x2 mixed ANOVA using FSL randomise
print("Running 2x2 mixed ANOVA...")
randomise = Randomise()
randomise.inputs.in_file = fc_4d_masked_output
randomise.inputs.mask = group_mask_file
randomise.inputs.design_mat = design_matrix_file
randomise.inputs.contrast = contrast_file
randomise.inputs.n_perm = 5000
randomise.inputs.tfce = True
randomise.inputs.vox_p_values = True
randomise.inputs.out_file = os.path.join(group_output_dir, 'group_mixed_anova_seed-PCC')
randomise_results = randomise.run()

# Threshold and visualize results
contrast_names = [
    'group_OCDgtHC', 'group_HCgtOCD',
    'session_followup_gt_baseline', 'session_baseline_gt_followup',
    'interaction_OCDdiff_gt_HCdiff', 'interaction_HCdiff_gt_OCDdiff'
]
for i, contrast_name in enumerate(contrast_names):
    tstat_file = randomise_results.outputs.tstat_files[i]
    corrp_file = randomise_results.outputs.t_corrected_p_files[i]
    thresh_output = os.path.join(group_output_dir, f'group_mixed_anova_seed-PCC_{contrast_name}_thresh.nii.gz')

    # Threshold at p < 0.05
    cluster = Cluster()
    cluster.inputs.in_file = corrp_file
    cluster.inputs.threshold = 0.95  # 1 - 0.05
    cluster.inputs.out_threshold_file = thresh_output
    cluster.run()

    # Visualize significant clusters
    plotting.plot_stat_map(
        thresh_output,
        bg_img=str(tpl_get('MNI152NLin6Asym', resolution=2, suffix='T1w')),
        output_file=os.path.join(group_output_dir, f'group_mixed_anova_seed-PCC_{contrast_name}_thresh.png'),
        threshold=0.95,
        title=f'Mixed ANOVA: {contrast_name} (p < 0.05, corrected)'
    )

    # Compute summary statistics
    stats = ImageStats()
    stats.inputs.in_file = thresh_output
    stats.inputs.op_string = '-M -v'
    stats_results = stats.run()
    summary = {
        'contrast': contrast_name,
        'n_subjects': len(valid_subjects),
        'mean_tstat': stats_results.outputs.out_stat[0],
        'n_significant_voxels': stats_results.outputs.out_stat[1]
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(group_output_dir, f'group_mixed_anova_seed-PCC_{contrast_name}_summary.csv'),
        index=False
    )

# Compute and save mean FC maps for each group and session
for group in ['HC', 'OCD']:
    for session in sessions:
        group_session_maps = [
            p for p in all_fc_maps if p.split('/')[-1].startswith(f'sub-') and session in p and
                                      subjects_df[subjects_df['subject_id'] == p.split('/')[-1].split('_')[0]][
                                          'group'].iloc[0] == group
        ]
        if group_session_maps:
            mean_fc_img = image.mean_img(group_session_maps)
            mean_fc_output = os.path.join(group_output_dir, f'group_{group}_{session}_seed-PCC_fcmap_mean.nii.gz')
            mean_fc_img.to_filename(mean_fc_output)

print(f"Mixed ANOVA analysis completed. Results saved in {group_output_dir}")