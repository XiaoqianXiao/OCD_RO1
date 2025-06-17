import os
import numpy as np
import pandas as pd
from nilearn import image, masking
from nipype.interfaces.fsl import Randomise, Merge
import argparse

# Arguments
parser = argparse.ArgumentParser(description='FC analysis: group diffs & symptom prediction')
parser.add_argument('--subjects_csv', type=str, required=True)
parser.add_argument('--clinical_csv', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--work_dir', type=str, required=True)
args = parser.parse_args()

# Setup
group_mask_file = '/home/xxqian/scratch/roi/tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz'
os.makedirs(args.work_dir, exist_ok=True)
os.makedirs(args.output_dir, exist_ok=True)

# Load metadata
df = pd.read_csv(args.subjects_csv)
df['subject_id'] = df['sub'].astype(str)
df = df[df['group'].isin(['HC', 'OCD'])]
df_clinical = pd.read_csv(args.clinical_csv)
df_clinical['subject_id'] = df_clinical['sub'].astype(str)
sessions = ['ses-baseline', 'ses-followup']

# Helpers
def get_fc_path(subject, session):
    return os.path.join(args.output_dir, f"{subject}_{session}_task-rest_seed-PCC_fcmap_avg.nii.gz")

def get_group(subject):
    return df[df['subject_id'] == subject]['group'].iloc[0]

def run_voxelwise_regression(input_imgs, y_values, prefix):
    design = np.array(y_values).reshape(-1, 1)
    mat_file = os.path.join(args.work_dir, f'{prefix}.mat')
    con_file = os.path.join(args.work_dir, f'{prefix}.con')
    np.savetxt(mat_file, design, fmt='%0.4f', header=f'/NumWaves 1\n/NumPoints {len(y_values)}\n/Matrix', comments='')
    with open(con_file, 'w') as f:
        f.write('/NumWaves 1\n/NumContrasts 1\n/Matrix\n1\n')

    merged = os.path.join(args.output_dir, f'{prefix}_4d.nii.gz')
    Merge(in_files=input_imgs, dimension='t', merged_file=merged).run()

    masker = masking.NiftiMasker(mask_img=group_mask_file, memory=args.work_dir)
    masked = masker.fit_transform(image.load_img(merged))
    masked_img = masker.inverse_transform(masked)
    masked_path = os.path.join(args.output_dir, f'{prefix}_masked.nii.gz')
    masked_img.to_filename(masked_path)

    rand = Randomise()
    rand.inputs.in_file = masked_path
    rand.inputs.mask = group_mask_file
    rand.inputs.design_mat = mat_file
    rand.inputs.contrast = con_file
    rand.inputs.n_perm = 5000
    rand.inputs.tfce = True
    rand.inputs.out_file = os.path.join(args.output_dir, prefix)
    rand.run()

# Valid subjects
valid = []
for sid in df['subject_id']:
    if all(os.path.exists(get_fc_path(sid, ses)) for ses in sessions):
        valid.append(sid)
df = df[df['subject_id'].isin(valid)]

# 1. Group diff at baseline
hc_paths = [get_fc_path(s, 'ses-baseline') for s in df[df['group'] == 'HC']['subject_id']]
ocd_paths = [get_fc_path(s, 'ses-baseline') for s in df[df['group'] == 'OCD']['subject_id']]

Merge(in_files=hc_paths, dimension='t', merged_file=os.path.join(args.output_dir, 'hc_baseline_4d.nii.gz')).run()
Merge(in_files=ocd_paths, dimension='t', merged_file=os.path.join(args.output_dir, 'ocd_baseline_4d.nii.gz')).run()

# Create mat/con for group diff
with open(os.path.join(args.work_dir, 'group.mat'), 'w') as f:
    f.write(f"/NumWaves 1\n/NumPoints {len(hc_paths) + len(ocd_paths)}\n/Matrix\n")
    f.writelines(['1\n'] * len(ocd_paths) + ['-1\n'] * len(hc_paths))
with open(os.path.join(args.work_dir, 'group.con'), 'w') as f:
    f.write("/NumWaves 1\n/NumContrasts 1\n/Matrix\n1\n")

# Merge and run group difference
Merge(in_files=ocd_paths + hc_paths, dimension='t',
      merged_file=os.path.join(args.output_dir, 'group_baseline_concat.nii.gz')).run()

rand = Randomise()
rand.inputs.in_file = os.path.join(args.output_dir, 'group_baseline_concat.nii.gz')
rand.inputs.mask = group_mask_file
rand.inputs.design_mat = os.path.join(args.work_dir, 'group.mat')
rand.inputs.contrast = os.path.join(args.work_dir, 'group.con')
rand.inputs.n_perm = 5000
rand.inputs.tfce = True
rand.inputs.out_file = os.path.join(args.output_dir, 'group_diff_baseline')
rand.run()

# 2. baseline FC vs symptom change
ocd_df = df_clinical[df_clinical['group'] == 'OCD'].copy()
ocd_df['delta_ybocs'] = ocd_df['ybocs_baseline'] - ocd_df['ybocs_followup']
baseline_fc = [get_fc_path(s, 'ses-baseline') for s in ocd_df['subject_id']]
run_voxelwise_regression(baseline_fc, ocd_df['delta_ybocs'], 'baselineFC_vs_deltaYBOCS')

# 3. FC change vs symptom change
change_maps = []
deltas = []
for _, row in ocd_df.iterrows():
    base = image.load_img(get_fc_path(row['subject_id'], 'ses-baseline'))
    follow = image.load_img(get_fc_path(row['subject_id'], 'ses-followup'))
    diff = image.math_img('img2 - img1', img1=base, img2=follow)
    out_path = os.path.join(args.work_dir, f"{row['subject_id']}_fc_change.nii.gz")
    diff.to_filename(out_path)
    change_maps.append(out_path)
    deltas.append(row['delta_ybocs'])

run_voxelwise_regression(change_maps, deltas, 'deltaFC_vs_deltaYBOCS')

print("All analyses completed.")
