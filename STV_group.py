import os
import numpy as np
import pandas as pd
from nilearn import image, masking
from nipype.interfaces.fsl import Randomise
import argparse
import glob

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
print(f"Subject IDs in CSV: {df['subject_id'].tolist()}")

# Helpers
def get_fc_path(subject, session):
    # Handle subject IDs with or without 'sub-' prefix
    if not subject.startswith('sub-'):
        subject = f"sub-{subject}"
    return os.path.join(args.output_dir, f"{subject}_{session}_task-rest_seed-PCC_fcmap_avg.nii.gz")

def get_group(subject):
    # Handle subject IDs with or without 'sub-' prefix
    if subject.startswith('sub-'):
        subject = subject.replace('sub-', '')
    return df[df['subject_id'] == subject]['group'].iloc[0]

def run_voxelwise_regression(input_imgs, y_values, prefix):
    design = np.array(y_values).reshape(-1, 1)
    mat_file = os.path.join(args.work_dir, f'{prefix}.mat')
    con_file = os.path.join(args.work_dir, f'{prefix}.con')
    np.savetxt(mat_file, design, fmt='%0.4f', header=f'/NumWaves 1\n/NumPoints {len(y_values)}\n/Matrix', comments='')
    with open(con_file, 'w') as f:
        f.write('/NumWaves 1\n/NumContrasts 1\n/Matrix\n1\n')

    merged = os.path.join(args.output_dir, f'{prefix}_4d.nii.gz')
    try:
        imgs = [image.load_img(p) for p in input_imgs]
        merged_img = image.concat_imgs(imgs)
        merged_img.to_filename(merged)
        print(f"Successfully merged files to: {merged}")
    except Exception as e:
        print(f"Error during nilearn merge for {prefix}: {e}")
        raise

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

# Valid subjects: Only include subjects with *fcmap_avg.nii.gz files
fc_files = glob.glob(os.path.join(args.output_dir, '*fcmap_avg.nii.gz'))
print(f"Found {len(fc_files)} *fcmap_avg.nii.gz files: {fc_files}")
subject_sessions = {}
for f in fc_files:
    filename = os.path.basename(f)
    if '_ses-' in filename and '_task-rest_seed-PCC_fcmap_avg.nii.gz' in filename:
        parts = filename.split('_')
        if len(parts) >= 2:
            subject = parts[0].replace('sub-', '')  # e.g., AOCD001
            session = parts[1]  # e.g., ses-baseline
            if subject not in subject_sessions:
                subject_sessions[subject] = []
            subject_sessions[subject].append(session)
print(f"Subject sessions: {subject_sessions}")

# Warn about subjects with files but not in CSV
csv_subjects = set(df['subject_id'])
file_subjects = set(subject_sessions.keys())
unmatched = file_subjects - csv_subjects
if unmatched:
    print(f"Warning: Found files for subjects not in subjects_csv: {unmatched}")

# Filter subjects for group analysis (requires only ses-baseline)
valid_group = []
for sid in df['subject_id']:
    sid_clean = sid.replace('sub-', '')
    if sid_clean in subject_sessions and 'ses-baseline' in subject_sessions[sid_clean]:
        valid_group.append(sid)
print(f"Valid subjects for group analysis: {valid_group}")

# Filter subjects for longitudinal analysis (requires both sessions)
valid_longitudinal = []
for sid in df['subject_id']:
    sid_clean = sid.replace('sub-', '')
    if sid_clean in subject_sessions and all(ses in subject_sessions[sid_clean] for ses in sessions):
        valid_longitudinal.append(sid)
print(f"Valid subjects for longitudinal analysis: {valid_longitudinal}")

if not valid_group:
    print("Warning: No subjects found with *fcmap_avg.nii.gz files for ses-baseline. Skipping group analysis.")
if not valid_longitudinal:
    print("Warning: No subjects found with *fcmap_avg.nii.gz files for both sessions. Skipping longitudinal analysis.")

# Filter DataFrame for group analysis
df = df[df['subject_id'].isin(valid_group)]
df_clinical = df_clinical[df_clinical['subject_id'].isin(valid_longitudinal)]

# 1. Group diff at baseline
if valid_group:
    hc_paths = [get_fc_path(s, 'ses-baseline') for s in df[df['group'] == 'HC']['subject_id']]
    ocd_paths = [get_fc_path(s, 'ses-baseline') for s in df[df['group'] == 'OCD']['subject_id']]

    # Validate input files
    print("Validating HC input files:")
    for path in hc_paths:
        if not os.path.exists(path):
            print(f"Error: Missing file: {path}")
        else:
            print(f"Found: {path}")
    if not hc_paths:
        print("Warning: No HC input files found. Skipping HC merge.")
    print("Validating OCD input files:")
    for path in ocd_paths:
        if not os.path.exists(path):
            print(f"Error: Missing file: {path}")
        else:
            print(f"Found: {path}")
    if not ocd_paths:
        print("Warning: No OCD input files found. Skipping OCD merge.")

    # Verify directories
    if not os.path.exists(args.output_dir):
        raise FileNotFoundError(f"Output directory does not exist: {args.output_dir}")
    if not os.access(args.output_dir, os.W_OK):
        raise PermissionError(f"Cannot write to output directory: {args.output_dir}")
    print(f"Output directory is writable: {args.output_dir}")
    if not os.path.exists(args.work_dir):
        raise FileNotFoundError(f"Work directory does not exist: {args.work_dir}")
    if not os.access(args.work_dir, os.W_OK):
        raise PermissionError(f"Cannot write to work directory: {args.work_dir}")
    print(f"Work directory is writable: {args.work_dir}")

    # Merge HC files using nilearn
    if hc_paths:
        hc_output = os.path.join(args.output_dir, 'hc_baseline_4d.nii.gz')
        print(f"Merging HC files to: {hc_output}")
        try:
            hc_imgs = [image.load_img(p) for p in hc_paths]
            merged_img = image.concat_imgs(hc_imgs)
            merged_img.to_filename(hc_output)
            print(f"Successfully merged HC files to: {hc_output}")
        except Exception as e:
            print(f"Error during nilearn merge for HC files: {e}")
            raise

    # Merge OCD files using nilearn
    if ocd_paths:
        ocd_output = os.path.join(args.output_dir, 'ocd_baseline_4d.nii.gz')
        print(f"Merging OCD files to: {ocd_output}")
        try:
            ocd_imgs = [image.load_img(p) for p in ocd_paths]
            merged_img = image.concat_imgs(ocd_imgs)
            merged_img.to_filename(ocd_output)
            print(f"Successfully merged OCD files to: {ocd_output}")
        except Exception as e:
            print(f"Error during nilearn merge for OCD files: {e}")
            raise

    # Create mat/con for group diff
    if hc_paths and ocd_paths:
        with open(os.path.join(args.work_dir, 'group.mat'), 'w') as f:
            f.write(f"/NumWaves 1\n/NumPoints {len(hc_paths) + len(ocd_paths)}\n/Matrix\n")
            f.writelines(['1\n'] * len(ocd_paths) + ['-1\n'] * len(hc_paths))
        with open(os.path.join(args.work_dir, 'group.con'), 'w') as f:
            f.write("/NumWaves 1\n/NumContrasts 1\n/Matrix\n1\n")

        # Merge and run group difference using nilearn
        concat_output = os.path.join(args.output_dir, 'group_baseline_concat.nii.gz')
        print(f"Merging all files to: {concat_output}")
        try:
            concat_imgs = [image.load_img(p) for p in ocd_paths + hc_paths]
            merged_img = image.concat_imgs(concat_imgs)
            merged_img.to_filename(concat_output)
            print(f"Successfully merged all files to: {concat_output}")
        except Exception as e:
            print(f"Error during nilearn merge for group concat: {e}")
            raise

        rand = Randomise()
        rand.inputs.in_file = concat_output
        rand.inputs.mask = group_mask_file
        rand.inputs.design_mat = os.path.join(args.work_dir, 'group.mat')
        rand.inputs.contrast = os.path.join(args.work_dir, 'group.con')
        rand.inputs.n_perm = 5000
        rand.inputs.tfce = True
        rand.inputs.out_file = os.path.join(args.output_dir, 'group_diff_baseline')
        rand.run()

# Longitudinal analyses
if valid_longitudinal:
    # 2. Baseline FC vs symptom change
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

if not valid_group and not valid_longitudinal:
    raise ValueError("No valid subjects found for any analysis. Check *fcmap_avg.nii.gz files in output directory.")

print("All analyses completed.")