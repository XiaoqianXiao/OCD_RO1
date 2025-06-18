import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import argparse
import logging

# Set up logging
log_file = '/scratch/xxqian/OCD/logs/NW_group.log'
#log_file = '/output/logs/NW_group.log'
#os.makedirs(os.path.dirname(log_file), exist_ok=True)
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
parser = argparse.ArgumentParser(description='Network-level FC group analysis')
parser.add_argument('--subjects_csv', type=str, required=True, help='Path to group.csv')
parser.add_argument('--clinical_csv', type=str, required=True, help='Path to clinical.csv')
parser.add_argument('--output_dir', type=str, default='/scratch/xxqian/OCD/NW_group', help='Output directory')
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Load metadata
df = pd.read_csv(args.subjects_csv)
df['subject_id'] = df['sub'].astype(str)
df = df[df['group'].isin(['HC', 'OCD'])]
df_clinical = pd.read_csv(args.clinical_csv)
df_clinical['subject_id'] = df_clinical['sub'].astype(str)
sessions = ['ses-baseline', 'ses-followup']
logger.info(f"Subject IDs in CSV: {df['subject_id'].tolist()}")


# Helpers
def get_network_fc_path(subject, session):
    """Get path to network FC CSV file."""
    if not subject.startswith('sub-'):
        subject = f"sub-{subject}"
    return os.path.join('/scratch/xxqian/OCD/NW_1stLevel',
                        f"{subject}_{session}_task-rest_power2011_network_fc_avg.csv")


def get_group(subject):
    """Get group label for a subject."""
    if subject.startswith('sub-'):
        subject = subject.replace('sub-', '')
    return df[df['subject_id'] == subject]['group'].iloc[0]


def run_ttest(fc_data_hc, fc_data_ocd, columns):
    """Run two-sample t-tests and apply FDR correction."""
    results = []
    for col in columns:
        hc_values = fc_data_hc[col].dropna()
        ocd_values = fc_data_ocd[col].dropna()
        if len(hc_values) < 2 or len(ocd_values) < 2:
            logger.warning(f"Skipping {col}: Insufficient data (HC: {len(hc_values)}, OCD: {len(ocd_values)})")
            continue
        t_stat, p_val = stats.ttest_ind(ocd_values, hc_values, equal_var=False)  # Welch's t-test
        results.append({
            'Feature': col,
            't_statistic': t_stat,
            'p_value': p_val,
            'OCD_mean': np.mean(ocd_values),
            'HC_mean': np.mean(hc_values),
            'OCD_n': len(ocd_values),
            'HC_n': len(hc_values)
        })
    if not results:
        return pd.DataFrame()
    results_df = pd.DataFrame(results)
    p_vals = results_df['p_value'].values
    _, p_vals_corr = fdrcorrection(p_vals, alpha=0.05)
    results_df['p_value_fdr'] = p_vals_corr
    return results_df


def run_regression(fc_data, y_values, columns):
    """Run linear regression and apply FDR correction."""
    results = []
    for col in columns:
        x = fc_data[col].dropna()
        y = y_values.loc[x.index].dropna()
        if len(x) < 2 or len(y) < 2:
            logger.warning(f"Skipping {col}: Insufficient data (n={len(x)})")
            continue
        x = x.values.reshape(-1, 1)
        y = y.values
        slope, intercept, r_value, p_val, _ = stats.linregress(x.flatten(), y)
        results.append({
            'Feature': col,
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'p_value': p_val,
            'n': len(x)
        })
    if not results:
        return pd.DataFrame()
    results_df = pd.DataFrame(results)
    p_vals = results_df['p_value'].values
    _, p_vals_corr = fdrcorrection(p_vals, alpha=0.05)
    results_df['p_value_fdr'] = p_vals_corr
    return results_df


# Validate subjects
fc_files = glob.glob(os.path.join('/scratch/xxqian/OCD/NW_1stLevel', '*_task-rest_power2011_network_fc_avg.csv'))
subject_sessions = {}
for f in fc_files:
    filename = os.path.basename(f)
    if '_ses-' in filename and '_task-rest_power2011_network_fc_avg.csv' in filename:
        parts = filename.split('_')
        subject = parts[0].replace('sub-', '')
        session = parts[1]
        if subject not in subject_sessions:
            subject_sessions[subject] = []
        subject_sessions[subject].append(session)
logger.info(f"Subject sessions: {subject_sessions}")

csv_subjects = set(df['subject_id'])
file_subjects = set(subject_sessions.keys())
unmatched = file_subjects - csv_subjects
if unmatched:
    logger.warning(f"Found files for subjects not in subjects_csv: {unmatched}")

valid_group = [sid for sid in df['subject_id'] if
               sid.replace('sub-', '') in subject_sessions and 'ses-baseline' in subject_sessions[
                   sid.replace('sub-', '')]]
valid_longitudinal = [sid for sid in df['subject_id'] if sid.replace('sub-', '') in subject_sessions and all(
    ses in subject_sessions[sid.replace('sub-', '')] for ses in sessions)]
logger.info(f"Valid subjects for group analysis: {valid_group}")
logger.info(f"Valid subjects for longitudinal analysis: {valid_longitudinal}")

if not valid_group:
    logger.warning("No subjects found for group analysis. Skipping.")
if not valid_longitudinal:
    logger.warning("No subjects found for longitudinal analysis. Skipping.")

# Filter DataFrames
df = df[df['subject_id'].isin(valid_group)]
df_clinical = df_clinical[df_clinical['subject_id'].isin(valid_longitudinal)]

# Load network FC data
network_columns = None  # Will be set from the first valid CSV
baseline_fc_data = []
for sid in valid_group:
    fc_path = get_network_fc_path(sid, 'ses-baseline')
    if os.path.exists(fc_path):
        fc_df = pd.read_csv(fc_path)
        if network_columns is None:
            network_columns = [col for col in fc_df.columns if col != 'Network']
        fc_df['subject_id'] = sid
        baseline_fc_data.append(fc_df)
    else:
        logger.warning(f"Missing baseline FC file: {fc_path}")
baseline_fc_data = pd.concat(baseline_fc_data, ignore_index=True) if baseline_fc_data else pd.DataFrame()

# 1. Group difference at baseline
if valid_group and not baseline_fc_data.empty:
    hc_data = baseline_fc_data[baseline_fc_data['subject_id'].isin(df[df['group'] == 'HC']['subject_id'])]
    ocd_data = baseline_fc_data[baseline_fc_data['subject_id'].isin(df[df['group'] == 'OCD']['subject_id'])]
    if not hc_data.empty and not ocd_data.empty:
        ttest_results = run_ttest(hc_data, ocd_data, network_columns)
        ttest_output = os.path.join(args.output_dir, 'group_diff_baseline_network_fc.csv')
        ttest_results.to_csv(ttest_output, index=False)
        logger.info(f"Saved group difference results: {ttest_output}")
    else:
        logger.warning("Insufficient HC or OCD data for group difference analysis.")

# Longitudinal analyses
if valid_longitudinal:
    ocd_df = df_clinical[df_clinical['subject_id'].isin(df[df['group'] == 'OCD']['subject_id'])].copy()
    ocd_df['delta_ybocs'] = ocd_df['ybocs_baseline'] - ocd_df['ybocs_followup']

    # 2. Baseline FC vs symptom change
    baseline_fc_ocd = baseline_fc_data[baseline_fc_data['subject_id'].isin(ocd_df['subject_id'])]
    if not baseline_fc_ocd.empty:
        regression_results = run_regression(baseline_fc_ocd.set_index('subject_id'),
                                            ocd_df.set_index('subject_id')['delta_ybocs'], network_columns)
        regression_output = os.path.join(args.output_dir, 'baselineFC_vs_deltaYBOCS_network_fc.csv')
        regression_results.to_csv(regression_output, index=False)
        logger.info(f"Saved baseline FC vs delta YBOCS results: {regression_output}")

    # 3. FC change vs symptom change
    fc_change_data = []
    for sid in valid_longitudinal:
        base_path = get_network_fc_path(sid, 'ses-baseline')
        follow_path = get_network_fc_path(sid, 'ses-followup')
        if os.path.exists(base_path) and os.path.exists(follow_path):
            base_fc = pd.read_csv(base_path)
            follow_fc = pd.read_csv(follow_path)
            change_fc = follow_fc.copy()
            for col in network_columns:
                change_fc[col] = follow_fc[col] - base_fc[col]
            change_fc['subject_id'] = sid
            fc_change_data.append(change_fc)
        else:
            logger.warning(f"Missing FC file for {sid}: baseline={base_path}, follow-up={follow_path}")
    fc_change_data = pd.concat(fc_change_data, ignore_index=True) if fc_change_data else pd.DataFrame()

    if not fc_change_data.empty:
        regression_results = run_regression(fc_change_data.set_index('subject_id'),
                                            ocd_df.set_index('subject_id')['delta_ybocs'], network_columns)
        regression_output = os.path.join(args.output_dir, 'deltaFC_vs_deltaYBOCS_network_fc.csv')
        regression_results.to_csv(regression_output, index=False)
        logger.info(f"Saved FC change vs delta YBOCS results: {regression_output}")

if not valid_group and not valid_longitudinal:
    logger.error("No valid subjects found for any analysis.")
    raise ValueError("No valid subjects found for any analysis.")

logger.info("All analyses completed.")