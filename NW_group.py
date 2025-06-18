import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='ROI-level FC group analysis')
parser.add_argument('--subjects_csv', type=str, required=True, help='Path to group.csv')
parser.add_argument('--clinical_csv', type=str, required=True, help='Path to clinical.csv')
parser.add_argument('--output_dir', type=str, default='/scratch/xxqian/OCD/NW_group', help='Output directory')
parser.add_argument('--input_dir', type=str, default='/input', help='Input directory for FC data')
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)


# Helper Functions
def get_roi_fc_path(subject, session, input_dir):
    """Get path to ROI FC CSV file."""
    if not subject.startswith('sub-'):
        subject = f"sub-{subject}"
    return os.path.join(input_dir, f"{subject}_{session}_task-rest_power2011_roi_fc_avg.csv")


def get_group(subject_id, metadata_df):
    """Get group label for a subject."""
    if subject_id.startswith('sub-'):
        subject_id = subject_id.replace('sub-', '')
    group = metadata_df[metadata_df['subject_id'] == subject_id]['group']
    return group.iloc[0] if not group.empty else None


def run_ttest(fc_data_hc, fc_data_ocd, columns):
    """Run two-sample t-tests with FDR correction."""
    results = []
    for col in columns:
        hc_values = fc_data_hc[col].dropna()
        ocd_values = fc_data_ocd[col].dropna()
        if len(hc_values) < 2 or len(ocd_values) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(ocd_values, hc_values, equal_var=False)
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
    """Run linear regression with FDR correction."""
    results = []
    for col in columns:
        x = fc_data[col].dropna()
        y = y_values.loc[x.index].dropna()
        if len(x) < 2 or len(y) < 2:
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


def load_and_validate_metadata(subjects_csv, clinical_csv):
    """Load and validate metadata CSVs."""
    try:
        df = pd.read_csv(subjects_csv)
        df['subject_id'] = df['sub'].astype(str)
        df = df[df['group'].isin(['HC', 'OCD'])]
    except Exception as e:
        raise ValueError(f"Failed to load subjects CSV: {e}")

    try:
        df_clinical = pd.read_csv(clinical_csv)
        df_clinical['subject_id'] = df_clinical['sub'].astype(str)
    except Exception as e:
        raise ValueError(f"Failed to load clinical CSV: {e}")

    return df, df_clinical


def validate_subjects(fc_dir, metadata_df):
    """Validate subjects based on available ROI FC files."""
    fc_files = glob.glob(os.path.join(fc_dir, '*_task-rest_power2011_roi_fc_avg.csv'))
    subject_sessions = {}
    for f in fc_files:
        filename = os.path.basename(f)
        if '_ses-' not in filename or '_task-rest_power2011_roi_fc_avg.csv' not in filename:
            continue
        parts = filename.split('_')
        subject = parts[0].replace('sub-', '')
        session = parts[1]
        subject_sessions.setdefault(subject, []).append(session)

    csv_subjects = set(metadata_df['subject_id'])
    file_subjects = set(subject_sessions.keys())
    sessions = ['ses-baseline', 'ses-followup']
    valid_group = [sid for sid in metadata_df['subject_id'] if
                   sid.replace('sub-', '') in subject_sessions and 'ses-baseline' in subject_sessions[
                       sid.replace('sub-', '')]]
    valid_longitudinal = [sid for sid in metadata_df['subject_id'] if
                          sid.replace('sub-', '') in subject_sessions and all(
                              ses in subject_sessions[sid.replace('sub-', '')] for ses in sessions)]

    return valid_group, valid_longitudinal, subject_sessions


def load_roi_fc_data(subject_ids, session, input_dir, fc_types=['within_network_FC', 'between_network_FC']):
    """Load ROI FC data for given subjects and session."""
    fc_data = []
    feature_columns = None
    for sid in subject_ids:
        fc_path = get_roi_fc_path(sid, session, input_dir)
        if not os.path.exists(fc_path):
            continue
        try:
            fc_df = pd.read_csv(fc_path)
            # Create unique feature identifier
            fc_df['feature_id'] = fc_df['network_name'] + '_' + fc_df['roi_name']
            if feature_columns is None:
                feature_columns = []
                for ftype in fc_types:
                    feature_columns.extend(fc_df['feature_id'].apply(lambda x: f"{x}_{ftype}").tolist())
            # Pivot to make features as columns
            fc_pivot = pd.DataFrame()
            for ftype in fc_types:
                temp = fc_df.pivot_table(
                    index=None,
                    columns='feature_id',
                    values=ftype
                ).reset_index(drop=True)
                temp.columns = [f"{col}_{ftype}" for col in temp.columns]
                fc_pivot = pd.concat([fc_pivot, temp], axis=1)
            fc_pivot['subject_id'] = sid
            fc_data.append(fc_pivot)
        except Exception as e:
            continue
    return pd.concat(fc_data, ignore_index=True) if fc_data else pd.DataFrame(), feature_columns


# Main Analysis
def main():
    # Load metadata
    df, df_clinical = load_and_validate_metadata(args.subjects_csv, args.clinical_csv)

    # Validate subjects
    valid_group, valid_longitudinal, _ = validate_subjects(args.input_dir, df)
    if not valid_group and not valid_longitudinal:
        raise ValueError("No valid subjects found for any analysis.")

    # Load baseline ROI FC data
    baseline_roi_fc_data, roi_feature_columns = load_roi_fc_data(valid_group, 'ses-baseline', args.input_dir)
    if baseline_roi_fc_data.empty:
        return

    # 1. Group difference at baseline
    if valid_group:
        hc_data = baseline_roi_fc_data[baseline_roi_fc_data['subject_id'].isin(df[df['group'] == 'HC']['subject_id'])]
        ocd_data = baseline_roi_fc_data[baseline_roi_fc_data['subject_id'].isin(df[df['group'] == 'OCD']['subject_id'])]
        if not hc_data.empty and not ocd_data.empty:
            ttest_results = run_ttest(hc_data, ocd_data, roi_feature_columns)
            if not ttest_results.empty:
                output_path = os.path.join(args.output_dir, 'group_diff_baseline_roi_fc.csv')
                ttest_results.to_csv(output_path, index=False)

    # 2. Longitudinal analyses
    if valid_longitudinal:
        ocd_df = df_clinical[df_clinical['subject_id'].isin(df[df['group'] == 'OCD']['subject_id'])].copy()
        ocd_df['delta_ybocs'] = ocd_df['ybocs_baseline'] - ocd_df['ybocs_followup']

        # Baseline FC vs symptom change
        baseline_fc_ocd = baseline_roi_fc_data[baseline_roi_fc_data['subject_id'].isin(ocd_df['subject_id'])]
        if not baseline_fc_ocd.empty:
            regression_results = run_regression(
                baseline_fc_ocd.set_index('subject_id'),
                ocd_df.set_index('subject_id')['delta_ybocs'],
                roi_feature_columns
            )
            if not regression_results.empty:
                output_path = os.path.join(args.output_dir, 'baselineFC_vs_deltaYBOCS_roi_fc.csv')
                regression_results.to_csv(output_path, index=False)

        # FC change vs symptom change
        fc_change_data = []
        for sid in valid_longitudinal:
            base_path = get_roi_fc_path(sid, 'ses-baseline', args.input_dir)
            follow_path = get_roi_fc_path(sid, 'ses-followup', args.input_dir)
            if not (os.path.exists(base_path) and os.path.exists(follow_path)):
                continue
            try:
                base_fc = pd.read_csv(base_path)
                follow_fc = pd.read_csv(follow_path)
                base_fc['feature_id'] = base_fc['network_name'] + '_' + base_fc['roi_name']
                follow_fc['feature_id'] = follow_fc['network_name'] + '_' + follow_fc['roi_name']
                change_fc = follow_fc.copy()
                for ftype in ['within_network_FC', 'between_network_FC']:
                    base_pivot = base_fc.pivot_table(index=None, columns='feature_id', values=ftype).reset_index(
                        drop=True)
                    follow_pivot = follow_fc.pivot_table(index=None, columns='feature_id', values=ftype).reset_index(
                        drop=True)
                    change_pivot = follow_pivot - base_pivot
                    change_pivot.columns = [f"{col}_{ftype}" for col in change_pivot.columns]
                    change_fc = pd.concat([change_fc, change_pivot], axis=1)
                change_fc['subject_id'] = sid
                fc_change_data.append(change_fc)
            except Exception:
                continue

        if fc_change_data:
            fc_change_data = pd.concat(fc_change_data, ignore_index=True)
            regression_results = run_regression(
                fc_change_data.set_index('subject_id'),
                ocd_df.set_index('subject_id')['delta_ybocs'],
                roi_feature_columns
            )
            if not regression_results.empty:
                output_path = os.path.join(args.output_dir, 'deltaFC_vs_deltaYBOCS_roi_fc.csv')
                regression_results.to_csv(output_path, index=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise