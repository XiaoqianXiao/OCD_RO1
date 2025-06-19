import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import argparse
import logging

# Configure logging
def setup_logging(output_dir):
    """Set up logging to console and file."""
    log_file = os.path.join(output_dir, 'roi_network_group_analysis.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging initialized. Output will be saved to %s", log_file)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='ROI-to-network FC group analysis')
parser.add_argument('--subjects_csv', type=str, required=True, help='Path to group.csv')
parser.add_argument('--clinical_csv', type=str, required=True, help='Path to clinical.csv')
parser.add_argument('--output_dir', type=str, default='/scratch/xxqian/OCD/NW_group', help='Output directory')
parser.add_argument('--input_dir', type=str, default='/scratch/xxqian/OCD', help='Input directory for FC data')
args = parser.parse_args()

# Create output directory and set up logging
os.makedirs(args.output_dir, exist_ok=True)
setup_logging(args.output_dir)
logging.info("Starting analysis with arguments: %s", vars(args))

# Helper Functions
def get_network_fc_path(subject, session, input_dir):
    """Get path to ROI-to-network FC CSV file."""
    if not subject.startswith('sub-'):
        subject = f"sub-{subject}"
    path = os.path.join(input_dir, f"{subject}_{session}_task-rest_power2011_network_fc_avg.csv")
    logging.debug("Generated FC path for subject %s, session %s: %s", subject, session, path)
    return path

def get_group(subject_id, metadata_df):
    """Get group label for a subject."""
    if subject_id.startswith('sub-'):
        subject_id = subject_id.replace('sub-', '')
    group = metadata_df[metadata_df['subject_id'] == subject_id]['group']
    if group.empty:
        logging.warning("No group found for subject %s", subject_id)
        return None
    logging.debug("Found group %s for subject %s", group.iloc[0], subject_id)
    return group.iloc[0]

def run_ttest(fc_data_hc, fc_data_ocd, feature_info):
    """Run two-sample t-tests with FDR correction for ROI-to-network FC."""
    logging.info("Running t-tests for %d ROI-to-network features", len(feature_info))
    results = []
    for feature, (net1, net2) in feature_info.items():
        hc_values = fc_data_hc[feature].dropna()
        ocd_values = fc_data_ocd[feature].dropna()
        logging.debug("Feature %s (%s_%s): HC n=%d, OCD n=%d", feature, net1, net2, len(hc_values), len(ocd_values))
        if len(hc_values) < 2 or len(ocd_values) < 2:
            logging.warning("Skipping feature %s (%s_%s) due to insufficient data (HC n=%d, OCD n=%d)",
                           feature, net1, net2, len(hc_values), len(ocd_values))
            continue
        t_stat, p_val = stats.ttest_ind(ocd_values, hc_values, equal_var=False)
        results.append({
            'ROI': feature,
            'network1': net1,
            'network2': net2,
            't_statistic': t_stat,
            'p_value': p_val,
            'OCD_mean': np.mean(ocd_values),
            'HC_mean': np.mean(hc_values),
            'OCD_n': len(ocd_values),
            'HC_n': len(hc_values)
        })
    if not results:
        logging.info("No t-test results generated (no valid features)")
        return pd.DataFrame()
    results_df = pd.DataFrame(results)
    p_vals = results_df['p_value'].values
    _, p_vals_corr = fdrcorrection(p_vals, alpha=0.05)
    results_df['p_value_fdr'] = p_vals_corr
    logging.info("Generated t-test results for %d features", len(results_df))
    return results_df

def run_regression(fc_data, y_values, feature_info):
    """Run linear regression with FDR correction for ROI-to-network FC."""
    logging.info("Running regressions for %d ROI-to-network features", len(feature_info))
    results = []
    fc_data.index = fc_data.index.astype(str)
    y_values.index = y_values.index.astype(str)
    common_subjects = fc_data.index.intersection(y_values.index)
    logging.debug("Common subjects for regression: %d", len(common_subjects))
    if not common_subjects.size:
        logging.warning("No common subjects for regression")
        return pd.DataFrame()
    fc_data = fc_data.loc[common_subjects]
    y_values = y_values.loc[common_subjects]

    for feature, (net1, net2) in feature_info.items():
        x = fc_data[feature].dropna()
        if x.empty:
            logging.warning("Skipping feature %s (%s_%s) due to empty data", feature, net1, net2)
            continue
        y = y_values.loc[x.index].dropna()
        logging.debug("Feature %s (%s_%s): n=%d", feature, net1, net2, len(y))
        if len(x) < 2 or len(y) < 2:
            logging.warning("Skipping feature %s (%s_%s) due to insufficient data (n=%d)", feature, net1, net2, len(y))
            continue
        x = x.values.reshape(-1, 1)
        y = y.values
        slope, intercept, r_value, p_val, _ = stats.linregress(x.flatten(), y)
        results.append({
            'ROI': feature,
            'network1': net1,
            'network2': net2,
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'p_value': p_val,
            'n': len(y)
        })
    if not results:
        logging.info("No regression results generated (no valid features)")
        return pd.DataFrame()
    results_df = pd.DataFrame(results)
    p_vals = results_df['p_value'].values
    _, p_vals_corr = fdrcorrection(p_vals, alpha=0.05)
    results_df['p_value_fdr'] = p_vals_corr
    logging.info("Generated regression results for %d features", len(results_df))
    return results_df

def load_and_validate_metadata(subjects_csv, clinical_csv):
    """Load and validate metadata CSVs."""
    logging.info("Loading metadata from %s and %s", subjects_csv, clinical_csv)
    try:
        df = pd.read_csv(subjects_csv)
        df['subject_id'] = df['sub'].astype(str)
        df = df[df['group'].isin(['HC', 'OCD'])]
        logging.info("Loaded %d subjects from %s", len(df), subjects_csv)
    except Exception as e:
        logging.error("Failed to load subjects CSV: %s", e)
        raise ValueError(f"Failed to load subjects CSV: {e}")

    try:
        df_clinical = pd.read_csv(clinical_csv)
        df_clinical['subject_id'] = df_clinical['sub'].astype(str)
        logging.info("Loaded %d clinical records from %s", len(df_clinical), clinical_csv)
    except Exception as e:
        logging.error("Failed to load clinical CSV: %s", e)
        raise ValueError(f"Failed to load clinical CSV: {e}")

    return df, df_clinical

def validate_subjects(fc_dir, metadata_df):
    """Validate subjects based on available ROI-to-network FC files."""
    logging.info("Validating subjects in FC directory %s", fc_dir)
    fc_files = glob.glob(os.path.join(fc_dir, '*_task-rest_power2011_network_fc_avg.csv'))
    logging.info("Found %d FC files", len(fc_files))
    subject_sessions = {}
    for f in fc_files:
        filename = os.path.basename(f)
        if '_ses-' not in filename or '_task-rest_power2011_network_fc_avg.csv' not in filename:
            logging.debug("Skipping invalid FC file: %s", filename)
            continue
        parts = filename.split('_')
        subject = parts[0].replace('sub-', '')
        session = parts[1]
        subject_sessions.setdefault(subject, []).append(session)
        logging.debug("Found subject %s, session %s in file %s", subject, session, filename)

    csv_subjects = set(metadata_df['subject_id'])
    file_subjects = set(subject_sessions.keys())
    logging.info("CSV subjects: %d, FC file subjects: %d, overlap: %d",
                len(csv_subjects), len(file_subjects), len(csv_subjects & file_subjects))
    sessions = ['ses-baseline', 'ses-followup']
    valid_group = [sid for sid in metadata_df['subject_id'] if
                   sid.replace('sub-', '') in subject_sessions and 'ses-baseline' in subject_sessions[
                       sid.replace('sub-', '')]]
    valid_longitudinal = [sid for sid in metadata_df['subject_id'] if
                          sid.replace('sub-', '') in subject_sessions and all(
                              ses in subject_sessions[sid.replace('sub-', '')] for ses in sessions)]
    logging.info("Valid subjects for group analysis: %d", len(valid_group))
    logging.info("Valid subjects for longitudinal analysis: %d", len(valid_longitudinal))
    return valid_group, valid_longitudinal, subject_sessions

def validate_network_fc_file(fc_path):
    """Validate that ROI-to-network FC file has required columns."""
    required_columns = {'ROI', 'network1', 'network2', 'fc_value'}
    try:
        df = pd.read_csv(fc_path)
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            logging.error("ROI-to-network FC file %s missing required columns: %s. Found columns: %s",
                         fc_path, missing_columns, list(df.columns))
            return False
        logging.debug("ROI-to-network FC file %s validated successfully", fc_path)
        return True
    except Exception as e:
        logging.error("Failed to validate ROI-to-network FC file %s: %s", fc_path, e)
        return False

def load_network_fc_data(subject_ids, session, input_dir):
    """Load ROI-to-network FC data for given subjects and session."""
    logging.info("Loading ROI-to-network FC data for %d subjects, session %s", len(subject_ids), session)
    fc_data = []
    feature_info = None
    valid_subjects = 0
    for sid in subject_ids:
        sid_no_prefix = sid.replace('sub-', '')
        fc_path = get_network_fc_path(sid_no_prefix, session, input_dir)
        if not os.path.exists(fc_path):
            logging.warning("ROI-to-network FC file not found: %s", fc_path)
            continue
        if not validate_network_fc_file(fc_path):
            continue
        try:
            fc_df = pd.read_csv(fc_path)
            logging.debug("Loaded ROI-to-network FC file %s with %d rows", fc_path, len(fc_df))
            # Create feature identifier and map networks
            fc_df['feature_id'] = fc_df['ROI']
            if feature_info is None:
                feature_info = {row['ROI']: (row['network1'], row['network2'])
                               for _, row in fc_df[['ROI', 'network1', 'network2']].drop_duplicates().iterrows()}
                logging.debug("Identified %d ROI-to-network feature columns with network mappings", len(feature_info))
            # Pivot to make features as columns
            fc_pivot = fc_df.pivot_table(
                index=None,
                columns='feature_id',
                values='fc_value'
            ).reset_index(drop=True)
            fc_pivot['subject_id'] = sid_no_prefix
            fc_data.append(fc_pivot)
            valid_subjects += 1
        except Exception as e:
            logging.error("Failed to process ROI-to-network FC file %s: %s", fc_path, e)
            continue
    if not fc_data:
        logging.warning("No valid ROI-to-network FC data loaded for session %s", session)
        return pd.DataFrame(), feature_info
    fc_data_df = pd.concat(fc_data, ignore_index=True)
    logging.info("Loaded ROI-to-network FC data for %d subjects, %d features", valid_subjects, len(feature_info))
    return fc_data_df, feature_info

# Main Analysis
def main():
    logging.info("Starting main ROI-to-network FC analysis")
    # Load metadata
    df, df_clinical = load_and_validate_metadata(args.subjects_csv, args.clinical_csv)

    # Normalize subject IDs
    df['subject_id'] = df['subject_id'].str.replace('sub-', '')
    df_clinical['subject_id'] = df_clinical['subject_id'].str.replace('sub-', '')
    logging.debug("Normalized subject IDs in metadata")

    # Validate subjects
    valid_group, valid_longitudinal, subject_sessions = validate_subjects(args.input_dir, df)
    if not valid_group and not valid_longitudinal:
        logging.error("No valid subjects found for any analysis")
        raise ValueError("No valid subjects found for any analysis.")

    # Load baseline ROI-to-network FC data
    baseline_fc_data, feature_info = load_network_fc_data(valid_group, 'ses-baseline', args.input_dir)
    if baseline_fc_data.empty:
        logging.warning("No baseline ROI-to-network FC data loaded. Skipping group and longitudinal analyses.")
        return

    # 1. Group difference at baseline
    if valid_group:
        hc_data = baseline_fc_data[baseline_fc_data['subject_id'].isin(df[df['group'] == 'HC']['subject_id'])]
        ocd_data = baseline_fc_data[baseline_fc_data['subject_id'].isin(df[df['group'] == 'OCD']['subject_id'])]
        logging.info("Group analysis: HC n=%d, OCD n=%d", len(hc_data), len(ocd_data))
        if not hc_data.empty and not ocd_data.empty:
            ttest_results = run_ttest(hc_data, ocd_data, feature_info)
            if not ttest_results.empty:
                output_path = os.path.join(args.output_dir, 'group_diff_baseline_roi_network_fc.csv')
                ttest_results.to_csv(output_path, index=False)
                logging.info("Saved t-test results to %s with columns: %s", output_path, list(ttest_results.columns))
            else:
                logging.info("No significant t-test results to save")
        else:
            logging.warning("Insufficient data for group analysis (HC empty: %s, OCD empty: %s)",
                           hc_data.empty, ocd_data.empty)

    # 2. Longitudinal analyses
    if valid_longitudinal:
        ocd_df = df_clinical[df_clinical['subject_id'].isin(df[df['group'] == 'OCD']['subject_id'])].copy()
        ocd_df['delta_ybocs'] = ocd_df['ybocs_baseline'] - ocd_df['ybocs_followup']
        logging.info("Longitudinal analysis: OCD subjects n=%d", len(ocd_df))

        # Baseline FC vs symptom change
        baseline_fc_ocd = baseline_fc_data[baseline_fc_data['subject_id'].isin(ocd_df['subject_id'])]
        logging.info("Baseline ROI-to-network FC for OCD: n=%d", len(baseline_fc_ocd))
        if not baseline_fc_ocd.empty:
            regression_results = run_regression(
                baseline_fc_ocd.set_index('subject_id'),
                ocd_df.set_index('subject_id')['delta_ybocs'],
                feature_info
            )
            if not regression_results.empty:
                output_path = os.path.join(args.output_dir, 'baselineFC_vs_deltaYBOCS_roi_network_fc.csv')
                regression_results.to_csv(output_path, index=False)
                logging.info("Saved baseline FC regression results to %s with columns: %s",
                            output_path, list(regression_results.columns))
            else:
                logging.info("No significant baseline FC regression results to save")

        # FC change vs symptom change
        fc_change_data = []
        for sid in valid_longitudinal:
            sid = sid.replace('sub-', '')
            base_path = get_network_fc_path(sid, 'ses-baseline', args.input_dir)
            follow_path = get_network_fc_path(sid, 'ses-followup', args.input_dir)
            if not (os.path.exists(base_path) and os.path.exists(follow_path)):
                logging.warning("Missing ROI-to-network FC files for subject %s (baseline: %s, followup: %s)",
                               sid, os.path.exists(base_path), os.path.exists(follow_path))
                continue
            if not (validate_network_fc_file(base_path) and validate_network_fc_file(follow_path)):
                continue
            try:
                base_fc = pd.read_csv(base_path)
                follow_fc = pd.read_csv(follow_path)
                logging.debug("Loaded baseline ROI-to-network FC (%d rows) and followup FC (%d rows) for %s",
                             len(base_fc), len(follow_fc), sid)
                base_fc['feature_id'] = base_fc['ROI']
                follow_fc['feature_id'] = follow_fc['ROI']
                base_pivot = base_fc.pivot_table(index=None, columns='feature_id', values='fc_value').reset_index(
                    drop=True)
                follow_pivot = follow_fc.pivot_table(index=None, columns='feature_id', values='fc_value').reset_index(
                    drop=True)
                change_pivot = follow_pivot - base_pivot
                change_pivot['subject_id'] = sid
                fc_change_data.append(change_pivot)
            except Exception as e:
                logging.error("Failed to process longitudinal ROI-to-network FC for subject %s: %s", sid, e)
                continue

        if fc_change_data:
            fc_change_data = pd.concat(fc_change_data, ignore_index=True)
            fc_change_data['subject_id'] = fc_change_data['subject_id'].astype(str)
            feature_columns = [col for col in fc_change_data.columns if col != 'subject_id']
            logging.info("Loaded ROI-to-network FC change data for %d subjects, %d features",
                        len(fc_change_data), len(feature_columns))
            regression_results = run_regression(
                fc_change_data.set_index('subject_id'),
                ocd_df.set_index('subject_id')['delta_ybocs'],
                feature_info
            )
            if not regression_results.empty:
                output_path = os.path.join(args.output_dir, 'deltaFC_vs_deltaYBOCS_roi_network_fc.csv')
                regression_results.to_csv(output_path, index=False)
                logging.info("Saved FC change regression results to %s with columns: %s",
                            output_path, list(regression_results.columns))
            else:
                logging.info("No significant FC change regression results to save")
        else:
            logging.warning("No ROI-to-network FC change data loaded for longitudinal analysis")

    logging.info("Main ROI-to-network FC analysis completed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("Main execution failed: %s", e)
        raise