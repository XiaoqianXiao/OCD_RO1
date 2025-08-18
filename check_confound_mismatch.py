#!/usr/bin/env python3
"""
Script to identify subjects with mismatched confound CSV rows and BOLD volumes.
This helps identify why some subjects fail in NW_1st.py processing.
"""

import os
import glob
import pandas as pd
import nibabel as nib
from pathlib import Path
import argparse
import logging

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def get_bold_volume_count(bold_file):
    """Get the number of volumes in a BOLD image."""
    try:
        img = nib.load(bold_file)
        return img.shape[3]  # 4th dimension is time
    except Exception as e:
        logging.error(f"Failed to load BOLD file {bold_file}: {e}")
        return None

def get_confound_row_count(confound_file):
    """Get the number of rows in a confound CSV file."""
    try:
        df = pd.read_csv(confound_file, sep='\t')
        return len(df)
    except Exception as e:
        logging.error(f"Failed to read confound file {confound_file}: {e}")
        return None

def check_subject_data_consistency(subject_dir, logger):
    """Check if a subject has consistent confound and BOLD data lengths."""
    subject_id = os.path.basename(subject_dir)
    logger.info(f"Checking {subject_id}...")
    
    # Find all sessions
    sessions = glob.glob(os.path.join(subject_dir, "ses-*"))
    if not sessions:
        logger.warning(f"No sessions found for {subject_id}")
        return None
    
    results = []
    
    for session_dir in sessions:
        session_name = os.path.basename(session_dir)
        func_dir = os.path.join(session_dir, "func")
        
        if not os.path.exists(func_dir):
            logger.warning(f"No func directory for {subject_id} {session_name}")
            continue
        
        # Find BOLD files
        bold_files = glob.glob(os.path.join(func_dir, "*_desc-preproc_bold.nii.gz"))
        confound_files = glob.glob(os.path.join(func_dir, "*_desc-confounds_regressors.tsv"))
        
        if not bold_files:
            logger.warning(f"No BOLD files found for {subject_id} {session_name}")
            continue
            
        if not confound_files:
            logger.warning(f"No confound files found for {subject_id} {session_name}")
            continue
        
        # Check each run
        for bold_file in bold_files:
            # Find matching confound file
            run_id = None
            if "_run-" in bold_file:
                run_match = bold_file.split("_run-")[1].split("_")[0]
                run_id = f"run-{run_match}"
            
            matching_confound = None
            for confound_file in confound_files:
                if run_id and run_id in confound_file:
                    matching_confound = confound_file
                    break
                elif not run_id and len(confound_files) == 1:
                    matching_confound = confound_file
                    break
            
            if not matching_confound:
                logger.warning(f"No matching confound file for {bold_file}")
                continue
            
            # Get counts
            bold_volumes = get_bold_volume_count(bold_file)
            confound_rows = get_confound_row_count(matching_confound)
            
            if bold_volumes is None or confound_rows is None:
                continue
            
            # Check consistency
            is_consistent = (bold_volumes == confound_rows)
            
            results.append({
                'subject': subject_id,
                'session': session_name,
                'run_id': run_id or 'unknown',
                'bold_file': os.path.basename(bold_file),
                'confound_file': os.path.basename(matching_confound),
                'bold_volumes': bold_volumes,
                'confound_rows': confound_rows,
                'is_consistent': is_consistent,
                'difference': bold_volumes - confound_rows
            })
            
            if not is_consistent:
                logger.warning(f"MISMATCH: {subject_id} {session_name} {run_id}: "
                             f"BOLD={bold_volumes}, Confounds={confound_rows}")
    
    return results

def main():
    """Main function to check all subjects."""
    parser = argparse.ArgumentParser(description='Check confound and BOLD data consistency')
    parser.add_argument('--bids-dir', type=str, 
                       default='/project/6079231/dliang55/R01_AOCD/derivatives/fmriprep-1.4.1',
                       help='BIDS derivatives directory')
    parser.add_argument('--output', type=str, default='confound_mismatch_report.csv',
                       help='Output CSV file for results')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info(f"Checking data consistency in: {args.bids_dir}")
    
    if not os.path.exists(args.bids_dir):
        logger.error(f"BIDS directory does not exist: {args.bids_dir}")
        return
    
    # Find all subjects
    subject_dirs = glob.glob(os.path.join(args.bids_dir, "sub-*"))
    if not subject_dirs:
        logger.error(f"No subjects found in {args.bids_dir}")
        return
    
    logger.info(f"Found {len(subject_dirs)} subjects")
    
    all_results = []
    
    # Check each subject
    for subject_dir in subject_dirs:
        results = check_subject_data_consistency(subject_dir, logger)
        if results:
            all_results.extend(results)
    
    if not all_results:
        logger.warning("No results generated")
        return
    
    # Create summary
    df = pd.DataFrame(all_results)
    
    # Summary statistics
    total_checks = len(df)
    consistent_checks = df['is_consistent'].sum()
    inconsistent_checks = total_checks - consistent_checks
    
    logger.info(f"Summary:")
    logger.info(f"  Total data checks: {total_checks}")
    logger.info(f"  Consistent: {consistent_checks}")
    logger.info(f"  Inconsistent: {inconsistent_checks}")
    logger.info(f"  Consistency rate: {consistent_checks/total_checks*100:.1f}%")
    
    # Find subjects with any inconsistencies
    inconsistent_subjects = df[~df['is_consistent']]['subject'].unique()
    logger.info(f"Subjects with inconsistencies: {len(inconsistent_subjects)}")
    
    if len(inconsistent_subjects) > 0:
        logger.info("Subjects with data mismatches:")
        for subject in sorted(inconsistent_subjects):
            subject_data = df[df['subject'] == subject]
            inconsistent_data = subject_data[~subject_data['is_consistent']]
            logger.info(f"  {subject}: {len(inconsistent_data)} inconsistent runs")
            
            for _, row in inconsistent_data.iterrows():
                logger.info(f"    {row['session']} {row['run_id']}: "
                          f"BOLD={row['bold_volumes']}, Confounds={row['confound_rows']} "
                          f"(diff={row['difference']})")
    
    # Save detailed results
    df.to_csv(args.output, index=False)
    logger.info(f"Detailed results saved to: {args.output}")
    
    # Save summary of inconsistent subjects
    if len(inconsistent_subjects) > 0:
        summary_file = args.output.replace('.csv', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("SUBJECTS WITH CONFOUND/BOLD MISMATCHES\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total subjects checked: {len(subject_dirs)}\n")
            f.write(f"Subjects with mismatches: {len(inconsistent_subjects)}\n")
            f.write(f"Consistency rate: {consistent_checks/total_checks*100:.1f}%\n\n")
            
            f.write("DETAILED MISMATCHES:\n")
            f.write("-" * 30 + "\n")
            for subject in sorted(inconsistent_subjects):
                subject_data = df[df['subject'] == subject]
                inconsistent_data = subject_data[~subject_data['is_consistent']]
                f.write(f"\n{subject}:\n")
                for _, row in inconsistent_data.iterrows():
                    f.write(f"  {row['session']} {row['run_id']}: "
                           f"BOLD={row['bold_volumes']}, Confounds={row['confound_rows']} "
                           f"(diff={row['difference']})\n")
        
        logger.info(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
