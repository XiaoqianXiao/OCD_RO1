#!/usr/bin/env python3
"""
Quick script to check for confound/BOLD volume mismatches.
Run this on the server to identify problematic subjects.
"""

import os
import glob
import pandas as pd
import nibabel as nib

def check_mismatches():
    """Check for confound/BOLD mismatches in the BIDS directory."""
    
    bids_dir = "/project/6079231/dliang55/R01_AOCD/derivatives/fmriprep-1.4.1"
    
    print("🔍 Checking for confound/BOLD volume mismatches...")
    print(f"📁 BIDS directory: {bids_dir}")
    print("=" * 80)
    
    mismatched_subjects = []
    
    # Find all subjects
    subject_dirs = glob.glob(os.path.join(bids_dir, "sub-*"))
    print(f"📊 Found {len(subject_dirs)} subjects")
    
    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        
        # Find all sessions
        sessions = glob.glob(os.path.join(subject_dir, "ses-*"))
        
        for session_dir in sessions:
            session_name = os.path.basename(session_dir)
            func_dir = os.path.join(session_dir, "func")
            
            if not os.path.exists(func_dir):
                continue
            
            # Find BOLD and confound files
            bold_files = glob.glob(os.path.join(func_dir, "*_desc-preproc_bold.nii.gz"))
            confound_files = glob.glob(os.path.join(func_dir, "*_desc-confounds_regressors.tsv"))
            
            if not bold_files or not confound_files:
                continue
            
            # Check each run
            for bold_file in bold_files:
                # Find matching confound file
                run_id = "unknown"
                if "_run-" in bold_file:
                    run_match = bold_file.split("_run-")[1].split("_")[0]
                    run_id = f"run-{run_match}"
                
                # Find matching confound file
                matching_confound = None
                for confound_file in confound_files:
                    if run_id != "unknown" and run_id in confound_file:
                        matching_confound = confound_file
                        break
                    elif run_id == "unknown" and len(confound_files) == 1:
                        matching_confound = confound_files[0]
                        break
                
                if not matching_confound:
                    continue
                
                try:
                    # Get BOLD volume count
                    img = nib.load(bold_file)
                    bold_volumes = img.shape[3]
                    
                    # Get confound row count
                    df = pd.read_csv(matching_confound, sep='\t')
                    confound_rows = len(df)
                    
                    # Check for mismatch
                    if bold_volumes != confound_rows:
                        mismatch_info = {
                            'subject': subject_id,
                            'session': session_name,
                            'run': run_id,
                            'bold_volumes': bold_volumes,
                            'confound_rows': confound_rows,
                            'difference': bold_volumes - confound_rows,
                            'bold_file': os.path.basename(bold_file),
                            'confound_file': os.path.basename(matching_confound)
                        }
                        mismatched_subjects.append(mismatch_info)
                        
                        print(f"❌ MISMATCH: {subject_id} {session_name} {run_id}")
                        print(f"   BOLD: {bold_volumes} volumes")
                        print(f"   Confounds: {confound_rows} rows")
                        print(f"   Difference: {mismatch_info['difference']}")
                        print(f"   BOLD file: {mismatch_info['bold_file']}")
                        print(f"   Confound file: {mismatch_info['confound_file']}")
                        print()
                        
                except Exception as e:
                    print(f"⚠️  Error processing {subject_id} {session_name} {run_id}: {e}")
                    continue
    
    # Summary
    print("=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)
    
    if mismatched_subjects:
        print(f"❌ Found {len(mismatched_subjects)} data mismatches")
        print(f"📊 Affects {len(set(s['subject'] for s in mismatched_subjects))} unique subjects")
        
        # Group by subject
        subjects_with_mismatches = {}
        for mismatch in mismatched_subjects:
            subject = mismatch['subject']
            if subject not in subjects_with_mismatches:
                subjects_with_mismatches[subject] = []
            subjects_with_mismatches[subject].append(mismatch)
        
        print("\n📝 Subjects with mismatches:")
        for subject in sorted(subjects_with_mismatches.keys()):
            mismatches = subjects_with_mismatches[subject]
            print(f"  {subject}: {len(mismatches)} mismatched runs")
            
            for mismatch in mismatches:
                print(f"    {mismatch['session']} {mismatch['run']}: "
                      f"BOLD={mismatch['bold_volumes']}, Confounds={mismatch['confound_rows']} "
                      f"(diff={mismatch['difference']})")
        
        # Save to file
        output_file = "confound_mismatch_subjects.txt"
        with open(output_file, 'w') as f:
            f.write("SUBJECTS WITH CONFOUND/BOLD MISMATCHES\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total mismatches found: {len(mismatched_subjects)}\n")
            f.write(f"Unique subjects affected: {len(subjects_with_mismatches)}\n\n")
            
            for subject in sorted(subjects_with_mismatches.keys()):
                mismatches = subjects_with_mismatches[subject]
                f.write(f"{subject}:\n")
                for mismatch in mismatches:
                    f.write(f"  {mismatch['session']} {mismatch['run']}: "
                           f"BOLD={mismatch['bold_volumes']}, Confounds={mismatch['confound_rows']} "
                           f"(diff={mismatch['difference']})\n")
                f.write("\n")
        
        print(f"\n💾 Detailed report saved to: {output_file}")
        
    else:
        print("✅ No confound/BOLD mismatches found!")
        print("All subjects have consistent data lengths.")

if __name__ == "__main__":
    check_mismatches()
