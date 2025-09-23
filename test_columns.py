#!/usr/bin/env python3
"""
Quick test to check what columns are being generated
"""

import pandas as pd
import os

# Test with a single subject
fc_file = "/Users/xiaoqianxiao/projects/OCD/results/NW_1st/sub-AOCD001_ses-baseline_task-rest_power_2011_roiroi_fc_avg.csv"
clinical_file = "/Users/xiaoqianxiao/projects/OCD/results/behav/clinical.csv"

# Load FC data
fc_df = pd.read_csv(fc_file)
print("FC file columns:", fc_df.columns.tolist())
print("FC file shape:", fc_df.shape)
print("Sample FC data:")
print(fc_df.head(3))

# Load clinical data
clinical_df = pd.read_csv(clinical_file)
print("\nClinical file columns:", clinical_df.columns.tolist())
print("Clinical file shape:", clinical_df.shape)
print("Sample clinical data:")
print(clinical_df.head(3))

# Check what the Power atlas summary should look like
print("\nExpected Power atlas columns:")
expected_columns = ['subjectID', 'group', 'ROI1', 'ROI2', 'fc_baseline', 'fc_followup', 'OCD_baseline', 'OCD_followup']
print(expected_columns)
