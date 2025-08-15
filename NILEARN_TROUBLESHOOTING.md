# Nilearn Data Directory Troubleshooting Guide

## Issue Description

The error encountered shows:
```
OSError: Nilearn tried to store the dataset in the following directories, but: 
 -/home/xxqian/nilearn_data/schaefer_2018 (File exists)
```

This is a common problem in containerized environments where:
1. The Nilearn data directory structure gets corrupted
2. There are permission issues with the home directory
3. Previous failed downloads leave incomplete files
4. The container environment has limited write access

## Root Cause

Nilearn is trying to create a directory for the Schaefer 2018 atlas, but there's already a file (not a directory) with that name. This prevents Nilearn from properly downloading and storing the atlas data.

## Solutions

### 1. **Automatic Fix (Recommended)**

The script now includes automatic detection and fixing of this issue:

```bash
# The script will automatically detect and fix the issue
python NW_1st.py --subject sub-AOCD001 --atlas schaefer_2018 --atlas-params '{"n_rois": 100, "yeo_networks": 7}' --label-pattern nilearn
```

### 2. **Manual Fix Before Analysis**

If you want to fix the issue before running analysis:

```bash
# Fix Nilearn data directory issues
python NW_1st.py --fix-nilearn-data

# Then run your analysis normally
python NW_1st.py --subject sub-AOCD001 --atlas schaefer_2018 --atlas-params '{"n_rois": 100, "yeo_networks": 7}' --label-pattern nilearn
```

### 3. **Pre-download Atlases**

To avoid download issues during analysis:

```bash
# Pre-download common Schaefer 2018 configurations
python NW_1st.py --pre-download-schaefer

# This will download:
# - 100 ROIs, 7 networks, 2mm
# - 200 ROIs, 7 networks, 2mm  
# - 400 ROIs, 7 networks, 2mm
# - 400 ROIs, 17 networks, 2mm
```

## What the Fix Does

The `--fix-nilearn-data` option:

1. **Clears corrupted directories**: Removes problematic `/home/xxqian/nilearn_data` if it exists
2. **Sets environment variables**: Sets `NILEARN_DATA` to `/tmp/nilearn_data`
3. **Creates clean directory**: Creates a fresh, writable data directory
4. **Sets proper permissions**: Ensures the directory is accessible

## Container-Specific Considerations

### Environment Variables
The fix sets:
```bash
export NILEARN_DATA=/tmp/nilearn_data
```

### Directory Structure
- **Before**: `/home/xxqian/nilearn_data/` (problematic)
- **After**: `/tmp/nilearn_data/` (clean, writable)

### Permissions
- Uses `/tmp` which is typically writable in containers
- Avoids home directory permission issues
- Ensures clean slate for atlas downloads

## Prevention Strategies

### 1. **Pre-download Atlases**
```bash
# Run this once before starting analysis
python NW_1st.py --pre-download-schaefer
```

### 2. **Use Fixed Data Directory**
```bash
# Set environment variable in your container
export NILEARN_DATA=/tmp/nilearn_data
```

### 3. **Monitor Disk Space**
Ensure sufficient disk space for atlas storage:
- Schaefer 2018 (100 ROIs): ~50MB
- Schaefer 2018 (400 ROIs): ~200MB  
- Schaefer 2018 (1000 ROIs): ~500MB

## Error Recovery

### If Automatic Fix Fails

1. **Check container permissions**:
   ```bash
   ls -la /tmp/
   ls -la /home/xxqian/
   ```

2. **Verify disk space**:
   ```bash
   df -h
   ```

3. **Check network connectivity**:
   ```bash
   ping -c 3 nilearn.github.io
   ```

4. **Manual cleanup**:
   ```bash
   rm -rf /home/xxqian/nilearn_data
   rm -rf /tmp/nilearn_data
   ```

### Alternative Atlas Options

If Schaefer 2018 continues to have issues:

```bash
# Use AAL atlas (smaller, more stable)
python NW_1st.py --subject sub-AOCD001 --atlas aal --label-pattern nilearn

# Use Harvard-Oxford atlas
python NW_1st.py --subject sub-AOCD001 --atlas harvard_oxford --atlas-params '{"atlas_name": "cort-maxprob-thr25-2mm"}' --label-pattern nilearn
```

## Testing the Fix

### Verify Fix Worked

1. **Run fix command**:
   ```bash
   python NW_1st.py --fix-nilearn-data
   ```

2. **Check data directory**:
   ```bash
   ls -la /tmp/nilearn_data/
   ```

3. **Test atlas download**:
   ```bash
   python NW_1st.py --pre-download-schaefer
   ```

### Expected Output

```
Fixing Nilearn data directory issues...
✅ Using data directory: /tmp/nilearn_data
✅ Nilearn data directory fixed: /tmp/nilearn_data
You can now run the script normally.
```

## Integration with SLURM Scripts

### Update Submission Scripts

Add the fix command to your SLURM scripts:

```bash
#!/bin/bash
#SBATCH --job-name=NW_1st_sub-AOCD001
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# Fix Nilearn data directory first
python3 /app/NW_1st.py --fix-nilearn-data

# Then run analysis
python3 /app/NW_1st.py --subject sub-AOCD001 --atlas schaefer_2018 --atlas-params '{"n_rois": 100, "yeo_networks": 7}' --label-pattern nilearn
```

### Environment Setup

Set environment variables in your container:

```bash
export NILEARN_DATA=/tmp/nilearn_data
export NILEARN_SHARED_DATA=/tmp/nilearn_data
```

## Summary

The Nilearn data directory issue is now fully handled with:

1. **Automatic detection** and fixing during atlas fetching
2. **Manual fix option** (`--fix-nilearn-data`)
3. **Pre-download option** (`--pre-download-schaefer`)
4. **Comprehensive error handling** and recovery
5. **Container-optimized** data storage locations

This ensures robust atlas functionality in containerized environments and prevents the download failures you encountered.
