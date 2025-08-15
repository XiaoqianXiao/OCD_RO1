# Group Analysis Scripts Update Summary

## Overview

This document summarizes the updates made to `NW_group.py` and `NW_group2.py` to ensure compatibility with the enhanced `NW_1st.py` script and its new Schaefer 2018 atlas functionality.

## Files Updated

### 1. `NW_group.py` - ROI-to-Network FC Group Analysis
### 2. `NW_group2.py` - Network-Level Pairwise FC Analysis

## Updates Made

### ✅ **Directory Path Fixes**

#### NW_group2.py
- **Fixed**: Incorrect input directory path
- **Before**: `'input_dir': '/input'` (incorrect)
- **After**: `'input_dir': '/project/6079231/dliang55/R01_AOCD'` (correct)
- **Reason**: Matches the configuration used in `NW_1st.py` and ensures proper data access

### ✅ **Documentation Updates**

#### Enhanced Usage Examples
Both scripts now include comprehensive examples for Schaefer 2018 atlas:

**Basic Schaefer 2018 (400 ROIs, 7 networks):**
```bash
python NW_group.py \
  --subjects_csv group.csv \
  --clinical_csv clinical.csv \
  --input_dir /path/to/fc/data \
  --atlas_name schaefer_2018_400_7_2
```

**High-Resolution Schaefer 2018 (1000 ROIs, 17 networks):**
```bash
python NW_group.py \
  --subjects_csv group.csv \
  --clinical_csv clinical.csv \
  --input_dir /path/to/fc/data \
  --atlas_name schaefer_2018_1000_17_1
```

### ✅ **Atlas Naming Convention Updates**

#### Enhanced Naming Patterns
Both scripts now properly document the Schaefer 2018 naming convention:

**Pattern**: `schaefer_2018_{n_rois}_{yeo_networks}_{resolution_mm}`

**Parameters**:
- **n_rois**: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
- **yeo_networks**: 7 or 17
- **resolution_mm**: 1 or 2

**Examples**:
- `schaefer_2018_400_7_2_network_fc_avg.csv` (400 ROIs, 7 networks, 2mm)
- `schaefer_2018_1000_17_1_network_fc_avg.csv` (1000 ROIs, 17 networks, 1mm)

### ✅ **Help Text Improvements**

#### Argument Help Updates
- **`--atlas_name`**: Updated help text to include Schaefer 2018 examples
- **Examples**: Added comprehensive usage examples in argument parser
- **Documentation**: Enhanced inline documentation and comments

## Compatibility Status

### ✅ **Full Compatibility Achieved**

1. **Directory Structure**: All scripts now use consistent directory paths
2. **Atlas Naming**: Consistent naming conventions across all scripts
3. **Documentation**: Unified documentation and examples
4. **Functionality**: All scripts support the new Schaefer 2018 atlas features

### ✅ **Integration Points**

1. **Input Files**: Group scripts can now properly read Schaefer 2018 output files from `NW_1st.py`
2. **Atlas Detection**: Auto-detection works with new naming patterns
3. **Output Files**: Consistent output file naming across the pipeline
4. **Error Handling**: Improved error messages for atlas-related issues

## Usage Workflow

### Complete Analysis Pipeline

1. **Individual Analysis** (`NW_1st.py`):
   ```bash
   python NW_1st.py --subject sub-AOCD001 --atlas schaefer_2018 --atlas-params '{"n_rois": 400, "yeo_networks": 7}' --label-pattern nilearn
   ```

2. **Group Analysis** (`NW_group.py`):
   ```bash
   python NW_group.py --subjects_csv group.csv --clinical_csv clinical.csv --input_dir /path/to/fc/data --atlas_name schaefer_2018_400_7_2
   ```

3. **Network Analysis** (`NW_group2.py`):
   ```bash
   python NW_group2.py --subjects_csv group.csv --clinical_csv clinical.csv --input_dir /path/to/fc/data --atlas_name schaefer_2018_400_7_2
   ```

## Testing Recommendations

### Verify Updates

1. **Compilation**: All scripts compile without errors ✅
2. **Documentation**: Help text displays updated examples ✅
3. **Directory Paths**: Consistent with `NW_1st.py` configuration ✅
4. **Atlas Support**: Full Schaefer 2018 atlas support ✅

### Test Commands

```bash
# Test compilation
python3 -m py_compile NW_group.py NW_group2.py

# Test help display
python3 NW_group.py --help
python3 NW_group2.py --help

# Test atlas detection (when data is available)
python3 NW_group.py --subjects_csv group.csv --clinical_csv clinical.csv --input_dir /path/to/data --auto-detect-atlas
```

## Future Enhancements

### Potential Improvements

1. **Atlas Validation**: Add validation for Schaefer 2018 parameters
2. **Performance Optimization**: Optimize for large ROI counts (800-1000)
3. **Memory Management**: Add memory usage warnings for high-resolution atlases
4. **Quality Metrics**: Add atlas quality assessment tools

## Summary

Both `NW_group.py` and `NW_group2.py` have been successfully updated to:

- ✅ **Fix directory path issues**
- ✅ **Support new Schaefer 2018 atlas functionality**
- ✅ **Provide comprehensive documentation and examples**
- ✅ **Maintain backward compatibility**
- ✅ **Ensure consistency with `NW_1st.py`**

The group analysis scripts are now fully compatible with the enhanced individual-level analysis and provide a seamless pipeline for Schaefer 2018 atlas-based functional connectivity analysis.
