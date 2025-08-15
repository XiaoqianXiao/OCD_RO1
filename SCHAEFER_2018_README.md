# Schaefer 2018 Atlas Support in NW_1st.py

## Overview

The `NW_1st.py` script now includes comprehensive support for the Schaefer 2018 atlas using Nilearn's `fetch_atlas_schaefer_2018` function. This atlas provides high-quality brain parcellations with 100-1000 ROIs organized by functional networks.

## Features

✅ **Automatic Atlas Downloading**: Uses Nilearn to automatically download and cache the atlas  
✅ **Flexible ROI Configuration**: Supports 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 ROIs  
✅ **Network Options**: Supports both 7 and 17 network configurations  
✅ **Resolution Options**: Supports both 1mm and 2mm spatial resolution  
✅ **Predefined Network Names**: Automatically generates appropriate network labels  
✅ **Parameter Validation**: Comprehensive error checking and validation  
✅ **Testing Tools**: Built-in configuration testing and validation  

## Network Configurations

### 7 Networks (Default)
- **Visual**: Visual processing areas
- **Somatomotor**: Motor and somatosensory areas  
- **Dorsal Attention**: Top-down attention control
- **Ventral Attention**: Bottom-up attention and salience
- **Limbic**: Emotional processing and memory
- **Frontoparietal**: Executive control and working memory
- **Default**: Default mode network

### 17 Networks (Extended)
Extended version with sub-networks for more detailed analysis:
- **Visual1, Visual2**: Primary and secondary visual areas
- **Somatomotor1, Somatomotor2**: Motor and sensory sub-networks
- **Dorsal Attention1, Dorsal Attention2**: Attention control sub-networks
- **Ventral Attention1, Ventral Attention2**: Salience and attention sub-networks
- **Limbic1, Limbic2**: Emotional and memory sub-networks
- **Frontoparietal1, Frontoparietal2**: Executive control sub-networks
- **Default1, Default2**: Default mode sub-networks
- **Temporal Parietal**: Language and social cognition
- **Orbital Frontal**: Decision making and reward
- **Cingulo-opercular**: Cognitive control and monitoring

## Usage Examples

### Basic Usage (Default Parameters)
```bash
python NW_1st.py \
  --subject sub-AOCD001 \
  --atlas schaefer_2018 \
  --label-pattern nilearn
```
**Default**: 400 ROIs, 7 networks, 2mm resolution

### Custom Parameters
```bash
python NW_1st.py \
  --subject sub-AOCD001 \
  --atlas schaefer_2018 \
  --atlas-params '{"n_rois": 200, "yeo_networks": 7, "resolution_mm": 2}' \
  --label-pattern nilearn
```

### High-Resolution Analysis
```bash
python NW_1st.py \
  --subject sub-AOCD001 \
  --atlas schaefer_2018 \
  --atlas-params '{"n_rois": 1000, "yeo_networks": 17, "resolution_mm": 1}' \
  --label-pattern nilearn
```

## Parameter Options

| Parameter | Values | Description |
|-----------|--------|-------------|
| `n_rois` | 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 | Number of ROIs |
| `yeo_networks` | 7, 17 | Number of functional networks |
| `resolution_mm` | 1, 2 | Spatial resolution in millimeters |

## Recommended Configurations

### For Exploratory Analysis
- **200 ROIs, 7 networks, 2mm**: Good balance of detail and computational efficiency
- **400 ROIs, 7 networks, 2mm**: Standard configuration, good for most analyses

### For Detailed Analysis
- **400 ROIs, 17 networks, 2mm**: More detailed network breakdown
- **600 ROIs, 17 networks, 2mm**: High detail with reasonable computational cost

### For High-Resolution Studies
- **1000 ROIs, 17 networks, 1mm**: Maximum detail (requires significant computational resources)

## Testing and Validation

### Test Configuration
```bash
python NW_1st.py --test-schaefer
```
This command tests:
- Atlas function availability
- Parameter validation
- Network name generation
- Configuration consistency

### List Available Atlases
```bash
python NW_1st.py --list-atlases
```
Shows all available Nilearn atlases and their parameters.

### Demo Script
```bash
python3 demo_schaefer_2018.py
```
Comprehensive demonstration of all features.

## Technical Details

### Atlas Loading
The script automatically:
1. Downloads the atlas using Nilearn (first time only)
2. Caches the atlas for future use
3. Generates appropriate network labels
4. Validates all parameters
5. Handles different Nilearn versions

### Network Label Generation
- **7 networks**: Uses predefined network names
- **17 networks**: Uses extended network names with sub-networks
- **Automatic**: No manual label file required
- **Consistent**: Labels are automatically mapped to ROI indices

### Memory Considerations
- **Low ROI counts (100-400)**: Minimal memory usage
- **Medium ROI counts (500-700)**: Moderate memory usage
- **High ROI counts (800-1000)**: Significant memory usage
- **1mm resolution**: 8x more memory than 2mm resolution

## Error Handling

### Common Issues
1. **Invalid parameters**: Script validates all parameters before processing
2. **Network mismatch**: Ensures n_rois ≥ yeo_networks
3. **Memory issues**: Warns about high memory usage configurations
4. **Download failures**: Provides clear error messages for network issues

### Troubleshooting
- Use `--test-schaefer` to verify configuration
- Use `--verbose` for detailed logging
- Check parameter ranges and combinations
- Ensure stable internet connection for first download

## Output Files

The script generates output files with names based on the atlas configuration:
- **Basic**: `schaefer_2018_400_7_2_roiroi_matrix_avg.npy`
- **Custom**: `schaefer_2018_200_7_2_roiroi_matrix_avg.npy`
- **High-res**: `schaefer_2018_1000_17_1_roiroi_matrix_avg.npy`

## Integration with Existing Pipeline

The Schaefer 2018 atlas integrates seamlessly with the existing analysis pipeline:
- **ROI-to-ROI FC**: Full correlation matrices
- **ROI-to-Network FC**: Network-level connectivity measures
- **Network Analysis**: Within/between network connectivity
- **Output Formats**: Compatible with existing group analysis scripts

## Performance Considerations

### Computational Cost
- **ROI count**: O(n²) complexity for correlation matrices
- **Network count**: Minimal impact on performance
- **Resolution**: 1mm requires 8x more memory than 2mm

### Recommended Workflows
1. **Start with defaults**: 400 ROIs, 7 networks, 2mm
2. **Scale up gradually**: Increase ROIs or networks as needed
3. **Monitor resources**: Use system monitoring tools
4. **Batch processing**: Process multiple subjects in parallel

## Future Enhancements

Planned improvements:
- **Custom network definitions**: User-defined network configurations
- **Atlas comparison tools**: Compare different atlas configurations
- **Performance optimization**: Memory-efficient processing for large ROIs
- **Quality metrics**: Atlas quality assessment tools

## Support

For issues or questions:
1. Check the troubleshooting section
2. Use the built-in testing tools
3. Review the comprehensive examples
4. Check the verbose logging output

---

**Note**: The Schaefer 2018 atlas is automatically downloaded and cached by Nilearn. Ensure you have sufficient disk space and a stable internet connection for the first download.
