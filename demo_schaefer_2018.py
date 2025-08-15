#!/usr/bin/env python3
"""
Demonstration script for Schaefer 2018 atlas functionality in NW_1st.py

This script shows how to use the enhanced Schaefer 2018 atlas support
without requiring the full fMRI processing pipeline.
"""

import json
import sys
import os

# Add the current directory to the path so we can import from NW_1st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demo_schaefer_2018():
    """Demonstrate the Schaefer 2018 atlas functionality."""
    print("Schaefer 2018 Atlas Functionality Demo")
    print("=" * 50)
    
    try:
        # Import the atlas configuration from NW_1st.py
        from NW_1st import NILEARN_ATLASES, validate_atlas_params
        
        if 'schaefer_2018' not in NILEARN_ATLASES:
            print("❌ Schaefer 2018 atlas is not available")
            return False
        
        atlas_info = NILEARN_ATLASES['schaefer_2018']
        print(f"✅ Atlas function: {atlas_info['function'].__name__}")
        print(f"✅ Default parameters: {atlas_info['default_params']}")
        print(f"✅ Available parameters: {atlas_info['param_options']}")
        print(f"✅ Network-based: {atlas_info['network_based']}")
        
        if 'network_names' in atlas_info:
            print(f"✅ Network names available for: {list(atlas_info['network_names'].keys())} networks")
            for n_networks, names in atlas_info['network_names'].items():
                print(f"   {n_networks} networks: {names}")
        else:
            print("❌ No predefined network names")
        
        # Test different parameter combinations
        print("\nTesting Parameter Combinations:")
        print("-" * 30)
        
        test_configs = [
            {'n_rois': 200, 'yeo_networks': 7, 'resolution_mm': 2},
            {'n_rois': 400, 'yeo_networks': 7, 'resolution_mm': 2},
            {'n_rois': 400, 'yeo_networks': 17, 'resolution_mm': 2},
            {'n_rois': 1000, 'yeo_networks': 17, 'resolution_mm': 1}
        ]
        
        for config in test_configs:
            try:
                validated = validate_atlas_params('schaefer_2018', config)
                print(f"✅ {config} -> {validated}")
            except Exception as e:
                print(f"❌ {config} -> Error: {e}")
        
        # Show usage examples
        print("\nUsage Examples:")
        print("-" * 20)
        print("1. Basic usage (default: 400 ROIs, 7 networks):")
        print("   python NW_1st.py --subject sub-AOCD001 --atlas schaefer_2018 --label-pattern nilearn")
        
        print("\n2. Custom parameters (200 ROIs, 7 networks):")
        print("   python NW_1st.py --subject sub-AOCD001 --atlas schaefer_2018 --atlas-params '{\"n_rois\": 200, \"yeo_networks\": 7}' --label-pattern nilearn")
        
        print("\n3. High-resolution (1000 ROIs, 17 networks):")
        print("   python NW_1st.py --subject sub-AOCD001 --atlas schaefer_2018 --atlas-params '{\"n_rois\": 1000, \"yeo_networks\": 17, \"resolution_mm\": 1}' --label-pattern nilearn")
        
        print("\n4. Test configuration:")
        print("   python NW_1st.py --test-schaefer")
        
        print("\n5. List all available atlases:")
        print("   python NW_1st.py --list-atlases")
        
        print("\n🎉 Schaefer 2018 Atlas is fully configured and ready to use!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure NW_1st.py is in the same directory")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def show_network_distribution():
    """Show how ROIs are distributed across networks for different configurations."""
    print("\nROI Distribution Across Networks:")
    print("=" * 40)
    
    configs = [
        (200, 7), (400, 7), (600, 7), (800, 7), (1000, 7),
        (200, 17), (400, 17), (600, 17), (800, 17), (1000, 17)
    ]
    
    for n_rois, n_networks in configs:
        rois_per_network = n_rois // n_networks
        remainder = n_rois % n_networks
        
        if remainder == 0:
            print(f"{n_rois} ROIs, {n_networks} networks: {rois_per_network} ROIs per network ✅")
        else:
            print(f"{n_rois} ROIs, {n_networks} networks: {rois_per_network} ROIs per network + {remainder} remainder ⚠️")

if __name__ == "__main__":
    print("Schaefer 2018 Atlas Demo for NW_1st.py")
    print("=" * 50)
    
    success = demo_schaefer_2018()
    
    if success:
        show_network_distribution()
        
        print("\n" + "=" * 50)
        print("The script is ready to use with the following features:")
        print("✅ Automatic atlas downloading via Nilearn")
        print("✅ Predefined network names for 7 and 17 networks")
        print("✅ Parameter validation and error checking")
        print("✅ Flexible ROI and network configurations")
        print("✅ Automatic network label generation")
        print("✅ Comprehensive error handling and logging")
        print("✅ Built-in testing and validation tools")
        
        print("\nTo use in your analysis:")
        print("1. Run with --test-schaefer to verify configuration")
        print("2. Use --list-atlases to see all available options")
        print("3. Start with default parameters and adjust as needed")
        print("4. Use --verbose for detailed logging during analysis")
    else:
        print("\n❌ Demo failed. Please check the configuration.")
        sys.exit(1)
