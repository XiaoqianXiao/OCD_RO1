#!/usr/bin/env python3
"""
Test script to verify Nilearn compatibility fixes in NW_1st.py
"""

import sys
import os

# Add the current directory to the path so we can import from NW_1st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        # Test basic imports
        from NW_1st import NILEARN_ATLASES, validate_atlas_params
        print("✅ Basic imports successful")
        
        # Test atlas configuration
        print(f"✅ Atlas configuration loaded: {len(NILEARN_ATLASES)} atlases")
        
        # Test parameter validation
        if 'schaefer_2018' in NILEARN_ATLASES:
            test_params = {'n_rois': 400, 'yeo_networks': 7, 'resolution_mm': 2}
            try:
                validated = validate_atlas_params('schaefer_2018', test_params)
                print(f"✅ Parameter validation successful: {validated}")
            except Exception as e:
                print(f"❌ Parameter validation failed: {e}")
                return False
        else:
            print("⚠️  Schaefer 2018 atlas not available")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_type_compatibility():
    """Test that type compatibility fixes work."""
    print("\nTesting type compatibility...")
    
    try:
        from NW_1st import fetch_nilearn_atlas
        print("✅ fetch_nilearn_atlas function imported successfully")
        
        # Check function signature
        import inspect
        sig = inspect.signature(fetch_nilearn_atlas)
        print(f"✅ Function signature: {sig}")
        
        return True
        
    except Exception as e:
        print(f"❌ Type compatibility test failed: {e}")
        return False

def test_atlas_configuration():
    """Test atlas configuration."""
    print("\nTesting atlas configuration...")
    
    try:
        from NW_1st import NILEARN_ATLASES
        
        for atlas_name, atlas_info in NILEARN_ATLASES.items():
            print(f"  {atlas_name}:")
            print(f"    Function: {atlas_info['function']}")
            print(f"    Default params: {atlas_info['default_params']}")
            print(f"    Network based: {atlas_info['network_based']}")
            
            if 'network_names' in atlas_info:
                print(f"    Network names: {list(atlas_info['network_names'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Atlas configuration test failed: {e}")
        return False

if __name__ == "__main__":
    print("Nilearn Compatibility Test for NW_1st.py")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_type_compatibility,
        test_atlas_configuration
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All compatibility tests passed!")
        print("The script should work correctly in the container.")
    else:
        print("❌ Some compatibility tests failed.")
        print("Please check the errors above.")
    
    sys.exit(0 if all_passed else 1)
