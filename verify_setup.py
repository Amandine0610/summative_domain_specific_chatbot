#!/usr/bin/env python3
"""
Verification script to test the HMM Activity Recognition setup.
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing module imports...")
    
    try:
        sys.path.append('src')
        
        from src.data_collection import ActivityDataGenerator
        print("✓ Data collection module imported successfully")
        
        from src.feature_extraction import FeatureExtractor
        print("✓ Feature extraction module imported successfully")
        
        from src.hmm_model import ActivityHMM
        print("✓ HMM model module imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of each module."""
    print("\nTesting basic functionality...")
    
    try:
        # Test data generator
        generator = ActivityDataGenerator(sampling_rate=50)
        print("✓ Data generator initialized")
        
        # Test feature extractor
        extractor = FeatureExtractor(window_size=2.0, overlap=0.5, sampling_rate=50)
        print("✓ Feature extractor initialized")
        
        # Test HMM model
        hmm_model = ActivityHMM(n_states=4)
        print("✓ HMM model initialized")
        
        return True
    except Exception as e:
        print(f"✗ Functionality error: {e}")
        return False

def check_file_structure():
    """Check that all required files are present."""
    print("\nChecking file structure...")
    
    required_files = [
        'src/data_collection.py',
        'src/feature_extraction.py', 
        'src/hmm_model.py',
        'notebooks/human_activity_hmm.ipynb',
        'demo_hmm_activity.py',
        'requirements.txt',
        'README.md'
    ]
    
    required_dirs = [
        'src',
        'notebooks',
        'data',
        'results'
    ]
    
    all_present = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
            all_present = False
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ Missing directory: {dir_path}/")
            all_present = False
    
    return all_present

def main():
    """Run all verification tests."""
    print("=" * 60)
    print("HMM Activity Recognition - Setup Verification")
    print("=" * 60)
    
    # Test file structure
    structure_ok = check_file_structure()
    
    # Test imports (only if we have dependencies)
    imports_ok = test_imports()
    
    # Test basic functionality (only if imports work)
    if imports_ok:
        functionality_ok = test_basic_functionality()
    else:
        functionality_ok = False
        print("\nSkipping functionality tests due to import errors")
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    print(f"File structure: {'✓ PASS' if structure_ok else '✗ FAIL'}")
    print(f"Module imports: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"Basic functionality: {'✓ PASS' if functionality_ok else '✗ FAIL'}")
    
    if structure_ok and imports_ok and functionality_ok:
        print("\n🎉 All tests passed! The setup is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run demo: python3 demo_hmm_activity.py")
        print("3. Or open notebook: jupyter notebook notebooks/human_activity_hmm.ipynb")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        if not imports_ok:
            print("   - Install dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    main()