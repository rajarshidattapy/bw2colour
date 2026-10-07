"""
Setup Verification Script
Run this to check if your environment is properly configured
"""

import sys
import os

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ⚠ Warning: Python 3.8+ recommended")
        return False
    return True

def check_dependencies():
    """Check required packages"""
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV (opencv-python)',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'skimage': 'scikit-image',
        'flask': 'Flask',
        'scipy': 'SciPy'
    }
    
    all_ok = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            all_ok = False
    
    return all_ok

def check_pytorch_cuda():
    """Check PyTorch and CUDA availability"""
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠ CUDA not available - will use CPU (slower)")
            return False
    except Exception as e:
        print(f"✗ PyTorch check failed: {e}")
        return False

def check_files():
    """Check if required files exist"""
    required_files = [
        'similarity_pytorch.py',
        'app.py',
        'index.html',
        'requirements.txt',
        'notebooks/unified_pipeline.ipynb'
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} NOT FOUND")
            all_exist = False
    
    return all_exist

def check_checkpoints():
    """Check if model checkpoints exist"""
    checkpoint_dir = 'checkpoints'
    required_checkpoints = [
        'restoration_final.pth',
        'colorization_final.pth'
    ]
    
    if not os.path.exists(checkpoint_dir):
        print(f"⚠ {checkpoint_dir}/ directory not found")
        print("  → You need to train models first or download pre-trained checkpoints")
        return False
    
    all_exist = True
    for checkpoint in required_checkpoints:
        path = os.path.join(checkpoint_dir, checkpoint)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✓ {checkpoint} exists ({size_mb:.1f} MB)")
        else:
            print(f"✗ {checkpoint} NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("  → Train models using notebooks/unified_pipeline.ipynb")
    
    return all_exist

def check_directories():
    """Check/create required directories"""
    dirs = ['uploads', 'outputs', 'checkpoints']
    
    for dir_name in dirs:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name}/ exists")
        else:
            os.makedirs(dir_name)
            print(f"✓ {dir_name}/ created")

def test_import_models():
    """Test importing model definitions"""
    try:
        from similarity_pytorch import SimilaritySubnet, VGG19Features
        print("✓ Can import SimilaritySubnet")
        return True
    except Exception as e:
        print(f"✗ Cannot import models: {e}")
        return False

def main():
    print("=" * 60)
    print("Setup Verification for Unified Pipeline")
    print("=" * 60)
    print()
    
    print("[1/7] Checking Python version...")
    python_ok = check_python_version()
    print()
    
    print("[2/7] Checking dependencies...")
    deps_ok = check_dependencies()
    print()
    
    print("[3/7] Checking PyTorch and CUDA...")
    cuda_ok = check_pytorch_cuda()
    print()
    
    print("[4/7] Checking required files...")
    files_ok = check_files()
    print()
    
    print("[5/7] Checking/creating directories...")
    check_directories()
    print()
    
    print("[6/7] Testing model imports...")
    import_ok = test_import_models()
    print()
    
    print("[7/7] Checking model checkpoints...")
    checkpoints_ok = check_checkpoints()
    print()
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    if python_ok and deps_ok and files_ok and import_ok:
        print("✓ Environment setup complete!")
        
        if checkpoints_ok:
            print("✓ Models ready - you can run: python app.py")
        else:
            print("⚠ Models not found - please train first:")
            print("  1. Open notebooks/unified_pipeline.ipynb in Colab")
            print("  2. Run all cells to train models")
            print("  3. Download checkpoints and place in checkpoints/")
        
        if cuda_ok:
            print("✓ CUDA available - inference will be fast")
        else:
            print("⚠ Using CPU - inference will be slower but still works")
    else:
        print("✗ Setup incomplete - please fix the issues above")
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements.txt")
    
    print()
    print("For detailed instructions, see:")
    print("  - QUICKSTART.md (quick guide)")
    print("  - README_UNIFIED.md (full documentation)")
    print("=" * 60)

if __name__ == "__main__":
    main()
