"""
Automated Dataset Download Script
Downloads datasets for old photo restoration and colorization training
"""

import os
import sys
import urllib.request
import zipfile
import tarfile
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(text.center(60))
    print("="*60 + "\n")

def download_file(url, filepath):
    """Download file with progress bar"""
    class DownloadProgressBar():
        def __init__(self):
            self.pbar = None

        def __call__(self, block_num, block_size, total_size):
            if not self.pbar:
                self.pbar = True
                print(f"Downloading... 0%", end='', flush=True)
            
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100)
            print(f"\rDownloading... {percent:.1f}%", end='', flush=True)
            
            if downloaded >= total_size:
                print("\rDownload complete!     ")
    
    urllib.request.urlretrieve(url, filepath, DownloadProgressBar())

def download_openphoto():
    """Download OpenPhoto Restore Dataset from HuggingFace"""
    print_header("Downloading OpenPhoto Restore Dataset")
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Error: 'datasets' library not installed")
        print("Install it with: pip install datasets")
        return False
    
    print("📦 Loading from HuggingFace...")
    print("Size: ~4.4 GB")
    print("This may take 10-20 minutes depending on your connection.\n")
    
    try:
        dataset = load_dataset("joshuachin/openphoto-restore-dataset")
        
        # Save to disk
        save_path = "datasets/openphoto_restore"
        os.makedirs(save_path, exist_ok=True)
        
        # Save images
        print("\n💾 Saving images to disk...")
        
        train_dir = os.path.join(save_path, "train")
        test_dir = os.path.join(save_path, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Save train images
        print(f"Saving {len(dataset['train'])} training images...")
        for i, example in enumerate(dataset['train']):
            pristine = example['pristine_image']
            damaged = example['damaged_image']
            
            pristine.save(os.path.join(train_dir, f"pristine_{i:04d}.png"))
            damaged.save(os.path.join(train_dir, f"damaged_{i:04d}.png"))
            
            if (i + 1) % 500 == 0:
                print(f"  Saved {i + 1} images...")
        
        # Save test images
        print(f"\nSaving {len(dataset['test'])} test images...")
        for i, example in enumerate(dataset['test']):
            pristine = example['pristine_image']
            damaged = example['damaged_image']
            
            pristine.save(os.path.join(test_dir, f"pristine_{i:04d}.png"))
            damaged.save(os.path.join(test_dir, f"damaged_{i:04d}.png"))
        
        print(f"\n✅ OpenPhoto dataset saved to: {save_path}")
        print(f"   - Train: {len(dataset['train'])} image pairs")
        print(f"   - Test: {len(dataset['test'])} image pairs")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

def download_div2k():
    """Download DIV2K dataset"""
    print_header("Downloading DIV2K Dataset")
    
    print("📦 High-resolution image dataset")
    print("Size: ~3 GB")
    print("This may take 5-15 minutes.\n")
    
    save_dir = "datasets/div2k"
    os.makedirs(save_dir, exist_ok=True)
    
    urls = {
        "DIV2K_train_HR.zip": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "DIV2K_valid_HR.zip": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
    }
    
    try:
        for filename, url in urls.items():
            filepath = os.path.join(save_dir, filename)
            
            # Check if already downloaded
            if os.path.exists(filepath.replace('.zip', '')):
                print(f"✓ {filename} already exists, skipping...")
                continue
            
            print(f"\n📥 Downloading {filename}...")
            download_file(url, filepath)
            
            # Unzip
            print(f"📂 Extracting {filename}...")
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(save_dir)
            
            # Remove zip file to save space
            os.remove(filepath)
            print(f"✓ Extracted and cleaned up")
        
        print(f"\n✅ DIV2K dataset saved to: {save_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading DIV2K: {e}")
        return False

def download_places365_small():
    """Download Places365 small version"""
    print_header("Downloading Places365-Small Dataset")
    
    print("⚠️  WARNING: This is a LARGE download (~36 GB)")
    print("It may take 1-3 hours depending on your connection.")
    print("Make sure you have at least 40 GB of free disk space.\n")
    
    proceed = input("Continue? (yes/no): ")
    if proceed.lower() not in ['yes', 'y']:
        print("Skipped Places365 download.")
        return False
    
    try:
        import tensorflow_datasets as tfds
        
        print("\n📦 Downloading via TensorFlow Datasets...")
        print("This will automatically download and prepare the dataset.\n")
        
        # This will auto-download
        dataset = tfds.load('places365_small', split='train', download=True)
        
        print("\n✅ Places365 downloaded successfully")
        print("   Location: ~/tensorflow_datasets/places365_small")
        return True
        
    except ImportError:
        print("❌ Error: TensorFlow Datasets not installed")
        print("Install with: pip install tensorflow-datasets")
        return False
    except Exception as e:
        print(f"❌ Error downloading Places365: {e}")
        return False

def verify_datasets():
    """Verify downloaded datasets"""
    print_header("Verifying Downloaded Datasets")
    
    datasets = {
        "OpenPhoto Restore": "datasets/openphoto_restore/train",
        "DIV2K": "datasets/div2k/DIV2K_train_HR",
    }
    
    for name, path in datasets.items():
        if os.path.exists(path):
            num_files = len([f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))])
            print(f"✅ {name}: {num_files} images found")
        else:
            print(f"❌ {name}: Not found")
    
    # Check Places365
    places_path = os.path.expanduser("~/tensorflow_datasets/places365_small")
    if os.path.exists(places_path):
        print(f"✅ Places365: Found at {places_path}")
    else:
        print(f"⚠️  Places365: Not downloaded")

def main():
    print_header("Dataset Download Script for Old Photo Restoration")
    
    print("This script will help you download training datasets.")
    print("You can choose which datasets to download.\n")
    
    print("Available datasets:")
    print("1. OpenPhoto Restore Dataset (4.4GB) ⭐ RECOMMENDED")
    print("   - 4,500 training image pairs (damaged + pristine)")
    print("   - 500 test pairs")
    print("   - Perfect for restoration training")
    print()
    print("2. DIV2K (3GB)")
    print("   - 900 high-resolution images")
    print("   - Great for both restoration and colorization")
    print()
    print("3. Places365-Small (36GB)")
    print("   - 1.8M scene images")
    print("   - Best for colorization diversity")
    print()
    print("4. Download ALL (Recommended for best results)")
    print()
    
    choice = input("Enter your choice (1-4, or 'q' to quit): ")
    
    if choice.lower() == 'q':
        print("Exiting...")
        return
    
    # Create base directory
    os.makedirs("datasets", exist_ok=True)
    
    success_count = 0
    total_count = 0
    
    if choice == "1" or choice == "4":
        total_count += 1
        if download_openphoto():
            success_count += 1
    
    if choice == "2" or choice == "4":
        total_count += 1
        if download_div2k():
            success_count += 1
    
    if choice == "3" or choice == "4":
        total_count += 1
        if download_places365_small():
            success_count += 1
    
    if choice not in ["1", "2", "3", "4"]:
        print("Invalid choice!")
        return
    
    # Verify downloads
    print()
    verify_datasets()
    
    # Summary
    print_header("Download Summary")
    print(f"✅ Successfully downloaded: {success_count}/{total_count} datasets")
    print(f"\n📁 Datasets saved in: ./datasets/")
    
    if success_count > 0:
        print("\n🎉 Ready to start training!")
        print("\nNext steps:")
        print("1. Open notebooks/unified_pipeline.ipynb in Google Colab")
        print("2. Upload your datasets to Colab (or mount Google Drive)")
        print("3. Update dataset paths in the notebook:")
        print("   - restoration_dataset = RestorationDataset('datasets/openphoto_restore/train', ...)")
        print("   - colorization_dataset = ColorizationDataset('datasets/div2k/DIV2K_train_HR', ...)")
        print("4. Run all cells to train!")
        print("\nTraining time: ~6-7 hours on Colab GPU")
    else:
        print("\n⚠️  No datasets downloaded successfully.")
        print("Please check your internet connection and try again.")
    
    print("\nFor more information, see DATASETS.md")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user.")
        print("You can run this script again to resume.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
