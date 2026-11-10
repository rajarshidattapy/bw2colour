# Dataset Guide for Old Photo Restoration + Colorization

This guide lists all recommended datasets with direct download links and usage instructions.

---

## 📊 Quick Summary

### For Restoration Training
1. **OpenPhoto Restore Dataset** (HuggingFace) - 5,000 pairs ⭐ RECOMMENDED
2. **DIV2K** - 900 high-quality images
3. **Your own old photos** - Real historical images

### For Colorization Training
1. **Places365** - 1.8M scene images ⭐ RECOMMENDED
2. **DIV2K** - 900 high-quality images
3. **ImageNet** - 1.2M diverse images
4. **COCO** - Natural images

---

## 🎯 Recommended Datasets (Easiest to Use)

### 1. OpenPhoto Restore Dataset ⭐ BEST FOR RESTORATION

**What it is:** Synthetic photo restoration dataset with 5,000 pristine + damaged image pairs

**Why use it:**
- ✅ Ready-made damaged/clean pairs
- ✅ Permissive CC-BY-4.0 license
- ✅ High quality 1024x1024 images
- ✅ Realistic damage simulation
- ✅ Easy to download via HuggingFace

**Download:**
```python
# In your notebook or Python script
from datasets import load_dataset

# Download dataset
dataset = load_dataset("joshuachin/openphoto-restore-dataset")
train_data = dataset['train']  # 4,500 images
test_data = dataset['test']    # 500 images

# Access images
for example in train_data:
    pristine_img = example['pristine_image']  # Clean image
    damaged_img = example['damaged_image']    # Damaged version
```

**Manual download:** https://huggingface.co/datasets/joshuachin/openphoto-restore-dataset

**Size:** ~4.42 GB

**Use for:** Restoration model training (pairs are perfect!)

---

### 2. Places365-Standard ⭐ BEST FOR COLORIZATION

**What it is:** 1.8 million images from 365 scene categories

**Why use it:**
- ✅ Huge diversity of scenes
- ✅ High quality color images
- ✅ Perfect for colorization training
- ✅ Widely used in research

**Download Options:**

**Option A: Small version (36 GB, faster)**
```bash
# Download 256x256 version
wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
tar -xf places365standard_easyformat.tar
```

**Option B: High-res version (105 GB)**
```bash
# Download full resolution
wget http://data.csail.mit.edu/places/places365/train_large_places365standard.tar
tar -xf train_large_places365standard.tar
```

**Option C: Using TensorFlow Datasets**
```python
import tensorflow_datasets as tfds

# Automatic download
dataset = tfds.load('places365_small', split='train', with_info=True)
```

**Official website:** http://places2.csail.mit.edu/download.html

**License:** Free for research and educational purposes

**Use for:** Colorization reference images training

---

### 3. DIV2K Dataset - HIGH QUALITY IMAGES

**What it is:** 900 high-resolution (2K) images for super-resolution and restoration

**Why use it:**
- ✅ Very high quality
- ✅ 2K resolution
- ✅ Good for both restoration and colorization
- ✅ Manageable size

**Download:**

**Official download:** https://data.vision.ee.ethz.ch/cvl/DIV2K/

**Direct links:**
```bash
# DIV2K Training Data (800 images)
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip

# DIV2K Validation Data (100 images)
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip

# Unzip
unzip DIV2K_train_HR.zip
unzip DIV2K_valid_HR.zip
```

**Using Python:**
```python
import tensorflow_datasets as tfds

dataset = tfds.load('div2k', split='train')
```

**Size:** ~3 GB

**License:** Academic research only

**Use for:** Both restoration and colorization training

---

## 🎨 Additional Colorization Datasets

### 4. COCO Dataset (Microsoft)

**What it is:** 330K images with diverse natural scenes

**Download:**
```bash
# 2017 Train images (18GB)
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
```

**Website:** https://cocodataset.org/#download

**Size:** ~20 GB

---

### 5. ImageNet (Subset)

**What it is:** 1.2M images across 1000 categories

**Download:** Requires registration at https://www.image-net.org/download.php

**Alternative - ImageNet subset:**
```python
# Using PyTorch
from torchvision.datasets import ImageNet

dataset = ImageNet(root='./data', split='train', download=False)
# Note: download=True doesn't work, manual download required
```

**Size:** ~150 GB (full), ~50 GB (subset)

---

## 🖼️ Specialized Datasets

### 6. CelebA (Face Images)

**What it is:** 200K celebrity face images

**Why use it:** If you focus on portrait restoration

**Download:**
```bash
# Google Drive link (requires gdrive tool or manual download)
# Visit: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
```

**Using PyTorch:**
```python
from torchvision.datasets import CelebA

dataset = CelebA(root='./data', split='train', download=True)
```

**Size:** ~1.4 GB

---

### 7. Flickr-Faces-HQ (FFHQ)

**What it is:** 70K high-quality face images at 1024×1024

**Download:** https://github.com/NVlabs/ffhq-dataset

**Size:** ~13 GB

**Use for:** High-quality portrait restoration

---

## 📥 Complete Setup Script

Save this as `download_datasets.py`:

```python
"""
Download datasets for old photo restoration and colorization
"""

import os
from datasets import load_dataset
import urllib.request
import zipfile

def download_openphoto():
    """Download OpenPhoto Restore Dataset from HuggingFace"""
    print("Downloading OpenPhoto Restore Dataset...")
    dataset = load_dataset("joshuachin/openphoto-restore-dataset")
    
    # Save to disk
    dataset.save_to_disk("datasets/openphoto_restore")
    print(f"✓ Saved {len(dataset['train'])} train images")
    print(f"✓ Saved {len(dataset['test'])} test images")

def download_div2k():
    """Download DIV2K dataset"""
    print("Downloading DIV2K...")
    os.makedirs("datasets/div2k", exist_ok=True)
    
    urls = [
        "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
    ]
    
    for url in urls:
        filename = url.split("/")[-1]
        filepath = f"datasets/div2k/{filename}"
        
        if not os.path.exists(filepath):
            print(f"  Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            
            # Unzip
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall("datasets/div2k/")
            print(f"  ✓ Extracted {filename}")

def download_places365_small():
    """Download Places365 small version"""
    print("Downloading Places365-Small...")
    print("⚠ This is a large download (~36 GB)")
    
    import tensorflow_datasets as tfds
    
    # This will auto-download
    dataset = tfds.load('places365_small', split='train', download=True)
    print("✓ Places365 downloaded via TensorFlow Datasets")

def main():
    print("=" * 60)
    print("Dataset Download Script")
    print("=" * 60)
    
    # Create directories
    os.makedirs("datasets", exist_ok=True)
    
    print("\nSelect datasets to download:")
    print("1. OpenPhoto Restore Dataset (4.4GB) - RECOMMENDED")
    print("2. DIV2K (3GB)")
    print("3. Places365-Small (36GB)")
    print("4. All of the above")
    
    choice = input("\nEnter choice (1-4): ")
    
    if choice == "1" or choice == "4":
        download_openphoto()
    
    if choice == "2" or choice == "4":
        download_div2k()
    
    if choice == "3" or choice == "4":
        download_places365_small()
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print("\nDatasets saved in: ./datasets/")
    print("\nNext steps:")
    print("1. Open notebooks/unified_pipeline.ipynb")
    print("2. Update dataset paths in the notebook")
    print("3. Start training!")

if __name__ == "__main__":
    main()
```

---

## 🚀 Quick Start (Recommended)

### Minimum Setup (Fast, ~5GB)

```bash
# For quick experimentation
python download_datasets.py
# Choose option 1 (OpenPhoto only)
```

**This gives you:**
- 4,500 training pairs for restoration
- 500 test pairs
- Ready to train in ~2-3 hours

### Optimal Setup (~40GB)

```bash
# For best results
python download_datasets.py
# Choose option 4 (All datasets)
```

**This gives you:**
- OpenPhoto for restoration training
- DIV2K for high-quality images
- Places365 for colorization diversity

---

## 📋 Integration with Training Notebook

In `unified_pipeline.ipynb`, update these paths:

```python
# For Restoration Training
restoration_dataset = RestorationDataset(
    'datasets/openphoto_restore/train',  # ← Update this
    transform=transform,
    add_synthetic_degradation=False  # Already have damaged images
)

# For Colorization Training
colorization_dataset = ColorizationDataset(
    'datasets/div2k/DIV2K_train_HR',  # ← Or use 'datasets/places365'
    transform=color_transform
)
```

---

## 💡 Dataset Recommendations by Use Case

### Personal/Family Photos
- **Use:** DIV2K + OpenPhoto
- **Why:** High quality, portrait-friendly
- **Training time:** ~4-5 hours

### Historical Archives
- **Use:** OpenPhoto + Places365
- **Why:** Scene diversity, realistic damage
- **Training time:** ~6-7 hours

### Portrait Focus
- **Use:** CelebA + FFHQ
- **Why:** Specialized for faces
- **Training time:** ~5-6 hours

### General Purpose
- **Use:** OpenPhoto + DIV2K + Places365 (subset)
- **Why:** Best overall coverage
- **Training time:** ~6-7 hours

---

## 🔍 Dataset Quality Checklist

Before using a dataset, verify:

- ✅ Images are high resolution (>256x256)
- ✅ License allows your use case
- ✅ Diverse subjects/scenes
- ✅ Good color distribution
- ✅ No corrupted files
- ✅ Sufficient quantity (>1000 images recommended)

---

## ⚡ Pro Tips

### 1. Start Small
Train on 100-200 images first to verify pipeline works

### 2. Mix Datasets
Combine multiple sources for better generalization:
```python
dataset = ConcatDataset([
    dataset_openphoto,
    dataset_div2k,
    dataset_places365_subset
])
```

### 3. Data Augmentation
Add synthetic degradation to clean images:
- Gaussian noise
- JPEG compression
- Random scratches
- Color shifts

### 4. Quality Over Quantity
500 high-quality images > 5000 low-quality images

### 5. Save Preprocessed Data
Save images at training resolution (256x256) to speed up loading:
```python
# Preprocess and save
for img in dataset:
    img_resized = img.resize((256, 256))
    img_resized.save(f'processed/{filename}')
```

---

## 🐛 Troubleshooting

### Problem: Download fails
**Solution:** Try manual download from website, then extract to `datasets/`

### Problem: Out of disk space
**Solution:** Use smaller datasets:
- Places365-Small instead of full
- DIV2K only (3GB)
- ImageNet subset

### Problem: Dataset takes too long to load
**Solution:** 
- Use LMDB format
- Preprocess to target size
- Use fewer workers in DataLoader

### Problem: License concerns
**Solution:** Check usage rights:
- Academic: Most datasets OK
- Commercial: Use CC-BY or public domain only

---

## 📞 Dataset Support

### OpenPhoto Issues
- GitHub: Check dataset repository
- HuggingFace: Forum support

### DIV2K Issues
- Official website: https://data.vision.ee.ethz.ch/cvl/DIV2K/

### Places365 Issues
- Email: places@csail.mit.edu

---

## 🎓 Citations

If you use these datasets in publications, cite them:

**OpenPhoto:**
```
@misc{chin2025openphoto,
  author = {Joshua Chin},
  title = {OpenPhoto Restore Dataset},
  year = {2025},
  url = {https://huggingface.co/datasets/joshuachin/openphoto-restore-dataset}
}
```

**DIV2K:**
```
@InProceedings{Agustsson_2017_CVPR_Workshops,
  author = {Agustsson, Eirikur and Timofte, Radu},
  title = {NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study},
  booktitle = {CVPR Workshops},
  year = {2017}
}
```

**Places365:**
```
@article{zhou2017places,
  title={Places: A 10 million Image Database for Scene Recognition},
  author={Zhou, Bolei and Lapedriza, Agata and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
  journal={IEEE TPAMI},
  year={2017}
}
```

---

**Happy Training! 🎨✨**

*Last updated: 2026-10-07*
