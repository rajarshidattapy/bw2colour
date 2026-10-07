# Unified Old Photo Restoration + Exemplar Colorization Pipeline

A complete deep learning pipeline that restores damaged/old black & white photographs and colorizes them using reference images. Built with pure Python/PyTorch for easy deployment and training.

## 🎯 Project Overview

This project combines two research implementations into one unified pipeline:

1. **Old Photo Restoration** (Wan et al., CVPR 2020)
   - Removes scratches, noise, blur, and artifacts
   - Repairs damaged regions
   - Enhances overall quality

2. **Deep Exemplar-based Colorization** (He et al., SIGGRAPH 2018)
   - Colorizes grayscale images using reference color images
   - Semantic-aware color transfer
   - Preserves original image structure

### Pipeline Flow

```
📷 Damaged B&W Photo → 🔧 Restoration → 🎨 Exemplar Colorization → 🌈 Final Color Image
```

## 🏗️ Architecture

### Key Components

1. **`similarity_pytorch.py`** - PyTorch-based similarity subnet
   - Replaces original Caffe implementation
   - VGG19-based feature extraction
   - Multi-scale semantic similarity computation

2. **`notebooks/unified_pipeline.ipynb`** - Training notebook
   - Complete training pipeline for both models
   - Dataset preparation and augmentation
   - Checkpoint saving and visualization
   - End-to-end inference testing

3. **`app.py`** - Flask web server
   - Loads trained checkpoints
   - REST API for image processing
   - Handles file uploads and downloads

4. **`index.html`** - Web interface
   - Beautiful, responsive UI
   - Drag-and-drop image upload
   - Real-time processing status
   - Side-by-side result comparison

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- Google Colab account (for training)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/bw2colour.git
cd bw2colour

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Step 1: Train Models on Google Colab

1. Open `notebooks/unified_pipeline.ipynb` in Google Colab
2. Upload your dataset or use the provided sample data
3. Run all cells to train both models (~6-7 hours on Colab GPU)
4. Download the trained checkpoints:
   - `checkpoints/restoration_final.pth`
   - `checkpoints/colorization_final.pth`

### Step 2: Run Local Inference

1. Extract downloaded checkpoints to `checkpoints/` directory

```bash
# Project structure should be:
bw2colour/
├── checkpoints/
│   ├── restoration_final.pth
│   └── colorization_final.pth
├── app.py
├── index.html
├── similarity_pytorch.py
└── requirements.txt
```

2. Start the Flask server:

```bash
python app.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

4. Upload your old photo and reference image, then click "Restore & Colorize"!

## 📊 Training Details

### Restoration Model
- **Architecture**: U-Net with skip connections
- **Loss Functions**: L1 + MSE + Perceptual (VGG19)
- **Training Time**: ~3-4 hours (50 epochs on Colab GPU)
- **Input/Output**: 256x256 RGB images

### Colorization Model
- **Architecture**: Deep Exemplar-based ColorNet
- **Input**: L channel (grayscale) + RGB reference
- **Output**: AB channels (color)
- **Training Time**: ~2-3 hours (50 epochs on Colab GPU)

### Similarity Subnet
- **Pretrained**: VGG19 on ImageNet
- **No Training Required**: Used for feature extraction only
- **Layers**: conv3_1, conv4_1 for multi-scale matching

## 📁 Project Structure

```
bw2colour/
├── notebooks/
│   └── unified_pipeline.ipynb     # Main training notebook
├── colorization/                   # Original colorization code
├── restoration/                    # Original restoration code
├── checkpoints/                    # Trained model weights
├── datasets/                       # Training data
│   ├── old_photos/
│   └── color_images/
├── uploads/                        # Temporary uploaded files
├── outputs/                        # Processed results
├── similarity_pytorch.py           # PyTorch similarity module
├── app.py                          # Flask web server
├── index.html                      # Web UI
├── requirements.txt                # Python dependencies
└── README_UNIFIED.md              # This file
```

## 🎨 Web Interface Features

- **Drag & Drop Upload**: Easy file selection
- **Live Preview**: See uploaded images immediately
- **Progress Indication**: Real-time processing status
- **4-Panel Comparison**: Original → Restored → Reference → Colorized
- **One-Click Download**: Save processed images
- **Responsive Design**: Works on desktop and mobile
- **Modern UI**: Beautiful gradient design with smooth animations

## 🔧 API Endpoints

### `GET /`
Serves the main HTML interface

### `GET /health`
Health check endpoint
```json
{
  "status": "ok",
  "device": "cuda",
  "models_loaded": true
}
```

### `POST /process`
Process images through the pipeline

**Request**: Multipart form data
- `old_photo`: File (damaged/old photo)
- `reference`: File (color reference image)

**Response**: JSON with base64 encoded images
```json
{
  "success": true,
  "original": "data:image/png;base64,...",
  "restored": "data:image/png;base64,...",
  "colorized": "data:image/png;base64,...",
  "reference": "data:image/png;base64,..."
}
```

### `GET /download/<filename>`
Download processed images

## 📈 Performance Tips

### For Training (Colab)
- Use GPU runtime (Runtime → Change runtime type → GPU)
- Reduce batch size if running out of memory
- Adjust `num_epochs` based on dataset size
- Monitor loss curves to avoid overfitting

### For Inference (Local)
- Use CUDA if available (automatic detection)
- Process images at 256x256 for speed
- Results are resized back to original dimensions
- Close other GPU-intensive applications

## 🐛 Troubleshooting

### Issue: "Models not loaded" warning
**Solution**: Ensure checkpoint files are in `checkpoints/` directory with correct names

### Issue: CUDA out of memory
**Solution**: 
- Reduce batch size in training
- Use smaller images
- Enable CPU mode (automatically falls back)

### Issue: Import errors
**Solution**: 
```bash
pip install --upgrade -r requirements.txt
```

### Issue: Slow inference
**Solution**:
- Check if CUDA is available: `torch.cuda.is_available()`
- Update GPU drivers
- Process images in smaller batches

## 📚 Research Papers

1. **Deep Exemplar-based Colorization**
   - He et al., SIGGRAPH 2018
   - Paper: https://arxiv.org/abs/1807.06587

2. **Bringing Old Photos Back to Life**
   - Wan et al., CVPR 2020
   - Paper: https://arxiv.org/abs/2004.09484

## 🎯 Future Improvements

- [ ] Add automatic color reference selection
- [ ] Implement real-time video processing
- [ ] Support batch processing
- [ ] Add more restoration options (super-resolution, denoising)
- [ ] Create mobile app
- [ ] Fine-tune on larger datasets
- [ ] Add model quantization for faster inference

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project combines code from:
- Microsoft's "Bringing Old Photos Back to Life" (MIT License)
- Microsoft's "Deep Exemplar-based Colorization" (MIT License)

See individual `LICENSE` files in `restoration/` and `colorization/` directories.

## 🙏 Acknowledgments

- Original restoration implementation by Ziyu Wan et al.
- Original colorization implementation by Mingming He et al.
- PyTorch and torchvision teams
- Google Colab for free GPU access

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the maintainer.

---

**Happy Restoring & Colorizing! 🎨✨**
