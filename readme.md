# AI-Based Image Restoration &amp; Colorization

A deep learning project for restoring and colorizing grayscale and historical photographs.

## Project Overview

The project combines three concepts:

1. **Automatic Image Colorization**  
 Converts grayscale images into realistic color images.
2. **Exemplar-Based Colorization**  
 Uses a reference color image to guide the colorization of a grayscale target image.
3. **Old Photo Restoration**  
 Restores degraded historical photographs by removing scratches, noise, blur, and other artifacts before colorization.

## Fast local demo (no training)

This repository now includes a pretrained DDColor demo for automatic colorization.

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`, upload one black-and-white image, and select **Colorize photo**. The compact DDColor model downloads from Hugging Face on its first run, then remains cached locally. The current demo colorizes images; it does not claim to repair scratches or perform reference-guided color transfer.

## Pipeline

```text
Input Image
     │
     ├── Grayscale ──────────► Automatic Colorization
     │
     ├── Grayscale + Reference ► Exemplar-Based Colorization
     │
     └── Damaged Old Photo ───► Restoration ──► Colorization
                                      │
                                      ▼
                              Final Color Image
```

## Research Papers

### 1. Deep Exemplar-based Colorization

He et al., 2018

- Paper: [https://arxiv.org/abs/1807.06587](https://arxiv.org/abs/1807.06587)
- Implementation: [https://github.com/msracver/Deep-Exemplar-based-Colorization](https://github.com/msracver/Deep-Exemplar-based-Colorization)

### 2. Bringing Old Photos Back to Life

Wan et al., CVPR 2020

- Paper: [https://arxiv.org/abs/2004.09484](https://arxiv.org/abs/2004.09484)
- Implementation: [https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life)

## Tech Stack

- Python
- PyTorch
- OpenCV
- NumPy
- Deep Learning
- Computer Vision

## Future Work

- Improve color realism and semantic consistency
- Compare quantitative metrics such as PSNR, SSIM and LPIPS
- Build an interactive web interface
- Experiment with modern generative colorization methods

