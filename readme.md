# AI-Based Image Restoration & Colorization

A deep learning project for restoring and colorizing grayscale and historical photographs.

## Project Overview

The project combines three concepts:

1. **Automatic Image Colorization**  
   Converts grayscale images into realistic color images.

2. **Exemplar-Based Colorization**  
   Uses a reference color image to guide the colorization of a grayscale target image.

3. **Old Photo Restoration**  
   Restores degraded historical photographs by removing scratches, noise, blur, and other artifacts before colorization.

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

- Paper: https://arxiv.org/abs/1807.06587
- Implementation: https://github.com/msracver/Deep-Exemplar-based-Colorization

### 2. Bringing Old Photos Back to Life
Wan et al., CVPR 2020

- Paper: https://arxiv.org/abs/2004.09484
- Implementation: https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life

## Tech Stack

- Python
- PyTorch
- OpenCV
- NumPy
- Deep Learning
- Computer Vision

## Results

> **Outputs will be added here.**

### Automatic Colorization

_TBD_

### Exemplar-Based Colorization

_TBD_

### Old Photo Restoration + Colorization

_TBD_

## Future Work

- Improve color realism and semantic consistency
- Compare quantitative metrics such as PSNR, SSIM and LPIPS
- Build an interactive web interface
- Experiment with modern generative colorization methods