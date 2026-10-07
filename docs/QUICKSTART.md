# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Train Models (Google Colab - 6-7 hours)

1. Open Google Colab: https://colab.research.google.com/

2. Upload `notebooks/unified_pipeline.ipynb` to Colab

3. Change runtime to GPU:
   - Runtime → Change runtime type → GPU → Save

4. Run all cells in order:
   - Environment Setup
   - Install Dependencies
   - Dataset Preparation (upload your images or use samples)
   - Train Restoration Model (~3-4 hours)
   - Train Colorization Model (~2-3 hours)
   - Test End-to-End Pipeline

5. Download trained checkpoints:
   ```python
   # Run this cell at the end of the notebook
   !zip -r checkpoints.zip checkpoints/
   ```
   - Download `checkpoints.zip` from Files panel
   - Extract to get `restoration_final.pth` and `colorization_final.pth`

### Step 2: Setup Local Environment (5 minutes)

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create checkpoints directory:**
   ```bash
   mkdir checkpoints
   ```

3. **Copy trained models:**
   - Place `restoration_final.pth` in `checkpoints/`
   - Place `colorization_final.pth` in `checkpoints/`

4. **Verify setup:**
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

### Step 3: Run the Web App (1 minute)

1. **Start Flask server:**
   ```bash
   python app.py
   ```

2. **Open browser:**
   ```
   http://localhost:5000
   ```

3. **Process your first image:**
   - Upload old/damaged photo (left)
   - Upload color reference (right)
   - Click "Restore & Colorize"
   - Wait ~30 seconds
   - Download results!

---

## 📝 Detailed Instructions

### Training Dataset Preparation

For best results, prepare your training data:

**Old Photos Dataset:**
- 500+ historical/old photographs
- Various degradation types (scratches, noise, blur)
- Different subjects (portraits, landscapes, buildings)
- Place in `datasets/old_photos/`

**Color Images Dataset:**
- 1000+ clean color images
- Diverse scenes and color palettes
- Similar subjects to target restoration
- Place in `datasets/color_images/`

**Or use public datasets:**
- [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) - High quality images
- [Places365](http://places2.csail.mit.edu/) - Scene images
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) - Face images

### Training Tips

**If you have limited time:**
- Reduce `num_epochs` to 20-30 (faster but lower quality)
- Use smaller dataset (100-200 images minimum)
- Train only colorization if you skip restoration

**If you have more GPU time:**
- Increase `num_epochs` to 100+
- Use larger batch sizes (if memory allows)
- Fine-tune on domain-specific data

**Monitoring training:**
- Watch loss curves (should decrease)
- Check intermediate outputs
- Save checkpoints every 10 epochs

### Local Inference Optimization

**For faster processing:**
```python
# In app.py, modify process_pipeline to use smaller size
transforms.Resize((128, 128))  # Instead of (256, 256)
```

**For better quality:**
```python
# Use larger size (requires more VRAM)
transforms.Resize((512, 512))
```

**CPU-only mode:**
- Automatically detected if no GPU
- ~2-3x slower than GPU
- Still works fine for single images

---

## 🎯 Common Use Cases

### Use Case 1: Restore Family Photos
```
Input: Scratched, faded family portrait from 1950s
Reference: Modern portrait with warm tones
Result: Restored and naturally colorized family photo
```

### Use Case 2: Historical Colorization
```
Input: Black & white historical photograph
Reference: Modern photo of similar scene
Result: Historically accurate colored image
```

### Use Case 3: Artistic Colorization
```
Input: Grayscale artistic photo
Reference: Vibrant color palette image
Result: Artistically stylized color version
```

---

## 💡 Pro Tips

### Choosing Good References

**Best practices:**
- ✅ Similar content (portrait → portrait)
- ✅ Desired color mood/palette
- ✅ High quality, well-lit images
- ✅ Clear colors and good contrast

**Avoid:**
- ❌ Very different subjects
- ❌ Poor quality or noisy references
- ❌ Extreme lighting or filters
- ❌ Black & white references

### Getting Best Results

1. **Pre-process your input:**
   - Crop to main subject
   - Remove borders
   - Adjust contrast if too flat

2. **Experiment with references:**
   - Try 2-3 different references
   - Compare results
   - Choose best colorization

3. **Post-process output:**
   - Adjust brightness/contrast
   - Fine-tune saturation
   - Sharpen if needed

---

## 🐛 Troubleshooting Quick Fixes

### Problem: Server won't start
```bash
# Kill any process using port 5000
# Windows:
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:5000 | xargs kill -9

# Or use different port in app.py:
app.run(port=8000)
```

### Problem: Out of memory during training
```python
# In notebook, reduce batch_size:
batch_size=4  # Instead of 8

# Or use gradient accumulation:
accumulation_steps = 2
```

### Problem: Poor colorization results
```
Solutions:
1. Train longer (more epochs)
2. Use better reference images
3. Increase training data
4. Fine-tune on specific domain
```

### Problem: Import errors
```bash
# Reinstall specific packages:
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Or use conda:
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

---

## 📊 Expected Results

**Training Progress:**
- Restoration: Loss should drop from ~1.0 to ~0.1
- Colorization: Loss should drop from ~0.5 to ~0.05
- Visual quality improves after epoch 20

**Inference Time:**
- GPU (CUDA): 5-10 seconds per image
- CPU: 20-40 seconds per image
- Batch processing: ~1-2 seconds per image

**Output Quality:**
- Restoration: Removes 80-90% of visible damage
- Colorization: Natural-looking colors
- Some artifacts may remain (expected)

---

## 🎓 Next Steps

After successful setup:

1. **Experiment**: Try different image types
2. **Fine-tune**: Train on your specific domain
3. **Optimize**: Improve speed/quality trade-offs
4. **Share**: Process images for friends/family
5. **Contribute**: Improve the codebase

---

## 📞 Need Help?

- Check `README_UNIFIED.md` for full documentation
- Review notebook comments for training details
- Inspect `app.py` for API customization
- Open GitHub issue for bugs

**Happy Restoring! 🎨✨**
