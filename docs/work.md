Now take the existing project and implement what we discussed.

I want ONE unified pipeline combining the two existing research implementations:

Old/damaged B&amp;W image

→ Old Photo Restoration

→ Restored grayscale

→ Deep Exemplar-based Colorization + color reference

→ Final color image

Keep the core ideas of both papers. The existing Caffe/C++ similarity subnet should be replaced with a PyTorch/Python implementation so the final project is purely Python.

Create:

1. `notebooks/unified_pipeline.ipynb`

   - This is the main file I will run directly on Google Colab.

   - It should contain the actual dataset setup, training/fine-tuning, checkpoint saving, and final end-to-end inference.

   - GPU support via PyTorch.

   - I am fine with 6–7 hours of training, so prioritize model quality/research fidelity over making training extremely fast.

   - Save trained models/checkpoints so I can download them and run inference locally.

2. [`app.py`](http://app.py)

   - Simple Flask server.

   - Loads the trained checkpoints.

   - Accepts an old/damaged image + color reference.

   - Runs the complete restoration → exemplar colorization pipeline.

   - No training inside the app.

3. `index.html`

   - Very basic UI.

   - Upload old/damaged image.

   - Upload color reference.

   - "Restore &amp; Colorize" button.

   - Show Original → Restored → Reference → Final output.

   - Plain HTML/CSS/JS only in index.html file

Keep everything simple and reuse the existing code wherever possible. Don't replace exemplar colorization with another method.

The important thing is: actually CREATE the `.ipynb`, [`app.py`](http://app.py), and `index.html` and wire them to the existing project. I will run the notebook on Colab, train the models, download the checkpoints, and then run `python [app.py](http://app.py)` locally.