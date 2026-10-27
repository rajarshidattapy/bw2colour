"""Local pretrained DDColor demo for automatic photo colorization.

The model downloads from Hugging Face on its first use; no training is required.
"""

import base64
import io
import sys
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
DDCOLOR_ROOT = ROOT / "services" / "ddcolor"
sys.path.insert(0, str(DDCOLOR_ROOT))

from ddcolor import ColorizationPipeline, DDColor  # noqa: E402

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
colorizer = None
model_lock = Lock()
MODEL_NAME = "piddnad/ddcolor_paper_tiny"


def load_model():
    """Load the smaller official DDColor model, downloading it once if needed."""
    global model, colorizer
    if colorizer is not None:
        return
    with model_lock:
        if colorizer is not None:
            return
        from huggingface_hub import PyTorchModelHubMixin

        class DDColorHF(DDColor, PyTorchModelHubMixin):
            def __init__(self, config=None, **kwargs):
                if isinstance(config, dict):
                    kwargs = {**config, **kwargs}
                super().__init__(**kwargs)

        print(f"Loading {MODEL_NAME} on {device}. First run downloads the model.")
        model = DDColorHF.from_pretrained(MODEL_NAME).to(device).eval()
        colorizer = ColorizationPipeline(model, input_size=512, device=device)
        print("DDColor is ready.")


def as_data_url(image: Image.Image) -> str:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


@app.get("/")
def index():
    return send_file(ROOT / "index.html")


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "device": str(device),
        "models_loaded": colorizer is not None,
        "model": MODEL_NAME,
    })


@app.post("/process")
def process():
    if "old_photo" not in request.files:
        return jsonify(error="Upload an image first."), 400
    try:
        original = Image.open(request.files["old_photo"].stream).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify(error="The uploaded file is not a readable image."), 400

    try:
        load_model()
        original_bgr = cv2.cvtColor(np.asarray(original), cv2.COLOR_RGB2BGR)
        output_bgr = colorizer.process(original_bgr)
        colorized = Image.fromarray(cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB))
        original_url = as_data_url(original)
        colorized_url = as_data_url(colorized)
        return jsonify(
            success=True,
            original=original_url,
            colorized=colorized_url,
            # Kept for compatibility with the existing comparison UI.
            restored=original_url,
            reference=original_url,
        )
    except Exception as error:
        app.logger.exception("Colorization failed")
        return jsonify(error=f"Colorization failed: {error}"), 500


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify(error="Image is too large (maximum 15 MB)."), 413


if __name__ == "__main__":
    print("DDColor PhotoRevive demo: http://localhost:5000")
    # Load before accepting requests so the browser can accurately show ready state.
    load_model()
    app.run(host="127.0.0.1", port=5000, debug=False)
