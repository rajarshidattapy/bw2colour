"""
Flask Application for Unified Old Photo Restoration + Exemplar Colorization
Loads trained checkpoints and provides REST API for inference
"""

from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from skimage import color
import io
import base64
import os
from pathlib import Path
import cv2

# Import models and similarity subnet
from similarity_pytorch import SimilaritySubnet


# ===== Model Definitions (same as in notebook) =====

class RestorationUNet(nn.Module):
    """U-Net architecture for old photo restoration"""
    
    def __init__(self, in_channels=3, out_channels=3):
        super(RestorationUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.dec4 = self.upconv_block(1024, 512)
        self.dec3 = self.upconv_block(512, 256)
        self.dec2 = self.upconv_block(256, 128)
        self.dec1 = self.upconv_block(128, 64)
        
        # Final output
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2, 2)
    
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4], dim=1) if d4.shape[2:] == e4.shape[2:] else d4
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1) if d3.shape[2:] == e3.shape[2:] else d3
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1) if d2.shape[2:] == e2.shape[2:] else d2
        d1 = self.dec1(d2)
        
        out = self.final(d1)
        return torch.tanh(out)


class ExampleColorNet(nn.Module):
    """Exemplar-based colorization network"""
    
    def __init__(self, ic):
        super(ExampleColorNet, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(ic, 32, 3, 1, 1), nn.ReLU(),
                                     nn.Conv2d(32, 64, 3, 1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv1_2norm = nn.BatchNorm2d(64, affine=False)
        self.conv1_2norm_ss = nn.Conv2d(64, 64, 1, 2, bias=False, groups=64)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv2_2norm = nn.BatchNorm2d(128, affine=False)
        self.conv2_2norm_ss = nn.Conv2d(128, 128, 1, 2, bias=False, groups=128)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3norm = nn.BatchNorm2d(256, affine=False)
        self.conv3_3norm_ss = nn.Conv2d(256, 256, 1, 2, bias=False, groups=256)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3norm = nn.BatchNorm2d(512, affine=False)
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_3norm = nn.BatchNorm2d(512, affine=False)
        self.conv6_1 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_2 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_3 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_3norm = nn.BatchNorm2d(512, affine=False)
        self.conv7_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_3norm = nn.BatchNorm2d(512, affine=False)
        self.conv8_1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.conv3_3_short = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_3norm = nn.BatchNorm2d(256, affine=False)
        self.conv9_1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv2_2_short = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv9_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv9_2norm = nn.BatchNorm2d(128, affine=False)
        self.conv10_1 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.conv1_2_short = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv10_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv10_ab = nn.Conv2d(128, 2, 1, 1)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        conv1_1 = torch.relu(self.conv1_1(x))
        conv1_2 = torch.relu(self.conv1_2(conv1_1))
        conv1_2norm = self.conv1_2norm(conv1_2)
        conv1_2norm_ss = self.conv1_2norm_ss(conv1_2norm)
        conv2_1 = torch.relu(self.conv2_1(conv1_2norm_ss))
        conv2_2 = torch.relu(self.conv2_2(conv2_1))
        conv2_2norm = self.conv2_2norm(conv2_2)
        conv2_2norm_ss = self.conv2_2norm_ss(conv2_2norm)
        conv3_1 = torch.relu(self.conv3_1(conv2_2norm_ss))
        conv3_2 = torch.relu(self.conv3_2(conv3_1))
        conv3_3 = torch.relu(self.conv3_3(conv3_2))
        conv3_3norm = self.conv3_3norm(conv3_3)
        conv3_3norm_ss = self.conv3_3norm_ss(conv3_3norm)
        conv4_1 = torch.relu(self.conv4_1(conv3_3norm_ss))
        conv4_2 = torch.relu(self.conv4_2(conv4_1))
        conv4_3 = torch.relu(self.conv4_3(conv4_2))
        conv4_3norm = self.conv4_3norm(conv4_3)
        conv5_1 = torch.relu(self.conv5_1(conv4_3norm))
        conv5_2 = torch.relu(self.conv5_2(conv5_1))
        conv5_3 = torch.relu(self.conv5_3(conv5_2))
        conv5_3norm = self.conv5_3norm(conv5_3)
        conv6_1 = torch.relu(self.conv6_1(conv5_3norm))
        conv6_2 = torch.relu(self.conv6_2(conv6_1))
        conv6_3 = torch.relu(self.conv6_3(conv6_2))
        conv6_3norm = self.conv6_3norm(conv6_3)
        conv7_1 = torch.relu(self.conv7_1(conv6_3norm))
        conv7_2 = torch.relu(self.conv7_2(conv7_1))
        conv7_3 = torch.relu(self.conv7_3(conv7_2))
        conv7_3norm = self.conv7_3norm(conv7_3)
        conv8_1 = self.conv8_1(conv7_3norm)
        conv3_3_short = self.conv3_3_short(conv3_3norm)
        conv8_1_comb = torch.relu(conv8_1 + conv3_3_short)
        conv8_2 = torch.relu(self.conv8_2(conv8_1_comb))
        conv8_3 = torch.relu(self.conv8_3(conv8_2))
        conv8_3norm = self.conv8_3norm(conv8_3)
        conv9_1 = self.conv9_1(conv8_3norm)
        conv2_2_short = self.conv2_2_short(conv2_2norm)
        conv9_1_comb = torch.relu(conv9_1 + conv2_2_short)
        conv9_2 = torch.relu(self.conv9_2(conv9_1_comb))
        conv9_2norm = self.conv9_2norm(conv9_2)
        conv10_1 = self.conv10_1(conv9_2norm)
        conv1_2_short = self.conv1_2_short(conv1_2norm)
        conv10_1_comb = torch.relu(conv10_1 + conv1_2_short)
        conv10_2 = self.leaky_relu(self.conv10_2(conv10_1_comb))
        conv10_ab = self.conv10_ab(conv10_2)
        pred_ab = torch.tanh(conv10_ab) * 100
        return pred_ab


# ===== Flask App Setup =====

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
CHECKPOINT_FOLDER = 'checkpoints'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Global model variables
restoration_model = None
colorization_model = None
similarity_subnet = None


def load_models():
    """Load trained models from checkpoints"""
    global restoration_model, colorization_model, similarity_subnet
    
    print("Loading models...")
    
    # Load restoration model
    restoration_model = RestorationUNet(in_channels=3, out_channels=3)
    restoration_checkpoint_path = os.path.join(CHECKPOINT_FOLDER, 'restoration_final.pth')
    
    if os.path.exists(restoration_checkpoint_path):
        restoration_model.load_state_dict(torch.load(restoration_checkpoint_path, map_location=device))
        restoration_model = restoration_model.to(device)
        restoration_model.eval()
        print("✓ Restoration model loaded")
    else:
        print(f"⚠ Warning: Restoration checkpoint not found at {restoration_checkpoint_path}")
    
    # Load colorization model
    colorization_model = ExampleColorNet(ic=4)
    colorization_checkpoint_path = os.path.join(CHECKPOINT_FOLDER, 'colorization_final.pth')
    
    if os.path.exists(colorization_checkpoint_path):
        colorization_model.load_state_dict(torch.load(colorization_checkpoint_path, map_location=device))
        colorization_model = colorization_model.to(device)
        colorization_model.eval()
        print("✓ Colorization model loaded")
    else:
        print(f"⚠ Warning: Colorization checkpoint not found at {colorization_checkpoint_path}")
    
    # Initialize similarity subnet
    similarity_subnet = SimilaritySubnet(device=device)
    print("✓ Similarity subnet initialized")
    
    print("All models loaded successfully!")


def lab2rgb(L, AB):
    """Convert LAB to RGB"""
    # Denormalize
    L = L * 100.0 + 50.0
    AB = AB * 110.0
    
    # Concatenate
    lab = np.concatenate([L, AB], axis=2)
    
    # Convert to RGB
    rgb = color.lab2rgb(lab)
    rgb = (rgb * 255).astype(np.uint8)
    
    return rgb


def process_pipeline(old_photo_path, reference_path):
    """
    Complete pipeline: Old/Damaged Photo → Restoration → Colorization
    """
    # Load images
    old_photo = Image.open(old_photo_path).convert('RGB')
    original_size = old_photo.size
    
    # 1. Restoration
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    old_tensor = transform(old_photo).unsqueeze(0).to(device)
    
    with torch.no_grad():
        restored_tensor = restoration_model(old_tensor)
    
    # Denormalize
    restored_tensor = (restored_tensor + 1) / 2
    restored_np = restored_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    restored_gray = color.rgb2gray(restored_np)
    
    # 2. Colorization
    reference = Image.open(reference_path).convert('RGB')
    reference = reference.resize((256, 256))
    reference_np = np.array(reference).astype(np.float32) / 255.0
    reference_tensor = torch.from_numpy(reference_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
    
    # Convert to L channel
    L = (restored_gray * 100.0 - 50.0) / 100.0
    L_tensor = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).float().to(device)
    
    # Compute similarity
    with torch.no_grad():
        sim_maps = similarity_subnet.compute_similarity(
            L_tensor, reference_tensor, layers=['conv3_1', 'conv4_1']
        )
    
    # Colorize
    input_tensor = torch.cat([L_tensor, reference_tensor], dim=1)
    
    with torch.no_grad():
        AB_pred = colorization_model(input_tensor)
    
    # Convert to RGB
    L_out = L_tensor.squeeze().cpu().numpy()
    AB_out = AB_pred.squeeze().cpu().numpy() / 110.0
    
    L_out = L_out[np.newaxis, :, :].transpose(1, 2, 0)
    AB_out = AB_out.transpose(1, 2, 0)
    
    colorized = lab2rgb(L_out, AB_out)
    
    # Resize back
    colorized_final = Image.fromarray(colorized).resize(original_size, Image.LANCZOS)
    restored_final = Image.fromarray((restored_np * 255).astype(np.uint8)).resize(original_size, Image.LANCZOS)
    
    return {
        'original': old_photo,
        'restored': restored_final,
        'colorized': colorized_final,
        'reference': reference
    }


def pil_to_base64(img):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


# ===== API Routes =====

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_file('index.html')


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'device': str(device),
        'models_loaded': restoration_model is not None and colorization_model is not None
    })


@app.route('/process', methods=['POST'])
def process_images():
    """
    Main endpoint to process images through the pipeline
    Expects: old_photo (file), reference (file)
    Returns: JSON with base64 encoded images
    """
    try:
        # Check if files are present
        if 'old_photo' not in request.files or 'reference' not in request.files:
            return jsonify({'error': 'Both old_photo and reference files are required'}), 400
        
        old_photo_file = request.files['old_photo']
        reference_file = request.files['reference']
        
        # Save uploaded files
        old_photo_path = os.path.join(UPLOAD_FOLDER, 'old_photo_temp.jpg')
        reference_path = os.path.join(UPLOAD_FOLDER, 'reference_temp.jpg')
        
        old_photo_file.save(old_photo_path)
        reference_file.save(reference_path)
        
        # Process through pipeline
        results = process_pipeline(old_photo_path, reference_path)
        
        # Convert to base64 for JSON response
        response = {
            'success': True,
            'original': pil_to_base64(results['original']),
            'restored': pil_to_base64(results['restored']),
            'colorized': pil_to_base64(results['colorized']),
            'reference': pil_to_base64(results['reference'])
        }
        
        # Save output files
        results['colorized'].save(os.path.join(OUTPUT_FOLDER, 'colorized_output.png'))
        results['restored'].save(os.path.join(OUTPUT_FOLDER, 'restored_output.png'))
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error processing images: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download processed images"""
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404


# ===== Main =====

if __name__ == '__main__':
    print("=" * 50)
    print("Old Photo Restoration + Colorization Server")
    print("=" * 50)
    
    # Load models on startup
    load_models()
    
    print("\nServer starting...")
    print("Open http://localhost:5000 in your browser")
    print("=" * 50)
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
