"""
PyTorch-based Similarity Subnet - Replaces Caffe implementation
Computes semantic similarity between reference and target images for exemplar colorization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from scipy import ndimage
import cv2


class VGG19Features(nn.Module):
    """Extract VGG19 features for semantic similarity computation"""
    
    def __init__(self):
        super(VGG19Features, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        
        # Extract feature layers
        self.conv1_1 = nn.Sequential(*list(vgg19.features[:2]))
        self.conv1_2 = nn.Sequential(*list(vgg19.features[2:4]))
        self.conv2_1 = nn.Sequential(*list(vgg19.features[4:7]))
        self.conv2_2 = nn.Sequential(*list(vgg19.features[7:9]))
        self.conv3_1 = nn.Sequential(*list(vgg19.features[9:12]))
        self.conv3_2 = nn.Sequential(*list(vgg19.features[12:14]))
        self.conv3_3 = nn.Sequential(*list(vgg19.features[14:16]))
        self.conv3_4 = nn.Sequential(*list(vgg19.features[16:18]))
        self.conv4_1 = nn.Sequential(*list(vgg19.features[18:21]))
        self.conv4_2 = nn.Sequential(*list(vgg19.features[21:23]))
        self.conv4_3 = nn.Sequential(*list(vgg19.features[23:25]))
        self.conv4_4 = nn.Sequential(*list(vgg19.features[25:27]))
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x, layers=['conv3_1', 'conv4_1']):
        """
        Extract features from specified layers
        Args:
            x: Input tensor (B, C, H, W)
            layers: List of layer names to extract features from
        Returns:
            Dictionary of layer_name: features
        """
        features = {}
        
        out = self.conv1_1(x)
        if 'conv1_1' in layers:
            features['conv1_1'] = out
        out = self.conv1_2(out)
        if 'conv1_2' in layers:
            features['conv1_2'] = out
            
        out = F.max_pool2d(out, 2, 2)
        
        out = self.conv2_1(out)
        if 'conv2_1' in layers:
            features['conv2_1'] = out
        out = self.conv2_2(out)
        if 'conv2_2' in layers:
            features['conv2_2'] = out
            
        out = F.max_pool2d(out, 2, 2)
        
        out = self.conv3_1(out)
        if 'conv3_1' in layers:
            features['conv3_1'] = out
        out = self.conv3_2(out)
        if 'conv3_2' in layers:
            features['conv3_2'] = out
        out = self.conv3_3(out)
        if 'conv3_3' in layers:
            features['conv3_3'] = out
        out = self.conv3_4(out)
        if 'conv3_4' in layers:
            features['conv3_4'] = out
            
        out = F.max_pool2d(out, 2, 2)
        
        out = self.conv4_1(out)
        if 'conv4_1' in layers:
            features['conv4_1'] = out
        out = self.conv4_2(out)
        if 'conv4_2' in layers:
            features['conv4_2'] = out
        out = self.conv4_3(out)
        if 'conv4_3' in layers:
            features['conv4_3'] = out
        out = self.conv4_4(out)
        if 'conv4_4' in layers:
            features['conv4_4'] = out
        
        return features


class SimilaritySubnet:
    """
    Compute semantic similarity maps between target and reference images
    Replaces the Caffe-based similarity subnet from the original implementation
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.vgg_model = VGG19Features().to(device)
        self.vgg_model.eval()
        
        # VGG normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    def normalize_vgg(self, x):
        """Normalize image for VGG"""
        return (x - self.mean) / self.std
    
    def compute_similarity(self, target_gray, reference_color, layers=['conv3_1', 'conv4_1']):
        """
        Compute multi-scale semantic similarity between target and reference
        
        Args:
            target_gray: Grayscale target image tensor (1, 1, H, W) in [0, 1]
            reference_color: Color reference image tensor (1, 3, H, W) in [0, 1]
            layers: Feature layers to use for similarity computation
            
        Returns:
            similarity_maps: Dictionary containing similarity maps for each layer
        """
        # Convert grayscale to 3-channel for VGG
        target_3ch = target_gray.repeat(1, 3, 1, 1)
        
        # Normalize for VGG
        target_norm = self.normalize_vgg(target_3ch)
        reference_norm = self.normalize_vgg(reference_color)
        
        # Extract features
        with torch.no_grad():
            target_features = self.vgg_model(target_norm, layers)
            reference_features = self.vgg_model(reference_norm, layers)
        
        similarity_maps = {}
        
        for layer_name in layers:
            target_feat = target_features[layer_name]
            ref_feat = reference_features[layer_name]
            
            # Compute similarity map using normalized cross-correlation
            sim_map = self.compute_similarity_map(target_feat, ref_feat)
            similarity_maps[layer_name] = sim_map
        
        return similarity_maps
    
    def compute_similarity_map(self, target_feat, ref_feat):
        """
        Compute normalized cross-correlation similarity map
        
        Args:
            target_feat: Target features (1, C, H_t, W_t)
            ref_feat: Reference features (1, C, H_r, W_r)
            
        Returns:
            similarity_map: (1, H_t, W_t, H_r*W_r) similarity scores
        """
        B, C, H_t, W_t = target_feat.shape
        _, _, H_r, W_r = ref_feat.shape
        
        # Normalize features
        target_norm = F.normalize(target_feat.view(B, C, -1), dim=1)  # (B, C, H_t*W_t)
        ref_norm = F.normalize(ref_feat.view(B, C, -1), dim=1)  # (B, C, H_r*W_r)
        
        # Compute cosine similarity
        similarity = torch.matmul(target_norm.transpose(1, 2), ref_norm)  # (B, H_t*W_t, H_r*W_r)
        
        # Reshape to spatial dimensions
        similarity_map = similarity.view(B, H_t, W_t, H_r * W_r)
        
        return similarity_map
    
    def generate_combo_file(self, target_gray, reference_color, warp_func=None):
        """
        Generate similarity data structure similar to original .combo file format
        
        Args:
            target_gray: Grayscale target image (H, W) numpy array [0, 255]
            reference_color: Color reference image (H, W, 3) numpy array [0, 255]
            warp_func: Optional warping function for bidirectional mapping
            
        Returns:
            combo_data: Dictionary containing similarity maps and warped images
        """
        # Convert to tensors
        target_tensor = torch.from_numpy(target_gray / 255.0).unsqueeze(0).unsqueeze(0).float().to(self.device)
        reference_tensor = torch.from_numpy(reference_color.transpose(2, 0, 1) / 255.0).unsqueeze(0).float().to(self.device)
        
        # Compute multi-scale similarities
        similarity_maps = self.compute_similarity(target_tensor, reference_tensor, layers=['conv3_1', 'conv4_1'])
        
        # Generate warped images (simplified - using simple matching for now)
        # In the original paper, they use Deep Image Analogy for better warping
        warp_ba, warp_aba = self.simple_warp(target_gray, reference_color, similarity_maps)
        
        # Generate error maps
        err_maps = self.compute_error_maps(target_gray, reference_color, warp_ba, warp_aba)
        
        combo_data = {
            'similarity_maps': similarity_maps,
            'warp_ba': warp_ba,  # Reference warped to target space
            'warp_aba': warp_aba,  # Target warped to ref and back
            'error_maps': err_maps,
            'target': target_gray,
            'reference': reference_color
        }
        
        return combo_data
    
    def simple_warp(self, target_gray, reference_color, similarity_maps):
        """
        Simple warping based on similarity maps
        For better results, use Deep Image Analogy or optical flow
        """
        # Use conv4_1 similarity for warping
        sim_map = similarity_maps['conv4_1'].squeeze(0)  # (H_t, W_t, H_r*W_r)
        H_t, W_t, _ = sim_map.shape
        H_r, W_r = reference_color.shape[:2]
        
        # Find best match for each target pixel
        best_matches = torch.argmax(sim_map, dim=2)  # (H_t, W_t)
        
        # Convert to reference coordinates
        ref_y = (best_matches // W_r).cpu().numpy()
        ref_x = (best_matches % W_r).cpu().numpy()
        
        # Resize to match target size if needed
        if H_t != target_gray.shape[0] or W_t != target_gray.shape[1]:
            ref_y = cv2.resize(ref_y.astype(np.float32), (target_gray.shape[1], target_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
            ref_x = cv2.resize(ref_x.astype(np.float32), (target_gray.shape[1], target_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Warp reference to target space
        warp_ba = np.zeros_like(reference_color)
        ref_y = np.clip(ref_y.astype(int), 0, H_r - 1)
        ref_x = np.clip(ref_x.astype(int), 0, W_r - 1)
        
        for c in range(3):
            warp_ba[:, :, c] = reference_color[ref_y, ref_x, c]
        
        # Simple approximation for warp_aba (would need bidirectional mapping)
        warp_aba = warp_ba.copy()
        
        return warp_ba, warp_aba
    
    def compute_error_maps(self, target_gray, reference_color, warp_ba, warp_aba):
        """Compute error maps at multiple scales"""
        # Convert to grayscale if needed
        if len(warp_ba.shape) == 3:
            warp_ba_gray = cv2.cvtColor(warp_ba.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            warp_aba_gray = cv2.cvtColor(warp_aba.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            warp_ba_gray = warp_ba
            warp_aba_gray = warp_aba
        
        # Compute errors at multiple scales
        error_maps = []
        for scale in [1, 0.5, 0.25, 0.125, 0.0625]:
            if scale != 1:
                h, w = int(target_gray.shape[0] * scale), int(target_gray.shape[1] * scale)
                target_scaled = cv2.resize(target_gray, (w, h))
                warp_ba_scaled = cv2.resize(warp_ba_gray, (w, h))
                warp_aba_scaled = cv2.resize(warp_aba_gray, (w, h))
            else:
                target_scaled = target_gray
                warp_ba_scaled = warp_ba_gray
                warp_aba_scaled = warp_aba_gray
            
            err_ba = np.abs(target_scaled.astype(float) - warp_ba_scaled.astype(float))
            err_ab = np.abs(target_scaled.astype(float) - warp_aba_scaled.astype(float))
            
            error_maps.append({
                'err_ba': err_ba,
                'err_ab': err_ab
            })
        
        return error_maps


def compute_similarity_for_colorization(target_gray_path, reference_color_path, device='cuda'):
    """
    High-level function to compute similarity data for colorization
    
    Args:
        target_gray_path: Path to grayscale target image
        reference_color_path: Path to color reference image
        device: 'cuda' or 'cpu'
    
    Returns:
        combo_data: Dictionary containing all similarity and warping information
    """
    # Load images
    target_gray = cv2.imread(target_gray_path, cv2.IMREAD_GRAYSCALE)
    reference_color = cv2.imread(reference_color_path, cv2.IMREAD_COLOR)
    reference_color = cv2.cvtColor(reference_color, cv2.COLOR_BGR2RGB)
    
    # Initialize similarity subnet
    similarity_subnet = SimilaritySubnet(device=device)
    
    # Compute similarity data
    combo_data = similarity_subnet.generate_combo_file(target_gray, reference_color)
    
    return combo_data
