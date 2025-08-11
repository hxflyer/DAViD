"""
Demo script to run the trained multi-task model checkpoint with samples from img/ directory.
Uses the same approach as vis_util.py with ground truth data loaded from img/.
"""

import torch
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import glob
import OpenEXR
import Imath

# Import model architecture
from multi_head_dpt import create_multi_head_dpt
from vis_util import align_depth_for_visualization


def load_model_checkpoint(checkpoint_path, device='cuda'):
    """
    Load the trained multi-task model from checkpoint.
    """
    # Create the model architecture
    model = create_multi_head_dpt(
        backbone="vitb16_384",
        features=256,
        use_bn=False,
        pretrained=True
    )
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state dict (handle different checkpoint formats)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded model from epoch: {epoch}")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Assume the entire checkpoint is the state dict
        state_dict = checkpoint
    
    # Handle DDP wrapped models (remove 'module.' prefix if present)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    # Load weights into model
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model


def load_exr_file(filepath):
    """Load EXR file using OpenEXR."""
    try:
        exr_file = OpenEXR.InputFile(filepath)
        header = exr_file.header()
        
        # Get image dimensions
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        # Get available channels
        channels = list(header['channels'].keys())
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        
        if 'Y' in channels:
            # Depth file - single Y channel
            channel_data = exr_file.channel('Y', FLOAT)
            data = np.frombuffer(channel_data, dtype=np.float32)
            data = data.reshape((height, width))
            
        elif all(c in channels for c in ['R', 'G', 'B']):
            # Normal file - RGB channels
            r_data = exr_file.channel('R', FLOAT)
            g_data = exr_file.channel('G', FLOAT)
            b_data = exr_file.channel('B', FLOAT)
            
            r = np.frombuffer(r_data, dtype=np.float32).reshape((height, width))
            g = np.frombuffer(g_data, dtype=np.float32).reshape((height, width))
            b = np.frombuffer(b_data, dtype=np.float32).reshape((height, width))
            
            data = np.stack([r, g, b], axis=2)  # [H, W, 3]
            
        else:
            raise ValueError(f"Unsupported EXR channels: {channels}")
        
        exr_file.close()
        return data
        
    except Exception as e:
        print(f"❌ Error loading EXR file {filepath}: {e}")
        return None


def load_samples_from_img_dir(img_dir="img", target_size=384):
    """
    Load 4 samples from img/ directory.
    
    Args:
        img_dir: Directory containing the sample files
        target_size: Target image size for model input
        
    Returns:
        List of sample dictionaries with 'rgb', 'depth', 'normals', 'alpha'
    """
    print(f"📊 Loading samples from: {img_dir}")
    
    # Find all sample files
    rgb_files = sorted(glob.glob(os.path.join(img_dir, "sample_*_rgb.png")))
    print(f"📈 Found {len(rgb_files)} samples")
    
    samples = []
    for rgb_file in rgb_files:
        # Extract sample name (e.g., "sample_01_0000428")
        sample_name = os.path.basename(rgb_file).replace("_rgb.png", "")
        
        # Define file paths
        depth_file = os.path.join(img_dir, f"{sample_name}_depth.exr")
        normal_file = os.path.join(img_dir, f"{sample_name}_normal.exr")
        alpha_file = os.path.join(img_dir, f"{sample_name}_alpha.png")
        
        
        # Load RGB
        rgb_image = Image.open(rgb_file).convert('RGB')
        rgb_image = rgb_image.resize((target_size, target_size), Image.LANCZOS)
        rgb_np = np.array(rgb_image).astype(np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb_np.transpose(2, 0, 1))  # [3, H, W]
        
        # Load depth EXR
        depth_np = load_exr_file(depth_file)

        depth_np = cv2.resize(depth_np, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        # Process depth: normalize to [0, 1] range (same as dataset processing)
        depth_valid = depth_np > 0
        if depth_valid.any():
            depth_clipped = np.clip(depth_np, 0, 300.0)  # Cap at 5 meters
            depth_min = 50.0   # 50cm minimum
            depth_max = 300.0  # 5m maximum
            depth_normalized = (depth_clipped - depth_min) / (depth_max - depth_min)
            depth_normalized = np.clip(depth_normalized, 0, 1)
            depth_np = np.where(depth_valid, depth_normalized, 0)
        depth_tensor = torch.from_numpy(depth_np).float()
        
        # Load normals EXR
        normals_np = load_exr_file(normal_file)
        normals_np = cv2.resize(normals_np, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        # Normalize normals to unit vectors
        norm = np.linalg.norm(normals_np, axis=2, keepdims=True)
        norm = np.where(norm > 0, norm, 1.0)
        normals_np = normals_np / norm
        normals_tensor = torch.from_numpy(normals_np.transpose(2, 0, 1)).float()  # [3, H, W]
        
        # Load alpha
        alpha_image = Image.open(alpha_file).convert('L')
        alpha_image = alpha_image.resize((target_size, target_size), Image.LANCZOS)
        alpha_np = np.array(alpha_image).astype(np.float32) / 255.0
        alpha_tensor = torch.from_numpy(alpha_np).unsqueeze(0).float()  # [1, H, W]
        
        sample = {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'normals': normals_tensor,
            'alpha': alpha_tensor,
            'sample_name': sample_name
        }
        samples.append(sample)
        
        print(f"  ✅ Loaded {sample_name}")
    
    print(f"📊 Successfully loaded {len(samples)} samples from {img_dir}")
    return samples


def save_inference_results(model, samples, device, output_dir="output"):
    """
    Save inference results using the same approach as vis_util.py.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # Prepare batch data
        batch_rgb = torch.stack([sample['rgb'] for sample in samples]).to(device)
        
        # Create batch dict
        batch = {
            'rgb': batch_rgb,
            'depth': torch.stack([sample['depth'] for sample in samples]),
            'normals': torch.stack([sample['normals'] for sample in samples]),
            'alpha': torch.stack([sample['alpha'] for sample in samples])
        }
        
        # Get model predictions
        outputs = model(batch_rgb)
        
        # Create figure with same layout as vis_util.py: 4 samples x 7 columns
        fig, axes = plt.subplots(4, 7, figsize=(21, 12))
        
        # Process each sample
        for row in range(len(samples)):
            sample_name = samples[row]['sample_name']
            
            # RGB
            rgb_vis = batch['rgb'][row].cpu().numpy().transpose(1, 2, 0)
            rgb_vis = np.clip(rgb_vis, 0, 1)
            axes[row, 0].imshow(rgb_vis)
            axes[row, 0].axis('off')
            if row == 0:
                axes[row, 0].set_title("RGB", fontsize=10)
            
            # Add sample label
            axes[row, 0].text(-0.1, 0.5, sample_name, rotation=90, va='center', ha='center', 
                            transform=axes[row, 0].transAxes, fontsize=10, fontweight='bold')
            
            # Depth GT and Pred with scale-shift alignment
            depth_gt = batch['depth'][row].cpu().numpy().squeeze()
            depth_pred_raw = outputs['depth'][row].cpu().numpy()
            
            # Get GT alpha mask ONLY for depth alignment
            alpha_gt = batch['alpha'][row].cpu().numpy().squeeze()
            if alpha_gt.ndim == 3 and alpha_gt.shape[0] == 1:
                alpha_gt = alpha_gt.squeeze(0)
            
            # Get predicted alpha mask for visualization masking
            alpha_pred = torch.sigmoid(outputs['alpha_logits'][row]).cpu().numpy()
            
            # Apply depth alignment using GT alpha
            depth_pred_aligned = align_depth_for_visualization(depth_pred_raw, depth_gt, alpha_gt)
            
            # Apply masked visualization using PREDICTED alpha
            depth_masked_vis = alpha_pred * depth_pred_aligned + (1 - alpha_pred) * depth_gt
            
            # Use 'turbo' colormap for much better contrast and perceptual uniformity
            axes[row, 1].imshow(depth_gt, cmap='turbo')
            axes[row, 1].axis('off')
            if row == 0:
                axes[row, 1].set_title("Depth GT", fontsize=10)
                
            axes[row, 2].imshow(depth_masked_vis, cmap='turbo')
            axes[row, 2].axis('off')
            if row == 0:
                axes[row, 2].set_title("Depth Pred", fontsize=10)
            
            # Normals GT and Pred
            normals_gt = batch['normals'][row].cpu().numpy().transpose(1, 2, 0)
            normals_pred = outputs['normals'][row].cpu().numpy().transpose(1, 2, 0)
            
            # Normalize both GT and predicted normals to unit vectors
            normals_gt_norm = np.linalg.norm(normals_gt, axis=2, keepdims=True)
            normals_gt_norm = np.where(normals_gt_norm > 0, normals_gt_norm, 1.0)
            normals_gt_normalized = normals_gt / normals_gt_norm
            
            normals_pred_norm = np.linalg.norm(normals_pred, axis=2, keepdims=True)
            normals_pred_norm = np.where(normals_pred_norm > 0, normals_pred_norm, 1.0)
            normals_pred_normalized = normals_pred / normals_pred_norm
            
            # Convert to visualization format [0, 1]
            normals_gt_vis = np.clip((normals_gt_normalized + 1) / 2, 0, 1)
            normals_pred_vis = np.clip((normals_pred_normalized + 1) / 2, 0, 1)
            
            # Apply masked visualization using PREDICTED alpha
            alpha_pred_3ch = np.stack([alpha_pred] * 3, axis=2)
            normals_masked_vis = alpha_pred_3ch * normals_pred_vis + (1 - alpha_pred_3ch) * normals_gt_vis
            
            axes[row, 3].imshow(normals_gt_vis)
            axes[row, 3].axis('off')
            if row == 0:
                axes[row, 3].set_title("Normals GT", fontsize=10)
                
            axes[row, 4].imshow(normals_masked_vis)
            axes[row, 4].axis('off')
            if row == 0:
                axes[row, 4].set_title("Normals Pred", fontsize=10)
            
            # Alpha GT and Pred
            alpha_gt_vis = batch['alpha'][row].cpu().numpy().squeeze()
            if alpha_gt_vis.ndim == 3 and alpha_gt_vis.shape[0] == 1:
                alpha_gt_vis = alpha_gt_vis.squeeze(0)
            
            axes[row, 5].imshow(alpha_gt_vis, cmap='gray', vmin=0, vmax=1)
            axes[row, 5].axis('off')
            if row == 0:
                axes[row, 5].set_title("Alpha GT", fontsize=10)
                
            axes[row, 6].imshow(alpha_pred, cmap='gray', vmin=0, vmax=1)
            axes[row, 6].axis('off')
            if row == 0:
                axes[row, 6].set_title("Alpha Pred", fontsize=10)
        
        # Same layout as vis_util.py
        plt.subplots_adjust(wspace=0.02, hspace=0.02)
        
        # Save results
        result_path = os.path.join(output_dir, 'demo_results.png')
        plt.savefig(result_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
    
    return result_path


def main():
    checkpoint_path = 'models/checkpoint_epoch_155.pth'
    img_dir = 'img'
    output_dir = 'output'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🚀 Running demo on device: {device}")
    
    # Load model
    try:
        model = load_model_checkpoint(checkpoint_path, device=device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    samples = load_samples_from_img_dir(img_dir, target_size=384)

    result_path = save_inference_results(model, samples, device, output_dir)
    
    print(f"\n🎯 Inference completed successfully!")
    print(f"📸 Results saved to: {result_path}")


if __name__ == "__main__":
    main()
