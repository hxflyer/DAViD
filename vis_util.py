"""
Visualization Utilities for Multi-task DPT Training
Contains functions for depth alignment and sample result visualization
"""

import torch
import os
import matplotlib.pyplot as plt
import numpy as np


def align_depth_for_visualization(depth_pred, depth_gt, mask):
    """
    Apply the same scale-shift alignment used in loss calculation for visualization.
    
    Args:
        depth_pred: Raw predicted depth [H, W] numpy array
        depth_gt: Ground truth depth [H, W] numpy array  
        mask: Alpha mask [H, W] numpy array
    
    Returns:
        Aligned depth prediction [H, W] numpy array
    """
    # Get valid pixels using mask (same as loss calculation)
    mask_valid = mask > 0
    num_valid = mask_valid.sum()
    
    if num_valid < 2:  # Need at least 2 points for least squares
        return depth_pred  # Return original if insufficient valid pixels
    
    # Extract valid depth values (working directly in depth space)
    pred_valid = depth_pred[mask_valid].reshape(-1, 1)  # [M, 1]
    gt_valid = depth_gt[mask_valid].reshape(-1, 1)      # [M, 1]
    
    # Set up least squares problem: A * [scale, shift]^T = gt_valid
    # where A = [pred_valid, ones]
    ones = np.ones_like(pred_valid)
    A = np.concatenate([pred_valid, ones], axis=1)  # [M, 2]
    
    try:
        # Solve normal equations: (A^T A) * h = A^T * gt_valid
        ATA = A.T @ A              # [2, 2]
        ATb = A.T @ gt_valid       # [2, 1]
        
        # Solve for optimal [scale, shift]
        h_opt = np.linalg.solve(ATA, ATb)  # [2, 1]
        scale = h_opt[0, 0]
        shift = h_opt[1, 0]
        
        # Apply scale and shift alignment to entire depth prediction
        depth_aligned = scale * depth_pred + shift
        
        return depth_aligned
        
    except np.linalg.LinAlgError:
        # Fallback: if matrix is singular, return original
        return depth_pred


def save_sample_results(model, train_batch, val_batch, epoch, device, output_dir="training_samples", rank=0):
    """
    Save sample predictions for visual inspection: 2 train + 2 val samples.
    
    Args:
        model: The trained model (or model.module for DDP)
        train_batch: Training batch dictionary with 'rgb', 'depth', 'normals', 'alpha'
        val_batch: Validation batch dictionary
        epoch: Current epoch number
        device: Device to run inference on
        output_dir: Directory to save visualization images
        rank: Process rank (for multi-GPU training, only rank 0 saves)
    """
    # Only save on rank 0 to avoid multiple saves in multi-GPU training
    if rank != 0:
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # Get 2 samples from train and 2 from validation
        train_rgb = train_batch['rgb'][:2].to(device)
        val_rgb = val_batch['rgb'][:2].to(device)
        
        # Get predictions
        train_outputs = model(train_rgb)
        val_outputs = model(val_rgb)
        
        # Create figure for 4 samples (2 train + 2 val)
        fig, axes = plt.subplots(4, 7, figsize=(21, 12))
        
        # Process samples
        samples_data = [
            (train_batch, train_outputs, 0, "Train"),
            (train_batch, train_outputs, 1, "Train"),
            (val_batch, val_outputs, 0, "Val"),
            (val_batch, val_outputs, 1, "Val")
        ]
        
        for row, (batch, outputs, idx, split) in enumerate(samples_data):
            # RGB
            rgb_vis = batch['rgb'][idx].cpu().numpy().transpose(1, 2, 0)
            rgb_vis = np.clip(rgb_vis, 0, 1)
            axes[row, 0].imshow(rgb_vis)
            axes[row, 0].axis('off')
            if row == 0:
                axes[row, 0].set_title("RGB", fontsize=10)
            
            # Add split label on the left
            axes[row, 0].text(-0.1, 0.5, split, rotation=90, va='center', ha='center', 
                            transform=axes[row, 0].transAxes, fontsize=12, fontweight='bold')
            
            # Depth GT and Pred with scale-shift alignment (same as loss calculation)
            depth_gt = batch['depth'][idx].cpu().numpy().squeeze()
            depth_pred_raw = outputs['depth'][idx].cpu().numpy()
            
            # Get alpha mask for visualization
            alpha_mask = batch['alpha'][idx].cpu().numpy().squeeze()
            if alpha_mask.ndim == 3 and alpha_mask.shape[0] == 1:
                alpha_mask = alpha_mask.squeeze(0)
            
            # Apply same scale-shift alignment as in loss calculation
            depth_pred_aligned = align_depth_for_visualization(depth_pred_raw, depth_gt, alpha_mask)
            
            # Normalize for visualization using aligned predictions
            d_min = np.percentile(np.concatenate([depth_gt.flatten(), depth_pred_aligned.flatten()]), 5)
            d_max = np.percentile(np.concatenate([depth_gt.flatten(), depth_pred_aligned.flatten()]), 95)
            
            if d_max > d_min:
                depth_gt_vis = np.clip((depth_gt - d_min) / (d_max - d_min), 0, 1)
                depth_pred_vis = np.clip((depth_pred_aligned - d_min) / (d_max - d_min), 0, 1)
            else:
                depth_gt_vis = depth_gt
                depth_pred_vis = depth_pred_aligned
            
            # Apply masked visualization: gt_mask * aligned_pred + (1-gt_mask)*gt
            depth_masked_vis = alpha_mask * depth_pred_vis + (1 - alpha_mask) * depth_gt_vis
            
            axes[row, 1].imshow(depth_gt_vis, cmap='plasma')
            axes[row, 1].axis('off')
            if row == 0:
                axes[row, 1].set_title("Depth GT", fontsize=10)
                
            axes[row, 2].imshow(depth_masked_vis, cmap='plasma')
            axes[row, 2].axis('off')
            if row == 0:
                axes[row, 2].set_title("Depth Pred", fontsize=10)
            
            # Normals GT and Pred with masked visualization
            normals_gt = batch['normals'][idx].cpu().numpy().transpose(1, 2, 0)
            normals_pred = outputs['normals'][idx].cpu().numpy().transpose(1, 2, 0)
            
            # Normalize both GT and predicted normals to unit vectors
            normals_gt_norm = np.linalg.norm(normals_gt, axis=2, keepdims=True)
            normals_gt_norm = np.where(normals_gt_norm > 0, normals_gt_norm, 1.0)  # Avoid division by zero
            normals_gt_normalized = normals_gt / normals_gt_norm
            
            normals_pred_norm = np.linalg.norm(normals_pred, axis=2, keepdims=True)
            normals_pred_norm = np.where(normals_pred_norm > 0, normals_pred_norm, 1.0)  # Avoid division by zero
            normals_pred_normalized = normals_pred / normals_pred_norm
            
            # Convert normalized normals to visualization format [0, 1]
            normals_gt_vis = np.clip((normals_gt_normalized + 1) / 2, 0, 1)
            normals_pred_vis = np.clip((normals_pred_normalized + 1) / 2, 0, 1)
            
            # Apply masked visualization: gt_mask * pred + (1-gt_mask)*gt
            # Expand alpha_mask to 3 channels for normals
            alpha_mask_3ch = np.stack([alpha_mask] * 3, axis=2)
            normals_masked_vis = alpha_mask_3ch * normals_pred_vis + (1 - alpha_mask_3ch) * normals_gt_vis
            
            axes[row, 3].imshow(normals_gt_vis)
            axes[row, 3].axis('off')
            if row == 0:
                axes[row, 3].set_title("Normals GT", fontsize=10)
                
            axes[row, 4].imshow(normals_masked_vis)
            axes[row, 4].axis('off')
            if row == 0:
                axes[row, 4].set_title("Normals Pred", fontsize=10)
            
            # Alpha GT and Pred
            alpha_gt = batch['alpha'][idx].cpu().numpy().squeeze()
            alpha_pred = torch.sigmoid(outputs['alpha_logits'][idx]).cpu().numpy()
            
            if alpha_gt.ndim == 3 and alpha_gt.shape[0] == 1:
                alpha_gt = alpha_gt.squeeze(0)
            
            axes[row, 5].imshow(alpha_gt, cmap='gray', vmin=0, vmax=1)
            axes[row, 5].axis('off')
            if row == 0:
                axes[row, 5].set_title("Alpha GT", fontsize=10)
                
            axes[row, 6].imshow(alpha_pred, cmap='gray', vmin=0, vmax=1)
            axes[row, 6].axis('off')
            if row == 0:
                axes[row, 6].set_title("Alpha Pred", fontsize=10)
        
        plt.subplots_adjust(wspace=0.02, hspace=0.02)
        sample_path = os.path.join(output_dir, f'epoch_{epoch+1:02d}_samples.png')
        plt.savefig(sample_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        print(f"📸 Sample results saved: {sample_path}")


def create_visualization_grid(samples_list, titles_list, save_path, colormap='viridis'):
    """
    Create a grid visualization for multiple samples and save to file.
    
    Args:
        samples_list: List of numpy arrays to visualize [N, H, W] or [N, H, W, C]
        titles_list: List of titles for each column
        save_path: Path to save the visualization
        colormap: Matplotlib colormap to use
    """
    num_samples = len(samples_list)
    num_cols = len(titles_list)
    
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(num_cols * 3, num_samples * 3))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    if num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, samples in enumerate(samples_list):
        for j, (sample, title) in enumerate(zip(samples, titles_list)):
            if len(sample.shape) == 3 and sample.shape[2] == 3:
                # RGB image
                axes[i, j].imshow(np.clip(sample, 0, 1))
            else:
                # Single channel
                axes[i, j].imshow(sample, cmap=colormap)
            
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(title, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📸 Visualization grid saved: {save_path}")


if __name__ == "__main__":
    # Test the visualization utilities
    print("Testing visualization utilities...")
    
    # Create dummy data for testing
    H, W = 64, 64
    depth_pred = np.random.randn(H, W) * 5 + 10
    depth_gt = np.random.randn(H, W) * 5 + 10
    mask = np.random.rand(H, W) > 0.3
    
    # Test depth alignment
    aligned_depth = align_depth_for_visualization(depth_pred, depth_gt, mask)
    print(f"Original depth range: [{depth_pred.min():.2f}, {depth_pred.max():.2f}]")
    print(f"Aligned depth range: [{aligned_depth.min():.2f}, {aligned_depth.max():.2f}]")
    print(f"GT depth range: [{depth_gt.min():.2f}, {depth_gt.max():.2f}]")
    
    print("✅ Visualization utilities working correctly!")
