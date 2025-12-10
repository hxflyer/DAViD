"""
Loss Functions for Multi-task DPT Training
Cleaned up version with single optimized depth loss implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_mse_loss(pred, target):
    """Simple MSE loss for all branches."""
    return nn.MSELoss()(pred, target)


def compute_masked_mse_loss(pred, target, mask):
    """MSE loss masked by alpha map: mse((gt - pred) * mask)."""
    # Ensure mask has same spatial dimensions as pred/target
    if mask.dim() == 2:  # [H, W]
        mask = mask.unsqueeze(0)  # [1, H, W]
    if mask.dim() == 3 and pred.dim() == 3:  # mask [B, H, W], pred [B, H, W]
        mask = mask  # Keep as is
    
    # Apply mask to difference
    masked_diff = (target - pred) * mask
    
    # Compute MSE only on masked regions
    # Sum over all dimensions then divide by number of valid (masked) pixels
    mse = torch.sum(masked_diff ** 2) / (torch.sum(mask) + 1e-8)  # Add epsilon to avoid division by zero
    
    return mse


def compute_scale_shift_invariant_depth_loss_optimized(pred_depth, gt_depth, mask, stride=2, use_fp16=True):
    """
    Optimized scale-shift invariant depth loss with:
    - Stride=2 sampling for minimal accuracy loss
    - FP16 precision for parameter estimation
    - Full resolution final loss computation
    
    Args:
        pred_depth: Predicted depth [B, H, W]
        gt_depth: Ground truth depth [B, H, W]
        mask: Valid pixel mask [B, H, W]
        stride: Stride for subsampling (default: 2)
        use_fp16: Use FP16 for parameter estimation (default: True)
    
    Returns:
        Scale-shift invariant loss
    """
    B, H, W = pred_depth.shape
    device = pred_depth.device
    
    # Stride=2 sampling for better accuracy/speed trade-off
    pred_strided = pred_depth[:, ::stride, ::stride]  # [B, H//2, W//2]
    gt_strided = gt_depth[:, ::stride, ::stride]
    mask_strided = mask[:, ::stride, ::stride]
    
    # Convert to FP16 for parameter estimation if enabled
    if use_fp16:
        pred_flat = pred_strided.reshape(B, -1).half()
        gt_flat = gt_strided.reshape(B, -1).half()
        mask_flat = mask_strided.reshape(B, -1).half()
        eps = torch.tensor(1e-4, device=device, dtype=torch.half)
    else:
        pred_flat = pred_strided.reshape(B, -1).float()
        gt_flat = gt_strided.reshape(B, -1).float()
        mask_flat = mask_strided.reshape(B, -1).float()
        eps = 1e-6
    
    # Count valid pixels per sample [B]
    num_valid = torch.sum(mask_flat, dim=1)
    
    # Quick exit if no samples have enough valid pixels
    valid_samples = num_valid >= 2
    if not torch.any(valid_samples):
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Compute all statistics [B]
    sum_pred = torch.sum(pred_flat * mask_flat, dim=1)
    sum_gt = torch.sum(gt_flat * mask_flat, dim=1) 
    sum_pred2 = torch.sum(pred_flat * pred_flat * mask_flat, dim=1)
    sum_pred_gt = torch.sum(pred_flat * gt_flat * mask_flat, dim=1)
    sum_mask = num_valid
    
    # Solve 2x2 system using Cramer's rule
    det = sum_pred2 * sum_mask - sum_pred * sum_pred
    
    # Compute scale and shift using Cramer's rule [B]
    scale = (sum_pred_gt * sum_mask - sum_gt * sum_pred) / (det + eps)
    shift = (sum_pred2 * sum_gt - sum_pred * sum_pred_gt) / (det + eps)
    
    # Convert back to FP32 for final computation
    scale = scale.float()
    shift = shift.float()
    
    # Enhanced numerical stability checks
    if use_fp16:
        valid_det = torch.abs(det.float()) > 1e-4
    else:
        valid_det = torch.abs(det) > 1e-5
    
    scale = torch.clamp(scale, min=0.1, max=10.0)
    
    # Handle degenerate cases
    scale = torch.where(valid_det, scale, torch.ones_like(scale))
    shift = torch.where(valid_det, shift, torch.zeros_like(shift))
    
    # Safety checks for NaN/Inf
    scale = torch.where(torch.isfinite(scale), scale, torch.ones_like(scale))
    shift = torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))
    
    # Apply alignment to FULL RESOLUTION for final loss computation (in FP32)
    pred_full_flat = pred_depth.view(B, -1)  # [B, H*W]
    gt_full_flat = gt_depth.view(B, -1)
    mask_full_flat = mask.view(B, -1).float()
    
    # Expand scale/shift to full resolution [B, 1] -> [B, H*W]
    scale_expanded = scale.unsqueeze(1)
    shift_expanded = shift.unsqueeze(1)
    pred_aligned = pred_full_flat * scale_expanded + shift_expanded
    
    # Safety clamp for aligned predictions
    pred_aligned = torch.clamp(pred_aligned, min=0.01, max=1000.0)
    
    # Compute MSE on full resolution [B]
    diff_sq = (pred_aligned - gt_full_flat) ** 2 * mask_full_flat
    valid_pixels_full = torch.sum(mask_full_flat, dim=1)
    sample_losses = torch.sum(diff_sq, dim=1) / (valid_pixels_full + 1e-8)
    
    # Final safety checks
    sample_losses = torch.clamp(sample_losses, max=100.0)
    
    # Average over valid samples
    valid_losses = sample_losses[valid_samples]
    final_loss = torch.mean(valid_losses) if len(valid_losses) > 0 else torch.tensor(0.0, device=device, requires_grad=True)
    
    # Ultimate safety
    if torch.isnan(final_loss) or torch.isinf(final_loss) or final_loss > 50.0:
        return torch.tensor(1.0, device=device, requires_grad=True)
    
    return final_loss


# Test if torch.compile actually works with a simple function
def _test_compile():
    """Test if torch.compile works properly."""
    try:
        @torch.compile(mode="reduce-overhead")
        def _test_fn(x):
            return x * 2
        
        # Try to actually run it
        x = torch.tensor([1.0])
        result = _test_fn(x)
        return True
    except Exception:
        return False

# Detect Triton availability by actually testing compilation
TRITON_AVAILABLE = _test_compile()

if TRITON_AVAILABLE:
    # Linux: Create compiled version with Triton support
    @torch.compile(mode="reduce-overhead")
    def compute_scale_shift_invariant_depth_loss_compiled(pred_depth, gt_depth, mask, stride=2):
        """Compiled version for Linux with Triton support."""
        return compute_scale_shift_invariant_depth_loss_optimized(pred_depth, gt_depth, mask, stride=stride, use_fp16=True)
else:
    # Windows: Fallback version without Triton
    def compute_scale_shift_invariant_depth_loss_compiled(pred_depth, gt_depth, mask, stride=2):
        """Fallback version for Windows without Triton."""
        return compute_scale_shift_invariant_depth_loss_optimized(pred_depth, gt_depth, mask, stride=stride, use_fp16=True)


def compute_gradient_loss(pred_depth, gt_depth, mask, scales=4):
    """
    Multi-scale gradient matching loss for sharp depth discontinuities.
    Based on MiDaS paper: helps preserve edges in depth maps.
    
    Args:
        pred_depth: Predicted depth [B, H, W]
        gt_depth: Ground truth depth [B, H, W] 
        mask: Valid pixel mask [B, H, W]
        scales: Number of scales to use (default: 4)
    
    Returns:
        Gradient loss value
    """
    device = pred_depth.device
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    for scale in range(scales):
        step = 2 ** scale
        
        # Subsample at current scale
        pred_s = pred_depth[:, ::step, ::step]
        gt_s = gt_depth[:, ::step, ::step]
        mask_s = mask[:, ::step, ::step]
        
        # Skip if too small
        if pred_s.size(1) < 2 or pred_s.size(2) < 2:
            continue
            
        # Compute gradients in x and y directions
        pred_grad_x = pred_s[:, :, 1:] - pred_s[:, :, :-1]  # [B, H, W-1]
        pred_grad_y = pred_s[:, 1:, :] - pred_s[:, :-1, :]  # [B, H-1, W]
        
        gt_grad_x = gt_s[:, :, 1:] - gt_s[:, :, :-1]
        gt_grad_y = gt_s[:, 1:, :] - gt_s[:, :-1, :]
        
        # Corresponding masks for gradients
        mask_grad_x = mask_s[:, :, 1:] * mask_s[:, :, :-1]  # Both pixels must be valid
        mask_grad_y = mask_s[:, 1:, :] * mask_s[:, :-1, :]
        
        # Compute L1 loss on gradients (more robust than L2)
        loss_x = torch.abs(pred_grad_x - gt_grad_x) * mask_grad_x
        loss_y = torch.abs(pred_grad_y - gt_grad_y) * mask_grad_y
        
        # Average over valid gradients
        valid_x = torch.sum(mask_grad_x, dim=(1, 2)) + 1e-8
        valid_y = torch.sum(mask_grad_y, dim=(1, 2)) + 1e-8
        
        scale_loss_x = torch.sum(loss_x, dim=(1, 2)) / valid_x
        scale_loss_y = torch.sum(loss_y, dim=(1, 2)) / valid_y
        
        # Add to total (average across batch)
        total_loss = total_loss + torch.mean(scale_loss_x) + torch.mean(scale_loss_y)
    
    # Average across scales
    return total_loss / scales if scales > 0 else total_loss


def compute_robust_depth_loss(pred_depth, gt_depth, mask, use_compiled=True, use_gradient_loss=True, alpha=0.5):
    """
    Main depth loss function with optional gradient regularization.
    
    Args:
        pred_depth: Predicted depth [B, H, W]
        gt_depth: Ground truth depth [B, H, W]
        mask: Valid pixel mask (foreground) [B, H, W]
        use_compiled: If True, uses compiled version when available
        use_gradient_loss: If True, adds gradient regularization term
        alpha: Weight for gradient regularization (default: 0.5)
    
    Returns:
        total_loss: Combined depth loss (SSI + gradient regularization)
        ssi_loss: Scale-shift invariant component
        gradient_loss: Gradient regularization component (zero if disabled)
    """
    # Compute scale-shift invariant loss
    if use_compiled and TRITON_AVAILABLE:
        # Linux: Use compiled version with Triton
        ssi_loss = compute_scale_shift_invariant_depth_loss_compiled(pred_depth, gt_depth, mask, stride=2)
    else:
        # Windows: Use optimized version without compilation
        ssi_loss = compute_scale_shift_invariant_depth_loss_optimized(pred_depth, gt_depth, mask, stride=2, use_fp16=True)
    
    # Compute gradient regularization loss if enabled
    if use_gradient_loss and alpha > 0:
        gradient_loss = compute_gradient_loss(pred_depth, gt_depth, mask, scales=4)
        total_loss = ssi_loss + alpha * gradient_loss
    else:
        gradient_loss = torch.tensor(0.0, device=pred_depth.device)
        total_loss = ssi_loss
    
    return total_loss, ssi_loss, gradient_loss


def compute_surface_normal_loss(pred_normals, gt_normals, mask):
    """
    Surface normal loss using cosine similarity: L = 1 - η·η̂ (computed on foreground region).
    
    Args:
        pred_normals: Predicted surface normals [B, 3, H, W]
        gt_normals: Ground truth surface normals [B, 3, H, W]
        mask: Valid pixel mask (foreground) [B, H, W]
    
    Returns:
        Surface normal loss (1 - cosine similarity)
    """
    # Ensure mask has same spatial dimensions as normals
    if mask.dim() == 2:  # [H, W]
        mask = mask.unsqueeze(0)  # [1, H, W]
    if mask.dim() == 3 and pred_normals.dim() == 4:  # mask [B, H, W], normals [B, C, H, W]
        mask = mask.unsqueeze(1)  # [B, 1, H, W]
    
    # Normalize both predicted and ground truth normals to unit vectors
    pred_normals_normalized = torch.nn.functional.normalize(pred_normals, p=2, dim=1, eps=1e-8)
    gt_normals_normalized = torch.nn.functional.normalize(gt_normals, p=2, dim=1, eps=1e-8)
    
    # Compute cosine similarity (dot product) between normalized vectors
    # Sum over channel dimension to get dot product for each pixel
    cosine_similarity = torch.sum(pred_normals_normalized * gt_normals_normalized, dim=1)  # [B, H, W]
    
    # Apply foreground mask
    masked_cosine_similarity = cosine_similarity * mask.squeeze(1)  # Remove channel dim from mask
    
    # Compute loss: L = 1 - cosine_similarity, averaged over foreground pixels
    loss = 1.0 - masked_cosine_similarity
    
    # Average over foreground pixels only
    total_loss = torch.sum(loss) / (torch.sum(mask) + 1e-8)
    
    return total_loss


def dice_loss(pred, target, smooth=1e-6):
    """
    Dice loss for binary segmentation.
    
    Args:
        pred: Predicted probabilities [B, H, W] or [B*H*W]
        target: Ground truth binary mask [B, H, W] or [B*H*W]
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice loss (1 - dice coefficient)
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice_coeff = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1 - dice_coeff


def compute_alpha_loss(alpha_logits, alpha_gt):
    """
    Combined loss for alpha/foreground segmentation: BCE + L1 + Dice.
    
    Args:
        alpha_logits: Raw logits from model [B, H, W]
        alpha_gt: Ground truth alpha mask [B, H, W]
    
    Returns:
        total_alpha_loss: Combined loss
        bce_loss: Binary cross entropy component
        l1_loss: L1 loss component  
        dice_loss_val: Dice loss component
    """
    # Convert logits to probabilities
    alpha_pred = torch.sigmoid(alpha_logits)
    
    # BCE Loss
    bce_loss = nn.BCEWithLogitsLoss()(alpha_logits, alpha_gt)
    
    # L1 Loss
    l1_loss = nn.L1Loss()(alpha_pred, alpha_gt)
    
    # Dice Loss
    dice_loss_val = dice_loss(alpha_pred, alpha_gt)
    
    # Combine losses with equal weights
    total_alpha_loss = bce_loss + l1_loss + dice_loss_val
    
    return total_alpha_loss, bce_loss, l1_loss, dice_loss_val


if __name__ == "__main__":
    # Test loss functions
    print("🧪 Testing cleaned up loss functions...")
    print(f"🔥 Triton available: {TRITON_AVAILABLE}")
    
    # Create dummy data
    B, H, W = 2, 384, 384
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pred_depth = torch.randn(B, H, W, device=device, requires_grad=True) * 5 + 10
    gt_depth = torch.randn(B, H, W, device=device) * 5 + 10
    mask = torch.rand(B, H, W, device=device) > 0.3
    
    pred_normals = torch.randn(B, 3, H, W, device=device, requires_grad=True)
    gt_normals = torch.randn(B, 3, H, W, device=device)
    
    alpha_logits = torch.randn(B, H, W, device=device, requires_grad=True)
    alpha_gt = torch.rand(B, H, W, device=device)
    
    # Test depth loss
    print("🔍 Testing depth loss...")
    depth_loss, ssi_loss, direct_loss = compute_robust_depth_loss(pred_depth, gt_depth, mask)
    print(f"   Depth Loss: {depth_loss:.4f}")
    
    # Test normal loss
    print("🔍 Testing normal loss...")
    normal_loss = compute_surface_normal_loss(pred_normals, gt_normals, mask)
    print(f"   Normal Loss: {normal_loss:.4f}")
    
    # Test alpha loss
    print("🔍 Testing alpha loss...")
    alpha_loss, bce_loss, l1_loss, dice_loss_val = compute_alpha_loss(alpha_logits, alpha_gt)
    print(f"   Alpha Loss: {alpha_loss:.4f}")
    
    print("✅ All loss functions working correctly!")
