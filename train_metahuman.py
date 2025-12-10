"""
Training Script for Dual-Task DPT Model with MetaHuman Dataset
Trains on MetaHuman dataset only with face mask and geo heads
Uses optimized dataset loader and model architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.amp import GradScaler, autocast
import time
import argparse
import random
from torch.utils.tensorboard import SummaryWriter

# Import our modules
from multi_head_dpt import create_dual_head_dpt
from metahuman_dataset2 import create_metahuman_dataloaders2
from loss import compute_alpha_loss


def save_metahuman_samples(model, train_batch, val_batch, epoch, device, output_dir="training_samples"):
    """
    Save sample predictions for MetaHuman dataset.
    Visualizes RGB, Mask, Base Geo, and n encoding images (sin components).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get samples (up to 8 if available)
    max_samples = 8
    train_rgb = train_batch['rgb'][:max_samples].to(device)
    val_rgb = val_batch['rgb'][:max_samples].to(device)
    
    model.eval()
    with torch.no_grad():
        train_outputs = model(train_rgb)
        val_outputs = model(val_rgb)
    
    # Infer n from geo channels
    # Total channels = 3 + 6 * n
    geo_channels = train_outputs['geo'].shape[1]
    n_encodings = (geo_channels - 3) // 6
    
    # Cols: RGB, MaskGT, MaskPred, (GeoGT_Base, GeoPred_Base), (GeoGT_Enc_i, GeoPred_Enc_i) * n
    num_cols = 3 + 2 + 2 * n_encodings
    
    # Rows: Train samples + Val samples
    num_train = train_rgb.shape[0]
    num_val = val_rgb.shape[0]
    total_rows = num_train + num_val
    
    fig, axes = plt.subplots(total_rows, num_cols, figsize=(num_cols * 2, total_rows * 2))
    fig.suptitle(f'MetaHuman Dataset - Epoch {epoch+1} (n={n_encodings})', fontsize=16, y=0.98)
    
    # Helper to get specific encoding image (sin component of freq k)
    def get_encoding_image(tensor, k, n):
        # tensor: [3 + 6*n, H, W]
        # Indices for sin component of frequency k:
        # C0 block starts at 3. Size 2*n (n sines, n cosines).
        # We want the k-th sine component.
        # C0: 3 + k
        # C1: 3 + 2*n + k
        # C2: 3 + 4*n + k
        
        idx0 = 3 + k
        idx1 = 3 + 2*n + k
        idx2 = 3 + 4*n + k
        
        c0 = tensor[idx0]
        c1 = tensor[idx1]
        c2 = tensor[idx2]
        
        img = np.stack([c0, c1, c2], axis=0)
        # Normalize -1..1 to 0..1
        img = (img + 1) / 2
        return np.clip(img.transpose(1, 2, 0), 0, 1)

    # Helper for base geo
    def get_base_geo(tensor):
        img = tensor[0:3]
        # Data is normalized to [-1, 1]. Un-normalize to [0, 1] for visualization.
        img = (img + 1) / 2
        return np.clip(img.transpose(1, 2, 0), 0, 1)

    # Process samples
    all_samples = []
    for i in range(num_train):
        all_samples.append((train_batch, train_outputs, i, "Train"))
    for i in range(num_val):
        all_samples.append((val_batch, val_outputs, i, "Val"))
        
    for row, (batch, outputs, idx, split) in enumerate(all_samples):
        col = 0
        
        # RGB
        rgb_vis = batch['rgb'][idx].cpu().numpy().transpose(1, 2, 0)
        rgb_vis = np.clip(rgb_vis, 0, 1)
        axes[row, col].imshow(rgb_vis)
        axes[row, col].axis('off')
        if row == 0: axes[row, col].set_title("RGB", fontsize=8)
        axes[row, col].text(-0.1, 0.5, split, rotation=90, va='center', ha='center', transform=axes[row, col].transAxes, fontsize=10, fontweight='bold')
        col += 1
        
        # Mask GT
        mask_gt = batch['face_mask'][idx].cpu().numpy().squeeze()
        if mask_gt.ndim == 3 and mask_gt.shape[0] == 1: mask_gt = mask_gt.squeeze(0)
        axes[row, col].imshow(mask_gt, cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
        if row == 0: axes[row, col].set_title("Mask GT", fontsize=8)
        col += 1
        
        # Mask Pred
        alpha_logits = outputs['alpha_logits'][idx].detach()
        alpha_pred = torch.sigmoid(alpha_logits).cpu().numpy().squeeze()
        axes[row, col].imshow(alpha_pred, cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
        if row == 0: axes[row, col].set_title("Mask Pred", fontsize=8)
        col += 1
        
        # Geo Data
        geo_gt = batch['geo'][idx].cpu().numpy()
        geo_pred = outputs['geo'][idx].detach().cpu().numpy()
        alpha_3ch = np.stack([alpha_pred] * 3, axis=2)
        
        # Base Geo GT
        axes[row, col].imshow(get_base_geo(geo_gt))
        axes[row, col].axis('off')
        if row == 0: axes[row, col].set_title("Geo Base GT", fontsize=8)
        col += 1
        
        # Base Geo Pred
        base_pred = get_base_geo(geo_pred)
        axes[row, col].imshow(np.clip(base_pred * alpha_3ch, 0, 1))
        axes[row, col].axis('off')
        if row == 0: axes[row, col].set_title("Geo Base Pred", fontsize=8)
        col += 1
        
        # Encodings
        for k in range(n_encodings):
            # GT
            gt_img = get_encoding_image(geo_gt, k, n_encodings)
            axes[row, col].imshow(gt_img)
            axes[row, col].axis('off')
            if row == 0: axes[row, col].set_title(f"Enc GT {k}", fontsize=8)
            col += 1
            
            # Pred
            pred_img = get_encoding_image(geo_pred, k, n_encodings)
            pred_masked = pred_img * alpha_3ch
            axes[row, col].imshow(np.clip(pred_masked, 0, 1))
            axes[row, col].axis('off')
            if row == 0: axes[row, col].set_title(f"Enc Pred {k}", fontsize=8)
            col += 1

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    sample_path = os.path.join(output_dir, f'epoch_{epoch+1:02d}_metahuman_samples.png')
    plt.savefig(sample_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📸 MetaHuman sample results saved: {sample_path}")


def train_model(load_pretrained=None):
    """Main training function for MetaHuman dataset."""
    print("🚀 Starting Dual-Head DPT training with MetaHuman dataset (Mask + Geo only)...")
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(f"runs/dpt_metahuman_{int(time.time())}")
    print(f"📝 TensorBoard logs will be saved to: {writer.log_dir}")
    
    # Configuration
    metahuman_data_root = [
        "G:/training1",
        #"G:/training2",
        #"G:/training3",
        #"G:/training4",
        # Add more paths here if needed
    ]
    batch_size = 8  # Reduced for mixed dataset training
    num_epochs = 200
    lr = 2e-4
    use_fp16 = False  # Disabled FP16 to prevent NaN in ViT attention
    
    # Positional Encoding configuration
    geo_encoding_n = 4  # User requested n=4 (frequencies 2, 4, 6, 8)
    
    # Calculate total output channels: 3 (base) + 3 channels * 2 (sin/cos) * n frequencies
    geo_output_channels = 3 + 6 * geo_encoding_n
    
    # Use GPU 1 (second GPU)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"📱 Using GPU 0: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"🔥 FP16 Training: {'Enabled' if use_fp16 else 'Disabled'}")
        print(f"📊 Batch size: {batch_size}")
        print(f"🔢 Geo Encoding N: {geo_encoding_n} (Total Geo Channels: {geo_output_channels})")
    else:
        device = torch.device('cpu')
        print("📱 Using CPU - no CUDA available")
    
    # Create data loaders for MetaHuman dataset
    print("📊 Creating MetaHuman dataset loaders (Mask + Geo)...")
    metahuman_train_loader, metahuman_val_loader = create_metahuman_dataloaders2(
        data_roots=metahuman_data_root,
        batch_size=batch_size,
        num_workers=2,
        image_size=512,  # MetaHuman uses 512
        train_ratio=0.95,
        geo_encoding_n=geo_encoding_n
    )
    print(f"🎯 MetaHuman - Train batches: {len(metahuman_train_loader)}, Val batches: {len(metahuman_val_loader)}")
    
    # Create model
    print("🧠 Creating Dual-Head DPT model...")
    model = create_dual_head_dpt(
        backbone="vitb16_384",
        features=256,
        use_bn=False,
        pretrained=True,
        geo_output_channels=geo_output_channels
    )
    model = model.to(device)
    
    # Auto-resume: Load best model if available, unless explicitly specified otherwise
    start_epoch = 0
    model_to_load = None
    
    if load_pretrained:
        # Explicit model specified via command line
        if os.path.exists(load_pretrained):
            model_to_load = load_pretrained
            print(f"📥 Loading explicitly specified model: {load_pretrained}")
        else:
            print(f"⚠️  Specified model file not found: {load_pretrained}")
            print(f"   Starting from scratch...")
    elif os.path.exists("best_model.pth"):
        # Auto-resume: Load best model if it exists
        model_to_load = "best_model.pth"
        print(f"🔄 Auto-resuming from: best_model.pth")
    else:
        print(f"🆕 Starting fresh training (no existing model found)")
    
    # Load the selected model
    if model_to_load:
        try:
            checkpoint = torch.load(model_to_load, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Loading from checkpoint file
                model.load_state_dict(checkpoint['model_state_dict'], strict=False) # Strict=False for DualHead loading from MultiHead
                start_epoch = checkpoint.get('epoch', 0) + 1
                print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 0)} (strict=False)")
            else:
                # Loading from state dict file (like best_model.pth)
                model.load_state_dict(checkpoint, strict=False)
                print(f"✅ Loaded model state dict successfully (resuming from epoch 0) (strict=False)")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print(f"   Starting from scratch...")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"⚙️  Total parameters: {total_params:,}")
    print(f"🎯 Trainable parameters: {trainable_params:,}")
    
    # Separate learning rates for backbone and heads
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'pretrained' in name:  # Original DPT backbone
            backbone_params.append(param)
        else:  # Task heads and fusion blocks
            head_params.append(param)
    
    # Conservative optimizer setup
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.1, 'weight_decay': 5e-5},  # Lower for backbone
        {'params': head_params, 'lr': lr, 'weight_decay': 5e-5}             # Normal for heads
    ])
    
    # Simple step decay scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=10,    # Reduce LR every 10 epochs
        gamma=0.7        # Multiply by 0.7
    )
    
    # Initialize FP16 scaler
    scaler = None
    if use_fp16 and torch.cuda.is_available():
        scaler = GradScaler()
        print("🔥 FP16 GradScaler initialized")
    
    print("🏃‍♂️ Starting training loop...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{start_epoch + num_epochs}")
        current_lr_backbone = optimizer.param_groups[0]['lr']
        current_lr_heads = optimizer.param_groups[1]['lr']
        print(f"🔧 Learning rates: backbone={current_lr_backbone:.2e}, heads={current_lr_heads:.2e}")
        
        # Training phase with MetaHuman dataset
        model.train()
        train_loss = 0.0
        train_batches = 0
        metahuman_train_sample_batch = None
        
        # Loss trackers for TensorBoard
        metahuman_train_losses = {'alpha': 0.0, 'geo': 0.0, 'total': 0.0}
        
        pbar = tqdm(metahuman_train_loader, desc="Training (MetaHuman)")
        for batch_idx, metahuman_batch in enumerate(pbar):
            optimizer.zero_grad()
            batch_loss = 0.0
            
            # Move MetaHuman data to device
            rgb = metahuman_batch['rgb'].to(device)
            face_mask_gt = metahuman_batch['face_mask'].to(device)  # Use face_mask as alpha
            geo_gt = metahuman_batch['geo'].to(device)
            
            # Handle face mask dimensions
            if face_mask_gt.dim() == 4 and face_mask_gt.shape[1] == 1:
                face_mask_gt = face_mask_gt.squeeze(1)
            
            # Forward pass - DualHeadDPT only computes mask and geo
            if use_fp16 and scaler is not None:
                with autocast(device_type=device.type):
                    outputs = model(rgb)
                    
                    alpha_logits = outputs['alpha_logits']
                    geo_pred = outputs['geo']
                    
                    # Compute losses with face mask
                    alpha_loss, _, _, _ = compute_alpha_loss(alpha_logits, face_mask_gt)
                    
                    # Compute geo losses
                    face_mask_gt_expanded = face_mask_gt.unsqueeze(1)
                    
                    # Apply hair mask if available (exclude hair region for flux/seedream)
                    valid_mask = face_mask_gt_expanded
                    if 'hair_mask' in metahuman_batch:
                        hair_mask = metahuman_batch['hair_mask'].to(device)
                        valid_mask = valid_mask * (1.0 - hair_mask)
                    geo_loss = F.l1_loss(geo_pred, geo_gt, reduction='none')
                    geo_loss = (geo_loss * valid_mask).mean()
                    
                    metahuman_loss = alpha_loss + geo_loss
                    batch_loss += metahuman_loss
                    
                    # Update MetaHuman train loss trackers
                    metahuman_train_losses['alpha'] += alpha_loss.item()
                    metahuman_train_losses['geo'] += geo_loss.item()
                    metahuman_train_losses['total'] += metahuman_loss.item()
            else:
                outputs = model(rgb)
                
                alpha_logits = outputs['alpha_logits']
                geo_pred = outputs['geo']
                
                # Compute losses with face mask
                alpha_loss, _, _, _ = compute_alpha_loss(alpha_logits, face_mask_gt)
                
                # Compute geo losses
                face_mask_gt_expanded = face_mask_gt.unsqueeze(1)
                
                # Apply hair mask if available
                valid_mask = face_mask_gt_expanded
                if 'hair_mask' in metahuman_batch:
                    hair_mask = metahuman_batch['hair_mask'].to(device)
                    valid_mask = valid_mask * (1.0 - hair_mask)

                geo_loss = F.l1_loss(geo_pred, geo_gt, reduction='none')
                geo_loss = (geo_loss * valid_mask).mean()
                
                metahuman_loss = alpha_loss + geo_loss
                batch_loss += metahuman_loss
                
                # Update MetaHuman train loss trackers
                metahuman_train_losses['alpha'] += alpha_loss.item()
                metahuman_train_losses['geo'] += geo_loss.item()
                metahuman_train_losses['total'] += metahuman_loss.item()
            
            # Save first MetaHuman batch for visualization
            if metahuman_train_sample_batch is None:
                metahuman_train_sample_batch = {
                    'rgb': metahuman_batch['rgb'],
                    'face_mask': metahuman_batch['face_mask'],
                    'geo': metahuman_batch['geo']
                }
            
            # Skip batch if no data was processed
            if batch_loss == 0.0:
                continue
            
            # Check for NaN/Inf before backward pass
            if torch.isnan(batch_loss) or torch.isinf(batch_loss) or batch_loss.item() > 1000:
                print(f"⚠️  Unstable loss detected: {batch_loss.item():.4f}, skipping batch {batch_idx}")
                print(f"   Alpha Loss: {alpha_loss.item():.4f}")
                print(f"   Geo Loss: {geo_loss.item():.4f}")
                print(f"   RGB Min/Max: {rgb.min().item():.4f} / {rgb.max().item():.4f}")
                print(f"   Face Mask GT Min/Max: {face_mask_gt.min().item():.4f} / {face_mask_gt.max().item():.4f}")
                print(f"   Geo GT Min/Max: {geo_gt.min().item():.4f} / {geo_gt.max().item():.4f}")
                print(f"   Geo Pred Min/Max: {geo_pred.min().item():.4f} / {geo_pred.max().item():.4f}")
                print(f"   Alpha Logits Min/Max: {alpha_logits.min().item():.4f} / {alpha_logits.max().item():.4f}")
                
                if torch.isnan(geo_gt).any(): print("   !!! Geo GT contains NaNs !!!")
                if torch.isnan(face_mask_gt).any(): print("   !!! Face Mask GT contains NaNs !!!")
                continue
            
            # Backward pass
            if use_fp16 and scaler is not None:
                scaler.scale(batch_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
            
            train_loss += batch_loss.item()
            train_batches += 1
            
            pbar.set_postfix({
                'Loss': f'{batch_loss.item():.4f}'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        metahuman_val_sample_batch = None
        
        # Loss trackers for TensorBoard
        metahuman_val_losses = {'alpha': 0.0, 'geo': 0.0, 'total': 0.0}
        
        with torch.no_grad():
            # Validate on MetaHuman dataset
            pbar = tqdm(metahuman_val_loader, desc="Validation (MetaHuman)")
            for batch_idx, batch in enumerate(pbar):
                if batch_idx == 0:
                    metahuman_val_sample_batch = {
                        'rgb': batch['rgb'],
                        'face_mask': batch['face_mask'],
                        'geo': batch['geo']
                    }
                
                rgb = batch['rgb'].to(device)
                face_mask_gt = batch['face_mask'].to(device)
                geo_gt = batch['geo'].to(device)
                
                if face_mask_gt.dim() == 4 and face_mask_gt.shape[1] == 1:
                    face_mask_gt = face_mask_gt.squeeze(1)
                
                outputs = model(rgb)
                alpha_logits = outputs['alpha_logits']
                geo_pred = outputs['geo']
                
                alpha_loss, _, _, _ = compute_alpha_loss(alpha_logits, face_mask_gt)
                
                # Compute geo losses
                face_mask_gt_expanded = face_mask_gt.unsqueeze(1)
                
                # Apply hair mask if available
                valid_mask = face_mask_gt_expanded
                if 'hair_mask' in batch:
                    hair_mask = batch['hair_mask'].to(device)
                    valid_mask = valid_mask * (1.0 - hair_mask)

                geo_loss = F.l1_loss(geo_pred, geo_gt, reduction='none')
                geo_loss = (geo_loss * valid_mask).mean()
                
                total_loss = alpha_loss + geo_loss
                
                # Update MetaHuman val loss trackers
                metahuman_val_losses['alpha'] += alpha_loss.item()
                metahuman_val_losses['geo'] += geo_loss.item()
                metahuman_val_losses['total'] += total_loss.item()
                
                val_loss += total_loss.item()
                val_batches += 1
                
                pbar.set_postfix({'Val Loss': f'{total_loss.item():.4f}'})
        
        # Log results
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        
        print(f"📊 Epoch {epoch+1} Results:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss: {avg_val_loss:.4f}")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Train_Total', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val_Total', avg_val_loss, epoch)
        writer.add_scalar('Learning_Rate/Backbone', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Learning_Rate/Heads', optimizer.param_groups[1]['lr'], epoch)
        
        # Log MetaHuman dataset losses
        if train_batches > 0:
            for loss_type, value in metahuman_train_losses.items():
                writer.add_scalar(f'MetaHuman/Train_{loss_type}', value / train_batches, epoch)
        if val_batches > 0:
            for loss_type, value in metahuman_val_losses.items():
                writer.add_scalar(f'MetaHuman/Val_{loss_type}', value / val_batches, epoch)
        
        # Save only best model (no checkpoints to save disk space)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"📈 Validation loss improved: {avg_val_loss:.4f} < {best_val_loss + (avg_val_loss - best_val_loss):.4f}")
            try:
                torch.save(model.state_dict(), "best_model.pth")
                print(f"🌟 New best model saved to 'best_model.pth' (Val Loss: {best_val_loss:.4f})")
            except (OSError, IOError) as e:
                print(f"⚠️  Failed to save best model due to disk space: {e}")
                print(f"   Continuing training without saving...")
        else:
            print(f"📊 No improvement: {avg_val_loss:.4f} >= {best_val_loss:.4f} (best so far)")
        
        # Step the scheduler
        scheduler.step()
        
        # Generate sample results
        if metahuman_train_sample_batch is not None and metahuman_val_sample_batch is not None:
             save_metahuman_samples(model, metahuman_train_sample_batch, metahuman_val_sample_batch, epoch, device)
    
    # Final model saving disabled to save disk space (only best_model.pth is kept)
        
    writer.close()
    print("📝 TensorBoard writer closed.")
    print(f"🎉 Training completed!")
    print(f"🌟 Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Multi-task DPT Model with MetaHuman Dataset')
    parser.add_argument('--load_pretrained', type=str, default=None,
                        help='Path to pretrained model file (e.g., best_model.pth)')
    
    args = parser.parse_args()
    train_model(load_pretrained=args.load_pretrained)
