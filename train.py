"""
Training Script for Multi-task DPT Model
Simple and clean training implementation with FP16 support
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import time
import argparse

# Import our modules
from multi_head_dpt import create_multi_head_dpt
from david_dataset import create_dataloaders
from loss import (
    compute_robust_depth_loss,
    compute_surface_normal_loss,
    compute_alpha_loss
)
from vis_util import save_sample_results


def train_model(load_pretrained=None):
    """Main training function."""
    print("🚀 Starting Multi-task DPT training...")
    
    # Configuration - combine all three dataset directories
    data_roots = [
        "F:/sy_human/SynthHuman_0000",
        "F:/sy_human/SynthHuman_0001",
        #"F:/sy_human/SynthHuman_0002",
        #"F:/sy_human/SynthHuman_0003",
        #"F:/sy_human/SynthHuman_0004",
        #"F:/sy_human/SynthHuman_0005"
    ]
    batch_size = 16
    num_epochs = 50
    lr = 2e-4  # Reduced learning rate
    use_fp16 = True  # Enable FP16 training for memory savings and speed
    
    # Use GPU 1 (second GPU)
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
        print(f"📱 Using GPU 1: {torch.cuda.get_device_name(1)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(1).total_memory / 1e9:.1f} GB")
        print(f"🔥 FP16 Training: {'Enabled' if use_fp16 else 'Disabled'}")
        print(f"📊 Batch size: {batch_size}")
    else:
        device = torch.device('cpu')
        print("📱 Using CPU - no CUDA available")
    
    # Create data loaders
    print("📊 Creating data loaders from combined datasets...")
    train_loader, val_loader = create_dataloaders(
        data_roots=data_roots,
        batch_size=batch_size,
        num_workers=2,
        image_size=384,
        train_ratio=0.95
    )
    print(f"🎯 Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print("🧠 Creating Multi-Head DPT model...")
    model = create_multi_head_dpt(
        backbone="vitb16_384",
        features=256,
        use_bn=False,
        pretrained=True
    )
    model = model.to(device)
    
    # Load pretrained model if specified
    start_epoch = 0
    if load_pretrained and os.path.exists(load_pretrained):
        print(f"📥 Loading pretrained model from: {load_pretrained}")
        try:
            checkpoint = torch.load(load_pretrained, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Loading from checkpoint file
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
            else:
                # Loading from state dict file (like best_model.pth)
                model.load_state_dict(checkpoint)
                print(f"✅ Loaded model state dict successfully")
        except Exception as e:
            print(f"❌ Failed to load pretrained model: {e}")
            print(f"   Starting from scratch...")
    elif load_pretrained:
        print(f"⚠️  Pretrained model file not found: {load_pretrained}")
        print(f"   Starting from scratch...")
    
    # Count parameters
    model_for_params = model
    total_params = sum(p.numel() for p in model_for_params.parameters())
    trainable_params = sum(p.numel() for p in model_for_params.parameters() if p.requires_grad)
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
    
    # Much more conservative optimizer setup
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.1, 'weight_decay': 1e-4},  # 1e-5 for backbone
        {'params': head_params, 'lr': lr, 'weight_decay': 1e-4}             # 1e-4 for heads
    ])
    
    # More conservative scheduler - simple step decay
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
        # Increase batch size for FP16 (more memory available)
        if batch_size == 8:
            batch_size = 12
            print(f"📊 Increased batch size to {batch_size} for FP16 training")
            # Recreate dataloaders with new batch size
            train_loader, val_loader = create_dataloaders(
                data_roots=data_roots,
                batch_size=batch_size,
                num_workers=2,
                image_size=384,
                train_ratio=0.95
            )
            print(f"🎯 Updated - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    print("🏃‍♂️ Starting training loop...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{start_epoch + num_epochs}")
        current_lr_backbone = optimizer.param_groups[0]['lr']
        current_lr_heads = optimizer.param_groups[1]['lr']
        print(f"🔧 Learning rates: backbone={current_lr_backbone:.2e}, heads={current_lr_heads:.2e}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        train_sample_batch = None
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            # Timing profiling to identify bottlenecks
            batch_start_time = time.time()
            
            # Save first batch for visualization
            if batch_idx == 0:
                train_sample_batch = {
                    'rgb': batch['rgb'],
                    'depth': batch['depth'],
                    'normals': batch['normals'],
                    'alpha': batch['alpha']
                }
            
            # Move to device
            data_load_time = time.time()
            rgb = batch['rgb'].to(device)
            depth_gt = batch['depth'].to(device)
            normals_gt = batch['normals'].to(device)
            alpha_gt = batch['alpha'].to(device)
            
            # Handle alpha dimensions
            if alpha_gt.dim() == 4 and alpha_gt.shape[1] == 1:
                alpha_gt = alpha_gt.squeeze(1)
            
            data_transfer_time = time.time()
            optimizer.zero_grad()
            
            # Debug: Print input tensor info
            if batch_idx == 0:
                print(f"🔍 DEBUG - Batch {batch_idx}:")
                print(f"   RGB shape: {rgb.shape}, dtype: {rgb.dtype}, device: {rgb.device}")
                print(f"   Depth GT shape: {depth_gt.shape}, dtype: {depth_gt.dtype}, range: [{depth_gt.min():.3f}, {depth_gt.max():.3f}]")
                print(f"   Normals GT shape: {normals_gt.shape}, dtype: {normals_gt.dtype}, range: [{normals_gt.min():.3f}, {normals_gt.max():.3f}]")
                print(f"   Alpha GT shape: {alpha_gt.shape}, dtype: {alpha_gt.dtype}, range: [{alpha_gt.min():.3f}, {alpha_gt.max():.3f}]")
            
            # Forward pass with FP16 autocast if enabled
            if use_fp16 and scaler is not None:
                with autocast():
                    try:
                        outputs = model(rgb)
                        depth_pred = outputs['depth']
                        normals_pred = outputs['normals']
                        alpha_logits = outputs['alpha_logits']
                        
                        # Compute losses with detailed timing profiling
                        forward_time = time.time()
                        
                        depth_loss, ssi_loss, direct_loss = compute_robust_depth_loss(depth_pred, depth_gt, alpha_gt)
                        depth_loss_time = time.time()
                        
                        normals_loss = compute_surface_normal_loss(normals_pred, normals_gt, alpha_gt)
                        normals_loss_time = time.time()
                        
                        alpha_loss, bce_loss, l1_loss, dice_loss = compute_alpha_loss(alpha_logits, alpha_gt)
                        alpha_loss_time = time.time()
                        
                        # Equal weighted total loss
                        total_loss = depth_loss + normals_loss + alpha_loss
                        
                        # Print timing info for first batch to identify bottleneck
                        if batch_idx == 0:
                            print(f"⏱️  Timing Profile:")
                            print(f"   Data loading: {(data_load_time - batch_start_time)*1000:.1f}ms")
                            print(f"   Data transfer: {(data_transfer_time - data_load_time)*1000:.1f}ms")
                            print(f"   Forward pass: {(forward_time - data_transfer_time)*1000:.1f}ms")
                            print(f"   Depth loss: {(depth_loss_time - forward_time)*1000:.1f}ms")
                            print(f"   Normals loss: {(normals_loss_time - depth_loss_time)*1000:.1f}ms")
                            print(f"   Alpha loss: {(alpha_loss_time - normals_loss_time)*1000:.1f}ms")
                    
                    except Exception as e:
                        print(f"❌ ERROR in FP16 forward pass: {e}")
                        print(f"   Batch idx: {batch_idx}")
                        import traceback
                        traceback.print_exc()
                        raise e
            else:
                # Standard FP32 forward pass
                try:
                    outputs = model(rgb)
                    depth_pred = outputs['depth']
                    normals_pred = outputs['normals']
                    alpha_logits = outputs['alpha_logits']
                    
                    # Compute losses with detailed timing profiling (FP32)
                    forward_time = time.time()
                    
                    depth_loss, ssi_loss, direct_loss = compute_robust_depth_loss(depth_pred, depth_gt, alpha_gt)
                    depth_loss_time = time.time()
                    
                    normals_loss = compute_surface_normal_loss(normals_pred, normals_gt, alpha_gt)
                    normals_loss_time = time.time()
                    
                    alpha_loss, bce_loss, l1_loss, dice_loss = compute_alpha_loss(alpha_logits, alpha_gt)
                    alpha_loss_time = time.time()
                    
                    # Equal weighted total loss
                    total_loss = depth_loss + normals_loss + alpha_loss
                    
                    # Print timing info for first batch to identify bottleneck (FP32)
                    if batch_idx == 0:
                        print(f"⏱️  Timing Profile (FP32):")
                        print(f"   Data loading: {(data_load_time - batch_start_time)*1000:.1f}ms")
                        print(f"   Data transfer: {(data_transfer_time - data_load_time)*1000:.1f}ms") 
                        print(f"   Forward pass: {(forward_time - data_transfer_time)*1000:.1f}ms")
                        print(f"   Depth loss: {(depth_loss_time - forward_time)*1000:.1f}ms")
                        print(f"   Normals loss: {(normals_loss_time - depth_loss_time)*1000:.1f}ms")
                        print(f"   Alpha loss: {(alpha_loss_time - normals_loss_time)*1000:.1f}ms")
                
                except Exception as e:
                    print(f"❌ ERROR in FP32 forward pass: {e}")
                    print(f"   Batch idx: {batch_idx}")
                    import traceback
                    traceback.print_exc()
                    raise e
            
            # Check for NaN/Inf before backward pass
            if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss.item() > 1000:
                print(f"⚠️  Unstable loss detected: {total_loss.item():.4f}, skipping batch {batch_idx}")
                continue
                
            # Backward pass with FP16 scaling if enabled
            if use_fp16 and scaler is not None:
                # FP16 backward pass
                scaler.scale(total_loss).backward()
                
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                if grad_norm > 10.0:
                    #print(f"⚠️  Large gradient norm detected: {grad_norm:.2f}, clipping more aggressively")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                
                # Update optimizer with scaling
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard FP32 backward pass
                total_loss.backward()
                
                # Check gradients for explosion
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                if grad_norm > 10.0:
                    #print(f"⚠️  Large gradient norm detected: {grad_norm:.2f}, clipping more aggressively")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                
                optimizer.step()
                
            train_loss += total_loss.item()
            train_batches += 1
            
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Depth': f'{depth_loss.item():.4f}',
                'Normals': f'{normals_loss.item():.4f}',
                'Alpha': f'{alpha_loss.item():.4f}',
            })
            
            # Break early for testing (remove for full training)
            #if batch_idx >= 300:
                #break
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_sample_batch = None
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for batch_idx, batch in enumerate(pbar):
                # Save first batch for visualization
                if batch_idx == 0:
                    val_sample_batch = {
                        'rgb': batch['rgb'],
                        'depth': batch['depth'],
                        'normals': batch['normals'],
                        'alpha': batch['alpha']
                    }
                
                rgb = batch['rgb'].to(device)
                depth_gt = batch['depth'].to(device)
                normals_gt = batch['normals'].to(device)
                alpha_gt = batch['alpha'].to(device)
                
                if alpha_gt.dim() == 4 and alpha_gt.shape[1] == 1:
                    alpha_gt = alpha_gt.squeeze(1)
                
                outputs = model(rgb)
                depth_pred = outputs['depth']
                normals_pred = outputs['normals']
                alpha_logits = outputs['alpha_logits']
                
                depth_loss, ssi_loss, direct_loss = compute_robust_depth_loss(depth_pred, depth_gt, alpha_gt)
                normals_loss = compute_surface_normal_loss(normals_pred, normals_gt, alpha_gt)
                # Combined alpha loss: BCE + L1 + Dice (consistent with training)
                alpha_loss, bce_loss, l1_loss, dice_loss = compute_alpha_loss(alpha_logits, alpha_gt)
                total_loss = depth_loss + normals_loss + alpha_loss
                
                val_loss += total_loss.item()
                val_batches += 1
                
                pbar.set_postfix({'Val Loss': f'{total_loss.item():.4f}'})
                
                # Break early for testing
                #if batch_idx >= 20:
                    #break
        
        # Log results
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        
        print(f"📊 Epoch {epoch+1} Results:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss: {avg_val_loss:.4f}")
        if False:
            # Save checkpoint
            checkpoint_path = f"checkpoint_epoch_{epoch+1:02d}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "best_model.pth")
                print(f"🌟 New best model saved (Val Loss: {best_val_loss:.4f})")
        
        # Step the scheduler at the end of each epoch (for StepLR)
        scheduler.step()
        
        # Generate sample results
        if train_sample_batch is not None and val_sample_batch is not None:
            save_sample_results(model, train_sample_batch, val_sample_batch, epoch, device)
    
    # Save final model
    if False:
        torch.save(model.state_dict(), "final_model.pth")
        
    print(f"🎉 Training completed! Final model saved: final_model.pth")
    print(f"🌟 Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Multi-task DPT Model')
    parser.add_argument('--load_pretrained', type=str, default=None,
                        help='Path to pretrained model file (e.g., best_model.pth)')
    
    args = parser.parse_args()
    train_model(load_pretrained=args.load_pretrained)
