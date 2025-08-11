"""
Multi-GPU Training Script for Multi-task DPT Model
Optimized for 4 GPU Linux training with DistributedDataParallel
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import socket
import datetime
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


# Loss functions are now imported from loss.py


def setup_distributed(rank, world_size):
    """Setup distributed training environment."""
    try:
        print(f"[Rank {rank}] Setting up distributed training...")
        
        # Initialize process group
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12355'
        
        # Set NCCL environment variables for better stability
        os.environ['NCCL_TIMEOUT'] = '1800'  # 30 minutes timeout instead of 10
        os.environ['NCCL_BLOCKING_WAIT'] = '1'  # Use blocking wait for better error detection
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'  # Enable async error handling
        os.environ['NCCL_DEBUG'] = 'WARN'  # Reduce verbosity but keep warnings
        
        print(f"[Rank {rank}] Initializing process group...")
        
        # Initialize process group with longer timeout
        dist.init_process_group(
            "nccl", 
            rank=rank, 
            world_size=world_size,
            timeout=datetime.timedelta(minutes=30)  # 30 minutes timeout
        )
        
        print(f"[Rank {rank}] Process group initialized successfully")
        
        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
            print(f"[Rank {rank}] Set CUDA device to {rank}")
        else:
            raise RuntimeError(f"CUDA not available on rank {rank}")
            
    except Exception as e:
        print(f"[Rank {rank}] ❌ Failed to setup distributed training: {e}")
        import traceback
        traceback.print_exc()
        raise e


def cleanup_distributed():
    """Clean up distributed training with robust error handling."""
    try:
        # Synchronize all processes before cleanup
        if dist.is_available() and dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
            dist.destroy_process_group()
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")
        # Continue cleanup even if distributed cleanup fails
        pass


def create_distributed_dataloaders(data_roots, batch_size, num_workers, image_size, train_ratio, rank, world_size):
    """Create distributed data loaders."""
    # Import the dataloader creation function
    from david_dataset import create_dataloaders
    
    # Create the dataloaders normally first
    train_loader, val_loader = create_dataloaders(
        data_roots=data_roots,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        train_ratio=train_ratio
    )
    
    # Get the datasets
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create optimized data loaders with distributed samplers
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,  # Prefetch more batches to prevent GPU starvation
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=4,  # Prefetch more batches
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    return train_loader, val_loader, train_sampler, val_sampler


def train_worker(rank, world_size, load_pretrained=None):
    """Worker function for distributed training."""
    print(f"🚀 Starting training worker {rank}/{world_size}...")
    
    try:
        # Setup distributed training
        setup_distributed(rank, world_size)
        
        print(f"[Rank {rank}] ✅ Distributed setup completed")
        
        # Configuration - Linux paths (adjust these to your actual data locations)
        data_roots = [
            "../../hy-tmp/data/00",
            "../../hy-tmp/data/01",
            "../../hy-tmp/data/02",
            "../../hy-tmp/data/03",
            "../../hy-tmp/data/04",
            "../../hy-tmp/data/05",
            "../../hy-tmp/data/06",
            "../../hy-tmp/data/07",
            "../../hy-tmp/data/08",
            "../../hy-tmp/data/09",
            "../../hy-tmp/data/10",
            "../../hy-tmp/data/11",
            "../../hy-tmp/data/12",
            "../../hy-tmp/data/13",
            "../../hy-tmp/data/14",
            "../../hy-tmp/data/15",
            "../../hy-tmp/data/16",
            "../../hy-tmp/data/17",
            "../../hy-tmp/data/18",
            # Add more dataset paths as needed
        ]
        
        print(f"[Rank {rank}] 📁 Data roots configured")
        
    except Exception as e:
        print(f"[Rank {rank}] ❌ Error during worker setup: {e}")
        import traceback
        traceback.print_exc()
        cleanup_distributed()
        raise e
    
    # Multi-GPU configuration
    batch_size_per_gpu = 24  # Batch size per GPU (total effective batch size = 8 * 4 = 32)
    num_epochs = 200
    lr = 3e-4
    use_fp16 = True
    
    # Set device
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        print(f"📱 Training on {world_size} GPUs")
        print(f"📊 Batch size per GPU: {batch_size_per_gpu}")
        print(f"📊 Total effective batch size: {batch_size_per_gpu * world_size}")
        print(f"🔥 FP16 Training: {'Enabled' if use_fp16 else 'Disabled'}")
        print(f"📍 Data roots: {len(data_roots)} datasets")
    
    # Create distributed data loaders
    try:
        print(f"[Rank {rank}] 📊 Creating distributed data loaders...")
        
        train_loader, val_loader, train_sampler, val_sampler = create_distributed_dataloaders(
            data_roots=data_roots,
            batch_size=batch_size_per_gpu,
            num_workers=2,  # Reduced to 2 to prevent hanging with many workers
            image_size=384,
            train_ratio=0.95,
            rank=rank,
            world_size=world_size
        )
        
        print(f"[Rank {rank}] ✅ Data loaders created successfully")
        print(f"[Rank {rank}] 🎯 Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
    except Exception as e:
        print(f"[Rank {rank}] ❌ Failed to create data loaders: {e}")
        import traceback
        traceback.print_exc()
        cleanup_distributed()
        raise e
    
    # Create model
    try:
        print(f"[Rank {rank}] 🧠 Creating Multi-Head DPT model...")
        
        model = create_multi_head_dpt(
            backbone="vitb16_384",
            features=256,
            use_bn=False,
            pretrained=True
        )
        
        print(f"[Rank {rank}] ✅ Model created, moving to device {device}")
        model = model.to(device)
        print(f"[Rank {rank}] ✅ Model moved to device successfully")
        
    except Exception as e:
        print(f"[Rank {rank}] ❌ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        cleanup_distributed()
        raise e
    
    # Load pretrained model if specified (only on rank 0 for logging)
    start_epoch = 0
    if load_pretrained and os.path.exists(load_pretrained):
        if rank == 0:
            print(f"📥 Loading pretrained model from: {load_pretrained}")
        try:
            checkpoint = torch.load(load_pretrained, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Loading from checkpoint file
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                if rank == 0:
                    print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
            else:
                # Loading from state dict file (like best_model.pth)
                model.load_state_dict(checkpoint)
                if rank == 0:
                    print(f"✅ Loaded model state dict successfully")
        except Exception as e:
            if rank == 0:
                print(f"❌ Failed to load pretrained model: {e}")
                print(f"   Starting from scratch...")
    elif load_pretrained and rank == 0:
        print(f"⚠️  Pretrained model file not found: {load_pretrained}")
        print(f"   Starting from scratch...")
    
    # Wrap model with DistributedDataParallel
    try:
        print(f"[Rank {rank}] 🔄 Wrapping model with DistributedDataParallel...")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        print(f"[Rank {rank}] ✅ Model wrapped with DDP successfully")
        
    except Exception as e:
        print(f"[Rank {rank}] ❌ Failed to wrap model with DDP: {e}")
        import traceback
        traceback.print_exc()
        cleanup_distributed()
        raise e
    
    # Count parameters (only on rank 0)
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"⚙️  Total parameters: {total_params:,}")
        print(f"🎯 Trainable parameters: {trainable_params:,}")
        print(f"🚀 Ready to start training!")
    
    # Separate learning rates for backbone and heads
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'pretrained' in name:  # Original DPT backbone
            backbone_params.append(param)
        else:  # Task heads and fusion blocks
            head_params.append(param)
    
    # Optimizer with different learning rates
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.1, 'weight_decay': 1e-4},  # Lower LR for backbone
        {'params': head_params, 'lr': lr, 'weight_decay': 1e-4}             # Higher LR for heads
    ])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=10,    # Reduce LR every 10 epochs
        gamma=0.7        # Multiply by 0.7
    )
    
    # Initialize FP16 scaler
    scaler = None
    if use_fp16:
        scaler = GradScaler()
        if rank == 0:
            print("🔥 FP16 GradScaler initialized")
    
    # Training loop
    if rank == 0:
        print("🏃‍♂️ Starting distributed training loop...")
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Set epoch for distributed sampler (ensures different shuffling each epoch)
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\n📅 Epoch {epoch+1}/{start_epoch + num_epochs}")
            current_lr_backbone = optimizer.param_groups[0]['lr']
            current_lr_heads = optimizer.param_groups[1]['lr']
            print(f"🔧 Learning rates: backbone={current_lr_backbone:.2e}, heads={current_lr_heads:.2e}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        train_sample_batch = None
        
        # Use tqdm only on rank 0
        if rank == 0:
            pbar = tqdm(train_loader, desc="Training")
        else:
            pbar = train_loader
        
        for batch_idx, batch in enumerate(pbar):
            # Timing profiling to identify bottlenecks
            batch_start_time = time.time()
            
            # Save first batch for visualization (rank 0 only)
            if batch_idx == 0 and rank == 0:
                train_sample_batch = {
                    'rgb': batch['rgb'],
                    'depth': batch['depth'],
                    'normals': batch['normals'],
                    'alpha': batch['alpha']
                }
            
            # Move to device
            data_load_time = time.time()
            rgb = batch['rgb'].to(device, non_blocking=True)
            depth_gt = batch['depth'].to(device, non_blocking=True)
            normals_gt = batch['normals'].to(device, non_blocking=True)
            alpha_gt = batch['alpha'].to(device, non_blocking=True)
            
            # Handle alpha dimensions
            if alpha_gt.dim() == 4 and alpha_gt.shape[1] == 1:
                alpha_gt = alpha_gt.squeeze(1)
            
            data_transfer_time = time.time()
            optimizer.zero_grad()
            
            # Forward pass with FP16 autocast if enabled
            if use_fp16 and scaler is not None:
                with autocast():
                    outputs = model(rgb)
                    depth_pred = outputs['depth']
                    normals_pred = outputs['normals']
                    alpha_logits = outputs['alpha_logits']
                    
                    # Compute losses with detailed timing profiling
                    forward_time = time.time()
                    
                    depth_loss, ssi_loss, direct_loss = compute_robust_depth_loss(depth_pred, depth_gt, alpha_gt)
                    depth_loss_time = time.time()
                    
                    # Safety check: prevent depth loss explosion
                    if torch.isnan(depth_loss) or torch.isinf(depth_loss) or depth_loss > 50.0:
                        if rank == 0:
                            print(f"⚠️  Depth loss explosion detected: {depth_loss.item():.4f}, using fallback")
                        depth_loss = torch.tensor(1.0, device=device, requires_grad=True)
                    
                    normals_loss = compute_surface_normal_loss(normals_pred, normals_gt, alpha_gt)
                    normals_loss_time = time.time()
                    
                    alpha_loss, bce_loss, l1_loss, dice_loss_val = compute_alpha_loss(alpha_logits, alpha_gt)
                    alpha_loss_time = time.time()
                    
                    # Equal weighted total loss with additional safety check
                    total_loss = depth_loss + normals_loss + alpha_loss
                    
                    # Final safety: clamp total loss to prevent training explosion
                    if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss > 100.0:
                        if rank == 0:
                            print(f"⚠️  Total loss explosion detected: {total_loss.item():.4f}, clamping to safe value")
                        total_loss = torch.clamp(total_loss, max=10.0)
                    
                    # Print timing info for first batch to identify bottleneck
                    if batch_idx == 0 and rank == 0:
                        print(f"⏱️  Timing Profile:")
                        print(f"   Data loading: {(data_load_time - batch_start_time)*1000:.1f}ms")
                        print(f"   Data transfer: {(data_transfer_time - data_load_time)*1000:.1f}ms")
                        print(f"   Forward pass: {(forward_time - data_transfer_time)*1000:.1f}ms")
                        print(f"   Depth loss: {(depth_loss_time - forward_time)*1000:.1f}ms")
                        print(f"   Normals loss: {(normals_loss_time - depth_loss_time)*1000:.1f}ms")
                        print(f"   Alpha loss: {(alpha_loss_time - normals_loss_time)*1000:.1f}ms")
            else:
                # Standard FP32 forward pass
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
                
                alpha_loss, bce_loss, l1_loss, dice_loss_val = compute_alpha_loss(alpha_logits, alpha_gt)
                alpha_loss_time = time.time()
                
                # Equal weighted total loss
                total_loss = depth_loss + normals_loss + alpha_loss
                
                # Print timing info for first batch to identify bottleneck (FP32)
                if batch_idx == 0 and rank == 0:
                    print(f"⏱️  Timing Profile (FP32):")
                    print(f"   Data loading: {(data_load_time - batch_start_time)*1000:.1f}ms")
                    print(f"   Data transfer: {(data_transfer_time - data_load_time)*1000:.1f}ms") 
                    print(f"   Forward pass: {(forward_time - data_transfer_time)*1000:.1f}ms")
                    print(f"   Depth loss: {(depth_loss_time - forward_time)*1000:.1f}ms")
                    print(f"   Normals loss: {(normals_loss_time - depth_loss_time)*1000:.1f}ms")
                    print(f"   Alpha loss: {(alpha_loss_time - normals_loss_time)*1000:.1f}ms")
            
            # Check for NaN/Inf before backward pass
            if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss.item() > 1000:
                if rank == 0:
                    print(f"⚠️  Unstable loss detected: {total_loss.item():.4f}, skipping batch {batch_idx}")
                continue
                
            # Backward pass with FP16 scaling if enabled
            if use_fp16 and scaler is not None:
                # FP16 backward pass
                scaler.scale(total_loss).backward()
                
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                # Update optimizer with scaling
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard FP32 backward pass
                total_loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                optimizer.step()
                
            train_loss += total_loss.item()
            train_batches += 1
            
            # Update progress bar only on rank 0
            if rank == 0 and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Depth': f'{depth_loss.item():.4f}',
                    'Normals': f'{normals_loss.item():.4f}',
                    'Alpha': f'{alpha_loss.item():.4f}',
                })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_sample_batch = None
        
        with torch.no_grad():
            if rank == 0:
                pbar = tqdm(val_loader, desc="Validation")
            else:
                pbar = val_loader
                
            for batch_idx, batch in enumerate(pbar):
                # Save first batch for visualization (rank 0 only)
                if batch_idx == 0 and rank == 0:
                    val_sample_batch = {
                        'rgb': batch['rgb'],
                        'depth': batch['depth'],
                        'normals': batch['normals'],
                        'alpha': batch['alpha']
                    }
                
                rgb = batch['rgb'].to(device, non_blocking=True)
                depth_gt = batch['depth'].to(device, non_blocking=True)
                normals_gt = batch['normals'].to(device, non_blocking=True)
                alpha_gt = batch['alpha'].to(device, non_blocking=True)
                
                if alpha_gt.dim() == 4 and alpha_gt.shape[1] == 1:
                    alpha_gt = alpha_gt.squeeze(1)
                
                outputs = model(rgb)
                depth_pred = outputs['depth']
                normals_pred = outputs['normals']
                alpha_logits = outputs['alpha_logits']
                
                depth_loss, ssi_loss, direct_loss = compute_robust_depth_loss(depth_pred, depth_gt, alpha_gt)
                
                # Safety check: prevent depth loss explosion during validation
                if torch.isnan(depth_loss) or torch.isinf(depth_loss) or depth_loss > 50.0:
                    depth_loss = torch.tensor(1.0, device=device)
                
                normals_loss = compute_surface_normal_loss(normals_pred, normals_gt, alpha_gt)
                alpha_loss, bce_loss, l1_loss, dice_loss_val = compute_alpha_loss(alpha_logits, alpha_gt)
                total_loss = depth_loss + normals_loss + alpha_loss
                
                # Safety check for total validation loss
                if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss > 100.0:
                    total_loss = torch.clamp(total_loss, max=10.0)
                
                val_loss += total_loss.item()
                val_batches += 1
                
                if rank == 0 and hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix({'Val Loss': f'{total_loss.item():.4f}'})
        
        # Synchronize and average losses across all GPUs with robust error handling
        if world_size > 1:
            train_loss_tensor = torch.tensor(train_loss / train_batches if train_batches > 0 else 0).to(device)
            val_loss_tensor = torch.tensor(val_loss / val_batches if val_batches > 0 else 0).to(device)
            
            try:
                if rank == 0:
                    print(f"🔄 Synchronizing losses across {world_size} GPUs...")
                
                # Add explicit synchronization before all_reduce
                torch.cuda.synchronize()
                
                # Synchronize with timeout handling
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                
                avg_train_loss = (train_loss_tensor / world_size).item()
                avg_val_loss = (val_loss_tensor / world_size).item()
                
                if rank == 0:
                    print(f"✅ Loss synchronization completed successfully")
                    
            except Exception as e:
                if rank == 0:
                    print(f"❌ Loss synchronization failed: {e}")
                    print(f"   Using local losses as fallback...")
                
                # Fallback to local losses if synchronization fails
                avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
                avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        else:
            avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        
        # Log results (rank 0 only)
        if rank == 0:
            print(f"📊 Epoch {epoch+1} Results:")
            print(f"   Train Loss: {avg_train_loss:.4f}")
            print(f"   Val Loss: {avg_val_loss:.4f}")
            
            # Save checkpoint with robust error handling (rank 0 only)
            if (epoch + 1) % 5 == 0 or epoch == 0:  # Save every 5 epochs and first epoch
                # Try to save to primary checkpoint directory
                checkpoint_dir = "../../hy-tmp/checkpoint/"
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1:02d}.pth")
                
                # Create checkpoint directory if it doesn't exist
                try:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                except Exception as e:
                    print(f"⚠️  Failed to create checkpoint directory {checkpoint_dir}: {e}")
                
                # Try to save checkpoint
                checkpoint_saved = False
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),  # Use .module for DDP
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss,
                    }, checkpoint_path)
                    print(f"💾 Checkpoint saved: {checkpoint_path}")
                    checkpoint_saved = True
                except Exception as e:
                    print(f"❌ Failed to save checkpoint to {checkpoint_path}: {e}")
                    
                    # Try fallback location (current directory)
                    try:
                        fallback_path = f"checkpoint_epoch_{epoch+1:02d}.pth"
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': avg_train_loss,
                            'val_loss': avg_val_loss,
                        }, fallback_path)
                        print(f"💾 Fallback checkpoint saved: {fallback_path}")
                        checkpoint_saved = True
                    except Exception as e2:
                        print(f"❌ Failed to save fallback checkpoint: {e2}")
                        print("⚠️  Checkpoint saving failed, but training continues...")
                
                # Save best model with error handling
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    
                    # Try to save to checkpoint directory first
                    best_model_paths = [
                        os.path.join(checkpoint_dir, "best_model.pth"),
                        "best_model.pth"  # Fallback to current directory
                    ]
                    
                    best_saved = False
                    for best_path in best_model_paths:
                        try:
                            torch.save(model.module.state_dict(), best_path)
                            print(f"🌟 New best model saved: {best_path} (Val Loss: {best_val_loss:.4f})")
                            best_saved = True
                            break
                        except Exception as e:
                            print(f"❌ Failed to save best model to {best_path}: {e}")
                    
                    if not best_saved:
                        print("⚠️  Best model saving failed, but training continues...")
        
        # Step the scheduler
        scheduler.step()
        
        # Generate sample results (rank 0 only)
        if rank == 0 and train_sample_batch is not None and val_sample_batch is not None:
            save_sample_results(model.module, train_sample_batch, val_sample_batch, epoch, device, rank=rank)
        
        # Wait for all processes to finish the epoch with timeout handling
        if world_size > 1:
            try:
                if rank == 0:
                    print(f"🔄 Waiting for all GPUs to finish epoch {epoch+1}...")
                
                # Add explicit CUDA synchronization before barrier
                torch.cuda.synchronize()
                
                # Use barrier with timeout handling
                dist.barrier()
                
                if rank == 0:
                    print(f"✅ All GPUs finished epoch {epoch+1} successfully")
                    
            except Exception as e:
                if rank == 0:
                    print(f"⚠️  Barrier synchronization failed: {e}")
                    print(f"   Continuing training (some GPUs may be out of sync)")
    
    # Save final model with error handling (rank 0 only)
    if rank == 0:
        final_model_paths = [
            "../../hy-tmp/checkpoint/final_model.pth",
            "final_model.pth"  # Fallback to current directory
        ]
        
        final_saved = False
        for final_path in final_model_paths:
            try:
                # Create directory if needed
                final_dir = os.path.dirname(final_path)
                if final_dir:
                    os.makedirs(final_dir, exist_ok=True)
                
                torch.save(model.module.state_dict(), final_path)
                print(f"🎉 Training completed! Final model saved: {final_path}")
                final_saved = True
                break
            except Exception as e:
                print(f"❌ Failed to save final model to {final_path}: {e}")
        
        if not final_saved:
            print("🎉 Training completed! (Final model saving failed)")
        
        print(f"🌟 Best validation loss: {best_val_loss:.4f}")
    
    # Clean up
    cleanup_distributed()


def train_model(load_pretrained=None):
    """Main training function that launches multi-GPU training."""
    world_size = torch.cuda.device_count()
    
    if world_size < 1:
        print("❌ No CUDA devices found! Please ensure you have CUDA-capable GPUs.")
        return
    
    print(f"🚀 Starting Multi-GPU training on {world_size} GPUs...")
    print("⚡ To run this script, use:")
    print("   python -m torch.distributed.launch --nproc_per_node=4 linux_train.py")
    print("   OR")
    print("   torchrun --nproc_per_node=4 linux_train.py")
    print()
    
    # Launch distributed training
    mp.spawn(train_worker, args=(world_size, load_pretrained), nprocs=world_size, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-GPU Training for Multi-task DPT Model')
    parser.add_argument('--load_pretrained', type=str, default=None,
                        help='Path to pretrained model file (e.g., best_model.pth)')
    
    args = parser.parse_args()
    
    # Check if we're in a distributed launch
    if "LOCAL_RANK" in os.environ:
        # Called by torch.distributed.launch or torchrun
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        train_worker(local_rank, world_size, args.load_pretrained)
    else:
        # Standard launch - use mp.spawn
        train_model(load_pretrained=args.load_pretrained)
