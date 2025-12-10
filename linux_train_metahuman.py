"""
Multi-GPU Training Script for Dual-Task DPT Model (MetaHuman Dataset)
Optimized for Linux training with DistributedDataParallel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import datetime
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Import our modules
from multi_head_dpt import create_dual_head_dpt
from metahuman_dataset2 import create_metahuman_dataloaders2
from loss import compute_alpha_loss

def save_metahuman_samples(model, train_batch, val_batch, epoch, device, rank=0, output_dir="training_samples"):
    """
    Save sample predictions for MetaHuman dataset.
    Visualizes RGB, Mask, Base Geo, and n encoding images (sin components).
    """
    if rank != 0:
        return
        
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
    if rank == 0:
        print(f"📸 MetaHuman sample results saved: {sample_path}")

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
        pass


def create_distributed_dataloaders(data_roots, batch_size, num_workers, image_size, train_ratio, rank, world_size, geo_encoding_n):
    """Create distributed data loaders for MetaHuman dataset."""
    
    # Create the dataloaders normally first
    train_loader, val_loader = create_metahuman_dataloaders2(
        data_roots=data_roots,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        train_ratio=train_ratio,
        geo_encoding_n=geo_encoding_n
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
        prefetch_factor=4,
        persistent_workers=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=4,
        persistent_workers=True
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
    batch_size_per_gpu = 8  # Adjusted for MetaHuman 512x512
    num_epochs = 200
    lr = 2e-4
    use_fp16 = True
    
    # Geo encoding configuration
    geo_encoding_n = 4
    geo_output_channels = 3 + 6 * geo_encoding_n
    
    # Set device
    device = torch.device(f'cuda:{rank}')
    
    # Initialize TensorBoard writer (only on rank 0)
    writer = None
    if rank == 0:
        writer = SummaryWriter(f"runs/dpt_metahuman_{int(time.time())}")
        print(f"📝 TensorBoard logs will be saved to: {writer.log_dir}")
        print(f"📱 Training on {world_size} GPUs")
        print(f"📊 Batch size per GPU: {batch_size_per_gpu}")
        print(f"📊 Total effective batch size: {batch_size_per_gpu * world_size}")
        print(f"🔥 FP16 Training: {'Enabled' if use_fp16 else 'Disabled'}")
        print(f"📍 Data roots: {len(data_roots)} datasets")
        print(f"🔢 Geo Encoding N: {geo_encoding_n} (Total Geo Channels: {geo_output_channels})")
    
    # Create distributed data loaders
    try:
        print(f"[Rank {rank}] 📊 Creating distributed data loaders...")
        
        train_loader, val_loader, train_sampler, val_sampler = create_distributed_dataloaders(
            data_roots=data_roots,
            batch_size=batch_size_per_gpu,
            num_workers=2,
            image_size=512,  # MetaHuman uses 512
            train_ratio=0.95,
            rank=rank,
            world_size=world_size,
            geo_encoding_n=geo_encoding_n
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
        print(f"[Rank {rank}] 🧠 Creating Dual-Head DPT model...")
        
        model = create_dual_head_dpt(
            backbone="vitb16_384",
            features=256,
            use_bn=False,
            pretrained=True,
            geo_output_channels=geo_output_channels
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
    
    # Load pretrained model
    start_epoch = 0
    if load_pretrained and os.path.exists(load_pretrained):
        if rank == 0:
            print(f"📥 Loading pretrained model from: {load_pretrained}")
        try:
            checkpoint = torch.load(load_pretrained, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                start_epoch = checkpoint.get('epoch', 0) + 1
                if rank == 0:
                    print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 0)} (strict=False)")
            else:
                model.load_state_dict(checkpoint, strict=False)
                if rank == 0:
                    print(f"✅ Loaded model state dict successfully (strict=False)")
        except Exception as e:
            if rank == 0:
                print(f"❌ Failed to load pretrained model: {e}")
                print(f"   Starting from scratch...")
    elif rank == 0:
        if os.path.exists("best_model.pth"):
             print(f"🔄 Auto-resuming from: best_model.pth")
             try:
                checkpoint = torch.load("best_model.pth", map_location=device)
                model.load_state_dict(checkpoint, strict=False)
                print("✅ Loaded best_model.pth")
             except:
                print("⚠️ Failed to load best_model.pth")
        else:
            print(f"🆕 Starting fresh training (no existing model found)")
    
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
    
    # Separate learning rates
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'pretrained' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    # Optimizer
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.1, 'weight_decay': 5e-5},
        {'params': head_params, 'lr': lr, 'weight_decay': 5e-5}
    ])
    
    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=10,
        gamma=0.7
    )
    
    # FP16 scaler
    scaler = None
    if use_fp16:
        scaler = GradScaler()
        if rank == 0:
            print("🔥 FP16 GradScaler initialized")
    
    if rank == 0:
        print("🏃‍♂️ Starting distributed training loop...")
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
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
        
        # Loss tracking
        local_losses = {'alpha': 0.0, 'geo': 0.0, 'total': 0.0}
        
        if rank == 0:
            pbar = tqdm(train_loader, desc="Training")
        else:
            pbar = train_loader
        
        for batch_idx, batch in enumerate(pbar):
            # Save first batch for visualization
            if batch_idx == 0 and rank == 0:
                train_sample_batch = {
                    'rgb': batch['rgb'],
                    'face_mask': batch['face_mask'],
                    'geo': batch['geo']
                }
            
            rgb = batch['rgb'].to(device, non_blocking=True)
            face_mask_gt = batch['face_mask'].to(device, non_blocking=True)
            geo_gt = batch['geo'].to(device, non_blocking=True)
            
            if face_mask_gt.dim() == 4 and face_mask_gt.shape[1] == 1:
                face_mask_gt = face_mask_gt.squeeze(1)
            
            optimizer.zero_grad()
            
            # Forward pass
            if use_fp16 and scaler is not None:
                with autocast():
                    outputs = model(rgb)
                    alpha_logits = outputs['alpha_logits']
                    geo_pred = outputs['geo']
                    
                    # Compute losses
                    alpha_loss, _, _, _ = compute_alpha_loss(alpha_logits, face_mask_gt)
                    
                    face_mask_gt_expanded = face_mask_gt.unsqueeze(1)
                    valid_mask = face_mask_gt_expanded
                    if 'hair_mask' in batch:
                        hair_mask = batch['hair_mask'].to(device)
                        valid_mask = valid_mask * (1.0 - hair_mask)

                    geo_loss = F.l1_loss(geo_pred, geo_gt, reduction='none')
                    geo_loss = (geo_loss * valid_mask).mean()
                    
                    total_loss = alpha_loss + geo_loss
            else:
                outputs = model(rgb)
                alpha_logits = outputs['alpha_logits']
                geo_pred = outputs['geo']
                
                alpha_loss, _, _, _ = compute_alpha_loss(alpha_logits, face_mask_gt)
                
                face_mask_gt_expanded = face_mask_gt.unsqueeze(1)
                valid_mask = face_mask_gt_expanded
                if 'hair_mask' in batch:
                    hair_mask = batch['hair_mask'].to(device)
                    valid_mask = valid_mask * (1.0 - hair_mask)

                geo_loss = F.l1_loss(geo_pred, geo_gt, reduction='none')
                geo_loss = (geo_loss * valid_mask).mean()
                
                total_loss = alpha_loss + geo_loss

            # Check for NaN/Inf
            if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss.item() > 1000:
                if rank == 0:
                    print(f"⚠️  Unstable loss detected: {total_loss.item():.4f}, skipping batch {batch_idx}")
                continue
                
            # Backward
            if use_fp16 and scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
            train_loss += total_loss.item()
            local_losses['alpha'] += alpha_loss.item()
            local_losses['geo'] += geo_loss.item()
            local_losses['total'] += total_loss.item()
            train_batches += 1
            
            if rank == 0 and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Geo': f'{geo_loss.item():.4f}',
                    'Alpha': f'{alpha_loss.item():.4f}'
                })
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_sample_batch = None
        val_local_losses = {'alpha': 0.0, 'geo': 0.0, 'total': 0.0}
        
        with torch.no_grad():
            if rank == 0:
                pbar = tqdm(val_loader, desc="Validation")
            else:
                pbar = val_loader
                
            for batch_idx, batch in enumerate(pbar):
                if batch_idx == 0 and rank == 0:
                    val_sample_batch = {
                        'rgb': batch['rgb'],
                        'face_mask': batch['face_mask'],
                        'geo': batch['geo']
                    }
                
                rgb = batch['rgb'].to(device, non_blocking=True)
                face_mask_gt = batch['face_mask'].to(device, non_blocking=True)
                geo_gt = batch['geo'].to(device, non_blocking=True)
                
                if face_mask_gt.dim() == 4 and face_mask_gt.shape[1] == 1:
                    face_mask_gt = face_mask_gt.squeeze(1)
                
                outputs = model(rgb)
                alpha_logits = outputs['alpha_logits']
                geo_pred = outputs['geo']
                
                alpha_loss, _, _, _ = compute_alpha_loss(alpha_logits, face_mask_gt)
                
                face_mask_gt_expanded = face_mask_gt.unsqueeze(1)
                valid_mask = face_mask_gt_expanded
                if 'hair_mask' in batch:
                    hair_mask = batch['hair_mask'].to(device)
                    valid_mask = valid_mask * (1.0 - hair_mask)

                geo_loss = F.l1_loss(geo_pred, geo_gt, reduction='none')
                geo_loss = (geo_loss * valid_mask).mean()
                
                total_loss = alpha_loss + geo_loss
                
                val_loss += total_loss.item()
                val_local_losses['alpha'] += alpha_loss.item()
                val_local_losses['geo'] += geo_loss.item()
                val_local_losses['total'] += total_loss.item()
                val_batches += 1
                
                if rank == 0 and hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix({'Val Loss': f'{total_loss.item():.4f}'})

        # Synchronization
        if world_size > 1:
            train_loss_tensor = torch.tensor(train_loss / train_batches if train_batches > 0 else 0).to(device)
            val_loss_tensor = torch.tensor(val_loss / val_batches if val_batches > 0 else 0).to(device)
            
            # Additional detail losses for logging (optional, focusing on total for sync to keep it simple or sync all)
            # For simplicity syncing totals, rank 0 will just log its own detailed breakdown approximation or sync them too.
            # Let's just sync totals for decision making
            
            try:
                torch.cuda.synchronize()
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                
                avg_train_loss = (train_loss_tensor / world_size).item()
                avg_val_loss = (val_loss_tensor / world_size).item()
            except Exception as e:
                if rank == 0:
                    print(f"❌ Loss synchronization failed: {e}")
                avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
                avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        else:
            avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        
        # Logging and Saving (Rank 0)
        if rank == 0:
            print(f"📊 Epoch {epoch+1} Results:")
            print(f"   Train Loss: {avg_train_loss:.4f}")
            print(f"   Val Loss: {avg_val_loss:.4f}")
            
            if writer:
                writer.add_scalar('Loss/Train_Total', avg_train_loss, epoch)
                writer.add_scalar('Loss/Val_Total', avg_val_loss, epoch)
                writer.add_scalar('Learning_Rate/Backbone', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('Learning_Rate/Heads', optimizer.param_groups[1]['lr'], epoch)
            
            # Save checkpoint
            checkpoint_dir = "../../hy-tmp/checkpoint/"
            try:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"metahuman_checkpoint_epoch_{epoch+1:02d}.pth")
                
                # Checkpoint dict
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                }
                
                # Only save occasional checkpoints or if it's the best
                if (epoch + 1) % 5 == 0:
                    torch.save(save_dict, checkpoint_path)
                    print(f"💾 Checkpoint saved: {checkpoint_path}")
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_path = "best_model.pth" # Save locally for easy access or in hy-tmp
                    try:
                        torch.save(model.module.state_dict(), best_path)
                        print(f"🌟 New best model saved: {best_path} (Val Loss: {best_val_loss:.4f})")
                    except Exception as e:
                        print(f"❌ Failed to save best model: {e}")
            
            except Exception as e:
                print(f"⚠️  Checkpoint saving issue: {e}")

        scheduler.step()
        
        # Visualize
        if rank == 0 and train_sample_batch is not None and val_sample_batch is not None:
            save_metahuman_samples(model.module, train_sample_batch, val_sample_batch, epoch, device, rank=rank)
        
        # Barrier
        if world_size > 1:
            try:
                torch.cuda.synchronize()
                dist.barrier()
            except Exception as e:
                if rank == 0:
                    print(f"⚠️  Barrier failed: {e}")

    # Final cleanup
    if rank == 0 and writer:
        writer.close()
    
    cleanup_distributed()


def train_model(load_pretrained=None):
    """Main training function that launches multi-GPU training."""
    world_size = torch.cuda.device_count()
    
    if world_size < 1:
        print("❌ No CUDA devices found! Please ensure you have CUDA-capable GPUs.")
        return
    
    print(f"🚀 Starting Multi-GPU training on {world_size} GPUs (MetaHuman Dataset)...")
    print("⚡ To run this script, use:")
    print("   python -m torch.distributed.launch --nproc_per_node=4 linux_train_metahuman.py")
    print("   OR")
    print("   torchrun --nproc_per_node=4 linux_train_metahuman.py")
    print()
    
    # Launch distributed training
    mp.spawn(train_worker, args=(world_size, load_pretrained), nprocs=world_size, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-GPU Training for MetaHuman DPT Model')
    parser.add_argument('--load_pretrained', type=str, default=None,
                        help='Path to pretrained model file (e.g., best_model.pth)')
    
    args = parser.parse_args()
    
    # Check if we're in a distributed launch
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        train_worker(local_rank, world_size, args.load_pretrained)
    else:
        train_model(load_pretrained=args.load_pretrained)
