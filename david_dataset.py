"""
SynthHuman Dataset Loader for DAViD Multi-Task Training
Handles RGB images, depth maps, surface normals, and alpha masks
"""

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from typing import Tuple, Dict, List
import torchvision.transforms as transforms
import OpenEXR
import Imath

# Enable OpenEXR support in OpenCV (though we'll use manual loading as fallback)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


class SynthHumanDataset(Dataset):
    """
    SynthHuman dataset for multi-task learning.
    Loads RGB images, depth maps (EXR), surface normals (EXR), and alpha masks (PNG).
    Supports multiple data directories for combining datasets.
    """
    
    def __init__(self, data_roots, split: str = 'train', image_size: int = 512, 
                 train_ratio: float = 0.8, augment: bool = True):
        """
        Args:
            data_roots: Path to SynthHuman dataset directory OR list of directories
            split: 'train' or 'val'  
            image_size: Target image size for training
            train_ratio: Ratio of data to use for training
            augment: Whether to apply data augmentation
        """
        # Support both single directory and list of directories
        if isinstance(data_roots, str):
            self.data_roots = [data_roots]
        else:
            self.data_roots = data_roots
            
        self.split = split
        self.image_size = image_size
        self.augment = augment
        
        # Find all samples across all directories
        self.samples = []  # List of (data_root, sample_id) tuples
        
        for data_root in self.data_roots:
            if not os.path.exists(data_root):
                print(f"⚠️  Warning: Data directory {data_root} does not exist, skipping...")
                continue
                
            rgb_files = glob.glob(os.path.join(data_root, "rgb_*.png"))
            
            for rgb_file in rgb_files:
                # Extract ID from filename (e.g., rgb_0000001.png -> 0000001)
                filename = os.path.basename(rgb_file)
                sample_id = filename.replace('rgb_', '').replace('.png', '')
                
                # Check if all required files exist
                depth_file = os.path.join(data_root, f"depth_{sample_id}.exr")
                normal_file = os.path.join(data_root, f"normal_{sample_id}.exr")
                alpha_file = os.path.join(data_root, f"alpha_{sample_id}.png")
                
                if all(os.path.exists(f) for f in [rgb_file, depth_file, normal_file, alpha_file]):
                    self.samples.append((data_root, sample_id))
        
        # Sort for consistent ordering
        self.samples.sort()
        
        # Improved train/val split: distribute samples from each directory
        # This ensures both train and val sets have samples from all directories
        train_samples = []
        val_samples = []
        
        # Group samples by data root
        from collections import defaultdict
        samples_by_root = defaultdict(list)
        for sample in self.samples:
            samples_by_root[sample[0]].append(sample)
        
        # Split each directory's samples proportionally
        for data_root, root_samples in samples_by_root.items():
            split_idx = int(len(root_samples) * train_ratio)
            train_samples.extend(root_samples[:split_idx])
            val_samples.extend(root_samples[split_idx:])
        
        # Use the appropriate split
        if split == 'train':
            self.samples = train_samples
        else:
            self.samples = val_samples
        
        # Print statistics only once per unique dataset configuration  
        dataset_key = f"{str(sorted([os.path.basename(d) for d in self.data_roots]))}_{train_ratio}"
        if not hasattr(SynthHumanDataset, '_printed_configs'):
            SynthHumanDataset._printed_configs = set()
        
        if dataset_key not in SynthHumanDataset._printed_configs:
            SynthHumanDataset._printed_configs.add(dataset_key)
            
            # Calculate statistics for all directories
            train_counts = {}
            val_counts = {}
            total_counts = {}
            
            for data_root, root_samples in samples_by_root.items():
                dir_name = os.path.basename(data_root)
                split_idx = int(len(root_samples) * train_ratio)
                train_counts[dir_name] = split_idx
                val_counts[dir_name] = len(root_samples) - split_idx
                total_counts[dir_name] = len(root_samples)
            
            print(f"\n📊 Dataset Statistics:")
            print(f"{'Directory':<20} {'Total':<8} {'Train':<8} {'Val':<8}")
            print("-" * 50)
            
            total_all = sum(total_counts.values())
            total_train = sum(train_counts.values())
            total_val = sum(val_counts.values())
            
            for dir_name in sorted(total_counts.keys()):
                print(f"{dir_name:<20} {total_counts[dir_name]:<8} {train_counts[dir_name]:<8} {val_counts[dir_name]:<8}")
            
            print("-" * 50)
            print(f"{'TOTAL':<20} {total_all:<8} {total_train:<8} {total_val:<8}")
            print(f"📁 Directories: {len(self.data_roots)}")
            print(f"🎯 Split ratio: {train_ratio:.1%} train / {1-train_ratio:.1%} val\n")
        
        # Data augmentation transforms
        if augment and split == 'train':
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.transform = None
            
    def __len__(self):
        return len(self.samples)
    
    def load_exr_manual(self, filepath: str) -> np.ndarray:
        """Load EXR file using manual OpenEXR (working method)."""
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
                print(f"⚠️  Unsupported EXR channels in {filepath}: {channels}")
                exr_file.close()
                return None
            
            exr_file.close()
            return data
            
        except Exception as e:
            print(f"❌ Error loading EXR file {filepath}: {e}")
            return None
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data_root, sample_id = self.samples[idx]
        
        # File paths
        rgb_path = os.path.join(data_root, f"rgb_{sample_id}.png")
        depth_path = os.path.join(data_root, f"depth_{sample_id}.exr")
        normal_path = os.path.join(data_root, f"normal_{sample_id}.exr")
        alpha_path = os.path.join(data_root, f"alpha_{sample_id}.png")
        
        # Load RGB image
        rgb_image = Image.open(rgb_path).convert('RGB')
        original_size = rgb_image.size
        
        # Load ground truth data using manual OpenEXR (working method)
        depth_map = self.load_exr_manual(depth_path)  # Load depth EXR
        normal_map = self.load_exr_manual(normal_path)  # Load normal EXR  
        alpha_mask = np.array(Image.open(alpha_path).convert('L'))  # [H, W]
        
        # Handle loading failures gracefully
        if depth_map is None:
            print(f"⚠️  Using dummy depth for sample {sample_id}")
            depth_map = np.zeros((512, 512), dtype=np.float32)
        elif len(depth_map.shape) == 3:
            # Sometimes depth comes as multi-channel, take first channel
            depth_map = depth_map[:, :, 0]
            
        if normal_map is None:
            print(f"⚠️  Using dummy normals for sample {sample_id}")
            normal_map = np.zeros((512, 512, 3), dtype=np.float32)
        elif len(normal_map.shape) == 2:
            # If somehow normals are grayscale, expand to 3 channels
            normal_map = np.stack([normal_map] * 3, axis=2)
        
        # Resize everything to target size
        rgb_image = rgb_image.resize((self.image_size, self.image_size), Image.BILINEAR)
        depth_map = cv2.resize(depth_map, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        normal_map = cv2.resize(normal_map, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        alpha_mask = cv2.resize(alpha_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        
        # Apply data augmentation
        if self.transform is not None:
            # Determine if horizontal flip should be applied
            apply_flip = False
            for transform in self.transform.transforms:
                if hasattr(transform, 'p') and hasattr(transform, '__class__') and 'RandomHorizontalFlip' in str(transform.__class__):
                    apply_flip = random.random() < transform.p
                    break
            
            # Apply consistent flip to all data
            if apply_flip:
                # Flip RGB
                rgb_image = rgb_image.transpose(Image.FLIP_LEFT_RIGHT)
                
                # Flip all ground truth data
                depth_map = np.fliplr(depth_map).copy()
                normal_map = np.fliplr(normal_map).copy()
                # Flip X component of normals (important for surface normals)
                normal_map[:, :, 0] = -normal_map[:, :, 0]
                alpha_mask = np.fliplr(alpha_mask).copy()
            
            # Apply other augmentations (color jitter, etc.) to RGB only
            for transform in self.transform.transforms:
                if not ('RandomHorizontalFlip' in str(transform.__class__)):
                    rgb_image = transform(rgb_image)
        
        # Convert to tensors
        rgb_tensor = transforms.ToTensor()(rgb_image)  # [3, H, W], [0, 1]
        
        # Process depth: improved normalization for better training
        depth_valid = depth_map > 0
        if depth_valid.any():
            # Clip extreme background values (65504 = infinity in EXR)
            # Focus on meaningful depth range (person + nearby objects)
            depth_clipped = np.clip(depth_map, 0, 300.0)  # Cap at 5 meters
            
            # Normalize meaningful range to [0, 1]
            depth_min = 50.0   # 50cm minimum (closer than typical person)
            depth_max = 300.0  # 5m maximum (reasonable indoor/outdoor range)
            depth_normalized = (depth_clipped - depth_min) / (depth_max - depth_min)
            depth_normalized = np.clip(depth_normalized, 0, 1)
            
            # Keep background as 1.0 (far), set invalid pixels to 0
            depth_map = np.where(depth_valid, depth_normalized, 0)
        depth_tensor = torch.from_numpy(depth_map).float()  # [H, W]
        
        # Process normals: ensure they're in [-1, 1] range and normalized
        normal_map = normal_map.astype(np.float32)
        # Normalize normal vectors
        norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
        norm = np.where(norm > 0, norm, 1.0)  # Avoid division by zero
        normal_map = normal_map / norm
        normal_tensor = torch.from_numpy(normal_map).permute(2, 0, 1).float()  # [3, H, W]
        
        # Process alpha: normalize to [0, 1]
        alpha_tensor = torch.from_numpy(alpha_mask).float() / 255.0  # [H, W]
        alpha_tensor = alpha_tensor.unsqueeze(0)  # [1, H, W]
        
        return {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'normals': normal_tensor,
            'alpha': alpha_tensor,
            'sample_id': sample_id
        }


def create_dataloaders(data_roots, batch_size: int = 2, num_workers: int = 4, 
                      image_size: int = 512, train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_roots: Path to SynthHuman dataset OR list of paths
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        image_size: Target image size
        train_ratio: Ratio of data for training
        
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = SynthHumanDataset(
        data_roots=data_roots,
        split='train',
        image_size=image_size,
        train_ratio=train_ratio,
        augment=True
    )
    
    val_dataset = SynthHumanDataset(
        data_roots=data_roots,
        split='val',
        image_size=image_size,
        train_ratio=train_ratio,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def test_dataset(data_root: str):
    """Test the dataset loader."""
    print("Testing SynthHuman dataset loader...")
    
    # Create dataset
    dataset = SynthHumanDataset(data_root, split='train', image_size=512)
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        # Test loading a sample
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"RGB shape: {sample['rgb'].shape}, range: [{sample['rgb'].min():.3f}, {sample['rgb'].max():.3f}]")
        print(f"Depth shape: {sample['depth'].shape}, range: [{sample['depth'].min():.3f}, {sample['depth'].max():.3f}]")
        print(f"Normals shape: {sample['normals'].shape}, range: [{sample['normals'].min():.3f}, {sample['normals'].max():.3f}]")
        print(f"Alpha shape: {sample['alpha'].shape}, range: [{sample['alpha'].min():.3f}, {sample['alpha'].max():.3f}]")
        print(f"Sample ID: {sample['sample_id']}")
        
        # Test dataloader
        train_loader, val_loader = create_dataloaders(data_root, batch_size=2)
        batch = next(iter(train_loader))
        print(f"\nBatch shapes:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {len(value)} items")
        
        print("✅ Dataset loader test passed!")
    else:
        print("❌ No samples found in dataset!")


if __name__ == "__main__":
    # Test the dataset
    data_root = "F:/sy_human/SynthHuman_0000"
    test_dataset(data_root)
