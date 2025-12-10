"""
MetaHuman Dataset Loader 2 for Dual-Task Training
Handles Color (primary), Face_Mask, and Geo only
Supports variants: gemini, flux, seedream
"""

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from typing import Tuple, Dict, List, Optional
import torchvision.transforms as transforms
import OpenEXR
import Imath

# Enable OpenEXR support in OpenCV
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def get_positional_encoding(x, n_freqs):
    """
    Apply periodic positional encoding to x.
    x: tensor of shape [C, H, W]
    n_freqs: number of frequencies. Frequencies are 2*k for k=1..n_freqs.
    Returns: tensor of shape [C * 2 * n_freqs, H, W]
    """
    # Linear frequencies: 2, 4, 6, 8...
    freq_bands = (torch.arange(1, n_freqs + 1, dtype=torch.float32, device=x.device) * 2.0)
    
    # x: [C, H, W] -> [C, 1, H, W]
    x_expanded = x.unsqueeze(1) 
    # freqs: [F] -> [1, F, 1, 1]
    freqs = freq_bands.view(1, -1, 1, 1)
    
    # Calculate x * freq * PI
    args = x_expanded * freqs * torch.pi
    
    # Sin/Cos
    pe_sin = torch.sin(args) # [C, F, H, W]
    pe_cos = torch.cos(args) # [C, F, H, W]
    
    # Stack: [C, 2*F, H, W]
    # cat along dim 1 (frequency dimension expanded)
    pe = torch.cat([pe_sin, pe_cos], dim=1)
    
    # Flatten channels: [C*2*F, H, W]
    # Merge dimensions 0 and 1
    pe = pe.view(-1, x.shape[1], x.shape[2])
    
    return pe


class MetahumanDataset2(Dataset):
    """
    MetaHuman dataset for dual-task learning (Mask + Geo).
    Loads Color (primary RGB input), Face_Mask, and Geo.
    Supports file variants with suffixes (gemini, flux, seedream).
    """
    
    def __init__(self, data_roots, split: str = 'train', image_size: int = 512, 
                 train_ratio: float = 0.8, augment: bool = True, geo_encoding_n: int = 4):
        """
        Args:
            data_roots: Path to MetaHuman dataset directory OR list of directories
            split: 'train' or 'val'  
            image_size: Target image size for training
            train_ratio: Ratio of data to use for training
            augment: Whether to apply data augmentation
            geo_encoding_n: Number of frequencies for positional encoding.
        """
        # Support both single directory and list of directories
        if isinstance(data_roots, str):
            self.data_roots = [data_roots]
        else:
            self.data_roots = data_roots
            
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.geo_encoding_n = geo_encoding_n
        
        # Find all samples across all directories
        self.samples = []  # List of (data_root, full_id, base_id, suffix) tuples
        
        for data_root in self.data_roots:
            if not os.path.exists(data_root):
                print(f"⚠️  Warning: Data directory {data_root} does not exist, skipping...")
                continue
                
            # Look for Color files as primary indicator of samples
            # Changed to Color_* to match user's data structure
            color_files = glob.glob(os.path.join(data_root, "Color_*"))
            # Filter out BaseColor or others if necessary (glob Color_* should avoid BaseColor)
            color_files = [f for f in color_files if os.path.basename(f).startswith("Color_")]
            
            # Debug: Print what files are found in each directory
            dir_name = os.path.basename(data_root)
            print(f"🔍 Checking directory '{dir_name}': found {len(color_files)} Color files")
            if len(color_files) == 0:
                print(f"   Path checked: {os.path.join(data_root, 'Color_*')}")
                try:
                    if os.path.exists(data_root):
                         print(f"   Directory contents (first 5): {os.listdir(data_root)[:5]}")
                except Exception as e:
                    print(f"   Error listing directory: {e}")

            for color_file in color_files:
                # Extract sample ID from filename
                filename = os.path.basename(color_file)
                # Filename is like Color_0_0.png or Color_0_0_gemini.png
                # Remove "Color_" prefix
                if filename.startswith("Color_"):
                    temp_name = filename[6:] # len("Color_") == 6
                else:
                    continue
                
                # Remove extension
                full_id = os.path.splitext(temp_name)[0]
                
                # Parse variants
                base_id = full_id
                suffix = None
                
                for s in ['_gemini', '_flux', '_seedream']:
                    if full_id.endswith(s):
                        base_id = full_id[:-len(s)]
                        suffix = s[1:] # 'gemini', 'flux', 'seedream'
                        break
                
                if full_id:
                    self.samples.append((data_root, full_id, base_id, suffix))
        
        # Remove duplicates and sort for consistent ordering
        self.samples = list(set(self.samples))
        self.samples.sort()
        
        # Improved train/val split: distribute samples from each directory
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
        
        # Enhanced data augmentation using TorchVision and custom augmentations
        if augment and split == 'train':
            self.augment_transforms = transforms.Compose([
                # Enhanced color jitter (includes brightness augmentation)
                transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.15, hue=0.05),
                # Gaussian blur using torchvision
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
            ])
            self.apply_augment = True
        else:
            self.augment_transforms = None
            self.apply_augment = False
            
    def __len__(self):
        return len(self.samples)
    
    def find_file_with_pattern(self, data_root: str, sample_id: str, pattern: str) -> Optional[str]:
        """Find file matching pattern for given sample ID."""
        extensions = ['.png', '.exr', '.jpg', '.jpeg']
        
        for ext in extensions:
            filepath = os.path.join(data_root, f"{pattern}_{sample_id}{ext}")
            if os.path.exists(filepath):
                return filepath
        
        return None
    
    def load_exr(self, exr_path: str) -> np.ndarray:
        """Load EXR file using OpenEXR."""
        try:
            exr_file = OpenEXR.InputFile(exr_path)
            header = exr_file.header()
            
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1
            
            channels_data = {}
            for channel in ['R', 'G', 'B']:
                if channel in header['channels']:
                    channel_type = header['channels'][channel].type
                    if channel_type == Imath.PixelType(Imath.PixelType.FLOAT):
                        pixel_data = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
                        channel_array = np.frombuffer(pixel_data, dtype=np.float32).reshape((height, width))
                    else:
                        pixel_data = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.HALF))
                        channel_array = np.frombuffer(pixel_data, dtype=np.float16).reshape((height, width))
                    channels_data[channel] = channel_array.astype(np.float32)
            
            if channels_data:
                # Stack channels
                if 'R' in channels_data and 'G' in channels_data and 'B' in channels_data:
                    img_array = np.stack([channels_data['R'], channels_data['G'], channels_data['B']], axis=2)
                elif 'R' in channels_data: # Single channel?
                    img_array = channels_data['R']
                    img_array = np.stack([img_array, img_array, img_array], axis=2) # Convert to 3ch
                else:
                    img_array = np.zeros((height, width, 3), dtype=np.float32)
                
                # Sanitize EXR data (remove NaNs/Infs)
                img_array = np.nan_to_num(img_array, nan=0.0, posinf=1.0, neginf=0.0)
                return img_array
            else:
                return np.zeros((height, width, 3), dtype=np.float32)
                
        except Exception as e:
            print(f"Error loading EXR {exr_path}: {e}")
            return np.zeros((512, 512, 3), dtype=np.float32)

    def load_image_safe(self, filepath: str, mode: str = 'RGB') -> np.ndarray:
        """Load image file."""
        if filepath is None or not os.path.exists(filepath):
            if mode == 'RGB':
                return np.zeros((512, 512, 3), dtype=np.uint8)
            else:
                return np.zeros((512, 512), dtype=np.uint8)
        
        # Check if EXR
        if filepath.lower().endswith('.exr'):
            return self.load_exr(filepath)
            
        try:
            image = Image.open(filepath).convert(mode)
            return np.array(image)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            if mode == 'RGB':
                return np.zeros((512, 512, 3), dtype=np.uint8)
            else:
                return np.zeros((512, 512), dtype=np.uint8)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data_root, full_id, base_id, suffix = self.samples[idx]
        
        # Load files
        color_path = self.find_file_with_pattern(data_root, full_id, "Color")
        if not color_path:
             color_path = self.find_file_with_pattern(data_root, full_id, "Color_Clothed")
             
        face_mask_path = self.find_file_with_pattern(data_root, base_id, "Face_Mask")
        geo_path = self.find_file_with_pattern(data_root, base_id, "Geo")
        
        # Load images
        rgb_array = self.load_image_safe(color_path, 'RGB')
        geo_array = self.load_image_safe(geo_path, 'RGB')
        face_mask = self.load_image_safe(face_mask_path, 'L')
        
        # Convert RGB to PIL for easy resizing/augmentation (if it's uint8)
        # If Color is EXR (float), we might need different handling, but usually Color is png/uint8
        if rgb_array.dtype == np.uint8:
            rgb_image = Image.fromarray(rgb_array)
        else:
            # If float, clip to 0-1 or 0-255? 
            # Assuming standard RGB is uint8. If it were float, we'd need to convert.
            # But let's assume it's uint8 PNG as per user listing.
            rgb_image = Image.fromarray((np.clip(rgb_array, 0, 1) * 255).astype(np.uint8))

        # Geo is likely float (EXR). Keep as numpy.
        
        # Load Hair_Mask
        hair_mask = None
        if suffix in ['flux', 'seedream']:
            hair_mask_path = self.find_file_with_pattern(data_root, base_id, "Hair_Mask")
            if hair_mask_path:
                hair_mask = self.load_image_safe(hair_mask_path, 'L')
        
        # Mixed augmentation strategy
        use_direct_resize = False
        if self.augment and self.split == 'train':
            use_direct_resize = random.random() < 0.3
        
        if use_direct_resize:
            # Path 1: Direct resize to 384x384
            target_size = 384
            
            rgb_image = rgb_image.resize((target_size, target_size), Image.BILINEAR)
            geo_array = cv2.resize(geo_array, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            face_mask = cv2.resize(face_mask, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            if hair_mask is not None:
                hair_mask = cv2.resize(hair_mask, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            
        else:
            # Path 2: Resize to 512x512, then crop to 384x384
            intermediate_size = 512
            crop_size = 384
            max_shift = 20
            
            # Resize
            rgb_image = rgb_image.resize((intermediate_size, intermediate_size), Image.BILINEAR)
            geo_array = cv2.resize(geo_array, (intermediate_size, intermediate_size), interpolation=cv2.INTER_LINEAR)
            face_mask = cv2.resize(face_mask, (intermediate_size, intermediate_size), interpolation=cv2.INTER_LINEAR)
            if hair_mask is not None:
                hair_mask = cv2.resize(hair_mask, (intermediate_size, intermediate_size), interpolation=cv2.INTER_LINEAR)
            
            # Crop logic
            if self.augment and self.split == 'train':
                max_left_shift = min(max_shift, (intermediate_size - crop_size) // 2)
                shift_x = random.randint(-max_left_shift, max_left_shift)
                max_top_shift = min(max_shift, (intermediate_size - crop_size) // 2)
                shift_y = random.randint(-max_top_shift, max_top_shift)
            else:
                shift_x = 0
                shift_y = 0
            
            center_x = intermediate_size // 2
            center_y = intermediate_size // 2
            left = center_x - crop_size // 2 + shift_x
            top = center_y - crop_size // 2 + shift_y
            right = left + crop_size
            bottom = top + crop_size
            
            left = max(0, min(left, intermediate_size - crop_size))
            top = max(0, min(top, intermediate_size - crop_size))
            right = left + crop_size
            bottom = top + crop_size
            
            rgb_image = rgb_image.crop((left, top, right, bottom))
            geo_array = geo_array[top:bottom, left:right]
            face_mask = face_mask[top:bottom, left:right]
            if hair_mask is not None:
                hair_mask = hair_mask[top:bottom, left:right]
        
        # Apply augmentations
        if self.apply_augment and self.augment_transforms is not None:
            rgb_image = self.augment_transforms(rgb_image)
            
            if random.random() < 0.6:
                rgb_np = np.array(rgb_image)
                noise_level = random.randint(5, 15)
                noise = np.random.normal(-noise_level, noise_level, rgb_np.shape).astype(np.int16)
                rgb_noisy = np.clip(rgb_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                rgb_image = Image.fromarray(rgb_noisy)
            
            if random.random() < 0.2:
                rgb_np = np.array(rgb_image)
                blurred = cv2.GaussianBlur(rgb_np, (0, 0), 1.0)
                sharpened = cv2.addWeighted(rgb_np, 1.5, blurred, -0.5, 0)
                rgb_image = Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8))
        
        # Convert to tensors
        rgb_tensor = transforms.ToTensor()(rgb_image)  # [3, H, W]
        # Geo is float array (H, W, 3). ToTensor will permute to (3, H, W).
        # It won't scale if dtype is float.
        geo_tensor = transforms.ToTensor()(geo_array)
        
        # Sanitize and Clamp to [0, 1] to ensure stability
        # Replace NaNs with 0, and clamp to valid range before normalization
        if torch.isnan(geo_tensor).any() or torch.isinf(geo_tensor).any():
             # print(f"⚠️ Warning: Found NaN/Inf in Geo tensor for {full_id}, sanitizing...")
             geo_tensor = torch.nan_to_num(geo_tensor, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Force range to [0, 1] to prevent loss explosion
        geo_tensor = torch.clamp(geo_tensor, 0.0, 1.0)
        
        # Normalize Base Geo from [0, 1] to [-1, 1] to match Tanh output range
        geo_tensor = (geo_tensor * 2.0) - 1.0
        
        # Apply positional encoding to Geo tensor
        # geo_tensor is [3, H, W]
        if self.geo_encoding_n > 0:
            # geo_encoding_n is the number of frequencies
            n_freqs = self.geo_encoding_n
            geo_encoded = get_positional_encoding(geo_tensor, n_freqs)
            # Result: [3*N, H, W]
            
            # Cat original and encoded: [3 + 3*2*n, H, W]
            geo_tensor = torch.cat([geo_tensor, geo_encoded], dim=0)
        
        # Process face mask
        face_mask_tensor = torch.from_numpy(face_mask).float() / 255.0  # [H, W]
        face_mask_tensor = face_mask_tensor.unsqueeze(0)  # [1, H, W]
        
        if hair_mask is not None:
            hair_mask_tensor = torch.from_numpy(hair_mask).float() / 255.0
            hair_mask_tensor = hair_mask_tensor.unsqueeze(0)
            
            return {
                'rgb': rgb_tensor,
                'geo': geo_tensor,
                'face_mask': face_mask_tensor,
                'hair_mask': hair_mask_tensor,
                'sample_id': full_id
            }
        
        return {
            'rgb': rgb_tensor,
            'geo': geo_tensor,
            'face_mask': face_mask_tensor,
            'sample_id': full_id
        }


def custom_collate_fn2(batch):
    """Custom collate function for MetahumanDataset2."""
    collated = {}
    
    for key in ['rgb', 'face_mask', 'geo']:
        if key in batch[0]:
            collated[key] = torch.stack([item[key] for item in batch])
    
    # Check if ANY item has hair_mask
    if any('hair_mask' in item for item in batch):
        hair_masks = []
        for item in batch:
            if 'hair_mask' in item:
                hair_masks.append(item['hair_mask'])
            else:
                # Fill missing with zeros
                hair_masks.append(torch.zeros_like(item['face_mask']))
        collated['hair_mask'] = torch.stack(hair_masks)
    
    collated['sample_id'] = [item['sample_id'] for item in batch]
    
    return collated


def create_metahuman_dataloaders2(data_roots, batch_size: int = 2, num_workers: int = 4, 
                                  image_size: int = 512, train_ratio: float = 0.8, geo_encoding_n: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for MetaHuman dataset 2 (Mask + Geo).
    """
    train_dataset = MetahumanDataset2(
        data_roots=data_roots,
        split='train',
        image_size=image_size,
        train_ratio=train_ratio,
        augment=True,
        geo_encoding_n=geo_encoding_n
    )
    
    val_dataset = MetahumanDataset2(
        data_roots=data_roots,
        split='val',
        image_size=image_size,
        train_ratio=train_ratio,
        augment=False,
        geo_encoding_n=geo_encoding_n
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn2
    )
    
    return train_loader, val_loader
