"""
Multi-Head DPT Model
Based on the original DPT architecture with multi-task output heads
for depth, surface normals, and alpha prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import types
import math

# Import original DPT components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'DPT'))

from DPT.dpt.base_model import BaseModel
from DPT.dpt.blocks import (
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class ResizerBlock(nn.Module):
    """
    Lightweight convolutional resizer block that halves resolution.
    Based on paper: rl = gl(rl−1) at layer l computes features at half resolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(ResizerBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class ResizerModule(nn.Module):
    """
    Full resizer module with 4 stacked resizer blocks.
    Processes original image size and generates multi-resolution features.
    """
    def __init__(self, input_channels=3, features=256):
        super(ResizerModule, self).__init__()
        
        # 4 resizer blocks as mentioned in paper
        self.resizer1 = ResizerBlock(input_channels, features//4)    # 1/2 resolution
        self.resizer2 = ResizerBlock(features//4, features//2)       # 1/4 resolution  
        self.resizer3 = ResizerBlock(features//2, features)          # 1/8 resolution
        self.resizer4 = ResizerBlock(features, features)             # 1/16 resolution
        
    def forward(self, x):
        """
        Forward pass through resizer blocks.
        Returns features at 4 different resolutions for decoder blocks.
        """
        r1 = self.resizer1(x)  # 1/2 resolution
        r2 = self.resizer2(r1)  # 1/4 resolution
        r3 = self.resizer3(r2)  # 1/8 resolution  
        r4 = self.resizer4(r3)  # 1/16 resolution
        
        return r1, r2, r3, r4


class ResidualConvUnit(nn.Module):
    """
    Residual Convolutional Unit (RConv) as specified in the paper.
    Used in decoder blocks for feature processing.
    """
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super(ResidualConvUnit, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual  # Residual connection
        out = self.relu(out)
        
        return out


class PaperDecoderBlock(nn.Module):
    """
    Paper-specific decoder block implementing the exact fusion pattern:
    dl_int = RConv(dl−1 + Interp(RConv(el)))
    dl = Conv([rl, Interp(dl_int)])
    
    Takes 3 inputs: encoder feature (el), resizer feature (rl), previous decoder output (dl-1)
    """
    def __init__(self, encoder_channels, resizer_channels, output_channels):
        super(PaperDecoderBlock, self).__init__()
        
        # Residual conv units for processing encoder features
        self.encoder_rconv = ResidualConvUnit(encoder_channels)
        
        # Residual conv unit for processing intermediate features
        self.intermediate_rconv = ResidualConvUnit(output_channels)
        
        # Standard conv unit for final processing after concatenation
        self.final_conv = nn.Sequential(
            nn.Conv2d(resizer_channels + output_channels, output_channels, 
                     kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 
                     kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
        # Projection layer to match encoder features to output channels
        self.encoder_proj = None
        if encoder_channels != output_channels:
            self.encoder_proj = nn.Conv2d(encoder_channels, output_channels, 
                                        kernel_size=1, stride=1, padding=0, bias=False)
    
    def forward(self, encoder_feature, resizer_feature, prev_decoder_output=None):
        """
        Forward pass following paper's decoder block pattern.
        
        Args:
            encoder_feature (el): Feature from ViT encoder
            resizer_feature (rl): Feature from resizer module  
            prev_decoder_output (dl-1): Previous decoder block output (None for first block)
        
        Returns:
            dl: Decoder block output
        """
        # Step 1: Process encoder feature with RConv
        processed_encoder = self.encoder_rconv(encoder_feature)
        
        # Project encoder feature to match output channels if needed
        if self.encoder_proj is not None:
            processed_encoder = self.encoder_proj(processed_encoder)
        
        # Step 2: Interpolate processed encoder to match resizer resolution
        target_size = (resizer_feature.size(2), resizer_feature.size(3))
        upsampled_encoder = F.interpolate(processed_encoder, size=target_size, 
                                        mode='bilinear', align_corners=True)
        
        # Step 3: Add with previous decoder output (if available)
        if prev_decoder_output is not None:
            # Interpolate previous decoder output to match current resolution
            prev_upsampled = F.interpolate(prev_decoder_output, size=target_size,
                                         mode='bilinear', align_corners=True)
            dl_int_input = prev_upsampled + upsampled_encoder
        else:
            dl_int_input = upsampled_encoder
        
        # Step 4: Process intermediate result with RConv
        dl_int = self.intermediate_rconv(dl_int_input)
        
        # Step 5: Concatenate with resizer features
        concat_features = torch.cat([resizer_feature, dl_int], dim=1)
        
        # Step 6: Final conv processing
        dl = self.final_conv(concat_features)
        
        return dl


class MultiHeadDPT(BaseModel):
    def __init__(
        self,
        features=256,
        backbone="vitb16_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):
        super(MultiHeadDPT, self).__init__()

        self.channels_last = channels_last
        self.features = features

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks using original DPT encoder
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            False,  # Set to true if you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        # Paper's architecture: Resizer + Paper-specific decoder blocks
        self.resizer = ResizerModule(input_channels=3, features=features)
        
        # Create paper-specific decoder blocks
        # Decoder block channel configurations based on resizer outputs
        resizer_channels = [features, features, features//2, features//4]  # r4, r3, r2, r1
        
        self.decoder_block4 = PaperDecoderBlock(features, resizer_channels[0], features)  # Deepest
        self.decoder_block3 = PaperDecoderBlock(features, resizer_channels[1], features)
        self.decoder_block2 = PaperDecoderBlock(features, resizer_channels[2], features)  
        self.decoder_block1 = PaperDecoderBlock(features, resizer_channels[3], features)  # Shallowest
        
        # Set refinenet blocks to None to avoid unused parameters
        self.scratch.refinenet1 = None
        self.scratch.refinenet2 = None
        self.scratch.refinenet3 = None
        self.scratch.refinenet4 = None

        # Multi-task output heads
        self.depth_head = self._make_depth_head(features)
        self.depth_head2 = self._make_depth_head(features)
        self.normal_head = self._make_normal_head(features)
        self.normal_head2 = self._make_normal_head(features)
        self.alpha_head = self._make_alpha_head(features)
        self.alpha_head2 = self._make_alpha_head(features)
        self.basecolor_head2 = self._make_basecolor_head(features)
        self.geo_head2 = self._make_geo_head(features)
        
    def _make_depth_head(self, features):
        """Create depth prediction head following original DPT style."""
        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),  # Non-negative depth values
        )
        return head

    def _make_normal_head(self, features):
        """Create surface normal prediction head."""
        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),  # Normals in [-1, 1] range
        )
        return head

    def _make_basecolor_head(self, features):
        """Create basecolor prediction head."""
        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),  # Basecolor in [0, 1] range
        )
        return head

    def _make_geo_head(self, features):
        """Create geo image prediction head."""
        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),  # Assuming geo image is also normalized to [0, 1]
        )
        return head

    def _make_alpha_head(self, features):
        """Create alpha/foreground prediction head."""
        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            # No activation - raw logits for BCE loss
        )
        return head

    def forward_features(self, x):
        """
        Common feature extraction steps (Encoder + Resizer + Decoder).
        """
        if self.channels_last == True:
            x = x.contiguous(memory_format=torch.channels_last)

        original_size = x.shape[2:]  # Store original input size, e.g., (H, W)
        
        # --- Encoder Stages ---
        
        # Step 1: ViT Encoder
        # Input must be cropped to the ViT's expected size (384x384)
        # x shape: [B, 3, H, W] - expected input is 384x512 or 384x384
        
        if x.shape[2:] == (384, 384):
            # Input is already 384x384, use directly
            vit_input = x
        else:
            # Input is 384x512, need to crop to 384x384
            crop_size = 384
            _, _, height, width = x.shape
            
            if width == 384 and height == 512:
                # Standard case: crop from 384x512 to 384x384
                if self.training:
                    # Random vertical shift during training (-10 to +10 pixels)
                    max_shift = 10
                    max_top_shift = min(max_shift, (height - crop_size) // 2)
                    shift_y = torch.randint(-max_top_shift, max_top_shift + 1, (1,)).item()
                else:
                    # Center crop during validation
                    shift_y = 0
                
                # Calculate crop coordinates
                center_y = height // 2
                top = center_y - crop_size // 2 + shift_y
                bottom = top + crop_size
                
                # Ensure crop stays within bounds
                top = max(0, min(top, height - crop_size))
                bottom = top + crop_size
                
                vit_input = x[:, :, top:bottom, :]
            else:
                # Fallback: resize if dimensions don't match expected format
                vit_input = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=True)
        
        # vit_input shape: [B, 3, 384, 384]
        
        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, vit_input)
        # For ViT-B/16, features are at 1/16th resolution of the 384x384 input
        # layer_1, layer_2, layer_3, layer_4 shapes: [B, 768, 24, 24]

        # Process encoder features through readout projections to reduce channel dimension
        layer_1_rn = self.scratch.layer1_rn(layer_1) # Shape: [B, 256, 24, 24]
        layer_2_rn = self.scratch.layer2_rn(layer_2) # Shape: [B, 256, 24, 24]
        layer_3_rn = self.scratch.layer3_rn(layer_3) # Shape: [B, 256, 24, 24]
        layer_4_rn = self.scratch.layer4_rn(layer_4) # Shape: [B, 256, 24, 24]
        
        # Step 2: Resizer Module
        # Processes the original image at its native resolution [B, 3, H, W]
        r1, r2, r3, r4 = self.resizer(x)
        # r1 shape: [B, 64, H/2, W/2], r2: [B, 128, H/4, W/4], r3: [B, 256, H/8, W/8], r4: [B, 256, H/16, W/16]
        
        # --- Decoder Stages ---
        
        # Step 3: Paper-specific decoder blocks fusion
        # The decoder progressively upsamples features, fusing ViT and Resizer outputs.
        
        # Decoder Block 4 (deepest): Fuses layer_4_rn and r4
        d4 = self.decoder_block4(layer_4_rn, r4, prev_decoder_output=None) # Shape: [B, 256, H/16, W/16]
        
        # Decoder Block 3: Fuses layer_3_rn, r3, and upsampled d4
        d3 = self.decoder_block3(layer_3_rn, r3, prev_decoder_output=d4) # Shape: [B, 256, H/8, W/8]
        
        # Decoder Block 2: Fuses layer_2_rn, r2, and upsampled d3
        d2 = self.decoder_block2(layer_2_rn, r2, prev_decoder_output=d3) # Shape: [B, 256, H/4, W/4]
        
        # Decoder Block 1 (shallowest): Fuses layer_1_rn, r1, and upsampled d2
        d1 = self.decoder_block1(layer_1_rn, r1, prev_decoder_output=d2) # Shape: [B, 256, H/2, W/2]
        
        # Final decoder output
        decoder_output = d1 # Shape: [B, 256, H/2, W/2]

        return decoder_output, original_size

    def forward(self, x, heads_to_compute="all"):
        """
        Forward pass implementing paper's architecture with ViT encoder + Resizer + Paper decoder blocks.
        
        Args:
            x: Input image tensor [B, 3, H, W] - can be any resolution
            heads_to_compute: Which heads to compute - "all", "head1", or "head2"
            
        Returns:
            Dictionary with requested head outputs
        """
        decoder_output, original_size = self.forward_features(x)

        # --- Prediction Heads ---
        
        results = {}
        
        # Compute head1 outputs (David dataset heads)
        if heads_to_compute in ["all", "head1"]:
            depth = self.depth_head(decoder_output)
            normals = self.normal_head(decoder_output)
            alpha_logits = self.alpha_head(decoder_output)
            
            # Ensure outputs match original input size
            depth = F.interpolate(depth, size=original_size, mode='bilinear', align_corners=True)
            normals = F.interpolate(normals, size=original_size, mode='bilinear', align_corners=True)
            alpha_logits = F.interpolate(alpha_logits, size=original_size, mode='bilinear', align_corners=True)
            
            # Remove channel dimension for depth and alpha
            depth = depth.squeeze(1)
            alpha_logits = alpha_logits.squeeze(1)
            
            results.update({
                'depth': depth,
                'normals': normals,
                'alpha_logits': alpha_logits,
            })
        
        # Compute head2 outputs (MetaHuman dataset heads)
        if heads_to_compute in ["all", "head2"]:
            depth2 = self.depth_head2(decoder_output)
            normals2 = self.normal_head2(decoder_output)
            alpha_logits2 = self.alpha_head2(decoder_output)
            basecolor2 = self.basecolor_head2(decoder_output)
            geo2 = self.geo_head2(decoder_output)
            
            # Ensure outputs match original input size
            depth2 = F.interpolate(depth2, size=original_size, mode='bilinear', align_corners=True)
            normals2 = F.interpolate(normals2, size=original_size, mode='bilinear', align_corners=True)
            alpha_logits2 = F.interpolate(alpha_logits2, size=original_size, mode='bilinear', align_corners=True)
            basecolor2 = F.interpolate(basecolor2, size=original_size, mode='bilinear', align_corners=True)
            geo2 = F.interpolate(geo2, size=original_size, mode='bilinear', align_corners=True)
            
            # Remove channel dimension for depth and alpha
            depth2 = depth2.squeeze(1)
            alpha_logits2 = alpha_logits2.squeeze(1)
            
            results.update({
                'depth2': depth2,
                'normals2': normals2,
                'alpha_logits2': alpha_logits2,
                'basecolor2': basecolor2,
                'geo2': geo2,
            })

        return results


class DualHeadDPT(MultiHeadDPT):
    """
    Dual-Head DPT model for MetaHuman dataset (Face Mask + Geo only).
    Supports expanded Geo output channels for positional encoding.
    """
    def __init__(
        self,
        features=256,
        backbone="vitb16_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
        geo_output_channels=3,
    ):
        self.geo_output_channels = geo_output_channels
        super(DualHeadDPT, self).__init__(
            features=features,
            backbone=backbone,
            readout=readout,
            channels_last=channels_last,
            use_bn=use_bn,
            enable_attention_hooks=enable_attention_hooks,
        )
        
        # Remove unused heads to save parameters
        del self.depth_head
        del self.depth_head2
        del self.normal_head
        del self.normal_head2
        del self.alpha_head
        del self.basecolor_head2
        
        # Keeping alpha_head2 and geo_head2
        
    def _make_geo_head(self, features):
        """Create geo image prediction head with dynamic output channels."""
        # Use self.geo_output_channels if available (set in __init__), else 3
        out_channels = getattr(self, 'geo_output_channels', 3)
        
        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),  # Bounded output [-1, 1] to match normalized target
        )
        return head
        
    def forward(self, x):
        decoder_output, original_size = self.forward_features(x)
        
        # Compute only mask and geo
        alpha_logits = self.alpha_head2(decoder_output)
        geo = self.geo_head2(decoder_output)
        
        # Interpolate
        alpha_logits = F.interpolate(alpha_logits, size=original_size, mode='bilinear', align_corners=True)
        geo = F.interpolate(geo, size=original_size, mode='bilinear', align_corners=True)
        
        alpha_logits = alpha_logits.squeeze(1)
        
        return {
            'alpha_logits': alpha_logits,
            'geo': geo
        }


def create_multi_head_dpt(
    backbone="vitb16_384",
    features=256,
    use_bn=False,
    pretrained=True
):
    """
    Create a multi-head DPT model based on original DPT architecture.
    """
    return MultiHeadDPT(
        backbone=backbone,
        features=features,
        use_bn=use_bn,
        enable_attention_hooks=False,
    )


def create_dual_head_dpt(
    backbone="vitb16_384",
    features=256,
    use_bn=False,
    pretrained=True,
    geo_output_channels=3,
):
    """
    Create a dual-head DPT model (Mask + Geo).
    """
    return DualHeadDPT(
        backbone=backbone,
        features=features,
        use_bn=use_bn,
        enable_attention_hooks=False,
        geo_output_channels=geo_output_channels,
    )


# Test function
if __name__ == "__main__":
    print("--- Testing MultiHeadDPT ---")
    model = create_multi_head_dpt()
    x = torch.randn(1, 3, 384, 512)
    
    with torch.no_grad():
        outputs = model(x)
    
    print("Multi-Head DPT model created successfully!")
    print(f"Input shape: {x.shape}")
    print(f"Depth output shape: {outputs['depth'].shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n--- Testing DualHeadDPT (Default 3 channels) ---")
    dual_model = create_dual_head_dpt()
    with torch.no_grad():
        dual_outputs = dual_model(x)
    print(f"Geo shape: {dual_outputs['geo'].shape}")
    
    print("\n--- Testing DualHeadDPT (60 channels) ---")
    dual_model_60 = create_dual_head_dpt(geo_output_channels=60)
    with torch.no_grad():
        dual_outputs_60 = dual_model_60(x)
    print(f"Geo shape: {dual_outputs_60['geo'].shape}")
    print(f"Total parameters: {sum(p.numel() for p in dual_model_60.parameters()):,}")
