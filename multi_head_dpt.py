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
        use_paper_architecture=True,  # New parameter to switch between architectures
    ):
        super(MultiHeadDPT, self).__init__()

        self.channels_last = channels_last
        self.features = features
        self.use_paper_architecture = use_paper_architecture

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

        if self.use_paper_architecture:
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
            
        else:
            # Original DPT feature fusion blocks (for backward compatibility)
            self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
            self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
            self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
            self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
            
            # Set resizer and decoder blocks to None to avoid unused parameters
            self.resizer = None
            self.decoder_block1 = None
            self.decoder_block2 = None
            self.decoder_block3 = None
            self.decoder_block4 = None

        # Multi-task output heads
        self.depth_head = self._make_depth_head(features)
        self.normal_head = self._make_normal_head(features)
        self.alpha_head = self._make_alpha_head(features)

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

    def forward(self, x):
        """
        Forward pass implementing paper's architecture with ViT encoder + Resizer + Paper decoder blocks.
        
        Args:
            x: Input image tensor [B, 3, H, W] - can be any resolution
            
        Returns:
            Dictionary with 'depth', 'normals', 'alpha_logits' keys
        """
        if self.channels_last == True:
            x = x.contiguous(memory_format=torch.channels_last)

        original_size = x.shape[2:]  # Store original input size
        
        if self.use_paper_architecture:
            # Paper's architecture implementation
            
            # Step 1: ViT Encoder - Fixed 384x384 input as per paper
            vit_input = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=True)
            layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, vit_input)

            # Process encoder features through readout projections
            layer_1_rn = self.scratch.layer1_rn(layer_1)
            layer_2_rn = self.scratch.layer2_rn(layer_2)
            layer_3_rn = self.scratch.layer3_rn(layer_3)
            layer_4_rn = self.scratch.layer4_rn(layer_4)
            
            # Step 2: Resizer - Process original resolution input
            r1, r2, r3, r4 = self.resizer(x)  # Multi-resolution features from original input
            
            # Step 3: Paper-specific decoder blocks fusion
            # Start from deepest level (r4, layer_4) and work upward
            
            # Decoder Block 4 (deepest): layer_4 + r4 -> d4
            d4 = self.decoder_block4(layer_4_rn, r4, prev_decoder_output=None)
            
            # Decoder Block 3: layer_3 + r3 + d4 -> d3  
            d3 = self.decoder_block3(layer_3_rn, r3, prev_decoder_output=d4)
            
            # Decoder Block 2: layer_2 + r2 + d3 -> d2
            d2 = self.decoder_block2(layer_2_rn, r2, prev_decoder_output=d3)
            
            # Decoder Block 1 (shallowest): layer_1 + r1 + d2 -> d1
            d1 = self.decoder_block1(layer_1_rn, r1, prev_decoder_output=d2)
            
            # Use final decoder output for predictions
            decoder_output = d1
            
        else:
            # Original DPT architecture (backward compatibility)
            layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

            layer_1_rn = self.scratch.layer1_rn(layer_1)
            layer_2_rn = self.scratch.layer2_rn(layer_2)
            layer_3_rn = self.scratch.layer3_rn(layer_3)
            layer_4_rn = self.scratch.layer4_rn(layer_4)

            # DPT feature fusion
            path_4 = self.scratch.refinenet4(layer_4_rn)
            path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
            path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
            path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
            
            decoder_output = path_1

        # Multi-task predictions from decoder output
        depth = self.depth_head(decoder_output)
        normals = self.normal_head(decoder_output)
        alpha_logits = self.alpha_head(decoder_output)

        # Ensure outputs match original input size (variable resolution support)
        depth = F.interpolate(depth, size=original_size, mode='bilinear', align_corners=True)
        normals = F.interpolate(normals, size=original_size, mode='bilinear', align_corners=True)
        alpha_logits = F.interpolate(alpha_logits, size=original_size, mode='bilinear', align_corners=True)

        # Remove channel dimension for depth and alpha
        depth = depth.squeeze(1)
        alpha_logits = alpha_logits.squeeze(1)

        return {
            'depth': depth,
            'normals': normals,
            'alpha_logits': alpha_logits,
        }


def create_multi_head_dpt(
    backbone="vitb16_384",
    features=256,
    use_bn=False,
    pretrained=True
):
    """
    Create a multi-head DPT model based on original DPT architecture.
    
    Args:
        backbone: ViT backbone architecture
        features: Number of features in fusion blocks
        use_bn: Whether to use batch normalization
        pretrained: Whether to use pretrained ViT weights
        
    Returns:
        MultiHeadDPT model
    """
    return MultiHeadDPT(
        backbone=backbone,
        features=features,
        use_bn=use_bn,
        enable_attention_hooks=False,
    )


# Test function
if __name__ == "__main__":
    model = create_multi_head_dpt()
    x = torch.randn(1, 3, 384, 384)
    
    with torch.no_grad():
        outputs = model(x)
    
    print("Multi-Head DPT model created successfully!")
    print(f"Input shape: {x.shape}")
    print(f"Depth output shape: {outputs['depth'].shape}")
    print(f"Depth range: [{outputs['depth'].min():.3f}, {outputs['depth'].max():.3f}]")
    print(f"Normals output shape: {outputs['normals'].shape}")
    print(f"Alpha logits output shape: {outputs['alpha_logits'].shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
