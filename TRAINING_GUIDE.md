# Training Guide for Multi-task DPT Model

This guide explains how to train the Multi-task Dense Prediction Transformer (DPT) model for depth estimation, surface normal prediction, and alpha/foreground segmentation.

## Files Overview

- `fixed_dpt_multitask.py` - The main model architecture
- `train.py` - Training script
- `david_dataset.py` - Dataset loader for SynthHuman data

## Prerequisites

### Required Packages
```bash
pip install torch torchvision timm tqdm matplotlib numpy opencv-python
```

### Data Setup
1. Download the SynthHuman dataset
2. Update the `data_root` path in `train.py` (line 134):
   ```python
   data_root = "F:/sy_human/SynthHuman_0000"  # Update this path
   ```

## Training the Model

### Basic Training
Simply run the training script:
```bash
python train.py
```

### Configuration Options

You can modify these parameters in `train.py`:

```python
# Training configuration
data_root = "F:/sy_human/SynthHuman_0000"  # Path to dataset
batch_size = 6                             # Adjust based on GPU memory
num_epochs = 20                            # Number of training epochs
lr = 8e-4                                  # Learning rate
```

### Model Configuration

The model can be configured in the `create_multitask_dpt_model()` call:

```python
model = create_multitask_dpt_model(
    backbone="vit_base_patch16_384",  # ViT backbone
    features=256,                     # Feature dimension
    use_bn=False,                     # Batch normalization
    pretrained=True,                  # Use pretrained ViT weights
    depth_scale=5.0                   # Depth scaling factor
)
```

## Training Features

### Advanced Training Techniques
- **Mixed Precision Training** - Uses automatic mixed precision for faster training
- **Gradient Clipping** - Prevents gradient explosion (max norm = 1.0)
- **Separate Learning Rates** - Lower LR for pretrained backbone, higher for task heads
- **BerHu Loss** - Better depth loss function for improved training
- **Cosine Similarity Loss** - For surface normals with proper normalization

### Output Files
- `checkpoint_epoch_XX.pth` - Checkpoints for each epoch
- `best_model.pth` - Best model based on validation loss
- `final_model.pth` - Final model after training
- `training_samples/` - Visual samples showing training progress

### Monitoring Training
- Training progress is displayed with progress bars
- Sample visualizations are saved each epoch
- Loss values are logged for both training and validation

## Model Architecture Details

### Input/Output
- **Input**: RGB images (3 channels, 384x384)
- **Outputs**:
  - `depth`: Depth maps (1 channel, positive values)
  - `normals`: Surface normals (3 channels, range [-1, 1])
  - `alpha_logits`: Foreground logits (1 channel, use sigmoid for probabilities)

### Key Improvements
- **Softplus Activation** for depth (ensures positive values)
- **Tanh Activation** for normals (proper range [-1, 1])
- **Better Weight Initialization** for stable training
- **Feature Fusion Blocks** for multi-scale information

## GPU Requirements

- **Minimum**: 8GB GPU memory for batch_size=6
- **Recommended**: 12GB+ for larger batch sizes
- Adjust `batch_size` if you encounter out-of-memory errors

## Tips for Better Training

1. **Start with smaller batch sizes** if memory is limited
2. **Monitor the sample outputs** in `training_samples/` directory
3. **Adjust depth_scale** based on your dataset's depth range
4. **Use the best_model.pth** for inference (lowest validation loss)

## Loading Trained Models

```python
from fixed_dpt_multitask import create_multitask_dpt_model
import torch

# Create model
model = create_multitask_dpt_model()

# Load trained weights
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Use for inference
with torch.no_grad():
    outputs = model(input_tensor)
    depth = outputs['depth']
    normals = outputs['normals']
    alpha = torch.sigmoid(outputs['alpha_logits'])
```

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce `batch_size` in `train.py`
2. **Dataset Path**: Ensure `data_root` points to correct directory
3. **Missing Dependencies**: Install required packages listed above
4. **NaN/Inf Loss**: Check data preprocessing and learning rates

### Early Stopping
The current script includes early breaks for testing:
- Training: breaks after 100 batches per epoch
- Validation: breaks after 20 batches per epoch

Remove these lines for full training:
```python
# Remove these lines for full training
if batch_idx >= 100:  # Line in training loop
    break
if batch_idx >= 20:   # Line in validation loop
    break
