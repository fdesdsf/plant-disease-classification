import torch
import numpy as np

print("=== CHECKING MODEL DETAILS ===")

# Load the model
state_dict = torch.load('best_model.pth', map_location='cpu')
print(f"Total layers: {len(state_dict)}")

# Check fc2 layer for number of classes
fc2_weight = state_dict['fc2.weight']
fc2_bias = state_dict['fc2.bias']

print(f"\nfc2.weight shape: {fc2_weight.shape}")  # [num_classes, input_features]
print(f"fc2.bias shape: {fc2_bias.shape}")      # [num_classes]

num_classes = fc2_weight.shape[0]
print(f"\n✓ Number of classes: {num_classes}")

# Check fc1 to understand dimensions
fc1_weight = state_dict['fc1.weight']
print(f"fc1.weight shape: {fc1_weight.shape}")
print(f"fc1 -> fc2 dimensions: {fc1_weight.shape[0]} -> {fc2_weight.shape[0]}")

# Check input dimensions from first conv layer
conv1_weight = state_dict['conv1.weight']
print(f"\nconv1.weight shape: {conv1_weight.shape}")
print(f"Input channels: {conv1_weight.shape[1]}")
print(f"Output channels: {conv1_weight.shape[0]}")
print(f"Kernel size: {conv1_weight.shape[2]}x{conv1_weight.shape[3]}")

# Show all layer shapes
print("\n=== ALL LAYER SHAPES ===")
for key, value in state_dict.items():
    if 'weight' in key or 'bias' in key:
        print(f"{key:20} shape: {list(value.shape)}")