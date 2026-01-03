import torch
import json

print("=== CHECKING MODEL DATASET ===")
print("\n1. Loading model weights...")

# Load the model state_dict
state_dict = torch.load('best_model.pth', map_location='cpu')

print("\n2. Model has 8 classes (we know from fc2.weight.shape)")

print("\n3. Common 8-class plant disease datasets:")
print("   A. Tomato diseases only (8 classes)")
print("   B. Potato + Tomato mix (8 classes)")
print("   C. Custom selection (8 classes)")

print("\n4. To find out, check your Colab for:")
print("   - Dataset download code")
print("   - Data loading code")
print("   - Class names printout")

print("\n5. Quick way: Look at your Colab and find lines like:")
print("   - dataset = ...")
print("   - train_dataset.classes")
print("   - print('Classes:', classes)")

print("\n6. For now, create class_names.json with 8 generic names.")
print("   We'll update later when you find the real names.")