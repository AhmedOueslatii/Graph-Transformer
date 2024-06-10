import torch

# Load the saved model state dict
saved_model_path = 'runs/May31_15-35-58_Oussama/checkpoints/model.pth'
state_dict = torch.load(saved_model_path)

# Print the keys and shapes in the state dict
for k, v in state_dict.items():
    print(f"Layer: {k}, Shape: {v.shape}")

