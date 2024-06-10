import torch
import torch.nn as nn
from models.GraphTransformer import Classifier
# Load the model


adj =  torch.load('graphs/simclr_files/Blast_PCRM_R14-0351/adj_s.pt').float()
features = torch.load('graphs/simclr_files/Blast_PCRM_R14-0351/features.pt')
print("Adjacency matrix shape:", adj.shape)
print("Feature matrix shape:", features.shape)

# Assume you have a labels tensor for the graph
label = torch.tensor([1])  # replace with actual label if available (0, 1, or 2)

# Create mask for the nodes (assuming all nodes are valid for simplicity)
mask = torch.ones(features.shape[0], dtype=torch.float32)


n_class = 3  # Replace with the actual number of classes
model = Classifier(n_class)

# Remove 'module.' prefix
state_dict = torch.load('graph_transformer/saved_models/GraphCAM.pth')
new_state_dict = {}
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k  # remove `module.`
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.eval()

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
features, label, adj, mask = features.to(device), label.to(device), adj.to(device), mask.to(device)

# Make prediction
with torch.no_grad():
    preds, labels, loss = model(features.unsqueeze(0), label, adj.unsqueeze(0), mask.unsqueeze(0))

# Print prediction
print(f'Predicted Labels: {preds}')