import torch

# Load the adjacency matrix
adj_matrix = torch.load('graphs/simclr_files/new2/adj_s.pt')
print("Adjacency Matrix:\n", adj_matrix)

# Load the features
features = torch.load('graphs/simclr_files/new2/features.pt')
print("Features Shape:", features.shape)
print("Features Data:\n", features)
