import torch

# Check MPS (Apple Metal) availability
print(f"Is MPS (Apple Metal) available? {torch.backends.mps.is_available()}")
print(f"Is MPS built? {torch.backends.mps.is_built()}")

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Test tensor operation on device
x = torch.rand(3, 3).to(device)
print(f"Test tensor on {device}:\n{x}")
