import torch

# Get the current device
device = torch.cuda.current_device()
print(f"PyTorch is using GPU: {device}")
