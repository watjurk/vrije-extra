import torch


print(torch.backends.mps.is_available()) #the MacOS is higher than 12.3+
print(torch.backends.mps.is_built()) #MPS is activated

# device = torch.device('mps')
# x = torch.rand((10000, 10000), dtype=torch.float32)
# y = torch.rand((10000, 10000), dtype=torch.float32)
# x = x.to(device)
# y = y.to(device)

# x * y