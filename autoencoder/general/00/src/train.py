import torch
import torch.utils.data

import torchvision.datasets
import torchvision.transforms.functional

import matplotlib.pyplot as plt
import numpy as np

import model
import tool

# torch.set_default_device("mps")

def transform_to_tensor(img):
    return torchvision.transforms.functional.pil_to_tensor(img)
    

dataset = torchvision.datasets.MNIST(root="~/pytorch/data", train=True, download=True, transform=transform_to_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

for image, label in dataloader:
    print(image.shape, label)
    exit()

model = model.AE()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-8)
loss_function = torch.nn.MSELoss()

epochs = 1
losses = []
for epoch in range(epochs):
    for image, label in dataset:
        print(image)
        image = image.to("mps")
        image = image.reshape(-1)

        reconstructed = model(image)
        # tool.show_MNIST_digit(reconstructed.cpu().detach(), label)

        loss = loss_function(reconstructed, image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Storing the losses in a list for plotting
        losses.append(loss)
    

# Defining the Plot Style
plt.style.use("fivethirtyeight")
plt.xlabel("Iterations")
plt.ylabel("Loss")

# Plotting the last 100 values
plt.plot(losses[-100:])
