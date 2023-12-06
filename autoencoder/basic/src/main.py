import torch
import torch.utils.data
import torchvision.datasets
import torchvision.transforms
from tqdm.auto import tqdm

import model
import tool

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
dataset = torchvision.datasets.MNIST(root="~/pytorch/data", train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


def train(model, epochs, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    loss_function = torch.nn.MSELoss()

    for _ in tqdm(range(epochs), desc="Epochs", position=0):
        for image, label in tqdm(dataloader, desc="Bach", position=1, leave=False):
            image = image.to(device)
            image = image.reshape(-1, 28 * 28)

            reconstructed = model(image)
            loss = loss_function(reconstructed, image)
            print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    model = model.AE()
    train(model, epochs=10)
