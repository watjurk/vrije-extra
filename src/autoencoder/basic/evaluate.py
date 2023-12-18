import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader


def evaluate_model(model: nn.Module, data_loader: DataLoader) -> float:
    loss_function = torch.nn.MSELoss()

    running_loss = 0
    for images, _ in data_loader:
        true_images = images
        predicted_images = model(true_images)
        running_loss += loss_function(predicted_images, true_images)

    return running_loss / len(data_loader)
