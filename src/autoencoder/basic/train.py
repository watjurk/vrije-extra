from typing import Optional

import torch
import wandb
from evaluate import evaluate_model
from fastprogress.fastprogress import master_bar, progress_bar
from torch import nn
from torch.utils.data.dataloader import DataLoader


def train_model(
    model: nn.Module,
    learning_rate,
    epochs,
    train_data_loader: DataLoader,
    wandb_enabled=False,
    validation_data_loader: Optional[DataLoader] = None,
    test_data_loader: Optional[DataLoader] = None,
):
    if wandb_enabled:
        assert validation_data_loader is not None
        assert test_data_loader is not None

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()

    for epoch in (mb := master_bar(range(epochs))):
        running_loss = 0
        for bach_of_data_images, _ in (pb := progress_bar(train_data_loader, parent=mb)):
            true_images = bach_of_data_images
            predicted = model(true_images)

            loss = loss_function(predicted, true_images)
            loss_item = loss.item()
            running_loss += loss_item
            pb.comment = f"Loss: {loss_item}"

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if wandb_enabled:
            model.eval()
            wandb.log(
                {
                    "epoch": epoch,
                    "loss": running_loss / len(train_data_loader),
                    "validation_loss": evaluate_model(model, validation_data_loader),
                    "test_loss": evaluate_model(model, test_data_loader),
                }
            )
            model.train()
        running_loss = 0
