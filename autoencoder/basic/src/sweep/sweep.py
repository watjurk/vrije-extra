import os
import sys
from pathlib import Path

# For now I will do this using relative imports.
# Then I will switch to Hydra.
# TODO: switch to Hydra.
sys.path.append(str(Path(os.path.abspath("")).parent.joinpath("data")))
sys.path.append(str(Path(os.path.abspath("")).parent.joinpath("model")))
sys.path.append(str(Path(os.path.abspath("")).parent))

from evaluate import evaluate_model
from train import train_model

import wandb
from MNIST import load_mnist_dataset
from MNISTAutoencoder import MNISTAutoencoder


def main():
    wandb.init()

    batch_size = wandb.config.batch_size
    number_of_layers = wandb.config.number_of_layers
    latent_space_size = wandb.config.latent_space_size
    epochs = wandb.config.epochs
    learning_rate = wandb.config.learning_rate

    _, (train_data_loader, validation_data_loader, test_data_loader) = load_mnist_dataset(batch_size)

    model = MNISTAutoencoder(latent_space_size, number_of_layers)

    model.train()
    train_model(
        model,
        learning_rate,
        epochs,
        train_data_loader,
        wandb_enabled=True,
        validation_data_loader=validation_data_loader,
        test_data_loader=test_data_loader,
    )

    model.eval()
    validation_loss = evaluate_model(model, validation_data_loader)
    test_loss = evaluate_model(model, test_data_loader)
    wandb.log(
        {
            "validation_loss": validation_loss,
            "test_loss": test_loss,
        }
    )


# Call the main function.
main()
