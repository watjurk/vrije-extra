import torch.utils.data
import torchvision
from torch.utils.data.dataloader import DataLoader


def load_mnist_dataset(batch_size: int):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )

    dataset = torchvision.datasets.MNIST("~/pytorch/data", train=True, transform=transforms)
    train_data, validation_data, test_data = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2])

    validation_data_loader = DataLoader(validation_data, batch_size)
    train_data_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size)
    return dataset, (train_data_loader, validation_data_loader, test_data_loader)
