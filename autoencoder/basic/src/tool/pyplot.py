import torch
import matplotlib.pyplot as plt


def show_MNIST_digit(x: torch.Tensor, y: str):
    x = x.reshape((28, 28))
    plt.title(y)
    plt.imshow(x, cmap=plt.cm.gray,origin="upper", interpolation="nearest")
    plt.show()
