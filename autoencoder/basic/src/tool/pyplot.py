import matplotlib.pyplot as plt


def show_MNIST_digit(digit):
    (x, y) = digit
    x = x.reshape((28, 28))
    plt.title(y)
    plt.imshow(x, cmap=plt.cm.gray, origin="upper", interpolation="nearest")
    plt.show()


def show_MNIST_digit_array(dataset):
    i = 0
    _, axs = plt.subplots(nrows=4, ncols=4, figsize=(4, 4))
    for i, ax in enumerate(axs.reshape(-1)):
        (data, target) = dataset[i]
        data = data.reshape((28, 28))

        ax.set_title(target)
        ax.imshow(data, cmap=plt.cm.gray, origin="upper", interpolation="nearest")
        ax.axis("off")
        i += 1
    plt.tight_layout()
    plt.show()
