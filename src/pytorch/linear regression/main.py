# Disclaimer:
# This is my implementation of LR, it 'works' but it's not a book-accurate version


import torch
import numpy as np
import matplotlib.pyplot as plt

# torch.set_default_device("mps")

class LineModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.a = torch.nn.Parameter(torch.rand(1))
        self.b = torch.nn.Parameter(torch.rand(1))


n_points = 30


points = np.random.random(size=(n_points, 2)) * 10
# print(points)
# points = [[8.954718, 1.04513474], [6.92419485, 7.28486949]]

ax = plt.subplot()
ax.set_ylim(0, 10)
ax.set_xlim(0, 10)


def distance_from_point(model: LineModel, point):
    A = model.a
    B = torch.tensor(-1)
    C = model.b
    x0, y0 = point[0], point[1]
    numerator = torch.abs(A * x0 + B * y0 + C)
    denominator = torch.sqrt(torch.pow(A, 2) + torch.pow(B, 2))
    return numerator / denominator


def loss_function(model: LineModel, points: points):
    total_d = torch.tensor([0.0])
    for point in points:
        d = distance_from_point(model, point)
        total_d += d * d
    return total_d / len(points)


def draw_line(ax, a, b):
    return ax.axline((0, b), slope=a)


model = LineModel()

optimizer = torch.optim.SGD(
    [
        {"params": [model.a], "lr": 0.00005},
        {"params": [model.b], "lr": 0.001},
    ],
    momentum=0.1,
)

for point in points:
    ax.scatter(point[0], point[1])

previous = None
for _ in range(100_000):
    loss = loss_function(model, points)
    print(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if previous is not None:
        previous.remove()
    previous = draw_line(ax, model.a.item(), model.b.item())

    plt.pause(1 / 30)
