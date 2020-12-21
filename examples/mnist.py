from pathlib import Path

import torch
from ignite.metrics import Average
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

from matches.loop import Loop
from matches.callbacks import *

data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
train_loader = DataLoader(
    MNIST(download=True, root=".", transform=data_transform, train=True),
    batch_size=10, shuffle=True
)
val_loader = DataLoader(
    MNIST(download=False, root=".", transform=data_transform, train=False),
    batch_size=10, shuffle=False
)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


model = Net()
optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.5)
criterion = nn.NLLLoss()
train_loss = Average()
valid_loss = Average()

loop = Loop(3, [
    BestModelSaver("valid_loss", Path("logs/mnist")),
    TqdmProgressCallback(),
])

loop.register("model", model)

loop.register("train_loss", train_loss)
loop.register("valid_loss", valid_loss)


def train(loop: Loop):
    for _ in loop.iterate_epochs():

        for x, y in loop.iterate_dataloader(train_loader, mode="train"):
            y_pred = model(x)

            loss: torch.Tensor = criterion(y_pred, y)
            optimizer.zero_grad()
            train_loss.update(loss.detach())
            loss.backward()
            optimizer.step()

        for x, y in loop.iterate_dataloader(val_loader):
            y_pred = model(x)

            loss: torch.Tensor = criterion(y_pred, y)
            valid_loss.update(loss)


loop.run(train)
