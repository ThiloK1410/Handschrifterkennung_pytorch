import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils import data as utils_data


train_data = utils_data.DataLoader(
                    datasets.MNIST("data", train=True, download=True, transform=transforms.Compose(
                        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                    batch_size=64,
                    shuffle=True
                        )
test_data = utils_data.DataLoader(
                    datasets.MNIST("data", train=False, download=True, transform=transforms.Compose(
                        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                    batch_size=64,
                    shuffle=True
                        )


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv_dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 60)
        self.fc2 = nn.Linear(60, 30)
        self.fc3 = nn.Linear(30, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = f.max_pool2d(x, 2)
        x = f.relu(x)
        x = self.conv2(x)
        x = self.conv_dropout(x)
        x = f.max_pool2d(x, 2)
        x = f.relu(x)
        print(x.size())
        exit()


model = Network()
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8)


def train(epoch):
    model.train()
    for batch_id, (data, target) in enumerate(train_data):
        data = data.cuda()
        target = target.cuda()
        data = Variable(data)
        target = Variable(target)

        optimizer.zero_grad()
        out = model(data)
        criterion = nn.NLLLoss
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print(f"EPOCHE: {epoch}")


for epoch in range(1, 30):
    train(epoch)
