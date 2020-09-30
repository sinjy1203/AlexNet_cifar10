import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.lrn = nn.LocalResponseNorm(size=5, k=2)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_features=2048, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=10)

        # nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)
        # nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01)
        # nn.init.normal_(self.conv3.weight, mean=0.0, std=0.01)
        # nn.init.normal_(self.conv4.weight, mean=0.0, std=0.01)
        # nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        # nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)

        # nn.init.ones_(self.conv2.bias)
        # nn.init.ones_(self.conv4.bias)
        # nn.init.ones_(self.fc1.bias)
        # nn.init.zeros_(self.conv1.bias)
        # nn.init.zeros_(self.conv3.bias)
        # nn.init.zeros_(self.fc2.bias)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.lrn(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.lrn(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = x.view(-1, 512*2*2)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc2(x)

        return x