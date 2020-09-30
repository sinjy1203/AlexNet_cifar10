import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import *
from util import *

lr = 0.01
weight_decay = 0.0005
momentum = 0.9
batch_size = 64
num_epochs = 50

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Net = AlexNet().to(device)

fn_loss = nn.CrossEntropyLoss().to(device)
fn_pred = lambda x: np.argmax(x, axis=-1)
optim = torch.optim.SGD(Net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

mean = 0.4733630004850899

train_transform = transforms.Compose([transforms.RandomCrop(size=28), transforms.ToTensor(),
                                      pca_color_aug(), transforms.Normalize(mean=mean, std=1)])
train_datasets = datasets.CIFAR10(root="./", download=True, train=True, transform=train_transform)
train_data_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)


for epoch in range(num_epochs):
    Net.train()

    loss_arr = []
    acc_arr = []
    for img, label in train_data_loader:
        img = img.to(device)
        label = label.to(device)
        output = Net(img)
        pred = fn_pred(output.detach().cpu().numpy())

        loss = fn_loss(output, label)

        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_arr += [loss.item()]
        acc = np.mean((pred == label.cpu().numpy()).astype(np.int))

        acc_arr += [acc]

    acc_arr = np.array(acc_arr)
    loss_arr = np.array(loss_arr)
    print("epoch: %04d | loss: %.4f | acc: %.4f" % (epoch, np.mean(loss_arr), np.mean(acc_arr)))

torch.save(Net.state_dict(), "/content/drive/My Drive/Colab Notebooks/training_AlexNet_cifar10/checkpoint/AlexNet.pth")