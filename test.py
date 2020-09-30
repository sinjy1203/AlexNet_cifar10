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
Net.load_state_dict(torch.load("/content/drive/My Drive/Colab Notebooks/training_AlexNet_cifar10/checkpoint/AlexNet.pth"))

fn_loss = nn.CrossEntropyLoss().to(device)
fn_pred = lambda x: np.argmax(x, axis=-1)
optim = torch.optim.SGD(Net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

mean = 0.4733630004850899

test_transform = transforms.Compose([FiveCrop(size=28), HorizontalFlip(),
                                    Normalize(mean=mean, std=1)])
test_datasets = datasets.CIFAR10(root="./", download=True, train=False, transform=test_transform)
test_data_loader = DataLoader(test_datasets, batch_size=batch_size, shuffle=False)


with torch.no_grad():
    Net.eval()

    loss_arr = []
    acc_arr = []
    for img, label in test_data_loader:
        img = img.transpose(0, 1).to(device)
        label = label.to(device)

        y_arr = torch.unsqueeze(Net(img[0]), 0)

        for i in range(1, 10):
            y = torch.unsqueeze(Net(img[i]), 0)
            y_arr = torch.cat((y_arr, y), 0)

        output = torch.sum(y_arr, dim=0)

        pred = fn_pred(output.detach().cpu().numpy())

        loss = fn_loss(output, label)

        loss_arr += [loss.item()]
        acc = np.mean((pred == label.cpu().numpy()).astype(np.int))

        acc_arr += [acc]

    acc_arr = np.array(acc_arr)
    loss_arr = np.array(loss_arr)
    print("Test || loss: %.4f | acc: %.4f" % (np.mean(loss_arr), np.mean(acc_arr)))