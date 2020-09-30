import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, ):