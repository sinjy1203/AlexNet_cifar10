import numpy as np
import torch
from torchvision import transforms


class pca_color_aug(object):
    def pca_color_aug_(self, image_array_input):
        assert image_array_input.ndim == 3 and image_array_input.shape[2] == 3
        assert image_array_input.dtype == np.uint8

        img = image_array_input.reshape(-1, 3).astype(np.float32)
        scaling_factor = np.sqrt(3.0 / np.sum(np.var(img, axis=0)))
        img *= scaling_factor

        cov = np.cov(img, rowvar=False)
        U, S, V = np.linalg.svd(cov)

        rand = np.random.randn(3) * 0.1
        delta = np.dot(U, rand * S)
        delta = (delta * 255.0).astype(np.int32)[np.newaxis, np.newaxis, :]

        img_out = np.clip(image_array_input + delta, 0, 255).astype(np.uint8)
        return img_out

    def __call__(self, img):
        img = img.numpy()
        img = img.transpose(1, 2, 0) * 255.0
        img = img.astype(np.uint8)
        img = self.pca_color_aug_(img)
        img = img / 255.0
        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)

        return img

class FiveCrop(object):
    def __init__(self, size):
        self.fivecrop = transforms.FiveCrop(size=size)
        self.totensor = transforms.ToTensor()

    def __call__(self, img):
        img_lst = self.fivecrop(img)
        img_tensor = self.totensor(img_lst[0]).reshape(1, 3, 28, 28)
        for i in range(1, 5):
            tensor = self.totensor(img_lst[i]).reshape(1, 3, 28, 28)
            img_tensor = torch.cat((img_tensor, tensor), 0)
        return img_tensor

class Normalize(object):
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, img_arr):
        for i in range(5):
            img_arr[i] = self.normalize(img_arr[i])

        return img_arr

class HorizontalFlip(object):
    def __init__(self):
        self.horizontalflip = transforms.RandomHorizontalFlip(p=1)

    def __call__(self, img_arr):
        img_arr_fliped = self.horizontalflip(img_arr)

        return torch.cat((img_arr, img_arr_fliped), 0)