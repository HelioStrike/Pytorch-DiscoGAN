import glob
import os
import torch

import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

#Defining a class as each image in the edge2shoes dataset(which is the dataset used to train the network) consists of images
#that contain the outline of the shoe on the left, and the real image of the shoe on the right.
#So here we're splitting each image into 2 images to used them as a training pair
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform_ = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w/2, h))
        img_B = img.crop((w/2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], 'RGB')
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], 'RGB')

        img_A = self.transform_(img_A)
        img_B = self.transform_(img_B)

        return {'A': img_A, 'B': img_B}

    def __len__(self):
        return len(self.files)
