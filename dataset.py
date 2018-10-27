import random, os, glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
import imageio
imageio.plugins.freeimage.download()
from torchvision import transforms, utils

from utility import *

#Note1: Pillow does not support RGB-32bit images

class LdrHdrDataset(Dataset):
    def __init__(self, data_dir, is_train):
        self.is_train = is_train

        #Read images
        assert os.path.exists(data_dir)
        filenames = glob.glob('{}/*.hdr'.format(data_dir))

        #Load images on memory
        self.hdr_images = []
        for filename in filenames:
            hdr_image = imageio.imread(filename, format='HDR-FI')
            self.hdr_images.append(hdr_image)

        #Transforms
        if self.is_train:
            #Train: Crop, Clip, Normalization, Camera curve, Rotate
            self.composed = transforms.Compose([RandomCrop((320, 320)),
                                            Clip(),
                                            Normalize((320, 320)),
                                            CameraCurve(),
                                            RandomRotate(),
                                            ToTensor()])
        else:
            #Test: Clip, Normalization, Camera curve
            self.composed = transforms.Compose([Clip(),
                                            Pad(),
                                            Normalize((1920, 1080)),
                                            CameraCurve(),
                                            ToTensor()])

    def __len__(self):
        return len(self.hdr_images)

    def __getitem__(self, idx):
        hdr_img = self.hdr_images[idx]
        ldr_img = self.composed(hdr_img)

        return ldr_img, hdr_img

if __name__ == "__main__":
    dataset = LdrHdrDataset('train/HDR', True)
