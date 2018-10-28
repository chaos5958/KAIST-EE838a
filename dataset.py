import random, os, glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
import imageio
imageio.plugins.freeimage.download()
from torchvision import transforms, datasets
from utility import *

#Note1: Pillow does not support RGB-32bit images

class LdrHdrDataset(Dataset):
    def __init__(self, data_dir, is_train, num_batch):
        self.is_train = is_train
        self.num_batch = num_batch

        #Read HDR images
        assert os.path.exists(data_dir)
        filenames = glob.glob('{}/*.hdr'.format(data_dir))

        #Load HDR images on memory
        self.hdr_images = []
        for filename in filenames:
            hdr_image = imageio.imread(filename, format='HDR-FI')
            self.hdr_images.append(hdr_image)

        #Transforms
        if self.is_train:
            #Train: Crop, Clip, Normalization, Camera curve, Rotate
            self.composed_ldr = transforms.Compose([RandomCrop((320, 320)),
                                            RandomRotate(),
                                            Clip(),
                                            Normalize((320, 320)),
                                            CameraCurve(),
                                            ToTensor()])
        else:
            #Test: Clip, Normalization, Camera curve
            self.composed_ldr = transforms.Compose([Pad(),
                                            Clip(),
                                            Normalize((1920, 1080)),
                                            CameraCurve(),
                                            ToTensor()])

    def __len__(self):
        if self.is_train:
            return self.num_batch * 1000
        else:
            return len(self.hdr_images)

    def __getitem__(self, idx=None):
        if self.is_train:
            idx = random.randint(0, len(self.hdr_images) - 1)
            img  = self.hdr_images[idx]

            ldr_img, hdr_img, normalized_value = self.composed_ldr(img)
        else:
            assert idx != None
            img = self.hdr_images[idx]

            ldr_img, hdr_img, normalized_value = self.composed_ldr(img)

        return ldr_img, hdr_img, normalized_value

if __name__ == "__main__":
    dataset = LdrHdrDataset('train/HDR', True)
