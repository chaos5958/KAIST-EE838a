import random, os, glob, sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torchvision.transforms.functional as TF
from PIL import Image

class DeblurDataset(Dataset):
    def __init__(self, rootdir, is_train):
        assert os.path.exists(rootdir)
        self.is_train = is_train

        #Read blur/sharp images
        self.blur_filenames = []
        self.sharp_filenames = []

        subdirs = os.listdir(rootdir)
        for subdir in subdirs:
            self.blur_filenames.extend(glob.glob('{}/{}/{}/*.png'.format(rootdir, subdir, 'blur_gamma')))
            self.sharp_filenames.extend(glob.glob('{}/{}/{}/*.png'.format(rootdir, subdir, 'sharp')))

        self.blur_filenames.sort()
        self.sharp_filenames.sort()

        assert len(self.blur_filenames) == len(self.sharp_filenames)

    def __len__(self):
        return len(self.sharp_filenames)

    def transform(self, blur_image, sharp_image):
        #Apply augmentation
        if self.is_train:
            #Crop
            i, j, h, w = transforms.RandomCrop.get_params(blur_image, output_size=(256, 256))
            blur_image = TF.crop(blur_image, i, j, h, w)
            sharp_image = TF.crop(sharp_image, i, j, h, w)

            #Rotate
            for _ in range(random.randint(1,3)):
                blur_image = TF.rotate(blur_image, 90)
                sharp_image = TF.rotate(sharp_image, 90)

            #Vertical flip
            if random.randint(0,1):
                blur_image = TF.vflip(blur_image)
                sharp_image = TF.vflip(sharp_image)

            #Horizontal flip
            if random.randint(0,1):
                blur_image = TF.hflip(blur_image)
                sharp_image = TF.hflip(sharp_image)

        #Generate multi-scale images
        blur_imgs = []
        sharp_imgs = []
        width, height = blur_image.size

        blur_imgs.append(TF.to_tensor(blur_image))
        sharp_imgs.append(TF.to_tensor(sharp_image))

        blur_imgs.append(TF.to_tensor(blur_image.resize((width//2, height//2), resample=Image.BICUBIC)))
        sharp_imgs.append(TF.to_tensor(sharp_image.resize((width//2, height//2), resample=Image.BICUBIC)))

        blur_imgs.append(TF.to_tensor(blur_image.resize((width//2, height//2), resample=Image.BICUBIC)))
        sharp_imgs.append(TF.to_tensor(sharp_image.resize((width//2, height//2), resample=Image.BICUBIC)))

        return blur_imgs, sharp_imgs

    def __getitem__(self, idx):
        blur_image = Image.open(self.blur_filenames[idx])
        sharp_image = Image.open(self.sharp_filenames[idx])
        blur_images, sharp_images = self.transform(blur_image, sharp_image)
        return blur_images, shapr_images

if __name__ == "__main__":
    dataset = DeblurDataset('train', True)
