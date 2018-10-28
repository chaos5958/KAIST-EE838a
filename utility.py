import torch
import numpy as np
import random, logging, os

HDR_MIN = 0
HDR_MAX = 100000
HDR_H = 1080
HDR_W = 1920
N_MEAN = 0.9
N_VAR = 0.1
SIG_MEAN = 0.6
SIG_VAR = 0.1
NORMALIZE_MIN = 0.05
NORMALIZE_MAX = 0.15

class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]

        return image

class RandomRotate(object):
    def __call__(self, image):

        for _ in range(random.randint(1,3)):
            image = np.rot90(image)

        if random.randint(0,1):
            image = np.flip(image, 0)

            image = np.flip(image, 1)

        return image

class Clip(object):
    def __call__(self, image):

        image = np.clip(image, a_min=HDR_MIN, a_max=HDR_MAX)

        return image

class Normalize(object):
    def __init__(self, image_size):
        assert isinstance(image_size, (tuple))
        self.num_pixels = image_size[0] * image_size[1]

    def __call__(self, image):
        image_mean = np.mean(image, axis=2)
        image_1d = image_mean.flatten()

        rand_idx = np.random.randint(self.num_pixels * NORMALIZE_MIN, self.num_pixels * NORMALIZE_MAX)
        normalized_value = image_1d[-rand_idx]
        image = image / normalized_value

        return image

class CameraCurve(object):
    def __call__(self, image):
        target = image #target (x Camera curve)

        n = np.clip(np.random.normal(N_MEAN, N_VAR), a_min=0, a_max=2.5)
        sig = np.clip(np.random.normal(SIG_MEAN, SIG_VAR), a_min=0, a_max=5)

        input = np.power(image, n)
        input = (1 + sig) * (input / (input + sig)) #input (o Camera curve)

        return (input, target)

class Pad(object):
    def __call__(self, image):
        image = np.pad(image, ((4,4), (0,0), (0, 0)), 'edge')

        return image

class ToTensor(object):
    def __call__(self, images):
        input = images[0]
        target = images[1]

        #input (Clip & Quantization)
        input = np.floor((np.clip(input, a_min=0, a_max=1) * 255 + 0.5)) / 255

        input = input.transpose((2, 0, 1))
        input = torch.from_numpy(input.copy())

        target = target.transpose((2, 0, 1))
        target = torch.from_numpy(target.copy())

        return (input, target)

def getLogger(save_dir, save_name):
    Logger = logging.getLogger(save_name)
    Logger.setLevel(logging.INFO)
    Logger.propagate = False

    filePath = os.path.join(save_dir, save_name)
    if os.path.exists(filePath):
        os.remove(filePath)

    fileHandler = logging.FileHandler(filePath)
    logFormatter = logging.Formatter('%(message)s')
    fileHandler.setFormatter(logFormatter)
    Logger.addHandler(fileHandler)

    return Logger
