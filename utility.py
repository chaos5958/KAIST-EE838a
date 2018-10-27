import numpy as np
import random

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

class Clip(object):
    def __call__(self, image):
        return np.clip(image, HDR_MIN, HDR_MAX)

class Normalize(object):
    def __init__(self, image_size):
        assert isinstance(image_size, (tuple))
        self.num_pixels = image_size[0] * image_size[1]

    def __call__(self, image):
        image_1d = image.flatten()
        image_1d.sort(axis=0)

        rand_idx = np.random.randint(self.num_pixels * NORMALIZE_MIN, self.num_pixels * NORMALIZE_MAX)
        normalized_value = image_1d[-rand_idx]

        return image / normalized_value

class CameraCurve(object):
    def __call__(self, image):
        n = np.normal(N_MEAN, N_VAR)
        sig = np.normal(SIG_MEAN, SIG_VAR)

        image = np.power(image, n)
        image = (1 + sig) * (image / image + sig)

        return image

class RandomRotate(object):
    def __call__(self, image):
        image = image.rotate(90 * random.randint(1,3))

        if random.randint(0,1):
            image = np.flip(image, 0)

        if random.randint(0,1):
            image = np.flip(image, 1)

        return image

class Pad(object):
    def __call__(self, image):
        image = np.pad(image, ((0,0), (4,4), (0, 0)), 'edge')

        return image

class ToTensor(object):
    def __call__(self, image):
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)
