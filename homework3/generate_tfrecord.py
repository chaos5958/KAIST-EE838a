import tensorflow as tf
import os, glob, random, sys
from PIL import Image
import numpy as np
from ops import load_image

from option import args

tf.enable_eager_execution()

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def resize_image(blur_image, sharp_image):
    with tf.device('cpu:0'):
        #Load
        blur_image = load_image(blur_image)
        blur_image = tf.expand_dims(blur_image, axis=0)
        sharp_image = load_image(sharp_image)
        sharp_image = tf.expand_dims(sharp_image, axis=0)

        _, height, width, _ = sharp_image.get_shape().as_list()

        #Resize
        blur_images = {}
        sharp_images = {}
        blur_images['x4'] = tf.image.resize_bicubic(blur_image, size=[height // 4, width // 4])

        blur_images['x2'] = tf.image.resize_bicubic(blur_image, size=[height // 2, width // 2])
        blur_images['x1'] = blur_image
        sharp_images['x4'] = tf.image.resize_bicubic(sharp_image, size=[height // 4, width // 4])
        sharp_images['x2'] = tf.image.resize_bicubic(sharp_image, size=[height // 2, width // 2])
        sharp_images['x1'] = sharp_image

        for idx, _ in blur_images.items():
            blur_images[idx] = tf.squeeze(blur_images[idx])
        for idx, _ in sharp_images.items():
            sharp_images[idx] = tf.squeeze(sharp_images[idx])

        return blur_images, sharp_images

def crop_resize_image(blur_image, sharp_image):
    with tf.device('cpu:0'):
        #Load
        blur_image = load_image(blur_image)
        sharp_image = load_image(sharp_image)

        #Stack
        stacked_image = tf.stack([blur_image, sharp_image], axis=0)

        #Crop
        stacked_cropped_image = tf.image.random_crop(stacked_image, [2, args.patch_size, args.patch_size, 3])

        #Augmentation
        stacked_cropped_image = tf.image.random_flip_left_right(stacked_cropped_image)
        stacked_cropped_image = tf.image.random_flip_up_down(stacked_cropped_image)
        for _ in range(random.randint(0,3)):
            stacked_cropped_image = tf.image.rot90(stacked_cropped_image)

        #Split
        blur_cropped_image, sharp_cropped_image = tf.split(stacked_cropped_image, num_or_size_splits=2, axis=0)


        #Resize
        blur_cropped_images = {}
        sharp_cropped_images = {}
        blur_cropped_images['x4'] = tf.image.resize_bicubic(blur_cropped_image, size=[args.patch_size // 4, args.patch_size // 4])

        blur_cropped_images['x2'] = tf.image.resize_bicubic(blur_cropped_image, size=[args.patch_size // 2, args.patch_size // 2])
        blur_cropped_images['x1'] = blur_cropped_image
        sharp_cropped_images['x4'] = tf.image.resize_bicubic(sharp_cropped_image, size=[args.patch_size // 4, args.patch_size // 4])
        sharp_cropped_images['x2'] = tf.image.resize_bicubic(sharp_cropped_image, size=[args.patch_size // 2, args.patch_size // 2])
        sharp_cropped_images['x1'] = sharp_cropped_image

        for idx, _ in blur_cropped_images.items():
            blur_cropped_images[idx] = tf.squeeze(blur_cropped_images[idx])
        for idx, _ in sharp_cropped_images.items():
            sharp_cropped_images[idx] = tf.squeeze(sharp_cropped_images[idx])

        return blur_cropped_images, sharp_cropped_images

#tfrecord
train_tf_records_filename = os.path.join(args.data_dir, 'train.tfrecords')
test_tf_records_filename = os.path.join(args.data_dir, 'test.tfrecords')
train_writer = tf.io.TFRecordWriter(train_tf_records_filename)
test_writer = tf.io.TFRecordWriter(test_tf_records_filename)

#load train images
subdirs = os.listdir('data/train')
train_blur_filenames = []
train_sharp_filenames = []

for subdir in subdirs:
    train_blur_filenames.extend(glob.glob('{}/{}/{}/*.png'.format('data/train', subdir, 'blur_gamma')))
    train_sharp_filenames.extend(glob.glob('{}/{}/{}/*.png'.format('data/train', subdir, 'sharp')))
train_blur_filenames.sort()
train_sharp_filenames.sort()

assert len(train_blur_filenames) == len(train_sharp_filenames)
assert len(train_blur_filenames) != 0

print('dataset length: {}'.format(len(train_blur_filenames)))

"""
for lr_filename, hr_filename in zip(train_lr_image_filenames, train_hr_image_filenames):
    with tf.device('cpu:0'):
        train_lr_images.append(load_image(lr_filename))
        train_hr_images.append(load_image(hr_filename))
"""

valid_blur_filenames = ['data/valid/GOPR0384_11_00/blur/000001.png', 'data/valid/GOPR0384_11_05/blur/004001.png', 'data/valid/GOPR0385_11_01/blur/003011.png']
valid_sharp_filenames = ['data/valid/GOPR0384_11_00/sharp/000001.png', 'data/valid/GOPR0384_11_05/sharp/004001.png', 'data/valid/GOPR0385_11_01/sharp/003011.png']

assert len(valid_blur_filenames) == len(valid_sharp_filenames)
assert len(valid_blur_filenames) != 0

"""
for lr_filename, hr_filename in zip(valid_lr_image_filenames, valid_hr_image_filenames):
    with tf.device('cpu:0'):
        valid_lr_images.append(load_image(lr_filename))
        valid_hr_images.append(load_image(hr_filename))
"""


print('dataset length: {}'.format(len(valid_blur_filenames)))

#generate training tfrecord
for i in range(args.num_patch):
    if i % 1000 == 0:
        print('Train TFRecord Process status: [{}/{}]'.format(i, args.num_patch))

    rand_idx = random.randint(0, len(train_blur_filenames) - 1)
    blur_images, sharp_images = crop_resize_image(train_blur_filenames[rand_idx], train_sharp_filenames[rand_idx])
    """
    Image.fromarray(np.uint8(blur_images['x4'].numpy()*255)).save('blur_x4.png')
    Image.fromarray(np.uint8(sharp_images['x4'].numpy()*255)).save('sharp_x4.png')
    Image.fromarray(np.uint8(blur_images['x2'].numpy()*255)).save('blur_x2.png')
    Image.fromarray(np.uint8(sharp_images['x2'].numpy()*255)).save('sharp_x2.png')
    Image.fromarray(np.uint8(blur_images['x1'].numpy()*255)).save('blur_x1.png')
    Image.fromarray(np.uint8(sharp_images['x1'].numpy()*255)).save('sharp_x1.png')
    sys.exit()
    """
    feature = {
        'blur_image_x4_raw': _bytes_feature(blur_images['x4'].numpy().tostring()),
        'blur_image_x2_raw': _bytes_feature(blur_images['x2'].numpy().tostring()),
        'blur_image_x1_raw': _bytes_feature(blur_images['x1'].numpy().tostring()),
        'sharp_image_x4_raw': _bytes_feature(sharp_images['x4'].numpy().tostring()),
        'sharp_image_x2_raw': _bytes_feature(sharp_images['x2'].numpy().tostring()),
        'sharp_image_x1_raw': _bytes_feature(sharp_images['x1'].numpy().tostring()),
        }

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    train_writer.write(tf_example.SerializeToString())
train_writer.close()

#generate validation tfrecord
for i in range(len(valid_blur_filenames)):
    if i % 10 == 0:
        print('Valid TFRecord Process status: [{}/{}]'.format(i, len(valid_blur_filenames)))

    blur_images, sharp_images = resize_image(valid_blur_filenames[i], valid_sharp_filenames[i])

    height, width, channel = sharp_images['x1'].get_shape().as_list()
    """
    Image.fromarray(np.uint8(blur_images['x4'].numpy()*255)).save('blur_x4.png')
    Image.fromarray(np.uint8(sharp_images['x4'].numpy()*255)).save('sharp_x4.png')
    Image.fromarray(np.uint8(blur_images['x2'].numpy()*255)).save('blur_x2.png')
    Image.fromarray(np.uint8(sharp_images['x2'].numpy()*255)).save('sharp_x2.png')
    Image.fromarray(np.uint8(blur_images['x1'].numpy()*255)).save('blur_x1.png')
    Image.fromarray(np.uint8(sharp_images['x1'].numpy()*255)).save('sharp_x1.png')
    sys.exit()
    """
    feature = {
        'blur_image_x4_raw': _bytes_feature(blur_images['x4'].numpy().tostring()),
        'blur_image_x2_raw': _bytes_feature(blur_images['x2'].numpy().tostring()),
        'blur_image_x1_raw': _bytes_feature(blur_images['x1'].numpy().tostring()),
        'sharp_image_x4_raw': _bytes_feature(sharp_images['x4'].numpy().tostring()),
        'sharp_image_x2_raw': _bytes_feature(sharp_images['x2'].numpy().tostring()),
        'sharp_image_x1_raw': _bytes_feature(sharp_images['x1'].numpy().tostring()),
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        }

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    test_writer.write(tf_example.SerializeToString())
test_writer.close()
