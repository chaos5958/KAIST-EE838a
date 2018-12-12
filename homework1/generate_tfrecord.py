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

def crop_image(lr_image, hr_image, scale):
    height, width, channel = lr_image.get_shape().as_list()
    rand_height = random.randint(0, height - args.patch_size - 1)
    rand_width = random.randint(0, width - args.patch_size - 1)
    lr_image_cropped = tf.image.crop_to_bounding_box(lr_image,
                                                    rand_height,
                                                    rand_width,
                                                    args.patch_size,
                                                    args.patch_size)
    hr_image_cropped = tf.image.crop_to_bounding_box(hr_image,
                                                    rand_height * scale,
                                                    rand_width * scale,
                                                    args.patch_size * scale,
                                                    args.patch_size * scale)

    return lr_image_cropped, hr_image_cropped

#tfrecord
train_tf_records_filename = os.path.join(args.data_dir, 'train.tfrecords')
test_tf_records_filename = os.path.join(args.data_dir, 'test.tfrecords')
train_writer = tf.io.TFRecordWriter(train_tf_records_filename)
test_writer = tf.io.TFRecordWriter(test_tf_records_filename)

#load train images
train_lr_image_path = os.path.join(args.data_dir, 'train', 'LR')
train_hr_image_path = os.path.join(args.data_dir, 'train', 'HR')
train_lr_image_filenames = glob.glob('{}/*.jpg'.format(train_lr_image_path))
train_hr_image_filenames = glob.glob('{}/*.jpg'.format(train_hr_image_path))
train_lr_images = []
train_hr_images = []

for lr_filename, hr_filename in zip(train_lr_image_filenames, train_hr_image_filenames):
    with tf.device('cpu:0'):
        train_lr_images.append(load_image(lr_filename))
        train_hr_images.append(load_image(hr_filename))

assert len(train_lr_images) == len(train_hr_images)
assert len(train_lr_images) != 0

print('dataset length: {}'.format(len(train_lr_images)))

#load valid images
valid_lr_image_path = os.path.join(args.data_dir, 'valid', 'LR')
valid_hr_image_path = os.path.join(args.data_dir, 'valid', 'HR')
valid_lr_image_filenames = glob.glob('{}/*.jpg'.format(valid_lr_image_path))
valid_hr_image_filenames = glob.glob('{}/*.jpg'.format(valid_hr_image_path))
valid_lr_images = []
valid_hr_images = []

for lr_filename, hr_filename in zip(valid_lr_image_filenames, valid_hr_image_filenames):
    with tf.device('cpu:0'):
        valid_lr_images.append(load_image(lr_filename))
        valid_hr_images.append(load_image(hr_filename))

assert len(valid_lr_images) == len(valid_hr_images)
assert len(valid_lr_images) != 0

print('dataset length: {}'.format(len(valid_lr_images)))

#generate training tfrecord
for i in range(args.num_patch):
    if i % 1000 == 0:
        print('Train TFRecord Process status: [{}/{}]'.format(i, args.num_patch))

    rand_idx = random.randint(0, len(train_lr_images) - 1)
    lr_image, hr_image = crop_image(train_lr_images[rand_idx], train_hr_images[rand_idx], args.scale)
    """
    Image.fromarray(np.uint8(lr_image.numpy()*255)).save('lr.png')
    Image.fromarray(np.uint8(hr_image.numpy()*255)).save('hr.png')
    sys.exit()
    """
    lr_image_shape = tf.shape(lr_image)
    hr_image_shape = tf.shape(hr_image)
    lr_binary_image = lr_image.numpy().tostring()
    hr_binary_image = hr_image.numpy().tostring()

    width, height, channel = lr_image.get_shape().as_list()
    if channel != 3:
        print('lr image {} has a problem'.format(rand_idx))
        print(lr_image_shape)
        continue

    width, height, channel = hr_image.get_shape().as_list()
    if channel != 3:
        print('hr image {} has a problem'.format(rand_idx))
        print(hr_image_shape)
        continue

    feature = {
        'lr_image_raw': _bytes_feature(lr_binary_image),
        'lr_height': _int64_feature(lr_image_shape[0]),
        'lr_width': _int64_feature(lr_image_shape[1]),
        'hr_image_raw': _bytes_feature(hr_binary_image),
        'hr_height': _int64_feature(hr_image_shape[0]),
        'hr_width': _int64_feature(hr_image_shape[1]),
        }

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    train_writer.write(tf_example.SerializeToString())
train_writer.close()

#generate validation tfrecord
for i in range(len(valid_lr_images)):
    if i % 10 == 0:
        print('Train TFRecord Process status: [{}/{}]'.format(i, len(valid_lr_images)))

    lr_image = valid_lr_images[i]
    hr_image = valid_hr_images[i]

    lr_image_shape = tf.shape(lr_image)
    hr_image_shape = tf.shape(hr_image)
    lr_binary_image = lr_image.numpy().tostring()
    hr_binary_image = hr_image.numpy().tostring()

    width, height, channel = lr_image.get_shape().as_list()
    if channel != 3:
        print('lr image {} has a problem'.format(rand_idx))
        print(lr_image_shape)
        continue

    width, height, channel = hr_image.get_shape().as_list()
    if channel != 3:
        print('hr image {} has a problem'.format(rand_idx))
        print(hr_image_shape)
        continue

    feature = {
        'lr_image_raw': _bytes_feature(lr_binary_image),
        'lr_height': _int64_feature(lr_image_shape[0]),
        'lr_width': _int64_feature(lr_image_shape[1]),
        'hr_image_raw': _bytes_feature(hr_binary_image),
        'hr_height': _int64_feature(hr_image_shape[0]),
        'hr_width': _int64_feature(hr_image_shape[1]),
        }

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    test_writer.write(tf_example.SerializeToString())
test_writer.close()
