import tensorflow as tf   # only work from tensorflow==1.9.0-rc1 and after
import os, glob
from option import args

class TFRecordDataset():
    def __init__(self, args):
        self.load_on_memory = args.load_on_memory
        self.num_batch = args.num_batch
        self.train_tfrecord_path = os.path.join(args.data_dir, 'train.tfrecords')
        self.test_tfrecord_path= os.path.join(args.data_dir, 'test.tfrecords')
        assert os.path.isfile(self.train_tfrecord_path)
        assert os.path.isfile(self.test_tfrecord_path)

    def _train_parse_function(self, example_proto):
        features = {'blur_image_x4_raw': tf.FixedLenFeature((), tf.string),
                'blur_image_x2_raw': tf.FixedLenFeature((), tf.string),
                'blur_image_x1_raw': tf.FixedLenFeature((), tf.string),
                'sharp_image_x4_raw': tf.FixedLenFeature((), tf.string),
                'sharp_image_x2_raw': tf.FixedLenFeature((), tf.string),
                'sharp_image_x1_raw': tf.FixedLenFeature((), tf.string)}
        parsed_features = tf.parse_single_example(example_proto, features)

        blur_images = {}
        blur_images['x4'] = tf.decode_raw(parsed_features['blur_image_x4_raw'], tf.float32)
        blur_images['x4'] = tf.reshape(blur_images['x4'], [args.patch_size // 4, args.patch_size // 4, 3])
        blur_images['x2'] = tf.decode_raw(parsed_features['blur_image_x2_raw'], tf.float32)
        blur_images['x2'] = tf.reshape(blur_images['x2'], [args.patch_size // 2, args.patch_size // 2, 3])
        blur_images['x1'] = tf.decode_raw(parsed_features['blur_image_x1_raw'], tf.float32)
        blur_images['x1'] = tf.reshape(blur_images['x1'], [args.patch_size, args.patch_size, 3])

        sharp_images = {}
        sharp_images['x4'] = tf.decode_raw(parsed_features['sharp_image_x4_raw'], tf.float32)
        sharp_images['x4'] = tf.reshape(sharp_images['x4'], [args.patch_size // 4, args.patch_size // 4, 3])
        sharp_images['x2'] = tf.decode_raw(parsed_features['sharp_image_x2_raw'], tf.float32)
        sharp_images['x2'] = tf.reshape(sharp_images['x2'], [args.patch_size // 2, args.patch_size // 2, 3])
        sharp_images['x1'] = tf.decode_raw(parsed_features['sharp_image_x1_raw'], tf.float32)
        sharp_images['x1'] = tf.reshape(sharp_images['x1'], [args.patch_size, args.patch_size, 3])

        return blur_images, sharp_images

    def _test_parse_function(self, example_proto):
        features = {'blur_image_x4_raw': tf.FixedLenFeature((), tf.string),
                'blur_image_x2_raw': tf.FixedLenFeature((), tf.string),
                'blur_image_x1_raw': tf.FixedLenFeature((), tf.string),
                'sharp_image_x4_raw': tf.FixedLenFeature((), tf.string),
                'sharp_image_x2_raw': tf.FixedLenFeature((), tf.string),
                'sharp_image_x1_raw': tf.FixedLenFeature((), tf.string),
                'height': tf.FixedLenFeature((), tf.int64),
                'width': tf.FixedLenFeature((), tf.int64)}
        parsed_features = tf.parse_single_example(example_proto, features)

        width = parsed_features['width']
        height = parsed_features['height']

        blur_images = {}
        blur_images['x4'] = tf.decode_raw(parsed_features['blur_image_x4_raw'], tf.float32)
        blur_images['x4'] = tf.reshape(blur_images['x4'], [height // 4, width // 4, 3])
        blur_images['x2'] = tf.decode_raw(parsed_features['blur_image_x2_raw'], tf.float32)
        blur_images['x2'] = tf.reshape(blur_images['x2'], [height // 2, width // 2, 3])
        blur_images['x1'] = tf.decode_raw(parsed_features['blur_image_x1_raw'], tf.float32)
        blur_images['x1'] = tf.reshape(blur_images['x1'], [height, width, 3])

        sharp_images = {}
        sharp_images['x4'] = tf.decode_raw(parsed_features['sharp_image_x4_raw'], tf.float32)
        sharp_images['x4'] = tf.reshape(sharp_images['x4'], [height // 4, width // 4, 3])
        sharp_images['x2'] = tf.decode_raw(parsed_features['sharp_image_x2_raw'], tf.float32)
        sharp_images['x2'] = tf.reshape(sharp_images['x2'], [height // 2, width // 2, 3])
        sharp_images['x1'] = tf.decode_raw(parsed_features['sharp_image_x1_raw'], tf.float32)
        sharp_images['x1'] = tf.reshape(sharp_images['x1'], [height, width, 3])

        return blur_images, sharp_images

    def create_train_dataset(self):
        dataset = tf.data.TFRecordDataset(self.train_tfrecord_path, num_parallel_reads=4)
        dataset = dataset.map(self._train_parse_function, num_parallel_calls=4)

        if self.load_on_memory:
            dataset = dataset.cache()

        dataset = dataset.shuffle(10000)
        dataset = dataset.repeat(None)
        dataset = dataset.batch(self.num_batch)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

    def create_test_dataset(self, num_sample=None):
        dataset = tf.data.TFRecordDataset(self.test_tfrecord_path)
        dataset = dataset.map(self._test_parse_function, num_parallel_calls=4)

        if num_sample is not None:
            dataset = dataset.take(num_sample)

        if self.load_on_memory:
            dataset = dataset.cache()

        dataset = dataset.repeat(1)
        dataset = dataset.batch(1)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

if __name__ == '__main__':
    tf.enable_eager_execution()
    with tf.device('/cpu:0'):
        dataset = TFRecordDataset(args)
        train_dataset = dataset.create_train_dataset()
        valid_dataset = dataset.create_test_dataset()

        for batch in train_dataset.take(1):
            print(tf.shape(batch[0]['x4']))
            print(tf.shape(batch[0]['x2']))
            print(tf.shape(batch[0]['x1']))

        for batch in valid_dataset.take(1):
            print(tf.shape(batch[0]['x4']))
            print(tf.shape(batch[0]['x2']))
            print(tf.shape(batch[0]['x1']))
