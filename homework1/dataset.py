import tensorflow as tf   # only work from tensorflow==1.9.0-rc1 and after
import os, glob

class TFRecordDataset():
    def __init__(self, args):
        self.load_on_memory = args.load_on_memory
        self.num_batch = args.num_batch
        self.train_tfrecord_path = os.path.join(args.data_dir, 'train.tfrecords')
        self.test_tfrecord_path= os.path.join(args.data_dir, 'test.tfrecords')
        assert os.path.isfile(self.train_tfrecord_path)
        assert os.path.isfile(self.test_tfrecord_path)

    def _train_parse_function(self, example_proto):
        features = {'lr_image_raw': tf.FixedLenFeature((), tf.string, default_value=""),
                'lr_height': tf.FixedLenFeature((), tf.int64),
                'lr_width': tf.FixedLenFeature((), tf.int64),
                'hr_image_raw': tf.FixedLenFeature((), tf.string, default_value=""),
                'hr_height': tf.FixedLenFeature((), tf.int64),
                'hr_width': tf.FixedLenFeature((), tf.int64)}
        parsed_features = tf.parse_single_example(example_proto, features)

        lr_image = tf.decode_raw(parsed_features['lr_image_raw'], tf.float32)
        hr_image = tf.decode_raw(parsed_features['hr_image_raw'], tf.float32)

        lr_image = tf.reshape(lr_image, [parsed_features['lr_height'], parsed_features['lr_width'], 3])
        hr_image = tf.reshape(hr_image, [parsed_features['hr_height'], parsed_features['hr_width'], 3])

        return lr_image, hr_image

    def _test_parse_function(self, example_proto):
        features = {'lr_image_raw': tf.FixedLenFeature((), tf.string, default_value=""),
                'lr_height': tf.FixedLenFeature((), tf.int64),
                'lr_width': tf.FixedLenFeature((), tf.int64),
                'hr_image_raw': tf.FixedLenFeature((), tf.string, default_value=""),
                'hr_height': tf.FixedLenFeature((), tf.int64),
                'hr_width': tf.FixedLenFeature((), tf.int64)}
        parsed_features = tf.parse_single_example(example_proto, features)

        lr_image = tf.decode_raw(parsed_features['lr_image_raw'], tf.float32)
        hr_image = tf.decode_raw(parsed_features['hr_image_raw'], tf.float32)

        lr_image = tf.reshape(lr_image, [parsed_features['lr_height'], parsed_features['lr_width'], 3])
        hr_image = tf.reshape(hr_image, [parsed_features['hr_height'], parsed_features['hr_width'], 3])

        return lr_image, hr_image

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
    """
    with tf.device('/cpu:0'):
        dataset = BigbuckbunnyV0_preprocess(args, 3)
        train_dataset = dataset.create_train_dataset()
        valid_dataset = dataset.create_test_dataset()

        for batch in train_dataset.take(1):
            print(tf.shape(batch[0]))
            print(tf.shape(batch[1]))
    """
