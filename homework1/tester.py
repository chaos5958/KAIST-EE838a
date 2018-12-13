import tensorflow as tf
import os, time, glob, sys
import random, ntpath
from importlib import import_module

import ops

tfe = tf.contrib.eager

def create_test_dataset(filenames):
    dataset = tf.data.Dataset.from_tensor_slices((filenames))
    dataset = dataset.map(ops.load_image, num_parallel_calls=4)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset

class Tester():
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.learning_rate = tfe.Variable(self.args.lr, dtype=tf.float32)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.root = tf.train.Checkpoint(optimizer=self.optimizer,
                            model=self.model,
                            optimizer_step=tf.train.get_or_create_global_step())

        lr_image_filenames = glob.glob('{}/LR/*.jpg'.format(args.test_dir))
        assert len(lr_image_filenames) != 0
        self.test_dataset = create_test_dataset(lr_image_filenames)
        self.save_dir = os.path.join(self.args.test_dir, 'SR')
        os.makedirs(self.save_dir, exist_ok=True)

    def load_model(self, checkpoint_path=None):
        if checkpoint_path is not None:
            self.root.restore(checkpoint_path)
        else:
            self.root.restore(tf.train.latest_checkpoint(self.args.model_dir))

    def test(self):
        with tf.device('gpu:{}'.format(self.args.gpu_idx)):
            for idx, input_image in enumerate(self.test_dataset):
                output_image = self.model(input_image)
                output_image = tf.clip_by_value(output_image, 0.0, 1.0)
                output_image = output_image * 255.0
                output_image = tf.cast(output_image, tf.uint8)
                output_image = tf.squeeze(output_image)

                with tf.device('cpu:0'):
                    output_image = tf.image.encode_png(output_image)
                    tf.io.write_file(os.path.join(self.save_dir, '{}.png'.format(idx)), output_image)
