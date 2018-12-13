import tensorflow as tf
import os, time, glob, sys
import random
from importlib import import_module

tfe = tf.contrib.eager

class Trainer():
    def __init__(self, args, model, dataset):
        self.args = args
        self.model = model
        self.learning_rate = tfe.Variable(self.args.lr, dtype=tf.float32)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.root = tf.train.Checkpoint(optimizer=self.optimizer,
                            model=self.model,
                            optimizer_step=tf.train.get_or_create_global_step())
        self.checkpoint_dir = os.path.join(self.args.model_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        board_dir = os.path.join(self.args.board_dir)
        os.makedirs(board_dir, exist_ok=True)
        self.writer = tf.contrib.summary.create_file_writer(board_dir)

        self.train_dataset = dataset.create_train_dataset()
        self.valid_dataset = dataset.create_test_dataset()

        self.training_loss = tfe.metrics.Mean("Training Loss")
        self.validation_loss = tfe.metrics.Mean("Validation Loss")
        self.validation_psnr = {}
        self.validation_psnr['x1'] = tfe.metrics.Mean("Validation PSNR (x1)")
        self.validation_psnr['x2'] = tfe.metrics.Mean("Validation PSNR (x2)")
        self.validation_psnr['x4'] = tfe.metrics.Mean("Validation PSNR (x4)")

    def apply_lr_decay(self):
        self.learning_rate.assign(self.learning_rate / 2)

    def load_model(self, checkpoint_path=None):
        if checkpoint_path is None:
            self.root.restore(checkpoint_path)
        else:
            self.root.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    def loss(self, output_images, target_images):
        loss_x1 = tf.losses.mean_squared_error(output_images['x1'], target_images['x1'])
        loss_x2 = tf.losses.mean_squared_error(output_images['x2'], target_images['x2'])
        loss_x4 = tf.losses.mean_squared_error(output_images['x4'], target_images['x4'])

        total_loss = (loss_x1 + loss_x2 + loss_x4) / 6
        return total_loss

    def save_model(self):
        checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        self.root.save(checkpoint_prefix)

    def train(self):
        with self.writer.as_default(), tf.contrib.summary.always_record_summaries(), tf.device('gpu:{}'.format(self.args.gpu_idx)):
            for input_images, target_images in self.train_dataset.take(self.args.num_batch_per_epoch):
                with tf.GradientTape() as tape:
                    output_images = self.model(input_images)

                    loss_value = self.loss(output_images, target_images)

                grads = tape.gradient(loss_value, self.model.variables)
                self.optimizer.apply_gradients(zip(grads, self.model.variables),
                                        global_step=tf.train.get_or_create_global_step())
                self.training_loss(loss_value)

            tf.contrib.summary.scalar('Average Traning Loss', self.training_loss.result())
            tf.contrib.summary.scalar('Learning rate', self.learning_rate)
            tf.contrib.summary.flush(self.writer)

            self.training_loss.init_variables()

    def validate(self):
        with self.writer.as_default(), tf.contrib.summary.always_record_summaries(), tf.device('gpu:{}'.format(self.args.gpu_idx)):
            for input_images, target_images in self.valid_dataset:
                output_images = self.model(input_images)

                output_loss_value = self.loss(output_images, target_images)
                output_images['x1'] = tf.clip_by_value(output_images['x1'], 0.0, 1.0)
                output_images['x2'] = tf.clip_by_value(output_images['x2'], 0.0, 1.0)
                output_images['x4'] = tf.clip_by_value(output_images['x4'], 0.0, 1.0)

                self.validation_loss(output_loss_value)
                self.validation_psnr['x1'](tf.image.psnr(output_images['x1'], target_images['x1'], max_val=1.0))
                self.validation_psnr['x2'](tf.image.psnr(output_images['x2'], target_images['x2'], max_val=1.0))
                self.validation_psnr['x4'](tf.image.psnr(output_images['x4'], target_images['x4'], max_val=1.0))

            tf.contrib.summary.scalar('Average Validation Loss', self.validation_loss.result())
            tf.contrib.summary.scalar('Average Validation PSNR (x1)', self.validation_psnr['x1'].result())
            tf.contrib.summary.scalar('Average Validation PSNR (x2)', self.validation_psnr['x2'].result())
            tf.contrib.summary.scalar('Average Validation PSNR (x4)', self.validation_psnr['x4'].result())
            tf.contrib.summary.flush(self.writer)

            self.validation_loss.init_variables()
            self.validation_psnr['x1'].init_variables()
            self.validation_psnr['x2'].init_variables()
            self.validation_psnr['x4'].init_variables()

    def visualize(self):
        with self.writer.as_default(), tf.contrib.summary.always_record_summaries(), tf.device('gpu:{}'.format(self.args.gpu_idx)):
            count = 0
            for input_images, target_images in self.valid_dataset:
                output_images = self.model(input_images)

                output_images['x1'] = tf.clip_by_value(output_images['x1'], 0.0, 1.0)
                output_images['x2'] = tf.clip_by_value(output_images['x2'], 0.0, 1.0)
                output_images['x4'] = tf.clip_by_value(output_images['x4'], 0.0, 1.0)

                tf.contrib.summary.image('Input{}_x1'.format(count), input_images['x1'])
                tf.contrib.summary.image('Output{}_x1'.format(count), output_images['x1'])
                tf.contrib.summary.image('Target{}_x1'.format(count), target_images['x1'])
                tf.contrib.summary.image('Input{}_x2'.format(count), input_images['x2'])
                tf.contrib.summary.image('Output{}_x2'.format(count), output_images['x2'])
                tf.contrib.summary.image('Target{}_x2'.format(count), target_images['x2'])
                tf.contrib.summary.image('Input{}_x4'.format(count), input_images['x4'])
                tf.contrib.summary.image('Output{}_x4'.format(count), output_images['x4'])
                tf.contrib.summary.image('Target{}_x4'.format(count), target_images['x4'])
                tf.contrib.summary.flush(self.writer)

                count += 1
