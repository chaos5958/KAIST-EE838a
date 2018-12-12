import tensorflow as tf
tf.enable_eager_execution()
import sys, time

from trainer import Trainer
from option import args
from model import Model
from dataset import TFRecordDataset

def loss_func(loss_type):
    assert loss_type in ['l1', 'l2']

    if loss_type == 'l1':
        return tf.losses.absolute_difference
    elif loss_type == 'l2':
        return tf.losses.mean_squared_error

model = Model()
loss = loss_func(args.loss_type)
dataset = TFRecordDataset(args)
trainer = Trainer(args, model, dataset, loss)

for epoch in range(args.num_epoch):
    print('[Train-{}epoch] Start'.format(epoch))
    start_time = time.time()
    trainer.train()
    print('[Train-{}epoch takes {} seconds] End'.format(epoch, time.time() - start_time))
    print('[Validation-{}epoch] Start'.format(epoch))
    start_time = time.time()
    trainer.validate()
    print('[Validation-{}epoch takes {} seconds] End'.format(epoch, time.time() - start_time))
    print('[Visualization-{}epoch] Start'.format(epoch))
    start_time = time.time()
    trainer.visualize(args.num_sample)
    print('[Visualization-{}epoch takes {} seconds] '.format(epoch, time.time() - start_time))
    trainer.save_model()

    #if epoch != 0 and epoch % args.lr_decay_epoch == 0:
    #    trainer.apply_lr_decay()
