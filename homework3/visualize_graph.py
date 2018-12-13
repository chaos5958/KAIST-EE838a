import tensorflow as tf

from model import Model
from option import args

with tf.Graph().as_default():
    init = tf.global_variables_initializer()
    input_images = {}
    x4 = tf.placeholder(tf.float32, [1, 64, 64, 3])
    x2 = tf.placeholder(tf.float32, [1, 128, 128, 3])
    x1 = tf.placeholder(tf.float32, [1, 256, 256, 3])
    input_images['x4'] = x4
    input_images['x2'] = x2
    input_images['x1'] = x1
    model = Model()
    output_image = model(input_images)
    with tf.Session() as sess:
        sess.run(init)
        summary_writer = tf.summary.FileWriter(args.board_dir, sess.graph)
