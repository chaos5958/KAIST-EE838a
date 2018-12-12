import tensorflow as tf

from model import Model
from option import args

with tf.Graph().as_default():
    init = tf.global_variables_initializer()
    input_images = tf.placeholder(tf.float32, [1, 100, 100, 3])
    model = Model()
    output_image = model(input_images)
    with tf.Session() as sess:
        sess.run(init)
        summary_writer = tf.summary.FileWriter(args.log_dir, sess.graph)
