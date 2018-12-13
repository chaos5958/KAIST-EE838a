import tensorflow as tf

def load_image(image):
    image = tf.io.read_file(image)
    image = tf.image.decode_image(image)
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image
