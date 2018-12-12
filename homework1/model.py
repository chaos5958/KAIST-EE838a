import tensorflow as tf

l2 = tf.keras.regularizers.l2

class ConvBlock(tf.keras.Model):
    def __init__(self, num_filters, kernel_size, weight_decay, add_act):
        super(ConvBlock, self).__init__()
        self.add_act = add_act
        self.conv1 = tf.keras.layers.Conv2D(num_filters,
                                        (kernel_size,kernel_size),
                                        padding='same',
                                        kernel_regularizer=l2(weight_decay))

        if self.add_act:
            self.relu1 = tf.keras.layers.ReLU()

    def call(self, x):
        output = self.conv1(x)

        if self.add_act:
            output = self.relu1(output)

        return output

class ResBlock(tf.keras.Model):
    def __init__(self, num_filters, kernel_size, weight_decay):
        super(ResBlock, self).__init__()
        self.conv_block1 = ConvBlock(num_filters, kernel_size, weight_decay, True)
        self.conv_block2 = ConvBlock(num_filters, kernel_size, weight_decay, False)

    def call(self, x):
        input = x
        output = self.conv_block1(x)
        output = self.conv_block2(output)
        output = input + output

        return output

def _subpixel_function(x):
    input = x[0]
    scale = x[1]
    return tf.depth_to_space(input, scale)

class Model(tf.keras.Model):
    def __init__(self, weight_decay=1e-04):
        super(Model, self).__init__()
        self.conv_head = ConvBlock(64, 7, weight_decay, True)
        self.resblocks = []

        for _ in range(4):
            self.resblocks.append(ResBlock(64, 3, weight_decay))

        self.conv_body1 = ConvBlock(64, 3, weight_decay, False)
        self.conv_body2 = ConvBlock(256, 3, weight_decay, False)
        self.subpixel = tf.keras.layers.Lambda(_subpixel_function)
        self.relu_tail = tf.keras.layers.ReLU()
        self.conv_tail = ConvBlock(3, 7, weight_decay, False)

    def call(self, x):
        input = self.conv_head(x) #used for global residual connections
        output = input

        for resblock in self.resblocks:
            output = resblock(output)

        output = self.conv_body1(output)
        output = input + output
        output = self.conv_body2(output)
        output = self.subpixel((output, 2))
        output = self.relu_tail(output)
        output = self.conv_tail(output)

        return output

if __name__ == "__main__":
    tf.enable_eager_execution()
    with tf.device('gpu:0'):
        model = Model()
        x = tf.random_normal([1, 100, 100, 3])
        y = model(x)
        print(tf.shape(y))
