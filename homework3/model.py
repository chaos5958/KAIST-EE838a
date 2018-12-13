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

class SubModel(tf.keras.Model):
    def __init__(self, weight_decay=1e-04):
        super(SubModel, self).__init__()
        self.conv_head = ConvBlock(64, 5, weight_decay, True)
        self.resblocks = []

        for _ in range(9):
            self.resblocks.append(ResBlock(64, 3, weight_decay))

        self.conv_tail = ConvBlock(3, 5, weight_decay, False)

    def call(self, x):
        output = self.conv_head(x) #used for global residual connections

        for resblock in self.resblocks:
            output = resblock(output)

        output = self.conv_tail(output)

        return output

class Model(tf.keras.Model):
    def __init__(self, weight_decay=1e-04):
        super(Model, self).__init__()

        self.submodel_x4 = SubModel()
        self.transpose_x4 = tf.keras.layers.Conv2DTranspose(3,
                                                kernel_size=(3,3),
                                                strides=(2,2),
                                                padding="same"
                                                )
        self.submodel_x2 = SubModel()
        self.transpose_x2 = tf.keras.layers.Conv2DTranspose(3,
                                                kernel_size=(3,3),
                                                strides=(2,2),
                                                padding="same"
                                                )
        self.submodel_x1 = SubModel()

    def call(self, input_images):
        output_images = {}
        output_x4 = self.submodel_x4(input_images['x4'])
        x2 = tf.concat([input_images['x2'], self.transpose_x4(output_x4)], axis=3)
        output_x2 = self.submodel_x2(x2)
        x1 = tf.concat([input_images['x1'], self.transpose_x2(output_x2)], axis=3)
        output_x1 = self.submodel_x1(x1)

        output_images['x4'] = output_x4
        output_images['x2'] = output_x2
        output_images['x1'] = output_x1

        return output_images

if __name__ == "__main__":
    tf.enable_eager_execution()
    with tf.device('gpu:0'):
        model = Model()
        x_4 = tf.random_normal([1, 64, 64, 3])
        x_2 = tf.random_normal([1, 128, 128, 3])
        x_1 = tf.random_normal([1, 256, 256, 3])
        y_4, y_2, y_1 = model(x_4, x_2, x_1)
        print(tf.shape(y_4))
        print(tf.shape(y_2))
        print(tf.shape(y_1))
