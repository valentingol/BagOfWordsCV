import tensorflow as tf
from tensorflow import keras
kl = keras.layers

class Mobilenet(keras.Model):
    def __init__(self, dense_unit, final_dropout, mobilenet_dropout=0.002):
        mobilenet = tf.keras.applications.MobileNet(
            input_shape=None,
            alpha=1.0,
            include_top=False,
            weights="imagenet",
            dropout=mobilenet_dropout
            )
        self.model = tf.keras.Sequential([
            mobilenet,
            kl.Flatten(),
            kl.Dense(dense_unit, activation='relu'),
            kl.Dropout(final_dropout),
            kl.Dense(1)
        ])

    def call(self, x, training=False):
        return self.model(x, training=training)

class VGGBlock(kl.Layer):
    def __init__(self, channels, dropout_rate):
        super().__init__()
        self.channels = channels
        self.dropout_rate = dropout_rate
        self.conv1 = kl.Conv2D(self.channels, 3, padding='same',
                               activation='relu',
                               kernel_initializer='he_normal')
        self.conv2 = kl.Conv2D(self.channels, 3, padding='same',
                               activation='relu',
                               kernel_initializer='he_normal')
        self.bn1 = kl.BatchNormalization()
        self.bn2 = kl.BatchNormalization()

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = kl.MaxPooling2D(2)(x)
        x = kl.Dropout(self.dropout_rate)(x, training)
        return x

class VGG(keras.Model):
    def __init__(self, blocks_channels, blocks_dropout_rates,
                 dense_units, final_dropout, num_classes):
        super().__init__()
        self.blocks = []
        self.final_dropout = final_dropout
        for channels, dropout in zip(blocks_channels, blocks_dropout_rates):
            self.blocks.append(VGGBlock(channels, dropout))
        self.lin1 = kl.Dense(dense_units, activation='relu',
                             kernel_initializer='he_normal')
        self.lin2 = kl.Dense(num_classes, kernel_initializer='glorot_normal')
        self.bn = kl.BatchNormalization()

    def call(self, x, training=False):
        for block in self.blocks:
            x = block(x, training)
        x = kl.GlobalAveragePooling2D()(x)
        x = kl.Flatten()(x)
        x = self.lin1(x)
        x = self.bn(x)
        x = kl.Dropout(self.final_dropout)(x, training)
        x = self.lin2(x)  # logits
        return x


class CNN (keras.Model):
    def __init__(self, channels_list, dropout_list, dense_units, num_classes):
        super().__init__()
        self.dropout_list = dropout_list
        self.convs = []
        for channel in channels_list:
            self.convs.append(kl.Conv2D(channel, 3, activation='relu',
                                        kernel_initializer='he_normal'))
        self.lin1 = kl.Dense(dense_units, activation='relu', kernel_initializer='he_normal')
        self.lin2 = kl.Dense(num_classes, kernel_initializer='glorot_normal')

    def call(self, x, training=False):
        for conv, dropout in zip(self.convs, self.dropout_list):
            x = conv(x)
            x = kl.Dropout(dropout)(x, training)
            x = kl.MaxPooling2D(2)(x)

        x = kl.Flatten()(x)
        x = self.lin1(x)
        x = self.lin2(x)
        return x
