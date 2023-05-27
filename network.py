import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D,Activation,Layer,Conv2DTranspose
from tensorflow.python.keras import layers,Model,Sequential
import numpy as np


class ConvModel(Layer):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv = Sequential([
            Conv2D(96, 3, 1, "same",kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            Activation(tf.nn.relu),
        ])
    def call(self, inputs, *args, **kwargs):
        return self.conv(inputs)


class FfdNet(Model):
    def __init__(self,layer):
        super(FfdNet, self).__init__()
        self.layer = layer
        self.init_layer = Conv2D(96,3,1,"same",kernel_initializer='he_normal')
        self.conv = [ConvModel() for _ in range(layer)]
        self.find_layer = Conv2D(12,1,1,"same",kernel_initializer='he_normal')

    def call(self, inputs,noise,training=None, mask=None):

        x = tf.concat([inputs,noise],axis=-1)
        x = self.init_layer(x)
        for i in range(self.layer):
            x = self.conv[i](x)
        x = self.find_layer(x)
        return x

