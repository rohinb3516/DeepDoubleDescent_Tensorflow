import tensorflow as tf
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPool2D,
    BatchNormalization,
    ReLU,
)


def make_convNet(
    input_shape, depth, n_classes=10, init_channels=64, layer_initializer=None
):
    """
    Returns A tensorflow Sequential Model with depth-1 Convolutional layers, and a final Softmax output layer.

    Parameters
    ----------
        input_shape - list
            Input dimensions of image data
        depth - int
            Number of layers in the network (including the dense output layer)
        N_Classes - int
            Output dimension of the final softmax layer.
        init_channels - int
            Number of filters in the network at layer 0.
        layer_initializer - str or tf.keras.initializer
            specify which method to use in initializing the conv net.

    Note: Depth will be limited by the input dimension, as the dimensions are halved after each layer.
    """
    conv_net = Sequential()

    if depth < 2:
        raise Exception("Conv Net Depth Must be greater than or equal to 2.")

    layer_init = layer_initializer if layer_initializer is not None else "he_uniform"

    # Each dimmension divides the input image dimension by 2 and doubles the # channels.
    conv_shapes = [input_shape] + [
        [
            input_shape[1] // (2 ** (i)),
            input_shape[2] // (2 ** (i)),
            init_channels * (2 ** i),
        ]
        for i in range(depth - 2)
    ]

    # for each layer apply a 3x3 convolution, batch normzliation, and relu. Use Max pooling after the first layer.
    for i in range(depth - 1):
        n_filters = init_channels * (2 ** (i))

        conv_net.add(
            Conv2D(
                filters=n_filters,
                input_shape=conv_shapes[i],
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                kernel_initializer=layer_init,
            )
        )
        conv_net.add(BatchNormalization(momentum=0.9, epsilon=1e-5, renorm=True))
        conv_net.add(ReLU())

        # This delays max pooling until the final 4 layers. After 4 layers of 2x2 (stride 2x2) max pooling the image dimension goes from 32 x 32 to 4x4.
        if depth - 5 < i:
            conv_net.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    conv_net.add(MaxPool2D(pool_size=(4, 4), strides=(4, 4)))
    conv_net.add(ReLU())
    conv_net.add(Flatten())
    conv_net.add(
        Dense(units=n_classes, activation="softmax", kernel_initializer=layer_init)
    )

    # used in identifying the model later on
    model_id = f"conv_net_depth_{depth}_width_{init_channels}"

    return conv_net, model_id
