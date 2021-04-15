import tensorflow as tf

from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPool2D,
    AveragePooling2D,
    BatchNormalization,
)
from tensorflow.keras import Model, Sequential


def make_resnet18_UniformHe(input_shape, k=64, num_classes=10):
    """ Returns a ResNet18 with width parameter k."""

    # Model_id to identify model set up when using tensorflow checkpoints.
    model_id = f"ResNet18_width_{k}"

    residual_block_params = [
        {"n_filters": k, "block_depth": 2, "stride": 1},
        {"n_filters": 2 * k, "block_depth": 2, "stride": 2},
        {"n_filters": 4 * k, "block_depth": 2, "stride": 2},
        {"n_filters": 8 * k, "block_depth": 2, "stride": 2},
    ]
    return (
        ResNet(input_shape, residual_block_params, num_classes, init_channels=k),
        model_id,
    )


class ResNet(Model):
    def __init__(
        self,
        input_shape,
        residual_block_params,
        n_classes=10,
        init_channels=64,
        layer_initializer=None,
    ):
        """
        Parameters
        ----------
            input_shape - list
                Input dimensions of image data
            residual_block_params - dict
                List of dict specifying # filters, # residual blocks, and stride for each residual block in the network.
                e.g.
                    [{'n_filters': 16, 'block_depth':3, 'stride': 1}, {'n_filters': 32, 'block_depth':3, 'stride': 1}]
            N_Classes - int
                Output dimension of the final softmax layer.
            init_channels - int
                Initial number of filters in the network prior to the Residual block layers.
            layer_initializer - tf.keras.initializers
                Optional Initializer to pass to the network. Default initialization is Kaiming Uniform
        """

        super(ResNet, self).__init__()

        self.n_classes = n_classes
        self._layer_init = (
            layer_initializer if layer_initializer is not None else "he_uniform"
        )

        self.conv_1 = Conv2D(
            filters=init_channels,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            input_shape=input_shape,
            kernel_initializer=self._layer_init,
            use_bias=False,
        )
        self.residual_blocks = []

        # Initialize the residual block layers using the parameter dictionaries.
        for param_dict in residual_block_params:
            self.residual_blocks.append(self._make_residual_block_layer(**param_dict))

        self.avgpool = AveragePooling2D(pool_size=4)
        self.flatten = Flatten()

        # Linear output -> Use from_logits = True with Sparse Categorical Cross Entropy.
        self.linear = Dense(units=n_classes)

    def _make_residual_block_layer(self, n_filters, block_depth, stride=1):
        # Define a sequential network which is composed of sequential residual blocks with the same # of filters

        res_block = tf.keras.Sequential()
        res_block.add(
            ResidualBlock(n_filters, stride=stride, layer_initializer=self._layer_init)
        )

        for _ in range(block_depth - 1):
            res_block.add(
                ResidualBlock(n_filters, stride=1, layer_initializer=self._layer_init)
            )

        return res_block

    def call(self, inputs, training=None, **kwargs):
        # training is used for layers which utilize Batch Normalization.

        x = self.conv_1(inputs)

        # Pass through residual blocks.
        for residual_block in self.residual_blocks:
            x = residual_block(x, training=training)

        x = self.avgpool(x)
        x = self.flatten(x)
        output = self.linear(x)

        return output


class ResidualBlock(tf.keras.layers.Layer):
    """
    Impliments A single Residual block component of a ResNet.

    The blocks are implimented as the pre-activation version of ResNets and expect the prior layer
    to have not applied either batch-normalization or ReLu activation to the output.

    Block Structure:

        Batch_Norm -> ReLu -> Convolution (3x3 / stride) -> Batch_Norm -> ReLU -> Convolution -> output
    """

    def __init__(self, n_filters, layer_initializer, stride=1):
        """
        Parameters
        ----------
            n_filters - int
                Width of the residual block. Number of channels for block convolutions.
            stride - int
                The stride used in the convolution layer. When 1 < stride, then the convolutions perform downsampling.
        """

        super(ResidualBlock, self).__init__()

        self._layer_init = (
            layer_initializer if layer_initializer is not None else "he_normal"
        )

        self.batch_norm_1 = BatchNormalization(momentum=0.9, epsilon=1e-5, renorm=True)
        self.conv_1 = Conv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            strides=stride,
            padding="same",
            kernel_initializer=self._layer_init,
            use_bias=False,
        )

        self.batch_norm_2 = BatchNormalization(momentum=0.9, epsilon=1e-5, renorm=True)
        self.conv_2 = Conv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            kernel_initializer=self._layer_init,
            use_bias=False,
        )

        # This is done on layers which reduce the image dimension.
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(
                Conv2D(
                    filters=n_filters,
                    kernel_size=(1, 1),
                    strides=stride,
                    kernel_initializer=self._layer_init,
                    use_bias=False,
                )
            )
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):

        # The input is from the Pre-activation of the prior layer.
        # Therefore we need to apply batch-norm and relu activation.
        x = self.batch_norm_1(inputs, training=training)
        x = tf.nn.relu(x)
        residual = self.downsample(x)

        x = self.conv_1(inputs)
        x = self.batch_norm_2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv_2(x)
        output = tf.keras.layers.add([residual, x])

        # We do not apply relu and batch norm to the output, as the next
        # resnet block expects pre-activation values.
        return output
