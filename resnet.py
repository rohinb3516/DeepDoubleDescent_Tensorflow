import tensorflow as tf

from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, MaxPool2D, AveragePooling2D, Softmax, BatchNormalization
)
from tensorflow.keras import Model, Sequential

def make_resnet18(input_shape, k=64, num_classes=10):
    ''' Returns a ResNet18 with width parameter k.'''
    
    # Model_id to identify model set up when using tensorflow checkpoints. 
    model_id = f'ResNet18_width_scale_{k}'
    
    residual_block_params = [
        {'n_filters': k, 'block_depth': 2, 'stride': 1},
        {'n_filters': 2*k, 'block_depth': 2, 'stride': 2},
        {'n_filters': 4*k, 'block_depth': 2, 'stride': 2},
        {'n_filters': 8*k, 'block_depth': 2, 'stride': 2},
    ]
    return ResNet(input_shape, residual_block_params, num_classes, init_channels=k), model_id
 

class ResNet(Model):
    def __init__(self, input_shape, residual_block_params, n_classes=10, init_channels=64):
        '''
            Parameters
            ----------
                input_shape - int
                    Input dimensions of image data
                residual_block_params - dict
                    List of dict specifying # filters, # residual blocks, and stride for each residual block in the network.  
                    e.g. 
                        [{'n_filters': 16, 'block_depth':3, 'stride': 1}, {'n_filters': 32, 'block_depth':3, 'stride': 1}]
                N_Classes - int
                    Output dimension of the final softmax layer.
                filter_n_0 - int
                    Initial number of filters in the network prior to the Residual block layers. 
                    
            Note: This implimentation varies slightly from that of Deep double descent and follows the architecture of 
                  Deep Residual Learning for Image Recognition, 2015 (https://arxiv.org/pdf/1512.03385.pdf)
        '''
        
        super(ResNet, self).__init__()     
        
        self.n_classes = n_classes
        
        self.conv_1 = Conv2D(filters=init_channels, kernel_size=(3, 3), strides=1, padding="same")
        self.batch_norm_1 = BatchNormalization()
        self.residual_blocks = []
      
        # Initialize the residual block layers using the parameter dictionaries.
        for param_dict in residual_block_params:
            self.residual_blocks.append(
                self._make_residual_block_layer(**param_dict)
            )

        self.avgpool = AveragePooling2D(pool_size=4)
        self.flatten = Flatten()
        self.softmax = Dense(units=self.n_classes, activation='softmax')
        
    def _make_residual_block_layer(self, n_filters, block_depth, stride=1):
        # Define a sequential network which is composed of sequential residual blocks with the same # of filters 
        
        res_block = tf.keras.Sequential()
        res_block.add(ResidualBlock(n_filters, stride=stride))

        for _ in range(block_depth):
            res_block.add(ResidualBlock(n_filters, stride=1))

        return res_block
                
        
    def call(self, inputs, training=None): 
        # training is used for layers which utilize Batch Normalization.
        
        x = self.conv_1(inputs)
        x = self.batch_norm_1(x, training=training)
        x = tf.nn.relu(x)
        
        # Pass through residual blocks.
        for residual_block in self.residual_blocks:
            x = residual_block(x, training=training)
            
        x = self.avgpool(x)
        x = self.flatten(x)
        output = self.softmax(x)

        return output


class ResidualBlock(tf.keras.layers.Layer):
    '''
        Impliments A single Residual block component of a ResNet.
    '''
    
    def __init__(self, n_filters, stride=1):
        """
            Parameters
            ----------
                n_filters - int 
                    Width of the residual block. Number of channels for block convolutions.
                stride - int
                    The stride used in the convolution layer. When 1 < stride, then the convolutions perform downsampling.
        """

        super(ResidualBlock, self).__init__()
        
        self.conv_1 = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=stride, padding="same")
        self.batch_norm_1 = BatchNormalization()
                
        self.conv_2 = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=1, padding="same")
        self.batch_norm_2 = BatchNormalization()
        
        # This is done for layers which reduce the filter dimensions. 
        # Use 1x1 convolutions when downsampling, and identity map otherwise.
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(Conv2D(filters=n_filters, kernel_size=(1, 1), strides=stride))
            self.downsample.add(BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)
        x = self.conv_1(inputs)
        x = self.batch_norm_1(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.conv_2(x)
        x = self.batch_norm_2(x, training=training)   
        
        output = tf.nn.relu(tf.keras.layers.add([residual, x]))
        return output