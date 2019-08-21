from keras.layers import Input
from keras.layers.core import Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation, Dropout, SpatialDropout2D
from keras.optimizers import Adam
from keras.optimizers import Adam
from keras.models import load_model
from keras import backend
from keras import layers
import pickle

class Models:
    
    def save_history(self, path, history):
        with open(path, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    
    def load_model(self, path):
        model = load_model(path)
        return model
    
    def batch_conv_layer(self, feature_size, inputs):
        conv1 = Conv2D(feature_size, (3, 3), activation=None, padding='same')(inputs)
        BN1 = BatchNormalization()(conv1)
        act1 = Activation('relu')(BN1)
        
        return act1
    
    def _make_divisible(self, v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v
    
    def _depthwise_conv_block(self, inputs, pointwise_conv_filters, alpha, depth_multiplier=1, strides=(1, 1), block_id=1):
        """Adds a depthwise convolution block.
        A depthwise convolution block consists of a depthwise conv,
        batch normalization, relu6, pointwise convolution,
        batch normalization and relu6 activation.
        # Arguments
            inputs: Input tensor of shape `(rows, cols, channels)`
                (with `channels_last` data format) or
                (channels, rows, cols) (with `channels_first` data format).
            pointwise_conv_filters: Integer, the dimensionality of the output space
                (i.e. the number of output filters in the pointwise convolution).
            alpha: controls the width of the network.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                     are used at each layer.
            depth_multiplier: The number of depthwise convolution output channels
                for each input channel.
                The total number of depthwise convolution output
                channels will be equal to `filters_in * depth_multiplier`.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution
                along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.
                Specifying any stride value != 1 is incompatible with specifying
                any `dilation_rate` value != 1.
            block_id: Integer, a unique identification designating
                the block number.
        # Input shape
            4D tensor with shape:
            `(batch, channels, rows, cols)` if data_format='channels_first'
            or 4D tensor with shape:
            `(batch, rows, cols, channels)` if data_format='channels_last'.
        # Output shape
            4D tensor with shape:
            `(batch, filters, new_rows, new_cols)`
            if data_format='channels_first'
            or 4D tensor with shape:
            `(batch, new_rows, new_cols, filters)`
            if data_format='channels_last'.
            `rows` and `cols` values might have changed due to stride.
        # Returns
            Output tensor of block.
        """
        channel_axis =-1
        
        in_channels = backend.int_shape(inputs)[channel_axis]
        pointwise_conv_filters = int(pointwise_conv_filters * alpha)

        if strides == (1, 1):
            x = inputs
        else:
            x = layers.ZeroPadding2D(((0, 1), (0, 1)),
                                     name='conv_pad_%d' % block_id)(inputs)
        x = layers.DepthwiseConv2D((3, 3),
                                   padding='same' if strides == (1, 1) else 'valid',
                                   depth_multiplier=depth_multiplier,
                                   strides=strides,
                                   use_bias=False,
                                   name='conv_dw_%d' % block_id)(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
        x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

        x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                          padding='same',
                          use_bias=False,
                          strides=(1, 1),
                          name='conv_pw_%d' % block_id)(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      name='conv_pw_%d_bn' % block_id)(x)
        return layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)

    def _inverted_res_block(self, inputs, expansion, stride, alpha, filters, block_id):
        channel_axis =-1

        in_channels = backend.int_shape(inputs)[channel_axis]
        pointwise_conv_filters = int(filters * alpha)
        pointwise_filters = self._make_divisible(pointwise_conv_filters, 8)
        x = inputs
        prefix = 'block_{}_'.format(block_id)

        if block_id:
            # Expand
            x = layers.Conv2D(expansion * in_channels,
                              kernel_size=1,
                              padding='same',
                              use_bias=False,
                              activation=None,
                              name=prefix + 'expand')(x)
            x = layers.BatchNormalization(axis=channel_axis,
                                          epsilon=1e-3,
                                          momentum=0.999,
                                          name=prefix + 'expand_BN')(x)
            x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
        else:
            prefix = 'expanded_conv_'

        # Depthwise
        if stride == 2:
            x = layers.ZeroPadding2D(padding=correct_pad(backend, x, 3),
                                     name=prefix + 'pad')(x)
        x = layers.DepthwiseConv2D(kernel_size=3,
                                   strides=stride,
                                   activation=None,
                                   use_bias=False,
                                   padding='same' if stride == 1 else 'valid',
                                   name=prefix + 'depthwise')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      epsilon=1e-3,
                                      momentum=0.999,
                                      name=prefix + 'depthwise_BN')(x)

        x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

        # Project
        x = layers.Conv2D(pointwise_filters,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None,
                          name=prefix + 'project')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      epsilon=1e-3,
                                      momentum=0.999,
                                      name=prefix + 'project_BN')(x)

        if in_channels == pointwise_filters and stride == 1:
            return layers.Add(name=prefix + 'add')([inputs, x])
        return x
    
    def get_unet_model_1(self, input_shape, classes, loss, metrics):
        inputs = Input(shape=input_shape)#, dtype=tf.float16)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)  #sigmoid
        #conv10 = Conv2D(classes, (1, 1), padding="valid")(conv9)
        #reshape1 = Reshape((input_shape[0]*input_shape[1], classes))(conv10)
        #x = Activation("softmax")(reshape1)

        model = Model(inputs=[inputs], outputs=[conv10])
        
        model.compile(optimizer=Adam(lr=1e-5), loss=loss, metrics=metrics)

        return model
    
    def get_unet_model_2(self, input_shape, classes, loss, metrics):
        inputs = Input(shape=input_shape)#, dtype=tf.float16)
        conv1 = self.batch_conv_layer(32, inputs)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = self.batch_conv_layer(64, pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = self.batch_conv_layer(128, pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = self.batch_conv_layer(256, pool3)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=3)
        conv7 = self.batch_conv_layer(128, up7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = self.batch_conv_layer(64, up8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = self.batch_conv_layer(32, up9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)  #sigmoid

        model = Model(inputs=[inputs], outputs=[conv10])
        
        model.compile(optimizer=Adam(lr=1e-5), loss=loss, metrics=metrics)

        return model
    
    def get_unet_model_3(self, input_shape, classes, loss, metrics):
        inputs = Input(shape=input_shape)#, dtype=tf.float16)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)  #sigmoid

        model = Model(inputs=[inputs], outputs=[conv10])
        
        model.compile(optimizer=Adam(lr=1e-5), loss=loss, metrics=metrics)

        return model
    
    def get_unet_model_4(self, input_shape, classes, loss, metrics):
        inputs = Input(shape=input_shape)#, dtype=tf.float16)
        conv1 = self.batch_conv_layer(32, inputs)
        conv1 = self.batch_conv_layer(32, conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.batch_conv_layer(64, pool1)
        conv2 = self.batch_conv_layer(64, conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.batch_conv_layer(128, pool2)
        conv3 = self.batch_conv_layer(128, conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self.batch_conv_layer(256, pool3)
        conv4 = self.batch_conv_layer(256, conv4)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=3)
        conv7 = self.batch_conv_layer(128, up7)
        conv7 = self.batch_conv_layer(128, conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = self.batch_conv_layer(64, up8)
        conv8 = self.batch_conv_layer(64, conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = self.batch_conv_layer(32, up9)
        conv9 = self.batch_conv_layer(32, conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)  #sigmoid

        model = Model(inputs=[inputs], outputs=[conv10])
        
        model.compile(optimizer=Adam(lr=1e-5), loss=loss, metrics=metrics)

        return model
    
    def get_unet_model_5(self, input_shape, classes, loss, metrics):
        inputs = Input(shape=input_shape)#, dtype=tf.float16)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)  #sigmoid

        model = Model(inputs=[inputs], outputs=[conv10])
        
        model.compile(optimizer=Adam(lr=1e-5), loss=loss, metrics=metrics)

        return model
    
    def get_unet_model_6(self, input_shape, classes, loss, metrics):
        inputs = Input(shape=input_shape)#, dtype=tf.float16)
        conv1 = self.batch_conv_layer(32, inputs)
        conv1 = self.batch_conv_layer(32, conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.batch_conv_layer(64, pool1)
        conv2 = self.batch_conv_layer(64, conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.batch_conv_layer(128, pool2)
        conv3 = self.batch_conv_layer(128, conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self.batch_conv_layer(256, pool3)
        conv4 = self.batch_conv_layer(256, conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        
        conv5 = self.batch_conv_layer(512, pool4)
        conv5 = self.batch_conv_layer(512, conv5)
        
        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = self.batch_conv_layer(256, up6)
        conv6 = self.batch_conv_layer(256, conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = self.batch_conv_layer(128, up7)
        conv7 = self.batch_conv_layer(128, conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = self.batch_conv_layer(64, up8)
        conv8 = self.batch_conv_layer(64, conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = self.batch_conv_layer(32, up9)
        conv9 = self.batch_conv_layer(32, conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)  #sigmoid

        model = Model(inputs=[inputs], outputs=[conv10])
        
        model.compile(optimizer=Adam(lr=1e-5), loss=loss, metrics=metrics)
        
        return model
    
    def get_unet_model_7(self, input_shape, classes, loss, metrics):
        #Mobile net v2 medium
        inputs = Input(shape=input_shape)#, dtype=tf.float16)
        conv1 = self.batch_conv_layer(32, inputs)
        conv1 = self.batch_conv_layer(32, conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self._inverted_res_block(pool1, filters=64, alpha=1.0, stride=1,expansion=6, block_id=1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self._inverted_res_block(pool2, filters=128, alpha=1.0, stride=1,expansion=6, block_id=3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self._inverted_res_block(pool3, filters=256, alpha=1.0, stride=1,expansion=6, block_id=5)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=3)
        conv7 = self._inverted_res_block(up7, filters=128, alpha=1.0, stride=1,expansion=6, block_id=11)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = self._inverted_res_block(up8, filters=64, alpha=1.0, stride=1,expansion=6, block_id=13)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = self.batch_conv_layer(32, up9)
        conv9 = self.batch_conv_layer(32, conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)  #sigmoid

        model = Model(inputs=[inputs], outputs=[conv10])
        
        model.compile(optimizer=Adam(lr=1e-5), loss=loss, metrics=metrics)

        return model
    
    def get_unet_model_8(self, input_shape, classes, loss, metrics):
        #Mobile net v2 large
        inputs = Input(shape=input_shape)#, dtype=tf.float16)
        conv1 = self.batch_conv_layer(32, inputs)
        conv1 = self.batch_conv_layer(32, conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self._inverted_res_block(pool1, filters=64, alpha=1.0, stride=1,expansion=6, block_id=1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self._inverted_res_block(pool2, filters=128, alpha=1.0, stride=1,expansion=6, block_id=3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self._inverted_res_block(pool3, filters=256, alpha=1.0, stride=1,expansion=6, block_id=5)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        
        conv5 = self._inverted_res_block(pool4, filters=512, alpha=1.0, stride=1,expansion=6, block_id=7)
        
        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = self._inverted_res_block(up6, filters=256, alpha=1.0, stride=1,expansion=6, block_id=9)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = self._inverted_res_block(up7, filters=128, alpha=1.0, stride=1,expansion=6, block_id=11)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = self._inverted_res_block(up8, filters=64, alpha=1.0, stride=1,expansion=6, block_id=13)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = self.batch_conv_layer(32, up9)
        conv9 = self.batch_conv_layer(32, conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)  #sigmoid

        model = Model(inputs=[inputs], outputs=[conv10])
        
        model.compile(optimizer=Adam(lr=1e-5), loss=loss, metrics=metrics)

        return model
    
    def get_unet_model_9(self, input_shape, classes, loss, metrics):
        # Mobile net v1 small
        inputs = Input(shape=input_shape)#, dtype=tf.float16)
        conv1 = self.batch_conv_layer(32, inputs)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = self._depthwise_conv_block(pool1, 64, alpha=1.0, depth_multiplier=1, block_id=1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = self._depthwise_conv_block(pool2, 128, alpha=1.0, depth_multiplier=1, block_id=2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = self._depthwise_conv_block(pool3, 256, alpha=1.0, depth_multiplier=1, block_id=3)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=3)
        conv7 = self._depthwise_conv_block(up7, 128, alpha=1.0, depth_multiplier=1, block_id=4)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = self._depthwise_conv_block(up8, 64, alpha=1.0, depth_multiplier=1, block_id=5)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = self.batch_conv_layer(32, up9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)  #sigmoid

        model = Model(inputs=[inputs], outputs=[conv10])
        
        model.compile(optimizer=Adam(lr=1e-5), loss=loss, metrics=metrics)

        return model
    
    def get_unet_model_10(self, input_shape, classes, loss, metrics):
        # Mobile net v1 medium
        inputs = Input(shape=input_shape)#, dtype=tf.float16)
        conv1 = self.batch_conv_layer(32, inputs)
        conv1 = self.batch_conv_layer(32, conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self._depthwise_conv_block(pool1, 64, alpha=1.0, depth_multiplier=1, block_id=1)
        conv2 = self._depthwise_conv_block(conv2, 64, alpha=1.0, depth_multiplier=1, block_id=2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self._depthwise_conv_block(pool2, 128, alpha=1.0, depth_multiplier=1, block_id=3)
        conv3 = self._depthwise_conv_block(conv3, 128, alpha=1.0, depth_multiplier=1, block_id=4)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self._depthwise_conv_block(pool3, 256, alpha=1.0, depth_multiplier=1, block_id=5)
        conv4 = self._depthwise_conv_block(conv4, 256, alpha=1.0, depth_multiplier=1, block_id=6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=3)
        conv7 = self._depthwise_conv_block(up7, 128, alpha=1.0, depth_multiplier=1, block_id=7)
        conv7 = self._depthwise_conv_block(conv7, 128, alpha=1.0, depth_multiplier=1, block_id=8)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = self._depthwise_conv_block(up8, 64, alpha=1.0, depth_multiplier=1, block_id=9)
        conv8 = self._depthwise_conv_block(conv8, 64, alpha=1.0, depth_multiplier=1, block_id=10)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = self.batch_conv_layer(32, up9)
        conv9 = self.batch_conv_layer(32, conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)  #sigmoid

        model = Model(inputs=[inputs], outputs=[conv10])
        
        model.compile(optimizer=Adam(lr=1e-5), loss=loss, metrics=metrics)

        return model