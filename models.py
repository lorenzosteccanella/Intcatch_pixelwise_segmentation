from keras.layers import Input
from keras.layers.core import Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation
from keras.optimizers import Adam
from keras.optimizers import Adam
from keras.models import load_model
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
        
        model.summary()

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
        
        model.summary()

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
        
        model.summary()

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
        
        model.summary()

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
        
        model.summary()

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
        
        model.summary()

        return model