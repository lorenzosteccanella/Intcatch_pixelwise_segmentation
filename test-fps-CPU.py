from models import Models
from losses import *
from Utils import Utils
import numpy as np
from keras import backend as K
import imageio
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # to make visible only the CPU

tf.logging.set_verbosity(tf.logging.ERROR)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)

from tensorflow.python.client import device_lib

devices = device_lib.list_local_devices()

print(devices)

filename = "video_input/lakegarda-may-9-prop-8.avi"
reader = imageio.get_reader(filename,  'ffmpeg')

model_class = Models()
path_models='./models_saved/Final_new_dataset/'
model_path_name= ['unet160--mediumBN_mobilenetv2--dice_coef_loss',
                  'unet160--small--dice_coef_loss',
                  'unet160--smallBN--dice_coef_loss',
                  'unet160--medium--dice_coef_loss',
                  'unet160--mediumBN--dice_coef_loss',
                  'unet160--large--dice_coef_loss',
                  'unet160--largeBN--dice_coef_loss',
                  'unet160--smallBN_mobilenetv1--dice_coef_loss',
                  'unet160--mediumBN_mobilenetv1--dice_coef_loss',
                  ]

for i in range(10):
        
    print('#'*30)
    print('Run n°',i)
    print('#'*30)
    print()
    
    for models_name in model_path_name:
        sess = tf.Session(config=config)
        set_session(sess)  # set this TensorFlow session as the default session for Keras
        model1=model_class.load_model(path_models+models_name)
        
        fps_speed = Utils.test_speed_from_video_v2(reader, model1, 160, 160, 100)
        print('-'*30)
        print("name of the model: ", models_name, "   fps speed: ", fps_speed)
        
        K.clear_session()

reader.close()
        
        

    
    
    
