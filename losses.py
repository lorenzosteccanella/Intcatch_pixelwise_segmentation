from keras import backend as K

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_2(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true_f)) + K.sum(K.square(y_pred_f)) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def dice_coef_loss_2(y_true, y_pred):
    return 1 - dice_coef_2(y_true, y_pred)

import keras.losses
keras.losses.dice_coef_loss=dice_coef_loss
keras.losses.dice_coef_loss_2=dice_coef_loss_2

import keras.metrics
keras.metrics.dice_coef=dice_coef
keras.metrics.dice_coef_2=dice_coef_2