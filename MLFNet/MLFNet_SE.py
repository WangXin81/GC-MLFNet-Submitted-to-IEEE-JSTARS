import os

from keras import layers, optimizers, models
from keras.regularizers import l2
from models.resnet50 import ResNet50
# from keras.applications.resnet50 import ResNet50
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dropout, concatenate, merge, UpSampling2D
from keras.optimizers import Adam


def SE(x):
    bs, h, w, c = x.get_shape().as_list()
    sequeeze = GlobalAveragePooling2D()(x)

    excitation = Dense(c//4)(sequeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(c)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, 1, c))(excitation)
    scale = Multiply()([x, excitation])
    return scale

def SEMLFNet(pretrained_weights=None, input_size=(256, 256, 3), classNum=6):
    H, W, C = input_size
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    base_model = ResNet50(H, W, C)
    if (pretrained_weights): base_model.load_weights(pretrained_weights)
    # base_model.load_weights('')
    # print(base_model.output)
    C1, C2, C3, C4, C5 = base_model.output
    P2 = Conv2D(256, (1, 1), padding='SAME', kernel_initializer='he_normal')(C2)
    P2 = BatchNormalization(axis=bn_axis)(P2)
    P2 = Activation('relu')(P2)
    P3 = Conv2D(256, (1, 1), padding='SAME', kernel_initializer='he_normal')(C3)
    P3 = BatchNormalization(axis=bn_axis)(P3)
    P3 = Activation('relu')(P3)
    P4 = Conv2D(256, (1, 1), padding='SAME', kernel_initializer='he_normal')(C4)
    P4 = BatchNormalization(axis=bn_axis)(P4)
    P4 = Activation('relu')(P4)
    P5 = Conv2D(256, (1, 1), padding='SAME', kernel_initializer='he_normal')(C5)
    P5 = BatchNormalization(axis=bn_axis)(P5)
    P5 = Activation('relu')(P5)


    P2 = MaxPooling2D(pool_size=(2,2))(P2)

    P3 = layers.add([P2, P3])
    P3 = Conv2D(256, (3, 3), padding='SAME', kernel_initializer='he_normal')(P3)
    P3 = BatchNormalization(axis=bn_axis)(P3)
    P3 = Activation('relu')(P3)
    P3 = SE(P3)
    P3 = MaxPooling2D(pool_size=(2, 2))(P3)

    P4 = layers.add([P3, P4])
    P4 = Conv2D(256, (3, 3), padding='SAME', kernel_initializer='he_normal')(P4)
    P4 = BatchNormalization(axis=bn_axis)(P4)
    P4 = Activation('relu')(P4)
    P4 = SE(P4)
    P4 = MaxPooling2D(pool_size=(2, 2))(P4)

    P5 = layers.add([P4, P5])
    P5 = Conv2D(256, (3, 3), padding='SAME', kernel_initializer='he_normal')(P5)
    P5 = BatchNormalization(axis=bn_axis)(P5)
    P5 = Activation('relu')(P5)
    P5 = SE(P5)

    P2 = GlobalAveragePooling2D()(P2)
    P3 = GlobalAveragePooling2D()(P3)
    P4 = GlobalAveragePooling2D()(P4)
    P5 = GlobalAveragePooling2D()(P5)
    out = concatenate([P2,P3,P4,P5], axis=-1)
    out = Dense(1024, activation='relu')(out)
    out = Dense(classNum, activation='sigmoid')(out)

    model = Model(input=base_model.input, output=out)
    # if (pretrained_weights): model.load_weights(pretrained_weights)
    return model