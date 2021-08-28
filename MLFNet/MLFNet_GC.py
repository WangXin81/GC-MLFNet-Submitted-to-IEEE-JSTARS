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


def GCM(x):
    """
    simplified non local net
    GCnet 发现在NLnet中图像每个点的全局上下文相近，只计算一个点的全局相似度，计算量减少1/hw
    :parameter x:input layers or tensor
    """
    bs, h, w, c = x.get_shape().as_list()
    input_x = x
    input_x = layers.Reshape((h*w, c))(input_x)  # [bs, H*W, C]
    # input_x = layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(input_x)  # [bs,C,H*W]
    # input_x = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(input_x)  # [bs,1,C,H*W]
    context_mask = layers.Conv2D(filters=1, kernel_size=(1, 1))(x) # [bs,h,w,1]
    context_mask = layers.Reshape((h*w, 1))(context_mask)
    context_mask = layers.Softmax(axis=1)(context_mask)  # [bs, H*W, 1]
    # context_mask = layers.Lambda(lambda x: tf.transpose(x, [0, 2, 1]))(context_mask)
    # context_mask = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(context_mask)
    context = layers.dot([input_x, context_mask],axes=1)  # [bs,1,c]
    context = layers.Reshape((1, 1, c))(context)
    # context_transform = layers.Conv2D(c, (1, 1))(context)
    # context_transform = LayerNormalization()(context_transform)
    # context_transform = layers.ReLU()(context_transform)
    # context_transform = layers.Conv2D(c, (1, 1))(context_transform)
    # context_transform=layers.Conv2D(c,kernel_size=(1,1))(context)
    x = layers.Add()([x,context])
    return x

def GCMLFNet(pretrained_weights=None, input_size=(256, 256, 3), classNum=6):
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
    P3 = GCM(P3)
    P3 = MaxPooling2D(pool_size=(2, 2))(P3)

    P4 = layers.add([P3, P4])
    P4 = Conv2D(256, (3, 3), padding='SAME', kernel_initializer='he_normal')(P4)
    P4 = BatchNormalization(axis=bn_axis)(P4)
    P4 = Activation('relu')(P4)
    P4 = GCM(P4)
    P4 = MaxPooling2D(pool_size=(2, 2))(P4)

    P5 = layers.add([P4, P5])
    P5 = Conv2D(256, (3, 3), padding='SAME', kernel_initializer='he_normal')(P5)
    P5 = BatchNormalization(axis=bn_axis)(P5)
    P5 = Activation('relu')(P5)
    P5 = GCM(P5)

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