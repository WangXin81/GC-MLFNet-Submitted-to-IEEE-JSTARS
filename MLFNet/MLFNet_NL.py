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

def NL(ip, intermediate_dim=None, compression=2,
                    mode='embedded', add_residual=True):
    """
    Adds a Non-Local block for self attention to the input tensor.
    Input tensor can be or rank 3 (temporal), 4 (spatial) or 5 (spatio-temporal).
    Arguments:
        ip: input tensor
        intermediate_dim: The dimension of the intermediate representation. Can be
            `None` or a positive integer greater than 0. If `None`, computes the
            intermediate dimension as half of the input channel dimension.
        compression: None or positive integer. Compresses the intermediate
            representation during the dot products to reduce memory consumption.
            Default is set to 2, which states halve the time/space/spatio-time
            dimension for the intermediate step. Set to 1 to prevent computation
            compression. None or 1 causes no reduction.
        mode: Mode of operation. Can be one of `embedded`, `gaussian`, `dot` or
            `concatenate`.
        add_residual: Boolean value to decide if the residual connection should be
            added or not. Default is True for ResNets, and False for Self Attention.
    Returns:
        a tensor of same shape as input
    """
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    ip_shape = K.int_shape(ip)

    if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
        raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

    if compression is None:
        compression = 1

    dim1, dim2, dim3 = None, None, None
    # check rank and calculate the input shape
    if len(ip_shape) == 3:  # temporal / time series data
        rank = 3
        batchsize, dim1, channels = ip_shape

    elif len(ip_shape) == 4:  # spatial / image data
        rank = 4

        if channel_dim == 1:
            batchsize, channels, dim1, dim2 = ip_shape
        else:
            batchsize, dim1, dim2, channels = ip_shape

    elif len(ip_shape) == 5:  # spatio-temporal / Video or Voxel data
        rank = 5

        if channel_dim == 1:
            batchsize, channels, dim1, dim2, dim3 = ip_shape
        else:
            batchsize, dim1, dim2, dim3, channels = ip_shape

    else:
        raise ValueError('Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)')
    # verify correct intermediate dimension specified
    if intermediate_dim is None:
        intermediate_dim = channels // 2

        if intermediate_dim < 1:
            intermediate_dim = 1

    else:
        intermediate_dim = int(intermediate_dim)

        if intermediate_dim < 1:
            raise ValueError('`intermediate_dim` must be either `None` or positive integer greater than 1.')

    if mode == 'gaussian':  # Gaussian instantiation
        x1 = Reshape((-1, channels))(ip)  # xi
        x2 = Reshape((-1, channels))(ip)  # xj
        f = dot([x1, x2], axes=2)
        f = Activation('softmax')(f)

    elif mode == 'dot':  # Dot instantiation
        # theta path
        theta = _convND(ip, rank, intermediate_dim)
        theta = Reshape((-1, intermediate_dim))(theta)
        # phi path
        phi = _convND(ip, rank, intermediate_dim)
        phi = Reshape((-1, intermediate_dim))(phi)

        f = dot([theta, phi], axes=2)

        size = K.int_shape(f)
        # scale the values to make it size invariant
        f = Lambda(lambda z: (1. / float(size[-1])) * z)(f)

    elif mode == 'concatenate':  # Concatenation instantiation
        raise NotImplementedError('Concatenate model has not been implemented yet')

    else:  # Embedded Gaussian instantiation
        # theta path
        theta = _convND(ip, rank, intermediate_dim)
        theta = Reshape((-1, intermediate_dim))(theta)
        # phi path
        phi = _convND(ip, rank, intermediate_dim)
        phi = Reshape((-1, intermediate_dim))(phi)

        if compression > 1:
            # shielded computation
            phi = MaxPool1D(compression)(phi)

        f = dot([theta, phi], axes=2)
        f = Activation('softmax')(f)
    # g path
    g = _convND(ip, rank, intermediate_dim)
    g = Reshape((-1, intermediate_dim))(g)

    if compression > 1 and mode == 'embedded':
        # shielded computation
        g = MaxPool1D(compression)(g)
    # compute output path
    y = dot([f, g], axes=[2, 1])
    # reshape to input tensor format
    if rank == 3:
        y = Reshape((dim1, intermediate_dim))(y)
    elif rank == 4:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, intermediate_dim))(y)
        else:
            y = Reshape((intermediate_dim, dim1, dim2))(y)
    else:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, dim3, intermediate_dim))(y)
        else:
            y = Reshape((intermediate_dim, dim1, dim2, dim3))(y)
    # project filters
    y = _convND(y, rank, channels)
    # residual connection
    if add_residual:
        y = add([ip, y])

    return y


def _convND(ip, rank, channels):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

    if rank == 3:
        x = Conv1D(channels, 1, padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    elif rank == 4:
        x = Conv2D(channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    else:
        x = Conv3D(channels, (1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    return x

def NLMLFNet(pretrained_weights=None, input_size=(256, 256, 3), classNum=6):
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
    P3 = NL(P3)
    P3 = MaxPooling2D(pool_size=(2, 2))(P3)

    P4 = layers.add([P3, P4])
    P4 = Conv2D(256, (3, 3), padding='SAME', kernel_initializer='he_normal')(P4)
    P4 = BatchNormalization(axis=bn_axis)(P4)
    P4 = Activation('relu')(P4)
    P4 = NL(P4)
    P4 = MaxPooling2D(pool_size=(2, 2))(P4)

    P5 = layers.add([P4, P5])
    P5 = Conv2D(256, (3, 3), padding='SAME', kernel_initializer='he_normal')(P5)
    P5 = BatchNormalization(axis=bn_axis)(P5)
    P5 = Activation('relu')(P5)
    P5 = NL(P5)

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