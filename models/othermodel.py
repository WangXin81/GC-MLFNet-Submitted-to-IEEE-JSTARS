
import numpy as np
import scipy.io as scio
import scipy.ndimage as im
import imageio
import matplotlib.pyplot as plt
import keras

from keras import models
from keras import layers
from keras import optimizers
from keras import applications
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, Model
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, Activation, Flatten, Conv2D, RepeatVector
from keras.layers import GlobalAveragePooling2D, BatchNormalization, ZeroPadding2D, UpSampling2D
from keras.models import Sequential
from keras.layers import Reshape, Add, Multiply, Lambda, AveragePooling2D
from keras.layers import concatenate
from keras.layers import MaxPooling2D, Dropout, Input, MaxPool2D
from keras.optimizers import SGD, Adam, Nadam, RMSprop, Adagrad
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers.advanced_activations import PReLU
from keras.activations import linear as linear_activation
from keras import initializers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import fbeta_score
from sklearn.metrics import classification_report,confusion_matrix
from  sklearn.metrics  import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from keras.utils import to_categorical
def VGGNET():
    vgg_model = VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))

    for layers in vgg_model.layers:
        layers.trainable = True

    model = Sequential()

    model.add(vgg_model)
    #model.add(GlobalAveragePooling2D())
    model.add(Flatten(name='flatten_1'))

    model.add(Dense(17, activation='sigmoid', name='dense_1'))

    return model

def CA_VGG_LSTM():
    vgg_model = VGG16(include_top=False, weights='imagenet', input_shape=(247, 242, 3))

    model = Sequential()

    for layer in tuple(vgg_model.layers[:-5]):
        layer_type = type(layer).__name__
    model.add(layer)

    model.add(Conv2D(512, (3, 3), activation='relu', name='block5_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))

    model.add(Conv2D(17, kernel_size=(1, 1), strides=(1, 1), kernel_initializer='glorot_uniform'))
    model.add(Reshape((17, 28*28), input_shape=(28, 28, 17)))

    model.add(LSTM(17, input_shape=(17, 28*28), activation='tanh', kernel_initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)))

    model.add(Dense(17, activation='sigmoid'))

    return model

def CA_VGG_BILSTM():
    vgg_model = VGG16(include_top=False, weights='imagenet', input_shape=(247, 242, 3))

    model = Sequential()

    for layer in tuple(vgg_model.layers[:-5]):
        layer_type = type(layer).__name__
    model.add(layer)

    model.add(Conv2D(512, (3, 3), activation='relu', name='block5_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))

    model.add(Conv2D(17, kernel_size=(1, 1), strides=(1, 1), kernel_initializer='glorot_uniform'))
    model.add(Reshape((17, 28*28), input_shape=(28, 28, 17)))

    model.add(Bidirectional(LSTM(17, input_shape=(17, 28*28), activation='tanh', kernel_initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)), merge_mode='sum'))

    model.add(Dense(17, activation='sigmoid'))

    return model

def GoogLeNet():
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
    # base_model = InceptionV1
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 17 classes
    predictions = Dense(17, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

def ResNet50():
    #base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(247, 242, 3))
    base_model = applications.resnet50.ResNet50(weights= 'imagenet', include_top=False, input_shape= (256,256,3))
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 17 classes
    predictions = Dense(17, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

def residual_block(input, input_channels=None, output_channels=None, kernel_size=(3, 3), stride=1):

    if output_channels is None:
        output_channels = input.get_shape()[-1].value
    if input_channels is None:
        input_channels = output_channels // 4

    strides = (stride, stride)

    x = BatchNormalization()(input)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, (1, 1))(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, kernel_size, padding='same', strides=stride)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(output_channels, (1, 1), padding='same')(x)

    if input_channels != output_channels or stride != 1:
        input = Conv2D(output_channels, (1, 1), padding='same', strides=strides)(input)

    x = Add()([x, input])
    return x

def attention_block(input, input_channels=None, output_channels=None, encoder_depth=1):

    p = 1
    t = 2
    r = 1

    if input_channels is None:
        input_channels = input.get_shape()[-1].value
    if output_channels is None:
        output_channels = input_channels

    # First Residual Block
    for i in range(p):
        input = residual_block(input)

    # Trunc Branch
    output_trunk = input
    for i in range(t):
        output_trunk = residual_block(output_trunk)

    # Soft Mask Branch

    ## encoder
    ### first down sampling
    output_soft_mask = MaxPool2D(padding='same')(input)  # 32x32
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)

    skip_connections = []
    for i in range(encoder_depth - 1):

        ## skip connections
        output_skip_connection = residual_block(output_soft_mask)
        skip_connections.append(output_skip_connection)
        # print ('skip shape:', output_skip_connection.get_shape())

        ## down sampling
        output_soft_mask = MaxPool2D(padding='same')(output_soft_mask)
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)

            ## decoder
    skip_connections = list(reversed(skip_connections))
    for i in range(encoder_depth - 1):
        ## upsampling
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)
        output_soft_mask = UpSampling2D()(output_soft_mask)
        ## skip connections
        output_soft_mask = Add()([output_soft_mask, skip_connections[i]])

    ### last upsampling
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)
    output_soft_mask = UpSampling2D()(output_soft_mask)

    ## Output
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Activation('sigmoid')(output_soft_mask)

    # Attention: (1 + output_soft_mask) * output_trunk
    output = Lambda(lambda x: x + 1)(output_soft_mask)
    output = Multiply()([output, output_trunk])  #

    # Last Residual Block
    for i in range(p):
        output = residual_block(output)

    return output

def ResAttentionNet56(shape=(256, 256, 3), n_channels=64, n_classes=17,
                      dropout=0):


    input_ = Input(shape=shape)
    x = Conv2D(n_channels, (7, 7), strides=(2, 2), padding='same')(input_) # 112x112
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)  # 56x56

    x = residual_block(x, output_channels=n_channels * 4)  # 56x56
    x = attention_block(x, encoder_depth=3)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 8, stride=2)  # 28x28
    x = attention_block(x, encoder_depth=2)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 16, stride=2)  # 14x14
    x = attention_block(x, encoder_depth=1)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 32, stride=2)  # 7x7
    x = residual_block(x, output_channels=n_channels * 32)
    x = residual_block(x, output_channels=n_channels * 32)

    pool_size = (x.get_shape()[1].value, x.get_shape()[2].value)
    x = AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x)
    if dropout:
        x = Dropout(dropout)(x)
    output = Dense(n_classes, activation='sigmoid')(x)

    model = Model(input_, output)
    
    return model

def my_model(shape=(256,256,3)):
    input_ = Input(shape=shape)
    a1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_)

    x1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(input_)
    x1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x1) 
    x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x1)
    x1 = concatenate([a1, x1])

    a2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(a1)
    x2 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x1)
    x2 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x2) 
    x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x2)
    x2 = concatenate([a2, x2])

    a3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(a2)
    x3 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x2)
    x3 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x3) 
    x3 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x3) 
    x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x3)
    x3 = concatenate([a3, x3])

    a4 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(a3)
    x4 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x3)
    x4 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x4) 
    x4 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x4) 
    x4 = BatchNormalization()(x4)
    x4 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x4)
    x4 = concatenate([a4, x4])

    a5 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(a4)
    x5 = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x4)
    x5 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x5) 
    x5 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x5)
    x5 = BatchNormalization()(x5)
    x5 = Activation('relu')(x5)
    x5 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x5)
    x5 = concatenate([a5, x5])

    pool_size = (x5.get_shape()[1].value, x5.get_shape()[2].value)
    x5 = AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x5)
    x5 = Flatten()(x5)
    #x5 = Dropout(0.50)(x5)

    pool_size = (x4.get_shape()[1].value, x4.get_shape()[2].value)
    x4 = AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x4)
    x4 = Flatten()(x4)
    #x4 = Dropout(0.50)(x4)

    pool_size = (x3.get_shape()[1].value, x3.get_shape()[2].value)
    x3 = AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x3)
    x3 = Flatten()(x3)
    #x3 = Dropout(0.50)(x3)

    pool_size = (x2.get_shape()[1].value, x2.get_shape()[2].value)
    x2 = AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x2)
    x2 = Flatten()(x2)
    #x2 = Dropout(0.50)(x2)

    pool_size = (x1.get_shape()[1].value, x1.get_shape()[2].value)
    x1 = AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x1)
    x1 = Flatten()(x1)
    #x1 = Dropout(0.50)(x1)

    x = concatenate([x1, x2, x3, x4, x5],axis=-1)

    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)

    output = Dense(17, activation='sigmoid')(x)

    model = Model(input_, output)
    return model
