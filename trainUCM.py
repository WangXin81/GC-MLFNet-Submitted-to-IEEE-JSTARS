#!/usr/bin/env python
# coding: utf-8
import keras
import numpy as np
import scipy.io as scio
import imageio
from keras import Model
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, Nadam, RMSprop, Adagrad
from sklearn.metrics import hamming_loss
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from models.mobilenet_v2 import MobileNetV2
from MLFNet.MLFNet_GC import GCMLFNet

premodel_path = 'pretrained/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
input_size = (256, 256, 3)
classnum = 17

OBSERVATIONS_FILE = 'UcmImages.npy'  # The file containing the data samples.
LABELS_FILE = 'UcmLabels.npy'  # The file containing the labels.
TESTING_DATA_NUM = 420

images = np.load(OBSERVATIONS_FILE)
labels = np.load(LABELS_FILE)

random_indices = np.arange(images.shape[0])
np.random.seed(42)
np.random.shuffle(random_indices)

labels = labels[random_indices]
images = images[random_indices]

test_set = images[:TESTING_DATA_NUM]
test_labels = labels[:TESTING_DATA_NUM]

train_set = images[TESTING_DATA_NUM:]
train_labels = labels[TESTING_DATA_NUM:]

# Parameters For Data Augmentation later
ROTATION_RANGE = 45
SHIFT_FRACTION = 0.2
SHEAR_RANGE = 0.0
ZOOM_RANGE = 0.0
HORIZONTAL_FLIP = True
VERTICAL_FILP = True

data_generator = ImageDataGenerator(
    rotation_range=ROTATION_RANGE, width_shift_range=SHIFT_FRACTION, height_shift_range=SHIFT_FRACTION,
    shear_range=SHEAR_RANGE, zoom_range=ZOOM_RANGE, horizontal_flip=HORIZONTAL_FLIP,
    vertical_flip=VERTICAL_FILP)
data_generator.fit(train_set)

import keras.backend as K
import tensorflow as tf
####################################
def cal_base(y_true, y_pred):
    y_pred_positive = K.round(K.clip(y_pred, 0, 1))
    y_pred_negative = 1 - y_pred_positive

    y_positive = K.round(K.clip(y_true, 0, 1))
    y_negative = 1 - y_positive

    TP = K.sum(y_positive * y_pred_positive)
    TN = K.sum(y_negative * y_pred_negative)

    FP = K.sum(y_negative * y_pred_positive)
    FN = K.sum(y_positive * y_pred_negative)

    return TP, TN, FP, FN

def acc(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    ACC = (TP + TN) / (TP + FP + FN + TN + K.epsilon())
    return ACC

def sensitivity(y_true, y_pred):
    """ recall """
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    SE = TP/(TP + FN + K.epsilon())
    return SE

def precision(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    PC = TP/(TP + FP + K.epsilon())
    return PC

def specificity(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    SP = TN / (TN + FP + K.epsilon())
    return SP

def f1_socre(y_true, y_pred):
    SE = sensitivity(y_true, y_pred)
    PC = precision(y_true, y_pred)
    F1 = 2 * SE * PC / (SE + PC + K.epsilon())
    return F1

# precision
def P(y_true, y_pred):
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.20), 'float32'))
    pred_positives = K.sum(K.cast(K.greater(K.clip(y_pred, 0, 1), 0.20), 'float32'))

    precision = true_positives / (pred_positives + K.epsilon())
    return precision


# recall
def R(y_true, y_pred):
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.20), 'float32'))
    poss_positives = K.sum(K.cast(K.greater(K.clip(y_true, 0, 1), 0.20), 'float32'))

    recall = true_positives / (poss_positives + K.epsilon())
    return recall


# f1-score
def F(y_true, y_pred):
    p_val = P(y_true, y_pred)
    r_val = R(y_true, y_pred)
    f_val = 2 * p_val * r_val / (p_val + r_val)

    return f_val

# Useful Callbacks:
def CALLBACKS():
    # lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, min_lr=10e-8, epsilon=0.01, verbose=1)
    # early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint('generatedmodel/UCM.hdf5', monitor='val_F', mode='max', verbose=1, save_best_only=True)
    callbacks = [
                # lr_reducer, early_stopper,
                 model_checkpoint]
    return callbacks


model = GCMLFNet(pretrained_weights = premodel_path,
            input_size = input_size,
            classNum = classnum)


# base_model = MobileNetV2(input_shape=input_size,
#                 include_top=False,
#                 weights='imagenet',
#                 input_tensor=None,
#                 pooling='avg',
#                 classes=17,
#                 backend=keras.backend,
#                 layers=keras.layers,
#                 models=keras.models,
#                 utils=keras.utils)
#
# x=base_model.output
# x=Dense(17,activation='sigmoid')(x)
# model=Model(inputs=base_model.input,outputs=x)


op = Adam(lr=3e-4)
model.compile(loss='binary_crossentropy', optimizer=op, metrics=['accuracy',P,R,F,precision,f1_socre,sensitivity,specificity])
model.summary()

from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
print('TensorFlow:', tf.__version__)

# model = tf.keras.applications.ResNet50()

forward_pass = tf.function(
    model.call,
    input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])

graph_info = profile(forward_pass.get_concrete_function().graph,
                        options=ProfileOptionBuilder.float_operation())

# The //2 is necessary since `profile` counts multiply and accumulate
# as two flops, here we report the total number of multiply accumulate ops
flops = graph_info.total_float_ops // 2
print('Flops: {:,}'.format(flops))

data_augmentation = 0
if data_augmentation == 1:
    model.fit_generator(data_generator.flow(train_set, train_labels, batch_size=8),
                         epochs=35,
                         steps_per_epoch=1680 // 8,
                         verbose=2,
                         validation_data=(test_set, test_labels),
                         callbacks=CALLBACKS())
else:
    model.fit(train_set, train_labels, batch_size=8, epochs=35,
               validation_data=(test_set, test_labels), callbacks=CALLBACKS())



