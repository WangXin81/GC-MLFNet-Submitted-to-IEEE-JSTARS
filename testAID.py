#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.io as scio
import imageio
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, Nadam, RMSprop, Adagrad
from sklearn.metrics import hamming_loss, multilabel_confusion_matrix, precision_recall_fscore_support, \
    balanced_accuracy_score, recall_score, fbeta_score
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, precision_score
from sklearn.metrics import accuracy_score
import keras.backend as K

####################################
from multilabelMetrics.examplebasedclassification import subsetAccuracy1, hammingLoss, recall1, precision1, accuracy1,fbeta1
from multilabelMetrics.examplebasedranking import rankingLoss, oneError, coverage, averagePrecision
from multilabelMetrics.labelbasedclassification import accuracyMacro, accuracyMicro, precisionMacro, precisionMicro, \
    recallMacro, recallMicro
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

def accuracy(y_true, y_pred):
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
####################################
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

# f-measure
def F(y_true, y_pred):
    p_val = P(y_true, y_pred)
    r_val = R(y_true, y_pred)
    f_val = 2 * p_val * r_val / (p_val + r_val)

    return f_val
# Accuracies:
def findMetrics(yTrue, yPred):
    # precision overall
    positive_predictions = np.count_nonzero(yPred)  # denominator
    true_positives = np.sum(np.logical_and(yTrue == 1, yPred == 1))  # numerator

    if positive_predictions == 0:
        precision = 0
    else:
        precision = true_positives / positive_predictions

    # recall overall
    relevant_positives = np.count_nonzero(yTrue)  # denominator

    recall = true_positives / relevant_positives

    # F Measure overall
    numerator = precision * recall
    denominator = precision + recall

    if denominator == 0:
        f_measure = 0
    else:
        f_measure = (2 * numerator) / denominator

    # precision per row/column
    positive_predictions_row = np.count_nonzero(yPred, axis=1)  # denominators
    positive_predictions_col = np.count_nonzero(yPred, axis=0)  # denominators

    true_positives_row = np.sum(np.logical_and(yTrue == 1, yPred == 1), axis=1)  # numerators
    true_positives_col = np.sum(np.logical_and(yTrue == 1, yPred == 1), axis=0)  # numerators

    positive_predictions_row = positive_predictions_row.astype('float')
    positive_predictions_col = positive_predictions_col.astype('float')

    true_positives_row = true_positives_row.astype('float')
    true_positives_col = true_positives_col.astype('float')

    precision_per_row = np.true_divide(true_positives_row, positive_predictions_row,
                                       out=np.zeros_like(true_positives_row), where=positive_predictions_row != 0)
    precision_per_col = np.true_divide(true_positives_col, positive_predictions_col,
                                       out=np.zeros_like(true_positives_col), where=positive_predictions_col != 0)
    avrg_precision_row = np.mean(precision_per_row)
    avrg_precision_col = np.mean(precision_per_col)

    # multi_label accuracy overall
    accuracy2 = true_positives / (np.sum(np.logical_or(yTrue == 1, yPred == 1)))

    acc2_denominator_row = np.sum(np.logical_or(yTrue == 1, yPred == 1), axis=1)

    acc2_denominator_row = acc2_denominator_row.astype('float')

    accuracy2_row = np.true_divide(true_positives_row, acc2_denominator_row, out=np.zeros_like(true_positives_row),
                                   where=acc2_denominator_row != 0)

    avrg_acc2_row = np.mean(accuracy2_row)

    # recall per row/column
    relevant_positives_row = np.count_nonzero(yTrue, axis=1)  # denominators
    relevant_positives_col = np.count_nonzero(yTrue, axis=0)  # denominators


    relevant_positives_row = relevant_positives_row.astype('float')
    relevant_positives_col = relevant_positives_col.astype('float')

    recall_per_row = np.true_divide(true_positives_row, relevant_positives_row, out=np.zeros_like(true_positives_row),
                                    where=relevant_positives_row != 0)
    recall_per_col = np.true_divide(true_positives_col, relevant_positives_col, out=np.zeros_like(true_positives_col),
                                    where=relevant_positives_col != 0)

    avrg_recall_row = np.mean(recall_per_row)
    avrg_recall_col = np.mean(recall_per_col)

    # F Measure per row
    numerator_row = avrg_precision_row * avrg_recall_row
    denominator_row = avrg_precision_row + avrg_recall_row

    if denominator_row == 0:
        f1_measure_row = 0
        f2_measure_row = 0
    else:
        f1_measure_row = (2 * numerator_row) / denominator_row
        f2_measure_row = ((5 * numerator_row) / ((4 * avrg_precision_row) + (avrg_recall_row)))

    print("Accuracy is :: " + str(avrg_acc2_row))
    print("F1 Score is :: " + str(f1_measure_row))
    print("F2 Score is :: " + str(f2_measure_row))
    print("Precision row :: " + str(avrg_precision_row))
    print("Recall row :: " + str(avrg_recall_row))
    print("Precision column :: " + str(avrg_precision_col))
    print("Recall column :: " + str(avrg_recall_col))
    return accuracy2, precision, recall, f_measure, avrg_precision_row, avrg_recall_row, f1_measure_row, f2_measure_row, avrg_precision_col, avrg_recall_col, avrg_acc2_row


# different threshold values
def thresholding1(test_set, test_labels):
    model = load_model('generatedmodel/MobileNetV2.hdf5', custom_objects={'P': P,'R': R, 'F':F, 'precision':precision,'f1_socre':f1_socre, 'sensitivity':sensitivity,'specificity':specificity})
    out = model.predict(test_set)
    out = np.array(out)
    # threshold = np.arange(0.1,0.9,0.05)
    # for t in threshold:

    y_pred = np.array([[1 if out[i, j] >= 0.5 else 0 for j in range(test_labels.shape[1])] for i in
                       range(len(test_labels))])

    print('hamming_loss:',hamming_loss(test_labels, y_pred))


    x = findMetrics(test_labels, y_pred)
    print(x)
    print('Classification Report:\n', classification_report(test_labels, y_pred), '\n')
    print('rankingLoss', rankingLoss(test_labels, y_pred))
    print('subsetAccuracy', subsetAccuracy1(test_labels, y_pred))
    print('hammingLoss', hammingLoss(test_labels, y_pred))
    print('accuracy1', accuracy1(test_labels, y_pred))
    print('precision1', precision1(test_labels, y_pred))
    print('recall1', recall1(test_labels, y_pred))
    print('fbeta1', fbeta1(test_labels, y_pred))
    print('oneError', oneError(test_labels, y_pred))
    print('coverage', coverage(test_labels, y_pred))
    print('averagePrecision', averagePrecision(test_labels, y_pred))

    print("label based")
    print('accuracyMacro', accuracyMacro(test_labels, y_pred))
    print('accuracyMicro', accuracyMicro(test_labels, y_pred))
    print('precisionMacro', precisionMacro(test_labels, y_pred))
    print('precisionMicro', precisionMicro(test_labels, y_pred))
    print('recallMacro', recallMacro(test_labels, y_pred))
    print('recallMicro', recallMicro(test_labels, y_pred))


OBSERVATIONS_FILE = 'AIDImages.npy'  # The file containing the data samples.
LABELS_FILE = 'AIDLabels.npy'  # The file containing the labels.
TESTING_DATA_NUM = 600

images = np.load(OBSERVATIONS_FILE)
labels = np.load(LABELS_FILE)

random_indices = np.arange(images.shape[0])
np.random.seed(42)
np.random.shuffle(random_indices)

labels = labels[random_indices]
images = images[random_indices]
test_set = images[:TESTING_DATA_NUM]
test_labels = labels[:TESTING_DATA_NUM]
print('shape',test_labels.shape)
thresholding1(test_set, test_labels)