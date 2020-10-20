import os
import datetime
import glob
import random
import sys
import imagecodecs
import cv2

import matplotlib.pyplot as plt
import skimage.io                                     #Used for imshow function
import skimage.transform                              #Used for resize function
from skimage.morphology import label                  #Used for Run-Length-Encoding RLE to create final submission

import pandas as pd
import tifffile as tiff

import keras
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv2DTranspose
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.merge import add, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import multi_gpu_model, plot_model
from keras import backend as K
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
import itertools
from sklearn.metrics import confusion_matrix
from skimage.transform import resize
from skimage import measure, io, img_as_ubyte
from skimage.color import label2rgb, rgb2gray
import numpy as nu


def true_positives(y, pred, th=0.5):
    """
    Count true positives.

    Args:
        y (nu.array): ground truth, size (n_examples)
        pred (nu.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        TP (int): true positives
    """
    TP = 0
    thresholded_preds = pred > th 
    TP = nu.sum((y == 1) & (thresholded_preds == 1))
   
    return TP

def true_negatives(y, pred, th=0.5):
    """
    Count true negatives.

    Args:
        y (nu.array): ground truth, size (n_examples)
        pred (nu.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        TN (int): true negatives
    """
    TN = 0
    thresholded_preds = pred > th
    TN = nu.sum((y == 0) & (thresholded_preds == 0))

    return TN

def false_positives(y, pred, th=0.5):
    """
    Count false positives.

    Args:
        y (nu.array): ground truth, size (n_examples)
        pred (nu.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        FP (int): false positives
    """
    FP = 0
    thresholded_preds = pred > th
    FP = nu.sum((y == 1) & (thresholded_preds == 0))

    return FP

def false_negatives(y, pred, th=0.5):
    """
    Count false positives.

    Args:
        y (nu.array): ground truth, size (n_examples)
        pred (nu.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        FN (int): false negatives
    """
    FN = 0
    thresholded_preds = pred > th
    FN = nu.sum((y == 0) & (thresholded_preds == 1))

    return FN
    
# Custom IoU metric
EPS = 1e-12
def mean_iou(y_true, y_pred):
    prec = []
    for t in nu.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.cast(y_pred > t , tf.int32)
        score, up_opt = tf.keras.metrics.MeanIoU(y_true, y_pred_, 2)
        K.get_session().run(tf.compat.v1.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def get_iou(gt, pr, n_classes):
    class_wise = nu.zeros(n_classes)
    for cl in range(n_classes):
        intersection = nu.sum(nu.logical_and((gt == cl), (pr == cl)))
        union = nu.sum(nu.maximum((gt == cl), (pr == cl)))
        iou = intersection / (union + EPS)
        class_wise[cl] = iou
    return class_wise


def get_recall(gt, pr, n_classes):
    class_wise = nu.zeros(n_classes)    
    for cl in range(n_classes):
        true_positives = nu.sum(nu.logical_and((gt == pr), (pr == cl)))
        possible_positives = nu.sum((gt == cl))
        recall = true_positives / (possible_positives + EPS)
        class_wise[cl] = recall
    return class_wise

def get_tp(gt, pr, n_classes):
    class_wise = nu.zeros(n_classes)    
    for cl in range(n_classes):
        true_positives = nu.sum(nu.logical_and((gt == pr), (pr == cl)))
        class_wise[cl] = true_positives
    return class_wise


def get_precision(gt, pr, n_classes):
    class_wise = nu.zeros(n_classes)
    for cl in range(n_classes):
        true_positives = nu.sum(nu.logical_and((gt == pr), (pr == cl)))
        predicted_positives = nu.sum((pr == cl))
        precision = true_positives / (predicted_positives + EPS)
        class_wise[cl] = precision
    return class_wise

# Custom loss function
# def dice_coef(y_true, y_pred):
#     smooth = 1.
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# def bce_dice_loss(y_true, y_pred):
#     return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

def dice_coef(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)

    return (numerator + 1) / (denominator + 1)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

def dice_loss(y_true, y_pred):
  numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
  denominator = tf.reduce_sum(y_true + y_pred, axis=-1)

  return 1 - (numerator + 1) / (denominator + 1)

def weighted_cross_entropy(beta):
  def convert_to_logits(y_pred):
      # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
      y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

      return tf.log(y_pred / (1 - y_pred))

  def loss(y_true, y_pred):
    y_pred = convert_to_logits(y_pred)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)

    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss)

  return loss
  
def dice(y, pred):
    th = 0.5
    TP = 0
    thresholded_preds = pred > th 
    TP = nu.sum((y == 1) & (thresholded_preds == 1))
    TN = 0
    thresholded_preds = pred > th
    TN = nu.sum((y == 0) & (thresholded_preds == 0))
    FP = 0
    thresholded_preds = pred > th
    FP = nu.sum((y == 1) & (thresholded_preds == 0))
    FN = 0
    thresholded_preds = pred > th
    FN = nu.sum((y == 0) & (thresholded_preds == 1))
    dice = 2*TP/(2*TP+FP+FN)
    
    return dice

def jac(y, pred):
    th = 0.5
    TP = 0
    thresholded_preds = pred > th 
    TP = nu.sum((y == 1) & (thresholded_preds == 1))
    TN = 0
    thresholded_preds = pred > th
    TN = nu.sum((y == 0) & (thresholded_preds == 0))
    FP = 0
    thresholded_preds = pred > th
    FP = nu.sum((y == 1) & (thresholded_preds == 0))
    FN = 0
    thresholded_preds = pred > th
    FN = nu.sum((y == 0) & (thresholded_preds == 1))
    jac = TP/(TP+FP+FN)
    
    return jac

def matriz_confusion(y, pred):
    TP = true_positives(y, pred, th=0.5)
    TN = true_negatives(y, pred, th=0.5)
    FP = false_positives(y, pred, th=0.5)
    FN = false_negatives(y, pred, th=0.5)
    matrix= [[TP, FP], [FN, TN]]
    matrix= nu.array(matrix)
    
    return matrix

def recall(y, pred):
    TP = true_positives(y, pred, th=0.5)
    TN = true_negatives(y, pred, th=0.5)
    FP = false_positives(y, pred, th=0.5)
    FN = false_negatives(y, pred, th=0.5)
    recall = TP/(TP+FN)
    
    return recall

def precision(y, pred):
    TP = true_positives(y, pred, th=0.5)
    TN = true_negatives(y, pred, th=0.5)
    FP = false_positives(y, pred, th=0.5)
    FN = false_negatives(y, pred, th=0.5)
    precision = TP/(TP+FP)
    
    return precision

def f1(y, pred):
    TP = true_positives(y, pred, th=0.5)
    TN = true_negatives(y, pred, th=0.5)
    FP = false_positives(y, pred, th=0.5)
    FN = false_negatives(y, pred, th=0.5)
    f1 = TP/(TP+((FN+FP)/2))
    
    return f1

def acurracy(y, pred):
    TP = true_positives(y, pred, th=0.5)
    TN = true_negatives(y, pred, th=0.5)
    FP = false_positives(y, pred, th=0.5)
    FN = false_negatives(y, pred, th=0.5)
    acurracy = (TP+TN)/(TP+FP+TN+FN)
    
    return acurracy

def specificity(y, pred):
    TP = true_positives(y, pred, th=0.5)
    TN = true_negatives(y, pred, th=0.5)
    FP = false_positives(y, pred, th=0.5)
    FN = false_negatives(y, pred, th=0.5)
    specificity = TN/(FP+TN)
    
    return specificity

def fallout(y, pred):
    TP = true_positives(y, pred, th=0.5)
    TN = true_negatives(y, pred, th=0.5)
    FP = false_positives(y, pred, th=0.5)
    FN = false_negatives(y, pred, th=0.5)
    fallout = FP/(FP+TN)
    
    return fallout

def fnr(y, pred):
    TP = true_positives(y, pred, th=0.5)
    TN = true_negatives(y, pred, th=0.5)
    FP = false_positives(y, pred, th=0.5)
    FN = false_negatives(y, pred, th=0.5)
    fnr = FN/(FN+TP)
    
    return fnr

def auc(y, pred):
    fallout = fallout(y, pred)
    fnr = fnr(y, pred)
    auc = 1 - (fallout+fnr)/2
    
    return auc

def roc(y, pred):
    recall = recall(y, pred)
    fallout = fallout(y, pred)
    roc = recall/fallout
    
    return roc

def gce(y, pred):
    TP = true_positives(y, pred, th=0.5)
    TN = true_negatives(y, pred, th=0.5)
    FP = false_positives(y, pred, th=0.5)
    FN = false_negatives(y, pred, th=0.5)
    n = TN+FP+FN+TP
    e1 = ( FN*(FN+ 2*TP)/(TP+FN) + FP*(FP + 2*TN)/(TN+FP) )
    e2 = ( FP*(FP+2*TP)/(TP+FP) + FN*(FN + 2*TN)/(TN+FN) )
    gce = min(e1,e2)/n
    
    return gce

def metricas(y, pred):
    print('Matriz de confusión:', matriz_confusion(y, pred),
          'Recall = %.5f' %recall(y, pred),
          'Precision = %.5f' %precision(y, pred),
          'DICE/F1 = %.5f' %dice(y, pred),
          'JAC/IoU = %.5f' %jac(y, pred),
          'Acurracy = %.5f' %acurracy(y, pred),
          'Specificity = %.5f' %specificity(y, pred),sep='\n')

def get_confusion_matrix_intersection_mats(groundtruth, predicted):
    """
    Returns a dictionary of 4 boolean numpy arrays containing True at TP, FP, FN, TN.
    """
    confusion_matrix_arrs = {}

    groundtruth_inverse = nu.logical_not(groundtruth)
    predicted_inverse = nu.logical_not(predicted)

    confusion_matrix_arrs["tp"] = nu.logical_and(groundtruth, predicted)
    confusion_matrix_arrs["tn"] = nu.logical_and(groundtruth_inverse, predicted_inverse)
    confusion_matrix_arrs["fp"] = nu.logical_and(groundtruth_inverse, predicted)
    confusion_matrix_arrs["fn"] = nu.logical_and(groundtruth, predicted_inverse)

    return confusion_matrix_arrs

def get_confusion_matrix_overlaid_mask(image, groundtruth, predicted):
    """
    Returns overlay the 'image' with a color mask where TP, FP, FN, TN are
    each a color given by the 'colors' dictionary
    """
    alpha = 0.5
    colors = {
        "tp": (0, 255, 255),  # cyan
        "fp": (255, 0, 255),  # magenta
        "fn": (255, 255, 0),  # yellow
        "tn": (0,0,0)  # black
    }
    
    print("Cyan - TP")
    print("Magenta - FP")
    print("Yellow - FN")
    print("Black - TN")
    image = cv2.cvtColor(image[:,:,0], cv2.COLOR_GRAY2RGB)
    masks = get_confusion_matrix_intersection_mats(groundtruth[:,:,0], predicted[:,:,0])
    color_mask = nu.zeros_like(image)
    for label, mask in masks.items():
        color = colors[label]
        mask_rgb = nu.zeros_like(image)
        mask_rgb[mask != 0] = color
        color_mask += mask_rgb
    return cv2.addWeighted(image, alpha, color_mask, 1 - alpha, 0)
    
def plot_confusion_matrix(cm, classes, normalize=False):
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:,nu.newaxis]
        title, fmt = 'Matriz de confusión normalizada', '.5f'
    else:
        title, fmt = 'Matriz de confusión sin normalizar', 'd'
    plt.figure(figsize=(7,7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Purples)
    plt.title(title)
    plt.colorbar()
    tick_marks = nu.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center", 
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Clase Verdadera')
    plt.xlabel('Clase Predicha')
    plt.show()
    
def conteo(y_true, y_pred):
    threshold = 0.5
    Nidos_r = []
    Nidos_p = []
    Area_r = []
    Area_p = []
    
    for i in range(0,len(y_true)):
        pred = img_as_ubyte(y_pred[i][:,:,0])

        label_pred= measure.label(pred > threshold, connectivity= pred.ndim)
        props_pred = measure.regionprops_table(label_pred, pred, 
                                          properties=['label','area','equivalent_diameter',
                                                      'mean_intensity','solidity','filled_area','centroid'])

        image = img_as_ubyte(rgb2gray(y_true[i][:,:,0]))
        label_image = measure.label(image > threshold, connectivity= image.ndim)
        props = measure.regionprops_table(label_image, image, 
                                      properties=['label','area','equivalent_diameter',
                                                  'mean_intensity','solidity','filled_area','centroid'])
        
        Nidos_r = nu.append(Nidos_r, len(props['label']))
        Area_r = nu.append(Area_r, nu.sum(props['area']))
        Nidos_p = nu.append(Nidos_p, len(props_pred['label']))
        Area_p = nu.append(Area_p, nu.sum(props_pred['area']))
    Diferencia_N = Nidos_p - Nidos_r
    Diferencia_A = Area_p - Area_r
    dic={'Nidos': Nidos_r, 'Nidos predichos': Nidos_p, 'Diferencia_N': Diferencia_N,
         'Área_total': Area_r, 'Área_predicha': Area_p, 'Diferencia_A': Diferencia_A}
    df=pd.DataFrame(dic)
    df.index.name = 'Imagen'
    return df