from __future__ import absolute_import
from keras.objectives import *
import keras.backend as K
import tensorflow as tf
import keras.losses

from metrics import DSSIMObjective, DSTSIMObjective, DSSIM_Sum, Sobel, DSSIM_SQU, DSSIM_ContentWeighted


def sobelLoss_mse(y_true, y_pred):
    sobel = Sobel()
    sobel_true = sobel(y_true)
    sobel_pred = sobel(y_pred)
    return K.mean(K.square(sobel_true - sobel_pred))


def superloss_1(y_true, y_pred):
    dssim = DSSIMObjective()
    loss_mae = keras.losses.mae(y_true, y_pred)
    loss_mse = keras.losses.mse(y_true, y_pred)
    loss_dssim = dssim(y_true, y_pred)
    return 0.75 * loss_dssim + 0.2 * loss_mae + 0.05 * loss_mse


def superloss_2(y_true, y_pred):
    dssim = DSSIMObjective()
    loss_mae = keras.losses.mae(y_true, y_pred)
    loss_mse = keras.losses.mse(y_true, y_pred)
    loss_dssim = dssim(y_true, y_pred)
    return 0.6 * loss_dssim + 0.3 * loss_mae + 0.1 * loss_mse


def superloss_3(y_true, y_pred):
    dssim = DSSIM_Sum(kernel_size=8)
    absolute_error = K.abs(y_true - y_pred)
    square_error = K.square(y_true - y_pred)
    loss_ssim = dssim(y_true, y_pred)
    return 0.4 * loss_ssim + 0.4 * absolute_error + 0.2 * square_error


def superloss_4(y_true, y_pred):
    dssim = DSSIMObjective(kernel_size=5)
    loss_mae = keras.losses.mae(y_true, y_pred)
    loss_mse = keras.losses.mse(y_true, y_pred)
    loss_dssim = dssim(y_true, y_pred)
    return 0.4 * loss_dssim + 0.4 * loss_mae + 0.2 * loss_mse


def superloss_5(y_true, y_pred):
    dssim = DSSIMObjective()
    loss_mae = keras.losses.mae(y_true, y_pred)
    loss_mse = keras.losses.mse(y_true, y_pred)
    loss_dssim = dssim(y_true, y_pred)
    return 0.3 * loss_dssim + 0.6 * loss_mae + 0.1 * loss_mse


def superloss_6(y_true, y_pred):
    dssim = DSSIMObjective(kernel_size=5)
    loss_mae = keras.losses.mae(y_true, y_pred)
    loss_mse = keras.losses.mse(y_true, y_pred)
    loss_dssim = dssim(y_true, y_pred)
    return 0.4 * loss_dssim + 0.5 * loss_mae + 0.1 * loss_mse


def superloss_7(y_true, y_pred):
    dssim = DSSIMObjective()
    loss_mae = keras.losses.mae(y_true, y_pred)
    loss_mse = keras.losses.mse(y_true, y_pred)
    loss_dssim = dssim(y_true, y_pred)
    return 0.5 * loss_dssim + 0.3 * loss_mae + 0.2 * loss_mse


def superloss_8(y_true, y_pred):
    dssim = DSSIMObjective()
    dstsim = DSTSIMObjective()
    loss_mae = keras.losses.mae(y_true, y_pred)
    loss_mse = keras.losses.mse(y_true, y_pred)
    loss_dssim = dssim(y_true, y_pred)
    loss_dstsim = dstsim(y_true, y_pred)
    #loss_dstsim = K.print_tensor(loss_dstsim, message='loss_stsim')
    #loss_dssim = K.print_tensor(loss_dssim, message='loss_ssim')
    return 0.4 * loss_mse + 0.4 * loss_dstsim + 0.2 * loss_mae


def superloss_9(y_true, y_pred):
    loss_sobel = sobelLoss_mse(y_true, y_pred)
    dssim = DSSIMObjective()
    loss_dssim = dssim(y_true, y_pred)
    loss_mae = keras.losses.mae(y_true, y_pred)
    loss_mse = keras.losses.mse(y_true, y_pred)
    return 0.4 * loss_sobel + 0.4 * loss_dssim + 0.1 * loss_mse + 0.1 * loss_mae

def superloss_10(y_true, y_pred):
    loss_sobel = sobelLoss_mse(y_true, y_pred)
    dssim = DSSIMObjective(kernel_size=5)
    loss_dssim = dssim(y_true, y_pred)
    loss_mae = keras.losses.mae(y_true, y_pred)
    loss_mse = keras.losses.mse(y_true, y_pred)
    return 0.3 * loss_sobel + 0.3 * loss_dssim + 0.2 * loss_mse + 0.2 * loss_mae

def superloss_11(y_true, y_pred):
    dssim = DSSIM_SQU(kernel_size=5)
    loss_mae = keras.losses.mae(y_true, y_pred)
    loss_mse = keras.losses.mse(y_true, y_pred)
    loss_dssim = dssim(y_true, y_pred)
    return 0.4 * loss_dssim + 0.5 * loss_mae + 0.1 * loss_mse

def superloss_12(y_true, y_pred):
    dssim = DSSIM_ContentWeighted(kernel_size=5)
    loss_mae = keras.losses.mae(y_true, y_pred)
    loss_mse = keras.losses.mse(y_true, y_pred)
    loss_dssim = dssim(y_true, y_pred)
    return 0.5 * loss_dssim + 0.4 * loss_mae + 0.1 * loss_mse