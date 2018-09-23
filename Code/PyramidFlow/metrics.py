import keras.backend as K
import tensorflow as tf
import keras.losses
import numpy
import math
import sys
from scipy import signal
from scipy import ndimage


def tf_decimalround(x, decimals=2):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier


def nan_to_num(x):
    return tf.where(tf.is_nan(x), tf.zeros_like(x), x)


class DSTSIMObjective():
    def __init__(self, k1=0.01, k2=0.03, k3=0.02, kernel_size=3,
                 max_value=1.0):
        self.__name__ = 'STSIMObjective'
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.max_value = max_value
        self.c1 = (self.k1 * self.max_value)**2
        self.c2 = (self.k2 * self.max_value)**2
        self.c3 = (self.k3 * self.max_value)**2
        self.dim_ordering = K.image_data_format()
        self.backend = K.backend()

    def __int_shape(self, x):
        return K.int_shape(x)

    def __call__(self, y_true, y_pred):
        #y_pred = K.print_tensor(y_pred, message='pred')
        kernel = [1, self.kernel_size, self.kernel_size, 1]
        y_true = K.reshape(y_true, [-1] + list(self.__int_shape(y_pred)[1:]))
        y_pred = K.reshape(y_pred, [-1] + list(self.__int_shape(y_pred)[1:]))

        patches_pred = tf.extract_image_patches(
            images=y_pred,
            ksizes=kernel,
            strides=kernel,
            rates=[1, 1, 1, 1],
            padding='SAME')
        patches_true = tf.extract_image_patches(
            images=y_true,
            ksizes=kernel,
            strides=kernel,
            rates=[1, 1, 1, 1],
            padding='SAME')

        # Get STSIM Line Shuffled Patches
        patches_pred_line_shuffled = tf.concat(
            [patches_pred[:, 1:, :, :], patches_pred[:, :1, :, :]], axis=1)
        patches_pred_line_shuffled = tf.reshape(patches_pred_line_shuffled,
                                                tf.shape(patches_pred))
        patches_pred_col_shuffled = tf.concat(
            [patches_pred[:, :, :1, :], patches_pred[:, :, 1:, :]], axis=2)
        patches_pred_col_shuffled = tf.reshape(patches_pred_col_shuffled,
                                               tf.shape(patches_pred))

        patches_true_line_shuffled = tf.concat(
            [patches_true[:, 1:, :, :], patches_true[:, :1, :, :]], axis=1)
        patches_true_line_shuffled = tf.reshape(patches_true_line_shuffled,
                                                tf.shape(patches_true))
        patches_true_col_shuffled = tf.concat(
            [patches_true[:, :, :1, :], patches_true[:, :, 1:, :]], axis=2)
        patches_true_col_shuffled = tf.reshape(patches_true_col_shuffled,
                                               tf.shape(patches_true))

        # Get mean
        u_true = K.mean(patches_true, axis=-1)
        u_pred = K.mean(patches_pred, axis=-1)
        u_true_ls = K.mean(patches_true_line_shuffled, axis=-1)
        u_pred_ls = K.mean(patches_pred_line_shuffled, axis=-1)
        u_true_cs = K.mean(patches_true_col_shuffled, axis=-1)
        u_pred_cs = K.mean(patches_pred_col_shuffled, axis=-1)
        # Get std dev
        std_true = K.std(patches_true, axis=-1)
        std_true_cs = K.std(patches_true_col_shuffled, axis=-1)
        std_true_ls = K.std(patches_true_line_shuffled, axis=-1)
        std_pred = K.std(patches_pred, axis=-1)
        std_pred_cs = K.std(patches_pred_col_shuffled, axis=-1)
        std_pred_ls = K.std(patches_pred_line_shuffled, axis=-1)
        covar_true_pred = K.mean(
            patches_true * patches_pred, axis=-1) - u_true * u_pred
        # Get variance
        var_true = K.var(patches_true, axis=-1)
        var_pred = K.var(patches_pred, axis=-1)
        #var_pred = K.print_tensor(var_pred, message='var_pred')

        # x: true; y: pred
        l_xy = (2 * u_true * u_pred + self.c1) / (
            K.square(u_true) + K.square(u_pred) + self.c1)
        c_xy = (2 * std_true * std_pred + self.c2) / (
            var_true + var_pred + self.c2)
        rx_01 = (K.mean(patches_true * patches_true_col_shuffled, axis=-1) -
                 u_true * u_true_cs + self.c3) / (
                     std_true * std_true_cs + self.c3)
        rx_10 = (K.mean(patches_true * patches_true_line_shuffled, axis=-1) -
                 u_true * u_true_ls + self.c3) / (
                     std_true * std_true_ls + self.c3)
        ry_01 = (K.mean(patches_pred * patches_pred_col_shuffled, axis=-1) -
                 u_pred * u_pred_cs + self.c3) / (
                     std_pred * std_pred_cs + self.c3)
        ry_10 = (K.mean(patches_pred * patches_pred_line_shuffled, axis=-1) -
                 u_pred * u_pred_ls + self.c3) / (
                     std_pred * std_pred_ls + self.c3)
        #ry_01 = K.print_tensor(ry_01, message='ry01')
        #ry_10 = K.print_tensor(ry_10, message='ry10')
        c_01_xy = 1.0 - 0.5 * abs(rx_01 - ry_01)
        c_10_xy = 1.0 - 0.5 * abs(rx_10 - ry_10)

        #stsim = K.sqrt(K.sqrt(l_xy * c_xy * c_01_xy * c_10_xy))
        stsim = K.sqrt(l_xy * c_xy * c_01_xy * c_10_xy)
        mean_stsim = K.mean(stsim)
        mean_dstsim = 1.0 - mean_stsim
        return mean_dstsim


class DSSIMObjective():
    def __init__(self, k1=0.01, k2=0.03, kernel_size=3, max_value=1.0):
        """
        Difference of Structural Similarity (DSSIM loss function). Clipped between 0 and 0.5
        Note : You should add a regularization term like a l2 loss in addition to this one.
        Note : In theano, the `kernel_size` must be a factor of the output size. So 3 could
               not be the `kernel_size` for an output of 32.
        # Arguments
            k1: Parameter of the SSIM (default 0.01)
            k2: Parameter of the SSIM (default 0.03)
            kernel_size: Size of the sliding window (default 3)
            max_value: Max value of the output (default 1.0)
        """
        self.__name__ = 'DSSIMObjective'
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.c1 = (self.k1 * self.max_value)**2
        self.c2 = (self.k2 * self.max_value)**2
        self.dim_ordering = K.image_data_format()
        self.backend = K.backend()

    def __int_shape(self, x):
        return K.int_shape(x)

    def __call__(self, y_true, y_pred):
        # There are additional parameters for this function
        # Note: some of the 'modes' for edge behavior do not yet have a gradient definition in the Theano tree
        #   and cannot be used for learning

        #print(y_true)

        kernel = [1, self.kernel_size, self.kernel_size, 1]
        y_true = K.reshape(y_true, [-1] + list(self.__int_shape(y_pred)[1:]))
        y_pred = K.reshape(y_pred, [-1] + list(self.__int_shape(y_pred)[1:]))

        #print(y_true)

        patches_pred = tf.extract_image_patches(
            images=y_pred,
            ksizes=kernel,
            strides=kernel,
            rates=[1, 1, 1, 1],
            padding='SAME')
        patches_true = tf.extract_image_patches(
            images=y_true,
            ksizes=kernel,
            strides=kernel,
            rates=[1, 1, 1, 1],
            padding='SAME')

        #print(patches_true)

        # Get mean
        u_true = K.mean(patches_true, axis=-1)
        u_pred = K.mean(patches_pred, axis=-1)
        # Get variance
        var_true = K.var(patches_true, axis=-1)
        var_pred = K.var(patches_pred, axis=-1)
        # Get std dev
        covar_true_pred = K.mean(
            patches_true * patches_pred, axis=-1) - u_true * u_pred

        #print(var_true)

        ssim = (2 * u_true * u_pred + self.c1) * \
            (2 * covar_true_pred + self.c2)
        denom = (K.square(u_true) + K.square(u_pred) + self.c1) * \
            (var_pred + var_true + self.c2)
        ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
        ssim = tf.clip_by_value(ssim, -1.0, 1.0)

        #print(ssim)

        #print(ssim.eval(session=tf.Session()))
        return K.mean((1.0 - ssim) / 2.0)


class DSSIM_ContentWeighted():
    def __init__(self, k1=0.01, k2=0.03, kernel_size=3, max_value=1.0):
        """
        Difference of Structural Similarity (DSSIM loss function). Clipped between 0 and 0.5
        Note : You should add a regularization term like a l2 loss in addition to this one.
        Note : In theano, the `kernel_size` must be a factor of the output size. So 3 could
               not be the `kernel_size` for an output of 32.
        # Arguments
            k1: Parameter of the SSIM (default 0.01)
            k2: Parameter of the SSIM (default 0.03)
            kernel_size: Size of the sliding window (default 3)
            max_value: Max value of the output (default 1.0)
        """
        self.__name__ = 'SSIM_ContentWeighted'
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.c1 = (self.k1 * self.max_value)**2
        self.c2 = (self.k2 * self.max_value)**2
        self.dim_ordering = K.image_data_format()
        self.backend = K.backend()

    def __int_shape(self, x):
        return K.int_shape(x)

    def __call__(self, y_true, y_pred):
        # There are additional parameters for this function
        # Note: some of the 'modes' for edge behavior do not yet have a gradient definition in the Theano tree
        #   and cannot be used for learning

        kernel = [1, self.kernel_size, self.kernel_size, 1]
        y_true = K.reshape(y_true, [-1] + list(self.__int_shape(y_pred)[1:]))
        y_pred = K.reshape(y_pred, [-1] + list(self.__int_shape(y_pred)[1:]))

        patches_pred = tf.extract_image_patches(
            images=y_pred,
            ksizes=kernel,
            strides=kernel,
            rates=[1, 1, 1, 1],
            padding='SAME')
        patches_true = tf.extract_image_patches(
            images=y_true,
            ksizes=kernel,
            strides=kernel,
            rates=[1, 1, 1, 1],
            padding='SAME')

        # Get mean
        u_true = K.mean(patches_true, axis=-1)
        u_pred = K.mean(patches_pred, axis=-1)
        # Get variance
        var_true = K.var(patches_true, axis=-1)
        var_pred = K.var(patches_pred, axis=-1)
        # Get std dev
        covar_true_pred = K.mean(
            patches_true * patches_pred, axis=-1) - u_true * u_pred


        ssim = (2 * u_true * u_pred + self.c1) * \
            (2 * covar_true_pred + self.c2)
        denom = (K.square(u_true) + K.square(u_pred) + self.c1) * \
            (var_pred + var_true + self.c2)
        ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
        ssim = tf.clip_by_value(ssim, -1.0, 1.0)

        content_weight = tf.log(
            tf.multiply(1.0 + var_true / self.c2, 1.0 + var_pred / self.c2))

        #print(ssim.eval(session=tf.Session()))
        return (1.0 - tf.reduce_sum(tf.multiply(ssim, content_weight)) / tf.reduce_sum(content_weight)) / 2.0


class DSSIM_SQU():
    def __init__(self, k1=0.01, k2=0.03, kernel_size=3, max_value=1.0):
        """
        Difference of Structural Similarity (DSSIM loss function). Clipped between 0 and 0.5
        Note : You should add a regularization term like a l2 loss in addition to this one.
        Note : In theano, the `kernel_size` must be a factor of the output size. So 3 could
               not be the `kernel_size` for an output of 32.
        # Arguments
            k1: Parameter of the SSIM (default 0.01)
            k2: Parameter of the SSIM (default 0.03)
            kernel_size: Size of the sliding window (default 3)
            max_value: Max value of the output (default 1.0)
        """
        self.__name__ = 'DSSIMObjective'
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.c1 = (self.k1 * self.max_value)**2
        self.c2 = (self.k2 * self.max_value)**2
        self.dim_ordering = K.image_data_format()
        self.backend = K.backend()

    def __int_shape(self, x):
        return K.int_shape(x)

    def __call__(self, y_true, y_pred):
        # There are additional parameters for this function
        # Note: some of the 'modes' for edge behavior do not yet have a gradient definition in the Theano tree
        #   and cannot be used for learning

        kernel = [1, self.kernel_size, self.kernel_size, 1]
        y_true = K.reshape(y_true, [-1] + list(self.__int_shape(y_pred)[1:]))
        y_pred = K.reshape(y_pred, [-1] + list(self.__int_shape(y_pred)[1:]))

        patches_pred = tf.extract_image_patches(
            images=y_pred,
            ksizes=kernel,
            strides=kernel,
            rates=[1, 1, 1, 1],
            padding='SAME')
        patches_true = tf.extract_image_patches(
            images=y_true,
            ksizes=kernel,
            strides=kernel,
            rates=[1, 1, 1, 1],
            padding='SAME')

        # Get mean
        u_true = K.mean(patches_true, axis=-1)
        u_pred = K.mean(patches_pred, axis=-1)
        # Get variance
        var_true = K.var(patches_true, axis=-1)
        var_pred = K.var(patches_pred, axis=-1)
        # Get std dev
        covar_true_pred = K.mean(
            patches_true * patches_pred, axis=-1) - u_true * u_pred

        ssim = (2 * u_true * u_pred + self.c1) * \
            (2 * covar_true_pred + self.c2)
        denom = (K.square(u_true) + K.square(u_pred) + self.c1) * \
            (var_pred + var_true + self.c2)
        ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
        ssim = tf.clip_by_value(ssim, -1.0, 1.0)
        dssim_squ = (K.square(2.0 - ssim) - 1.0) / 3.0
        #print(ssim.eval(session=tf.Session()))
        return K.mean(dssim_squ)


class DSSIM_Sum():
    def __init__(self, k1=0.01, k2=0.03, kernel_size=3, max_value=1.0):
        """
        Difference of Structural Similarity (DSSIM loss function). Clipped between 0 and 0.5
        Note : You should add a regularization term like a l2 loss in addition to this one.
        Note : In theano, the `kernel_size` must be a factor of the output size. So 3 could
               not be the `kernel_size` for an output of 32.
        # Arguments
            k1: Parameter of the SSIM (default 0.01)
            k2: Parameter of the SSIM (default 0.03)
            kernel_size: Size of the sliding window (default 3)
            max_value: Max value of the output (default 1.0)
        """
        self.__name__ = 'DSSIM_Sum'
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.c1 = (self.k1 * self.max_value)**2
        self.c2 = (self.k2 * self.max_value)**2
        self.dim_ordering = K.image_data_format()
        self.backend = K.backend()

    def __int_shape(self, x):
        return K.int_shape(x)

    def __call__(self, y_true, y_pred):
        # There are additional parameters for this function
        # Note: some of the 'modes' for edge behavior do not yet have a gradient definition in the Theano tree
        #   and cannot be used for learning

        kernel = [1, self.kernel_size, self.kernel_size, 1]
        y_true = K.reshape(y_true, [-1] + list(self.__int_shape(y_pred)[1:]))
        y_pred = K.reshape(y_pred, [-1] + list(self.__int_shape(y_pred)[1:]))

        patches_pred = tf.extract_image_patches(
            images=y_pred,
            ksizes=kernel,
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='SAME')
        patches_true = tf.extract_image_patches(
            images=y_true,
            ksizes=kernel,
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='SAME')

        # Get mean
        u_true = K.mean(patches_true, axis=-1)
        u_pred = K.mean(patches_pred, axis=-1)
        # Get variance
        var_true = K.var(patches_true, axis=-1)
        var_pred = K.var(patches_pred, axis=-1)
        # Get std dev
        covar_true_pred = K.mean(
            patches_true * patches_pred, axis=-1) - u_true * u_pred

        ssim = (2 * u_true * u_pred + self.c1) * \
            (2 * covar_true_pred + self.c2)
        denom = (K.square(u_true) + K.square(u_pred) + self.c1) * \
            (var_pred + var_true + self.c2)
        ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
        return K.sum((1.0 - ssim) / 2.0)


class Sobel():
    def __init__(self):
        self.sobelFilter = K.variable([[[[1., 1.]], [[0., 2.]], [[-1., 1.]]],
                                       [[[2., 0.]], [[0., 0.]], [[-2., 0.]]],
                                       [[[1., -1.]], [[0., -2.]], [[-1.,
                                                                    -1.]]]])

    def expandedSobel(self, inputTensor):
        inputChannels = K.reshape(
            K.ones_like(inputTensor[0, 0, 0, :]), (1, 1, -1, 1))
        return self.sobelFilter * inputChannels

    def __call__(self, x):
        filt = self.expandedSobel(x)
        sobel = K.depthwise_conv2d(x, filt)
        return sobel


def psnr(img1, img2):
    mse = numpy.mean((img1 - img2)**2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = numpy.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:
                       size // 2 + 1]
    g = numpy.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1 
    and img2 (images are assumed to be uint8)
    
    This function attempts to mimic precisely the functionality of ssim.m a 
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(numpy.float64)
    img2 = img2.astype(numpy.float64)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255  #bitdepth of image
    C1 = (K1 * L)**2
    C2 = (K2 * L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2
    if cs_map:
        return numpy.mean(
            (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) /
             ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)),
             (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)))
    else:
        return numpy.mean(((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) /
                          ((mu1_sq + mu2_sq + C1) *
                           (sigma1_sq + sigma2_sq + C2)))
