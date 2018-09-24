import tensorflow as tf
import keras
from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Lambda, BatchNormalization, SeparableConv2D, Dense
from keras import regularizers
from keras import initializers

from metrics import DSSIMObjective

def bilinear_interp(im, x, y, name):
    with tf.variable_scope(name):
        x = tf.reshape(x, [-1])
        y = tf.reshape(y, [-1])

        # constants
        num_batch = tf.shape(im)[0]
        _, height, width, channels = im.get_shape().as_list()

        x = tf.to_float(x)
        y = tf.to_float(y)

        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        zero = tf.constant(0, dtype=tf.int32)

        max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')
        max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
        x = (x + 1.0) * (width_f - 1.0) / 2.0
        y = (y + 1.0) * (height_f - 1.0) / 2.0

        # Sampling
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        dim2 = width
        dim1 = width * height

        # Create base index
        base = tf.range(num_batch) * dim1
        base = tf.reshape(base, [-1, 1])
        base = tf.tile(base, [1, height * width])
        base = tf.reshape(base, [-1])

        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # Use indices to look up pixels
        im_flat = tf.reshape(im, tf.stack([-1, channels]))
        im_flat = tf.to_float(im_flat)
        pixel_a = tf.gather(im_flat, idx_a)
        pixel_b = tf.gather(im_flat, idx_b)
        pixel_c = tf.gather(im_flat, idx_c)
        pixel_d = tf.gather(im_flat, idx_d)

        # Interpolate the values
        x1_f = tf.to_float(x1)
        y1_f = tf.to_float(y1)

        wa = tf.expand_dims((x1_f - x) * (y1_f - y), 1)
        wb = tf.expand_dims((x1_f - x) * (1.0 - (y1_f - y)), 1)
        wc = tf.expand_dims((1.0 - (x1_f - x)) * (y1_f - y), 1)
        wd = tf.expand_dims((1.0 - (x1_f - x)) * (1.0 - (y1_f - y)), 1)

        output = tf.add_n(
            [wa * pixel_a, wb * pixel_b, wc * pixel_c, wd * pixel_d])
        output = tf.reshape(
            output, shape=tf.stack([num_batch, height, width, channels]))
        return output


def meshgrid(height, width):
    """Tensorflow meshgrid function.
  """
    with tf.variable_scope('meshgrid'):
        x_t = tf.matmul(
            tf.ones(shape=tf.stack([height, 1])),
            tf.transpose(
                tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(
            tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
            tf.ones(shape=tf.stack([1, width])))
        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))
        # grid_x = tf.reshape(x_t_flat, [1, height, width, 1])
        # grid_y = tf.reshape(y_t_flat, [1, height, width, 1])
        grid_x = tf.reshape(x_t_flat, [1, height, width])
        grid_y = tf.reshape(y_t_flat, [1, height, width])
        return grid_x, grid_y

def scale_mask(mask):
    mask = 0.5 * (1.0 + mask)
    mask = tf.tile(mask, [1, 1, 1, 3])
    return mask


def combine_pyramid_4p(x):
    conv8 = x[:, :, :, :5]
    input_image = x[:, :, :, 5:]

    #pyramid Flow
    flow = conv8[:, :, :, 0:2]
    mask_12 = tf.expand_dims(conv8[:, :, :, 2], 3)
    mask_34 = tf.expand_dims(conv8[:, :, :, 3], 3)
    mask_c = tf.expand_dims(conv8[:, :, :, 4], 3)

    base_x, base_y = meshgrid(tf.shape(conv8)[1], tf.shape(conv8)[2])
    base_x = tf.tile(base_x, [tf.shape(conv8)[0], 1, 1])
    base_y = tf.tile(base_y, [tf.shape(conv8)[0], 1, 1])

    flow = 0.25 * flow

    coor_x_1 = base_x - 2 * flow[:, :, :, 0]
    coor_y_1 = base_y - 2 * flow[:, :, :, 1]

    coor_x_2 = base_x - flow[:, :, :, 0]
    coor_y_2 = base_y - flow[:, :, :, 1]

    coor_x_3 = base_x + flow[:, :, :, 0]
    coor_y_3 = base_y + flow[:, :, :, 1]

    coor_x_4 = base_x + 2 * flow[:, :, :, 0]
    coor_y_4 = base_y + 2 * flow[:, :, :, 1]

    output_1 = bilinear_interp(input_image[:, :, :, 0:3], coor_x_1, coor_y_1,
                               'interpolate')
    output_2 = bilinear_interp(input_image[:, :, :, 3:6], coor_x_2, coor_y_2,
                               'interpolate')
    output_3 = bilinear_interp(input_image[:, :, :, 6:9], coor_x_3, coor_y_3,
                               'interpolate')
    output_4 = bilinear_interp(input_image[:, :, :, 9:12], coor_x_4, coor_y_4,
                               'interpolate')

    mask_12 = scale_mask(mask_12)
    mask_34 = scale_mask(mask_34)
    mask_c = scale_mask(mask_c)
    o_12 = tf.multiply(mask_12, output_1) + tf.multiply(
        1.0 - mask_12, output_2)
    o_34 = tf.multiply(mask_34, output_3) + tf.multiply(
        1.0 - mask_34, output_4)
    output = tf.multiply(mask_c, o_12) + tf.multiply(1.0 - mask_c, o_34)

    return output

def lambda_upsampling(src, target):
    _, target_w, target_h, _ = target.shape
    return Lambda(lambda x: tf.image.resize_nearest_neighbor(x, (target_w, target_h)))(src)

def pyramid_model(input_shape):
    inputs = Input(input_shape)
    input_scale = Lambda(lambda x: x * 2.0 - 1.0)(inputs)
    
    # Mask Model
    conv1 = Conv2D(
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding='same',
        data_format='channels_last',
        activation='relu')(input_scale)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)
    conv2 = Conv2D(
        filters=128,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding='same',
        activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu')(pool3)
    up4 = lambda_upsampling(conv4, conv3)
    #up4 = UpSampling2D(size=(2, 2))(conv4)
    concat4 = concatenate([up4, conv3])
    conv5 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu')(concat4)
    up5 = lambda_upsampling(conv5, conv2)
    #up5 = UpSampling2D(size=(2, 2))(conv5)
    concat6 = concatenate([up5, conv2])
    conv6 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu')(concat6)
    up6 = lambda_upsampling(conv6, conv1)
    #up6 = UpSampling2D(size=(2, 2))(conv6)
    concat7 = concatenate([up6, conv1])
    conv7 = Conv2D(
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding='same',
        activation='relu')(concat7)
    conv8 = Conv2D(
        filters=3,
        kernel_size=(5, 5),
        strides=(1, 1),
        activation='tanh',
        padding='same')(conv7)

    # Flow model
    f_input_pool1 = MaxPooling2D(pool_size=(2, 2))(input_scale)
    f_input_pool2 = MaxPooling2D()(f_input_pool1)
    f_input_pool3 = MaxPooling2D()(f_input_pool2)

    f_p0_conv1 = Conv2D(
        filters=64,
        kernel_size=(3,3),
        padding='same',
        activation='relu',
        data_format='channels_last')(inputs)
    f_p0_pool1 = MaxPooling2D()(f_p0_conv1)
    f_p0_conv2 = Conv2D(
        filters=128,
        kernel_size=(3,3),
        padding='same',
        activation='relu')(f_p0_pool1)
    f_p0_pool2 = MaxPooling2D()(f_p0_conv2)
    f_p0_conv3 = Conv2D(
        filters=256,
        kernel_size=(3,3),
        padding='same',
        activation='relu')(f_p0_pool2)
    f_p0_pool3 = MaxPooling2D()(f_p0_conv3)
    f_p0_conv4 = Conv2D(
        filters=512,
        kernel_size=(3,3),
        padding='same',
        activation='relu')(f_p0_pool3)
    f_p0_up4 = lambda_upsampling(f_p0_conv4, f_p0_conv3)
    #f_p0_up4 = UpSampling2D()(f_p0_conv4)
    f_p0_concat4 = concatenate([f_p0_up4,f_p0_conv3])
    f_p0_conv5 = Conv2D(
        filters=256,
        kernel_size=(3,3),
        padding='same',
        activation='relu')(f_p0_concat4)
    f_p0_up5 = lambda_upsampling(f_p0_conv5, f_p0_conv2)
    #f_p0_up5 = UpSampling2D()(f_p0_conv5)
    f_p0_concat5 = concatenate([f_p0_up5, f_p0_conv2])
    f_p0_conv6 = Conv2D(
        filters=128,
        kernel_size=(3,3),
        padding='same',
        activation='relu')(f_p0_concat5)
    f_p0_up6 = lambda_upsampling(f_p0_conv6, f_p0_conv1)
    #f_p0_up6 = UpSampling2D()(f_p0_conv6)
    f_p0_concat6 = concatenate([f_p0_up6, f_p0_conv1])
    f_p0_conv7 = Conv2D(
        filters=64,
        kernel_size=(3,3),
        padding='same',
        activation='relu')(f_p0_concat6)
    f_p0_flow = Conv2D(
        filters=2,
        kernel_size=(3,3),
        padding='same',
        activation='tanh')(f_p0_conv7)
    
    f_p1_conv1 = Conv2D(
        filters=128,
        kernel_size=(3,3),
        padding='same',
        activation='relu',
        data_format='channels_last')(f_input_pool1)
    f_p1_pool1 = MaxPooling2D()(f_p1_conv1)
    f_p1_conv2 = Conv2D(
        filters=256,
        kernel_size=(3,3),
        padding='same',
        activation='relu')(f_p1_pool1)
    f_p1_pool2 = MaxPooling2D()(f_p1_conv2)
    f_p1_conv3 = Conv2D(
        filters=512,
        kernel_size=(3,3),
        padding='same',
        activation='relu')(f_p1_pool2)
    f_p1_up3 = lambda_upsampling(f_p1_conv3, f_p1_conv2)
    #f_p1_up3 = UpSampling2D()(f_p1_conv3)
    f_p1_concat3 = concatenate([f_p1_up3, f_p1_conv2])
    f_p1_conv4 = Conv2D(
        filters=256,
        kernel_size=(3,3),
        padding='same',
        activation='relu')(f_p1_concat3)
    f_p1_up4 = lambda_upsampling(f_p1_conv4, f_p1_conv1)
    #f_p1_up4 = UpSampling2D()(f_p1_conv4)
    f_p1_concat4 = concatenate([f_p1_up4, f_p1_conv1])
    f_p1_conv5 = Conv2D(
        filters=128,
        kernel_size=(3,3),
        padding='same',
        activation='relu')(f_p1_concat4)
    f_p1_flow = Conv2D(
        filters=2,
        kernel_size=(3,3),
        padding='same',
        activation='tanh')(f_p1_conv5)

    f_p2_conv1 = Conv2D(
        filters=256,
        kernel_size=(3,3),
        padding='same',
        activation='relu',
        data_format='channels_last')(f_input_pool2)
    f_p2_pool1 = MaxPooling2D()(f_p2_conv1)
    f_p2_conv2 = Conv2D(
        filters=512,
        kernel_size=(3,3),
        padding='same',
        activation='relu')(f_p2_pool1)
    f_p2_conv3 = Conv2D(
        filters=512,
        kernel_size=(3,3),
        padding='same',
        activation='relu')(f_p2_conv2)
    f_p2_up3 = UpSampling2D()(f_p2_conv3)
    f_p2_conv4 = Conv2D(
        filters=256,
        kernel_size=(3,3),
        padding='same',
        activation='relu')(f_p2_up3)
    f_p2_flow = Conv2D(
        filters=2,
        kernel_size=(3,3),
        padding='same',
        activation='tanh')(f_p2_conv4)
    
    f_p3_conv1 = Conv2D(
        filters=512,
        kernel_size=(3,3),
        padding='same',
        activation='relu',
        data_format='channels_last')(f_input_pool3)
    f_p3_conv2 = Conv2D(
        filters=512,
        kernel_size=(3,3),
        padding='same',
        activation='relu')(f_p3_conv1)
    f_p3_conv3 = Conv2D(
        filters=512,
        kernel_size=(3,3),
        padding='same',
        activation='relu')(f_p3_conv2)
    f_p3_flow = Conv2D(
        filters=2,
        kernel_size=(3,3),
        padding='same',
        activation='tanh')(f_p3_conv3)

    #f_p3_flow_up = UpSampling2D()(f_p3_flow)
    f_p3_flow_up = lambda_upsampling(f_p3_flow, f_p2_flow)
    f_p2_flow_concat = concatenate([f_p2_flow, f_p3_flow_up])
    f_p2_flow_conv1 = Conv2D(
        filters=64,
        kernel_size=(3,3),
        padding='same',
        activation='relu')(f_p2_flow_concat)
    f_p2_flow_conv2 = Conv2D(
        filters=64,
        kernel_size=(3,3),
        padding='same',
        activation='relu')(f_p2_flow_conv1)
    f_p2_flow_combine = Conv2D(
        filters=2,
        kernel_size=(3,3),
        padding='same',
        activation='tanh')(f_p2_flow_conv2)

    f_p2_flow_up = lambda_upsampling(f_p2_flow_combine, f_p1_flow)
    #f_p2_flow_up = UpSampling2D()(f_p2_flow_combine)
    f_p1_flow_concat = concatenate([f_p1_flow, f_p2_flow_up])
    f_p1_flow_conv1 = Conv2D(
        filters=128,
        kernel_size=(3,3),
        padding='same',
        activation='relu')(f_p1_flow_concat)
    f_p1_flow_conv2 = Conv2D(
        filters=128,
        kernel_size=(3,3),
        padding='same',
        activation='relu')(f_p1_flow_conv1)
    f_p1_flow_combine = Conv2D(
        filters=2,
        kernel_size=(3,3),
        padding='same',
        activation='tanh')(f_p1_flow_conv2)

    f_p1_flow_up = lambda_upsampling(f_p1_flow_combine, f_p0_flow)
    #f_p1_flow_up = UpSampling2D()(f_p1_flow_combine)
    f_p0_flow_concat = concatenate([f_p0_flow, f_p1_flow_up])
    f_p0_flow_conv1 = Conv2D(
        filters=512,
        kernel_size=(3,3),
        padding='same',
        activation='relu')(f_p0_flow_concat)
    f_p0_flow_conv2 = Conv2D(
        filters=512,
        kernel_size=(3,3),
        padding='same',
        activation='relu')(f_p0_flow_conv1)
    f_p0_flow_combine = Conv2D(
        filters=2,
        kernel_size=(3,3),
        padding='same',
        activation='tanh')(f_p0_flow_conv2)
    
    flow_and_mask = concatenate([f_p0_flow_combine, conv8])

    #pyramid Prepare
    pre_pyramid = concatenate([flow_and_mask, inputs])
    pyramid_layer = Lambda(function=combine_pyramid_4p)(pre_pyramid)

    model = Model(inputs=inputs, outputs=pyramid_layer)

    return model

    