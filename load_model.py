from functools import reduce  # In Python 3, reduce() was moved into functools

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Input
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Activation, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras.utils import plot_model
from scipy import signal

# -----------------------------------------------------------------------------------------

# Functions from src/utils.py:


def gabor_fn(ksize, sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma
    # Bounding box
    nstds = 3
    xmax = ksize[0] / 2
    ymax = ksize[1] / 2
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gb_cos = np.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) * np.cos(
        2 * np.pi / Lambda * x_theta + psi
    )
    gb_sin = np.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) * np.sin(
        2 * np.pi / Lambda * x_theta + psi
    )
    return gb_cos, gb_sin


def gabor_bank(stride=2, Lambda=8):
    filters_cos = np.ones([25, 25, int(180 / stride)], dtype=float)
    filters_sin = np.ones([25, 25, int(180 / stride)], dtype=float)
    for n, i in enumerate(range(-90, 90, stride)):
        theta = i * np.pi / 180.0
        kernel_cos, kernel_sin = gabor_fn((24, 24), 4.5, -theta, Lambda, 0, 0.5)
        filters_cos[..., n] = kernel_cos
        filters_sin[..., n] = kernel_sin
    filters_cos = np.reshape(filters_cos, [25, 25, 1, -1])
    filters_sin = np.reshape(filters_sin, [25, 25, 1, -1])
    return filters_cos, filters_sin


def gausslabel(length=180, stride=2):
    gaussian_pdf = signal.gaussian(length + 1, 3)
    label = np.reshape(np.arange(stride / 2, length, stride), [1, 1, -1, 1])
    y = np.reshape(np.arange(stride / 2, length, stride), [1, 1, 1, -1])
    delta = np.array(np.abs(label - y), dtype=int)
    delta = np.minimum(delta, length - delta) + length / 2
    delta = delta.astype(np.int32)  # Ensure integer type for indexing
    return gaussian_pdf[delta]


# -----------------------------------------------------------------------------------------


# Functions from src/train_test_deploy.py:


def img_normalization(img_input, m0=0.0, var0=1.0):
    m = K.mean(img_input, axis=[1, 2, 3], keepdims=True)
    var = K.var(img_input, axis=[1, 2, 3], keepdims=True)
    after = K.sqrt(var0 * tf.square(img_input - m) / var)
    image_n = tf.where(tf.greater(img_input, m), m0 + after, m0 - after)
    return image_n


def atan2(y_x):
    y, x = y_x[0], y_x[1] + K.epsilon()
    atan = tf.atan(y / x)
    angle = tf.where(tf.greater(x, 0.0), atan, tf.zeros_like(x))
    angle = tf.where(tf.logical_and(tf.less(x, 0.0), tf.greater_equal(y, 0.0)), atan + np.pi, angle)
    angle = tf.where(tf.logical_and(tf.less(x, 0.0), tf.less(y, 0.0)), atan - np.pi, angle)
    return angle


def merge_mul(x):
    return reduce(lambda x, y: x * y, x)


def merge_sum(x):
    return reduce(lambda x, y: x + y, x)


def reduce_sum(x):
    return K.sum(x, axis=-1, keepdims=True)


def merge_concat(x):
    return tf.concat(x, 3)


def select_max(x):
    x = x / (K.max(x, axis=-1, keepdims=True) + K.epsilon())
    x = tf.where(tf.greater(x, 0.999), x, tf.zeros_like(x))  # select the biggest one
    x = x / (K.sum(x, axis=-1, keepdims=True) + K.epsilon())  # prevent two or more ori is selected
    return x


def conv_bn_prelu(bottom, w_size, name, strides=(1, 1), dilation_rate=(1, 1)):
    if dilation_rate == (1, 1):
        conv_type = "conv"
    else:
        conv_type = "atrousconv"
    top = Conv2D(
        w_size[0],
        (w_size[1], w_size[2]),
        kernel_regularizer=l2(5e-5),
        padding="same",
        strides=strides,
        dilation_rate=dilation_rate,
        name=conv_type + name,
    )(bottom)
    top = BatchNormalization(name="bn-" + name)(top)
    top = PReLU(alpha_initializer="zero", shared_axes=[1, 2], name="prelu-" + name)(top)
    return top


# find highest peak using gaussian
def ori_highest_peak(y_pred, length=180):
    glabel = gausslabel(length=length, stride=2).astype(np.float32)
    ori_gau = K.conv2d(y_pred, glabel, padding="same")
    return ori_gau


def get_main_net(input_shape=(512, 512, 1), weights_path=None, mode="deploy"):
    img_input = Input(input_shape)
    bn_img = Lambda(img_normalization, name="img_norm")(img_input)
    # feature extraction VGG
    conv = conv_bn_prelu(bn_img, (64, 3, 3), "1_1")
    conv = conv_bn_prelu(conv, (64, 3, 3), "1_2")
    conv = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv)

    conv = conv_bn_prelu(conv, (128, 3, 3), "2_1")
    conv = conv_bn_prelu(conv, (128, 3, 3), "2_2")
    conv = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv)

    conv = conv_bn_prelu(conv, (256, 3, 3), "3_1")
    conv = conv_bn_prelu(conv, (256, 3, 3), "3_2")
    conv = conv_bn_prelu(conv, (256, 3, 3), "3_3")
    conv = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv)

    # multi-scale ASPP
    scale_1 = conv_bn_prelu(conv, (256, 3, 3), "4_1", dilation_rate=(1, 1))
    ori_1 = conv_bn_prelu(scale_1, (128, 1, 1), "ori_1_1")
    ori_1 = Conv2D(90, (1, 1), padding="same", name="ori_1_2")(ori_1)
    seg_1 = conv_bn_prelu(scale_1, (128, 1, 1), "seg_1_1")
    seg_1 = Conv2D(1, (1, 1), padding="same", name="seg_1_2")(seg_1)

    scale_2 = conv_bn_prelu(conv, (256, 3, 3), "4_2", dilation_rate=(4, 4))
    ori_2 = conv_bn_prelu(scale_2, (128, 1, 1), "ori_2_1")
    ori_2 = Conv2D(90, (1, 1), padding="same", name="ori_2_2")(ori_2)
    seg_2 = conv_bn_prelu(scale_2, (128, 1, 1), "seg_2_1")
    seg_2 = Conv2D(1, (1, 1), padding="same", name="seg_2_2")(seg_2)

    scale_3 = conv_bn_prelu(conv, (256, 3, 3), "4_3", dilation_rate=(8, 8))
    ori_3 = conv_bn_prelu(scale_3, (128, 1, 1), "ori_3_1")
    ori_3 = Conv2D(90, (1, 1), padding="same", name="ori_3_2")(ori_3)
    seg_3 = conv_bn_prelu(scale_3, (128, 1, 1), "seg_3_1")
    seg_3 = Conv2D(1, (1, 1), padding="same", name="seg_3_2")(seg_3)

    # sum fusion for ori
    ori_out = Lambda(merge_sum)([ori_1, ori_2, ori_3])
    ori_out_1 = Activation("sigmoid", name="ori_out_1")(ori_out)
    ori_out_2 = Activation("sigmoid", name="ori_out_2")(ori_out)

    # sum fusion for segmentation
    seg_out = Lambda(merge_sum)([seg_1, seg_2, seg_3])
    seg_out = Activation("sigmoid", name="seg_out")(seg_out)
    # ----------------------------------------------------------------------------
    # enhance part
    filters_cos, filters_sin = gabor_bank(stride=2, Lambda=8)
    filter_img_real = Conv2D(
        filters_cos.shape[3],
        (filters_cos.shape[0], filters_cos.shape[1]),
        weights=[filters_cos, np.zeros([filters_cos.shape[3]])],
        padding="same",
        name="enh_img_real_1",
    )(img_input)
    filter_img_imag = Conv2D(
        filters_sin.shape[3],
        (filters_sin.shape[0], filters_sin.shape[1]),
        weights=[filters_sin, np.zeros([filters_sin.shape[3]])],
        padding="same",
        name="enh_img_imag_1",
    )(img_input)
    ori_peak = Lambda(ori_highest_peak)(ori_out_1)
    ori_peak = Lambda(select_max)(ori_peak)  # select max ori and set it to 1
    upsample_ori = UpSampling2D(size=(8, 8))(ori_peak)
    seg_round = Activation("softsign")(seg_out)
    upsample_seg = UpSampling2D(size=(8, 8))(seg_round)
    mul_mask_real = Lambda(merge_mul)([filter_img_real, upsample_ori])
    enh_img_real = Lambda(reduce_sum, name="enh_img_real_2")(mul_mask_real)
    mul_mask_imag = Lambda(merge_mul)([filter_img_imag, upsample_ori])
    enh_img_imag = Lambda(reduce_sum, name="enh_img_imag_2")(mul_mask_imag)
    enh_img = Lambda(atan2, name="phase_img")([enh_img_imag, enh_img_real])
    enh_seg_img = Lambda(merge_concat, name="phase_seg_img")([enh_img, upsample_seg])
    # ----------------------------------------------------------------------------
    # mnt part
    mnt_conv = conv_bn_prelu(enh_seg_img, (64, 9, 9), "mnt_1_1")
    mnt_conv = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(mnt_conv)

    mnt_conv = conv_bn_prelu(mnt_conv, (128, 5, 5), "mnt_2_1")
    mnt_conv = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(mnt_conv)

    mnt_conv = conv_bn_prelu(mnt_conv, (256, 3, 3), "mnt_3_1")
    mnt_conv = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(mnt_conv)

    mnt_o_1 = Lambda(merge_concat)([mnt_conv, ori_out_1])
    mnt_o_2 = conv_bn_prelu(mnt_o_1, (256, 1, 1), "mnt_o_1_1")
    mnt_o_3 = Conv2D(180, (1, 1), padding="same", name="mnt_o_1_2")(mnt_o_2)
    mnt_o_out = Activation("sigmoid", name="mnt_o_out")(mnt_o_3)

    mnt_w_1 = conv_bn_prelu(mnt_conv, (256, 1, 1), "mnt_w_1_1")
    mnt_w_2 = Conv2D(8, (1, 1), padding="same", name="mnt_w_1_2")(mnt_w_1)
    mnt_w_out = Activation("sigmoid", name="mnt_w_out")(mnt_w_2)

    mnt_h_1 = conv_bn_prelu(mnt_conv, (256, 1, 1), "mnt_h_1_1")
    mnt_h_2 = Conv2D(8, (1, 1), padding="same", name="mnt_h_1_2")(mnt_h_1)
    mnt_h_out = Activation("sigmoid", name="mnt_h_out")(mnt_h_2)

    mnt_s_1 = conv_bn_prelu(mnt_conv, (256, 1, 1), "mnt_s_1_1")
    mnt_s_2 = Conv2D(1, (1, 1), padding="same", name="mnt_s_1_2")(mnt_s_1)
    mnt_s_out = Activation("sigmoid", name="mnt_s_out")(mnt_s_2)

    if mode == "deploy":
        model = Model(
            inputs=[
                img_input,
            ],
            outputs=[enh_img_real, ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out],
        )
    else:
        model = Model(
            inputs=[
                img_input,
            ],
            outputs=[ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out],
        )
    if weights_path:
        model.load_weights(weights_path, by_name=True)
    return model


# -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    model = get_main_net(weights_path="models/released_version/Model.model", mode="deploy")
    model.summary()
