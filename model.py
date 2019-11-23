import tensorflow as tf
import numpy as np
from conf_tab import config
batch_norm = config.TRAIN.batch_norm
kernel_size = config.TRAIN.kernel_size
# import src.flownet2.flownet2 as flownet2
# batch_size = config.TRAIN.per_gpu_batch_size
# from src.training_schedules import LONG_SCHEDULE
from copy import deepcopy
# from skimage.io import imread
# from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
from my_pwc_net import nn

def conv2d_padding_same(inputs, numfilter, kernel_size=3, trainable=True, activate=None):
    return tf.layers.conv2d(inputs, numfilter, kernel_size, padding='same', kernel_initializer=tf.variance_scaling_initializer(),
                            trainable=trainable, activation=activate)


def batchnorm(inputs, training):
    return tf.layers.batch_normalization(inputs, training=training)

def maxpool2d_same(inputs, poolsize=2, stride=2):
    return tf.layers.max_pooling2d(inputs, pool_size=poolsize, strides=stride, padding='same')


def bilinear_interp(im, x, y, name):
    """Perform bilinear sampling on im given x, y coordinates

    This function implements the differentiable sampling mechanism with
    bilinear kernel. Introduced in https://arxiv.org/abs/1506.02025, equation
    (5).

    x,y are tensors specfying normalized coorindates [-1,1] to sample from im.
    (-1,1) means (0,0) coordinate in im. (1,1) means the most bottom right pixel.
    Args:
      im: Tensor of size [batch_size, height, width, depth]
      x: Tensor of size [batch_size, height, width, 1]
      y: Tensor of size [batch_size, height, width, 1]
      name: String for the name for this opt.
    Returns:
      Tensor of size [batch_size, height, width, depth]
    """
    with tf.variable_scope(name):
        x = tf.reshape(x, [-1])
        y = tf.reshape(y, [-1])

        # constants
        num_batch = tf.shape(im)[0]
        # _, height, width, channels = im.get_shape().as_list()
        height, width, channels = tf.shape(im)[1], tf.shape(im)[2], tf.shape(im)[3]
        # x = tf.to_float(x)
        # y = tf.to_float(y)

        # height_f = tf.cast(height, 'float32')
        # width_f = tf.cast(width, 'float32')
        zero = tf.constant(0, dtype=tf.int32)

        max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')
        max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
        # x = (x + 1.0) * (width_f - 1.0) / 2.0
        # y = (y + 1.0) * (height_f - 1.0) / 2.0

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

        output = tf.add_n([wa * pixel_a, wb * pixel_b, wc * pixel_c, wd * pixel_d])
        # output = tf.reshape(output, shape=tf.stack([num_batch, height, width, channels]))
        output = tf.reshape(output, shape=tf.shape(im))
        return output


def meshgrid(height, width):
    """Tensorflow meshgrid function.
    """
    with tf.variable_scope('meshgrid'):
        x_t = tf.matmul(
            tf.ones(shape=tf.stack([height, 1])),
            tf.transpose(
                tf.expand_dims(
                    tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(
            tf.expand_dims(
                tf.linspace(-1.0, 1.0, height), 1),
            tf.ones(shape=tf.stack([1, width])))
        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        grid_x = tf.reshape(x_t_flat, [1, height, width])
        grid_y = tf.reshape(y_t_flat, [1, height, width])
        return grid_x, grid_y

def res_block(inputs, training=True, trainable=True, reuse=False):
    # with tf.variable_scope('res_block', reuse=reuse):
    h = inputs
    with tf.variable_scope('batchnorm1', reuse=reuse):
        h = batchnorm(h, training=training)
        h = conv2d_padding_same(h, 64, kernel_size=3, trainable=trainable, activate=tf.nn.relu)
    with tf.variable_scope('batchnorm2', reuse=reuse):
        h = batchnorm(h, training=training)
        h = conv2d_padding_same(h, 64, kernel_size=3, trainable=trainable, activate=None)
    h = h + inputs
    h = tf.nn.relu(h)
    return h

def ctx_net(inputs, name, training=True, reuse=False, trainable=True):
    # with tf.variable_scope('context_net_{}'.format(name)):
    h = inputs
    h = batchnorm(h, training=training)
    h = conv2d_padding_same(h, 64, kernel_size=7, trainable=trainable, activate=tf.nn.relu)
    h1 = h
    h = res_block(h, training=training, trainable=trainable)
    h2 = h
    h = res_block(h, training=training, trainable=trainable)
    h = tf.concat([h, h1, h2], axis=-1)
    return h

def rect_net(inputs, training=True, reuse=False, trainable=True):
    with tf.variable_scope('rect_net'):
        h = inputs
        h = batchnorm(h, training=training)
        h = conv2d_padding_same(h, 64, kernel_size=3, trainable=trainable, activate=tf.nn.relu)
        for i in range(3):
            h = res_block(h, training=training, trainable=trainable)
        h = batchnorm(h, training=training)
        h = conv2d_padding_same(h, 3, kernel_size=3, trainable=trainable, activate=None)
    return h

def u_net(inputs, name, training=True, reuse=False, trainable=True, out_size=2):
    h = inputs
    filter_nums = [64, 128, 256, 256]
    # with tf.variable_scope('flow_net_{}'.format(name)):
    mid_feat = []
    for k, n_dim in enumerate(filter_nums):
        h = batchnorm(h, training)
        h = conv2d_padding_same(h, n_dim, activate=tf.nn.relu, trainable=trainable)
        if k != len(filter_nums) - 1:
            mid_feat.append(h)
            h = maxpool2d_same(h)

    for n_dim, pre_f in zip(filter_nums[:-1][::-1], mid_feat[::-1]):
        h = batchnorm(h, training)
        shape_f = tf.shape(pre_f)
        h = tf.image.resize_bilinear(h, (shape_f[1], shape_f[2]))
        h = tf.concat([h, pre_f], axis=-1)
        h = conv2d_padding_same(h, n_dim, activate=tf.nn.relu, trainable=trainable)

    h = batchnorm(h, training)
    h = conv2d_padding_same(h, out_size, kernel_size=3, activate=None, trainable=trainable)

    return h

def to_flow(inputs, name, training=True, reuse=False, trainable=True, out_size=2):
    h = inputs
    h = batchnorm(h, training)
    h = conv2d_padding_same(h, 64, kernel_size=3, activate=tf.nn.relu, trainable=trainable)
    h = batchnorm(h, training)
    h = conv2d_padding_same(h, out_size, kernel_size=3, activate=None, trainable=trainable)

    return h


def adaptive_warp(tgt_img, flow, kernel):
    '''
    :param tgt_img: shape = [b, h, w, 3]
    :param flow: shape = [b, h, w, 2]
    :param kernel: shape = [b, h, w, 25]
    :return src_img: shape = [b, h, w, 3]
    '''
    h, w, c = tf.shape(tgt_img)[1], tf.shape(tgt_img)[2], tf.shape(tgt_img)[3]
    '''
    kernel_size = kernel.shape[3]//2

    kernel_h = kernel[:,:,:,:kernel_size]
    kernel_v = kernel[:,:,:,kernel_size:]

    kernel_h = tf.tile(tf.expand_dims(kernel_h, -1), [1, 1, 1, 1, kernel_size])
    kernel_v = tf.tile(tf.expand_dims(kernel_v, -2), [1, 1, 1, kernel_size, 1])
    
    kernel_reshape = tf.reshape(tf.multiply(kernel_h, kernel_v), [-1, h, w, 1, kernel_size*kernel_size])
    '''
    kernel_reshape = tf.expand_dims(kernel, -2)
    warped_img = tf.contrib.image.dense_image_warp(image=tgt_img, flow=flow)
    input_x_padding = tf.pad(warped_img, [[0, 0], [kernel_size//2, kernel_size//2], [kernel_size//2, kernel_size//2], [0, 0]])

    src_img = None
    for i in range(kernel_size):
        for j in range(kernel_size):
            slice = tf.slice(input_x_padding, [0, i, j, 0], [-1, h, w, -1])
            if src_img == None:
                src_img = slice * kernel_reshape[:, :, :, :, i * kernel_size + j]
            else:
                src_img += slice * kernel_reshape[:, :, :, :, i * kernel_size + j]

    return src_img#, p_node

def downsampling(inputs, num_f, training=True, trainable=True):
    '''
    bn->relu->conv
    :param inputs:
    :param training:
    :param trainable:
    :return:
    '''
    h = inputs
    for i in range(2):
        # h = batchnorm(h, training=training)
        h = tf.nn.relu(h)
        h = conv2d_padding_same(h, numfilter=num_f, trainable=trainable, activate=None)
        if i==0:h = maxpool2d_same(h)
    return h

def upsampling(inputs, num_f, training=True, trainable=True):
    '''
    bn->relu->conv
    :param inputs:
    :param num_f:
    :param training:
    :param trainable:
    :return:
    '''
    h = inputs
    for i in range(2):
        # h = batchnorm(h, training=training)
        h = tf.nn.relu(h)
        h = conv2d_padding_same(h, numfilter=num_f, trainable=trainable, activate=None)
        if i==0: h = tf.image.resize_bilinear(h, size=[tf.shape(h)[1]*2, tf.shape(h)[2]*2])
    return h

def lateral(inputs, training=True, trainable=True):
    '''
    bn->relu->conv
    :return:
    '''
    h = inputs
    _, _, _, num_f = inputs.get_shape().as_list()
    for i in range(2):
        # h = batchnorm(h, training)
        h = tf.nn.relu(h)
        h = conv2d_padding_same(h, numfilter=num_f, trainable=trainable, activate=None)
    return h + inputs

def in_grid(inputs, num_f=32, training=True, trainable=True):
    h = inputs
    # h = batchnorm(h, training)
    h = conv2d_padding_same(h, numfilter=num_f, trainable=trainable, activate=tf.nn.relu)
    return h

def out_grid(inputs, num_f=3, training=True, trainable=True):
    h = inputs
    # h = batchnorm(h, training)
    h = conv2d_padding_same(h, numfilter=num_f, trainable=trainable, activate=None)
    return h

def gridnet(inputs, training=True, trainable=True):
    ch_sizes = [32, 64, 96]
    down_first_hs, down_second_hs, down_third_hs = [], [], []
    h = inputs
    '''
        downsample
    '''
    for i in range(3):
        if len(down_first_hs)==0:
            down_first_hs.append(in_grid(h, num_f=ch_sizes[0], training=training, trainable=trainable))
        else:
            down_first_hs.append(lateral(down_first_hs[-1], training=training, trainable=trainable))

    for i in range(3):
        tmp_h = downsampling(down_first_hs[i], num_f=ch_sizes[1], training=training, trainable=trainable)
        if len(down_second_hs): tmp_h += down_second_hs[-1]
        down_second_hs.append(tmp_h)

    for i in range(3):
        tmp_h = downsampling(down_second_hs[i], num_f=ch_sizes[2], training=training, trainable=trainable)
        if len(down_third_hs): tmp_h += down_third_hs[-1]
        down_third_hs.append(tmp_h)

    '''
        upsample
    '''
    for i in range(3, 6, 1):
        down_third_hs.append(lateral(down_third_hs[-1], training=training, trainable=trainable))

    for i in range(3, 6, 1):
        down_second_hs.append(lateral(upsampling(down_third_hs[i], num_f=ch_sizes[1], training=training, trainable=trainable) +
                                      down_second_hs[-1], training=training, trainable=trainable))

    for i in range(3, 6, 1):
        down_first_hs.append(lateral(upsampling(down_second_hs[i], num_f=ch_sizes[0], training=training, trainable=trainable) +
                                     down_first_hs[-1], training=training, trainable=trainable))

    return out_grid(down_first_hs[-1], num_f=3, training=training, trainable=trainable)

def synthesis_net(inputs, name='synthesis_net', training=True, trainable=True, out_size = 3, reuse=True):
    with tf.variable_scope(name, reuse=reuse):
        h = inputs
        h = batchnorm(h, training)
        h = conv2d_padding_same(h, 128, kernel_size=7, trainable=trainable, activate=tf.nn.relu)

        h1 = batchnorm(h, training)
        h1 = conv2d_padding_same(h1, 128, trainable=trainable, activate=tf.nn.relu)
        h1 = batchnorm(h1, training)
        h1 = conv2d_padding_same(h1, 128, trainable=trainable, activate=None)
        h += h1
        h = tf.nn.relu(h)

        h2 = batchnorm(h, training)
        h2 = conv2d_padding_same(h2, 128, trainable=trainable, activate=tf.nn.relu)
        h2 = batchnorm(h2, training)
        h2 = conv2d_padding_same(h2, 128, trainable=trainable, activate=None)
        h += h2
        h = tf.nn.relu(h)

        h3 = batchnorm(h, training)
        h3 = conv2d_padding_same(h3, 128, trainable=trainable, activate=tf.nn.relu)
        h3 = batchnorm(h3, training)
        h3 = conv2d_padding_same(h3, 128, trainable=trainable, activate=None)
        h += h3
        h = tf.nn.relu(h)

        h = batchnorm(h, training)
        h = conv2d_padding_same(h, out_size, trainable=trainable, activate=None)
    return h


def pyr_synthesis_net(warped_img1_pyr, warped_ctx1_pyr, training=True, reuse=False, trainable=True):
    last_f = None
    for i, (warped_img1, warped_ctx1) in enumerate(
            zip(warped_img1_pyr, warped_ctx1_pyr)):
        h = tf.concat([warped_img1, warped_ctx1], axis=-1)
        with tf.variable_scope('res_block_{}'.format(i), reuse=reuse):
            h = batchnorm(h, training=training)
            h = conv2d_padding_same(h, 64, trainable=trainable, activate=tf.nn.relu)
            h = res_block(h, training=training, trainable=trainable, reuse=reuse)
        if last_f is not None: h += last_f
        last_f = tf.image.resize_bilinear(h, [tf.shape(h)[1] * 2, tf.shape(h)[2] * 2])
    h = to_flow(h, name=None, training=training, reuse=reuse, trainable=trainable, out_size=3)
    return h

def model_interpolation(first_img_t, end_img_t, ctx_net, training=True, reuse=False, trainable=True):
    '''
    Compute Flow
    '''
    x_tnsr1 = tf.stack([first_img_t, end_img_t], axis=1)
    flow_pred1, flow_pyr1 = nn(x_tnsr1, reuse=reuse)
    flow_pred1 = flow_pred1[:, :, :, ::-1]
    x_tnsr2 = tf.stack([end_img_t, first_img_t], axis=1)
    flow_pred2, flow_pyr2 = nn(x_tnsr2, reuse=True)
    flow_pred2 = flow_pred2[:, :, :, ::-1]

    with tf.variable_scope('interpolation_net', reuse=reuse):
        t = 0.5

        '''
        Compute Mask
        '''
        mask = u_net(tf.concat([first_img_t, end_img_t], axis=-1), name='mask', out_size=1, training=training, reuse=reuse, trainable=trainable)
        mask = (mask + 1.0) * 0.5
        # mask1 = mask[:,:,:,0:1]
        # mask2 = mask[:,:,:,1: ]

        '''
        Compute Context
        '''
        # ctx1 = ctx_net.conv1_2(first_img_t)
        # ctx2 = ctx_net.conv1_2(end_img_t)

        '''
        Compute Kernel
        '''
        kernels = u_net(tf.concat([first_img_t, end_img_t], axis=-1), name='kernel', training=training, reuse=reuse, trainable=trainable, out_size=kernel_size*kernel_size*2)
        kernel1 = kernels[:, :, :, :kernel_size*kernel_size]
        kernel2 = kernels[:, :, :, kernel_size*kernel_size:]

        '''
        Warp Context and Input Images
        '''
        warped_img1 = tf.reshape(adaptive_warp(first_img_t, flow_pred1 * t, kernel1), tf.shape(first_img_t))
        warped_img2 = tf.reshape(adaptive_warp(end_img_t, flow_pred2 * (1-t), kernel2), tf.shape(end_img_t))

        # warped_ctx1 = tf.reshape(adaptive_warp(ctx1, flow_pred1*t, kernel1), tf.shape(ctx1))
        # warped_ctx2 = tf.reshape(adaptive_warp(ctx2, flow_pred2*(1-t), kernel2), tf.shape(ctx2))

        '''
        Fuse the Warped Images
        '''
        # res_img = synthesis_net(tf.concat([warped_img1, warped_ctx1, warped_img2, warped_ctx2], axis=-1), training=training, trainable=trainable, out_size=3, reuse=reuse)
        res_img = warped_img1 * mask + warped_img2 * (1.0 - mask)
    return res_img

if __name__ == '__main__':
    pass