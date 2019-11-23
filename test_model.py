# coding: utf-8
import numpy as np
# import time
import os
import tensorflow as tf
from conf_tab import config
# import argparse
from model import model_interpolation
from vgg19 import Vgg19

# import cv2
# from PIL import Image
# from conf_tab import config
import time
# import src.flownet2.flownet2 as flownet2
# from src.training_schedules import LONG_SCHEDULE
# from scipy.misc import imread
# from model import gridnet
import dataset_mul_tab as dataset_mul
# from copy import deepcopy
# from skimage.io import imread
# from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
# os.environ['CUDA_VISIBLE_DEVICES']='0'
# gpu_devices = ['/device:GPU:0']
# controller = '/device:GPU:0'
# ckpt_path = '../tfoptflow/tfoptflow/models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'
#
# # Configure the model for inference, starting with the default options
# nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
# nn_opts['verbose'] = True
# nn_opts['ckpt_path'] = ckpt_path
# nn_opts['gpu_devices'] = gpu_devices
# nn_opts['controller'] = controller
#
# # We're running the PWC-Net-large model in quarter-resolution mode
# # That is, with a 6 level pyramid, and upsampling of level 2 by 4 in each dimension as the final flow prediction
# nn_opts['use_dense_cx'] = True
# nn_opts['use_res_cx'] = True
# nn_opts['pyr_lvls'] = 6
# nn_opts['flow_pred_lvl'] = 2
# nn_opts['batch_size'] = 4

# traindata_list = config.TRAIN.data_path
#
# first_names, mid_names, end_names = [], [], []
# traindata_base = os.path.join(os.path.dirname(traindata_list), 'sequences')
#
# with open(traindata_list) as f:
#     while True:
#         cur_name = f.readline().strip()
#         if not cur_name: break
#         video_segment_path = os.path.join(traindata_base, cur_name)
#         first_names.append(os.path.join(video_segment_path, 'im1.png'))
#         mid_names.append(os.path.join(video_segment_path, 'im2.png'))
#         end_names.append(os.path.join(video_segment_path, 'im3.png'))
#
# first_names_p = tf.placeholder(tf.string, shape=[None])
# mid_names_p = tf.placeholder(tf.string, shape=[None])
# end_names_p = tf.placeholder(tf.string, shape=[None])
#
# first_img, second_img, end_img, iterator = dataset_mul.mkdataset(first_names_p, mid_names_p, end_names_p)
# sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
# sess.run(tf.global_variables_initializer())
# sess.run(iterator.initializer, feed_dict={first_names_p: first_names,
#                                                       mid_names_p: mid_names,
#                                                       end_names_p: end_names})
# for i in range(3):
#     import pdb; pdb.set_trace();
#     cur_i = sess.run(first_img)
from PIL import Image
# import cv2
# a = np.random.normal(size=[3, 224, 224, 3])
# b = np.random.normal(size=[224, 224, 3])
# print(a[0].shape)
# res = cv2.resize(b, (8,8))
from my_pwc_net import nn
x_tnsr1 = tf.placeholder(dtype=tf.float32, shape=[None,2, None, None, 3])
x_tnsr2 = tf.placeholder(dtype=tf.float32, shape=[None,2, None, None, 3])
reuse = False
flow_pred1, flow_pyr1 = nn(x_tnsr1, reuse=reuse)
flow_pred2, flow_pyr2 = nn(x_tnsr2, reuse=True)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
sess.run(tf.global_variables_initializer())
len = 30
a = np.random.normal(size=[1,2, 512, 640, 3])
b = np.random.normal(size=[1,2, 512, 640, 3])
start = time.time()
for i in range(len):
    sess.run([flow_pred1, flow_pred2], feed_dict={x_tnsr1: a,
                                                  x_tnsr2: a})
runtime = time.time() - start
print('runtime: ', runtime*1./len)
'''
flow1_np = np.random.normal(size=[16, 224, 224, 2])
flow2_np = np.random.normal(size=[16, 224, 224, 2])
first_img_np = np.random.normal(size=[16, 224, 224, 3])
end_img_np = np.random.normal(size=[16, 224, 224, 3])
# coor_x_1 = tf.random.normal(shape=[16, 256, 256, 1], dtype=tf.float32)
# coor_y_1 = tf.random.normal(shape=[16, 256, 256, 1], dtype=tf.float32)
first_img_p = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32)
end_img_p = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32)
first_flow_p = tf.placeholder(shape=[None, None, None, 2], dtype=tf.float32)
end_flow_p = tf.placeholder(shape=[None, None, None, 2], dtype=tf.float32)
# if loss_type == 'feature_reconstruct':
vgg_data_dict = np.load(config.TRAIN.vgg19_npy_path, encoding='latin1').item()

vgg = Vgg19(vgg_data_dict)

mid_img_t = model_interpolation(first_img_p, end_img_p, first_flow_p, end_flow_p, ctx_net=vgg, training=True)
'''

# h, p_node = bilinear_interp(first_img_t, coor_x_1, coor_y_1, 'interpolate')
# y = bilinear_interp(first_img_t, coor_x_1, coor_y_1, 'interpolate')
# batch_size = 4
# H = 10
# W = 8
# width_x = tf.expand_dims(tf.expand_dims(tf.linspace(-1.0, 1.0, W), axis=0), axis=0)
# grid_x = tf.tile(width_x, [batch_size, H, 1])
# height_y = tf.expand_dims(tf.expand_dims(tf.linspace(-1.0, 1.0, H), axis=0), axis=-1)
# grid_y = tf.tile(height_y, [batch_size, 1, W])

# grid_x_back, grid_y_back =  meshgrid(H, W)
# flow_p = tf.placeholder(shape=[None, None, None, 2], dtype=tf.float32)
# res = tf.contrib.image.dense_image_warp(image=first_img_t, flow=flow_p)
# inputs = {}
# net = flownet2.FlowNet2()
# input_a = imread(inputs_a_path)
# input_b = imread(inputs_b_path)
# input_a = input_a[..., [2, 1, 0]]
# input_b = input_b[..., [2, 1, 0]]
# if input_a.max() > 1.0:
#     input_a = input_a / 255.0
# if input_b.max() > 1.0:
#     input_b = input_b / 255.0
#
# inputs = {
#     'input_a': tf.tile(tf.expand_dims(tf.constant(input_a, dtype=tf.float32), 0), [4, 1, 1, 1]),
#     'input_b': tf.tile(tf.expand_dims(tf.constant(input_b, dtype=tf.float32), 0), [4, 1, 1, 1])
# }
# def conv2d_padding_same(inputs, numfilter, kernel_size=3, trainable=True, activate=None):
#     return tf.layers.conv2d(inputs, numfilter, kernel_size, padding='same', kernel_initializer=tf.variance_scaling_initializer(),
#                             trainable=trainable, activation=activate)

# out = net.model(inputs, training_schedule=LONG_SCHEDULE, trainable=False)
# b, h, w, c = 4, 20, 20, 3
# tgt_img, flow, kernel =  np.random.normal(size=[b, h, w, c]), np.random.normal(size=[b, h, w, 2]), np.random.normal(size=[b, h, w, 10])

# src_img, p_node = adaptive_warp(tgt_img, flow, kernel)
# mid = model_interpolation(first_img_t, end_img_t, training=True)
# first_img_p = tf.placeholder(tf.float32, shape=[None, None, None, 3])
# first_img_t = np.random.normal(size=[4, 8, 8, 3])
# out = conv2d_padding_same(first_img_p, 2)
# def flow_net(x_np):
#     model = ModelPWCNet(mode='test', options=nn_opts)
#     with model.graph.as_default():
#         x_tnsr_p, x_adapt_info = model.adapt_x(x_np)
#         if x_adapt_info is not None:
#             y_adapt_info = (x_adapt_info[0], x_adapt_info[2], x_adapt_info[3], 2)
#         else:
#             y_adapt_info = None
#         feed_dict = {model.x_tnsr: x_tnsr_p}
#         y_hat = model.sess.run(model.y_hat_test_tnsr, feed_dict=feed_dict)
#         y_hats, _ = model.postproc_y_hat_test(y_hat, y_adapt_info)
#     return y_hats

# first_img_t = np.random.normal(size=[4, 256, 256, 3])
# end_img_t = np.random.normal(size=[4, 256, 256, 3])
#
# first_img_p = tf.placeholder(tf.float32, shape=[None, None, None, 3])
# end_img_p = tf.placeholder(tf.float32, shape=[None, None, None, 3])
'''
# out = model_interpolation(first_img_p, end_img_p)
x_np = np.random.uniform(low=0., high=1., size=[4, 2, 14, 14, 3])
# y_hats = flow_net(x_np)
# print(np.max(y_hats), np.min(y_hats))
# print(y_hats.shape)
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
sess.run(tf.global_variables_initializer())
tmp = sess.run(mid_img_t, feed_dict={first_img_p:first_img_np, end_img_p:end_img_np, first_flow_p: flow1_np, end_flow_p: flow2_np})
print(tmp.shape)
'''
# sess.run(y_hats)
# first_img, end_img = sess.run([first_img_p, end_img_p], feed_dict={first_img_p: first_img_t, end_img_p: end_img_t})
# model = ModelPWCNet(mode='test', options=nn_opts)
# x_tnsr_p = np.random.normal(size=[4, 2, 256, 256, 3])
# # with model.graph.as_default():
# x_tnsr_p = tf.convert_to_tensor(x_tnsr_p, tf.float32)
# # flow_pred, flow_pyr = model.nn(x_tnsr_p)
# flow_pred_tnsr, flow_pyr_tnsr = model.cal_flow(x_tnsr_p)
#
# model.sess.run(flow_pyr_tnsr)

# t = time.time()
# np_grid_x, np_grid_x_back = sess.run([grid_x, grid_x_back])
# np_grid_y, np_grid_y_back = sess.run([grid_y, grid_y_back])
# assert(np.all(np_grid_x!=np_grid_x_back)),'fuck'
# assert(np.all(np_grid_y!=np_grid_y_back)),'fuck'
# print(sess.run(res, feed_dict={first_img_p:first_img_t, end_img_p:end_img_t, flow_p:flow_np}).shape)
# print(sess.run(h).shape)
# print(sess.run(y).shape)
# print(sess.run(mid).shape)
# # sess.run(p_node)
# print(sess.run(out["flow"]).max())
# print(sess.run(out["flow"]).mean())
# print(sess.run(out["flow"]).min())
# print(sess.run(out["flow"]).shape)
# print(sess.run(out, feed_dict={first_img_p: first_img_t, end_img_p: end_img_t} ).shape)
# print("consumed: ", time.time() - t)
# width_x = tf.expand_dims(tf.expand_dims(tf.linspace(-1.0, 1.0, 10), axis=0), axis=0)
# grid_x = tf.tile(width_x, [16, 10, 1])
# print(sess.run(grid_x).shape)
# params = tf.constant([[1, 2], [3, 4]])
# indices = tf.constant([[[1, 0], [1, 2]],[[1, 0], [1, 2]]])
# out = tf.gather_nd(params=params, indices=indices)
# print(sess.run(out))