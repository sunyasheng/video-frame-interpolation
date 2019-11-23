import tensorflow as tf
from conf_tab import config
import os
import dataset_mul_tab as dataset_mul
from model import model_interpolation
import numpy as np
import time
from vgg19 import Vgg19
import datetime
from tensorflow.python import debug as tf_debug
import sys
from copy import deepcopy
import cv2
from pwc_tab import pwc_opt
from PIL import Image
ckpt_path = '../tfoptflow/tfoptflow/models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'

beta1 = config.TRAIN.beta1
beta2 = config.TRAIN.beta2
lr_init = config.TRAIN.lr_init
pwc_lr_init = config.TRAIN.pwc_lr_init
lr_decay = config.TRAIN.lr_decay
checkpoint_path = config.TRAIN.checkpoint_path
loss_type = config.TRAIN.loss_type
optim_type = config.TRAIN.optim_type
debug = config.DEBUG
per_gpu_batch_size = config.TRAIN.per_gpu_batch_size

timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")



def optimistic_restore(session, save_file):
    """
    restore only those variable that exists in the model
    :param session:
    :param save_file:
    :return:
    """
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for
                      var in tf.global_variables()
                      if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0],tf.global_variables()),tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                print("going to restore.var_name:",var_name,";saved_var_name:",saved_var_name)
                restore_vars.append(curr_var)
            else:
                print("variable not trained.var_name:",var_name)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

def get_config(config):
    config_str = 'bn:{},opt:{},bs:{},ims:{},lr:{},lt:{},ks:{}'.format(
                                                  config.TRAIN.batch_norm,
                                                  config.TRAIN.optim_type,
                                                  config.TRAIN.per_gpu_batch_size,
                                                  config.TRAIN.image_input_size,
                                                  config.TRAIN.lr_init,
                                                  config.TRAIN.loss_type,
                                                  config.TRAIN.kernel_size)
    return config_str

def pad_img(img, pyr_lvls):
    ##img shape: [h, w, c]
    _, pad_h = divmod(img.shape[0], 2**pyr_lvls)
    if pad_h != 0:
        pad_h = 2 ** pyr_lvls - pad_h
    _, pad_w = divmod(img.shape[1], 2**pyr_lvls)
    if pad_w != 0:
        pad_w = 2 ** pyr_lvls - pad_w
    if pad_h != 0 or pad_w != 0:
        padding = [(0, pad_h), (0, pad_w), (0, 0)]
        img = np.pad(img, padding, mode='constant', constant_values=0.)
    return img

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    print(local_device_protos)
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # compute average gradient for every variable
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)

        average_grads.append(grad_and_var)

    return average_grads

def get_variables_with_name(name=None, train_only=True, printable=False):
    print("  [*] geting variables with %s" % name)
    # tvar = tf.trainable_variables() if train_only else tf.all_variables()
    if train_only:
        t_vars = tf.trainable_variables()
    else:
        try:  # TF1.0+
            t_vars = tf.global_variables()
        except:  # TF0.12
            t_vars = tf.all_variables()

    if name is None:
        d_vars = [var for var in t_vars]
    else:
        d_vars = [var for var in t_vars if name in var.name]
    if printable:
        for idx, v in enumerate(d_vars):
            print("  got {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))
    return d_vars

def params_count(name=None, train_only=False):
    if train_only:
        t_vars = tf.trainable_variables()
    else:
        try:  # TF1.0+
            t_vars = tf.global_variables()
        except:  # TF0.12
            t_vars = tf.all_variables()

    total_params = 0

    for variable in t_vars:
        if not name in variable.name: continue
        shape = variable.get_shape()
        variable_paramters = 1
        for dim in shape:
            variable_paramters *= dim.value
        total_params += variable_paramters
    param_type = 'trainable' if train_only else 'total'
    print('{} include {} {} params.'.format(name, total_params, param_type))

def build_model(first_img_t, mid_img_t, end_img_t, vgg_data_dict=None, reuse_all=False):

    first_img_t = tf.cast(first_img_t, tf.float32)/255.0
    mid_img_t = tf.cast(mid_img_t, tf.float32)/255.0
    end_img_t = tf.cast(end_img_t, tf.float32)/255.0

    assert vgg_data_dict is not None, 'Invalid vgg data dict'
    vgg = Vgg19(vgg_data_dict)
    pred_mid_img = model_interpolation(first_img_t, end_img_t, ctx_net=vgg, reuse=reuse_all)

    if loss_type is 'feature_reconstruct':
        pred_feat = vgg.relu4_4(pred_mid_img)
        gt_feat = vgg.relu4_4(mid_img_t)
        l1_loss = tf.reduce_mean(tf.square(gt_feat - pred_feat), axis=[0, 1, 2, 3])
    else:
        l1_loss = tf.losses.absolute_difference(pred_mid_img, mid_img_t)

    summary = [tf.summary.image('first_img', first_img_t), tf.summary.image('end_img', end_img_t),
               tf.summary.image('gt_mid_img', mid_img_t), tf.summary.image('pred_mid_img', tf.clip_by_value(pred_mid_img, 0.0, 1.0)),
               tf.summary.scalar('l1_loss', l1_loss)]

    return l1_loss, summary

def train(args):
    global num_gpu
    global batch_size

    gpu_ids = get_available_gpus()
    num_gpu = len(gpu_ids)

    batch_size = per_gpu_batch_size*num_gpu

    print("reading images")
    traindata_list = config.TRAIN.data_path
    first_names, mid_names, end_names = [], [], []
    traindata_base = os.path.join(os.path.dirname(traindata_list), 'sequences')

    with open(traindata_list) as f:
        while True:
            cur_name = f.readline().strip()
            if not cur_name: break
            video_segment_path = os.path.join(traindata_base, cur_name)
            first_names.append(os.path.join(video_segment_path, 'im1.png'))
            mid_names.append(os.path.join(video_segment_path, 'im2.png'))
            end_names.append(os.path.join(video_segment_path, 'im3.png'))

    first_names_p = tf.placeholder(tf.string, shape=[None])
    mid_names_p = tf.placeholder(tf.string, shape=[None])
    end_names_p = tf.placeholder(tf.string, shape=[None])

    print("building models...")

    # print("gpu nums: ", num_gpu)
    with tf.device('/cpu:0'):
        with tf.variable_scope('learning_rate'):
            lr_v = tf.Variable(lr_init, trainable=False)
        if optim_type == 'Adamax':
            opt = tf.contrib.opt.AdaMaxOptimizer(lr_v, beta1=beta1, beta2=beta2)
            pwcnet_opt = tf.contrib.opt.AdaMaxOptimizer(pwc_lr_init, beta1=beta1, beta2=beta2)
        else:
            opt=tf.train.AdamOptimizer(lr_v, beta1=beta1, beta2=beta2)
            pwcnet_opt = tf.train.AdamOptimizer(pwc_lr_init, beta1=beta1, beta2=beta2)
        vgg_data_dict = np.load(config.TRAIN.vgg19_npy_path, encoding='latin1').item()

        reuse_all = False

        tower_it = []
        tower_grads, tower_pwc_grads = [], []
        tower_loss = []
        for d in range(0, num_gpu):
            print("dealing {}th gpu".format(d))
            with tf.device('/gpu:%s' % d):
                with tf.name_scope('%s_%s' % ('tower', d)):
                    print("prepare dataset!!!")
                    first_img_t, mid_img_t, end_img_t, iterator_gpu \
                        = dataset_mul.mkdataset(first_names_p, mid_names_p, end_names_p,
                                                  int(batch_size / num_gpu), gpu_ind=d,
                                                  num_gpu=num_gpu)
                    print("build model!!!")
                    mse_loss_gpu, summary \
                        = build_model(first_img_t, mid_img_t, end_img_t, vgg_data_dict, reuse_all=reuse_all)
                    if not reuse_all:
                        vars_trainable = get_variables_with_name(name='interpolation_net', train_only = True)
                        grads = opt.compute_gradients(mse_loss_gpu, var_list=vars_trainable)
                        pwc_vars_trainable = get_variables_with_name(name='pwcnet', train_only= True)
                        pwc_grads = opt.compute_gradients(mse_loss_gpu, var_list=pwc_vars_trainable)

                    for i, (g, v) in enumerate(grads):
                        if g is not None:
                            grads[i] = (tf.clip_by_norm(g, 5), v)
                    for i, (g, v) in enumerate(pwc_grads):
                        if g is not None:
                            pwc_grads[i] = (tf.clip_by_norm(g, 5), v)

                    tower_grads.append(grads)
                    tower_pwc_grads.append(pwc_grads)
                    tower_loss.append(mse_loss_gpu)

                    reuse_all = True
                    tower_it.append(iterator_gpu)
        if num_gpu == 1:
            with tf.device('/gpu:0'):
                mse_loss = tf.reduce_mean(tf.stack(tower_loss, 0), 0)
                mean_grads = average_gradients(tower_grads)
                mean_pwc_grads = average_gradients(tower_pwc_grads)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='.*?interpolation_net')
                update_pwc_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='.*?pwcnet')
                with tf.control_dependencies(update_ops):
                    minimize_op = opt.apply_gradients(mean_grads)
                with tf.control_dependencies(update_pwc_ops):
                    minimize_pwc_op = pwcnet_opt.apply_gradients(mean_pwc_grads)

        else:
            mse_loss = tf.reduce_mean(tf.stack(tower_loss, 0), 0)
            mean_grads = average_gradients(tower_grads)
            mean_pwc_grads = average_gradients(tower_pwc_grads)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='.*?interpolation_net')
            update_pwc_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='.*?pwcnet')
            with tf.control_dependencies(update_ops):
                minimize_op = opt.apply_gradients(mean_grads)
            with tf.control_dependencies(update_pwc_ops):
                minimize_pwc_op = pwcnet_opt.apply_gradients(mean_pwc_grads)

        print('trainable variables:')
        print(vars_trainable)
        print('pwc trainable variables:')
        print(pwc_vars_trainable)

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        if debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=200)
        lr_str = timestamp + ' ' + get_config(config) + ',gn:{}'.format(num_gpu)
        if not os.path.exists(checkpoint_path + lr_str):
            os.makedirs(checkpoint_path + lr_str)

        optimistic_restore(sess, pwc_opt.ckpt_path)
        if args.pretrained:
            saver.restore(sess, checkpoint_path + args.lr_str + '/interpolate.ckpt-' + str(args.modeli))

        sess.run(tf.assign(lr_v, lr_init))

        n_epoch = 30
        decay_every = 10

        summary_ops = tf.summary.merge(summary)
        summary_writer = tf.summary.FileWriter(checkpoint_path + lr_str + '/summary', sess.graph)
        len_train = len(first_names)

        for iterator in tower_it:
            sess.run(iterator.initializer, feed_dict={first_names_p: first_names,
                                                      mid_names_p: mid_names,
                                                      end_names_p: end_names})

    for epoch in range(0, n_epoch):
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f " % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f " % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        for it in range(int(len_train/batch_size)):
            errM, _, _, summary = sess.run([mse_loss, minimize_op, minimize_pwc_op, summary_ops], feed_dict={first_names_p:first_names,
                                                                                         mid_names_p:mid_names,
                                                                                         end_names_p:end_names})

            if (it+int(len_train/batch_size)*epoch)%10==0:
                summary_writer.add_summary(summary, it+int(len_train/batch_size)*epoch)

            sys.stdout.flush()
            print("Epoch [%2d/%2d] %4d time: %4.4fs, loss:  %5.5f" %
                  (epoch, n_epoch, it, time.time() - epoch_time, errM))
            epoch_time = time.time()

            if (it + int(len_train/batch_size)*epoch)%1000 == 0:
                saver.save(sess, checkpoint_path + lr_str + '/interpolate.ckpt', global_step=(it + int(len_train/batch_size)*epoch))
'''
def test(args):
    # pwc_model = ModelPWCNet(mode='test', options=nn_opts)

    if args.lr_str:
        lr_str = args.lr_str
    gpu_ids = get_available_gpus()
    num_gpu = len(gpu_ids)
    num_gpu = 1
    batch_size = per_gpu_batch_size * num_gpu

    print("reading images")
    traindata_list = config.TEST.data_path
    first_names, mid_names, end_names = [], [], []
    traindata_base = os.path.join(os.path.os.path.dirname(traindata_list), 'sequences')

    with open(traindata_list) as f:
        while True:
            cur_name = f.readline().strip()
            if not cur_name: break
            video_segment_path = os.path.join(traindata_base, cur_name)
            first_names.append(os.path.join(video_segment_path, 'im1.png'))
            mid_names.append(os.path.join(video_segment_path, 'im2.png'))
            end_names.append(os.path.join(video_segment_path, 'im3.png'))

    first_names_p = tf.placeholder(tf.string, shape=[None])
    mid_names_p = tf.placeholder(tf.string, shape=[None])
    end_names_p = tf.placeholder(tf.string, shape=[None])

    first_img_p = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    mid_img_p = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    end_img_p = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    first_flow_p = tf.placeholder(tf.float32, shape=[None, None, None, 2])
    end_flow_p = tf.placeholder(tf.float32, shape=[None, None, None, 2])

    with tf.device('/cpu:0'):
        reuse_all = False
        tower_it = []
        vgg_data_dict = None
        if loss_type == 'feature_reconstruct':
            vgg_data_dict = np.load(config.TRAIN.vgg19_npy_path, encoding='latin1').item()

        for d in range(0, num_gpu):
            with tf.device('/gpu:%s' % d):
                with tf.name_scope('%s_%s' % ('tower', d)):
                    print("prepare dataset!!!")
                    first_img_t, mid_img_t, end_img_t, iterator_gpu \
                        = dataset_mul.mkdataset(first_names_p, mid_names_p, end_names_p,
                                                int(batch_size / num_gpu), gpu_ind=d,
                                                num_gpu=num_gpu)
                    print("build model!!!")
                    mse_loss_gpu, summary \
                        = build_model(first_img_p, mid_img_p, end_img_p, first_flow_p, end_flow_p, vgg_data_dict,
                                      reuse_all=reuse_all)
                    tower_it.append(iterator_gpu)
                    reuse_all = True
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    summary_ops = tf.summary.merge(summary)
    summary_writer = tf.summary.FileWriter(checkpoint_path + 'test' + '/summary', sess.graph)

    lines = []
    # outfile = open(os.path.join(config.TEST.res_dir, 'test_res.txt'), 'w')

    for modeli in range(94000, 95000, 10000):
        try:
            saver.restore(sess, checkpoint_path + lr_str + '/interpolate.ckpt-' + str(modeli))
        except Exception as ex:
            print(ex)
            break

        for iterator in tower_it:
            sess.run(iterator.initializer, feed_dict={first_names_p: first_names,
                                                      mid_names_p: mid_names,
                                                      end_names_p: end_names})
        loss_list = []
        try:
            for i in range(0, len(first_names), batch_size):
                [first_img_np, mid_img_np, end_img_np] = sess.run([first_img_t, mid_img_t, end_img_t])
                [first_img_np, mid_img_np, end_img_np] = list(map(lambda x:x.astype(np.uint8), [first_img_np, mid_img_np, end_img_np]))
                first_flow_np = flow_net(pwc_model, np.stack([first_img_np, end_img_np], axis=1))
                end_flow_np = flow_net(pwc_model, np.stack([end_img_np, first_img_np], axis=1))
                # import pdb; pdb.set_trace();
                first_flow_np, end_flow_np = first_flow_np[:, :, :, ::-1], end_flow_np[:, :, :, ::-1]
                errM, summary = sess.run([mse_loss_gpu, summary_ops], feed_dict={first_img_p:first_img_np,
                                                                                         mid_img_p:mid_img_np,
                                                                                         end_img_p:end_img_np,
                                                                                         first_flow_p:first_flow_np,
                                                                                         end_flow_p:end_flow_np})
                summary_writer.add_summary(summary, i)
                print("iteration: ", modeli, ' cur_batch: ', i, ' cur_loss: ', errM)
                loss_list.append(errM)
        except Exception as ex:
            print(str(ex))
            break

        print("iteration: ", modeli, ' mean loss:', np.mean(loss_list))
        sys.stdout.flush()
        lines.append('iteration: ' + str(modeli) + ' mean loss:' + str(np.mean(loss_list)))
    #     outfile.write(lines[-1])
    #     outfile.write('\n')
    # outfile.close()
'''

def build_model_test(input_frames):
    first_frames = input_frames[:-1]
    end_frames = input_frames[1:]
    first_frames = tf.cast(first_frames, tf.float32)*1.0/255.0
    end_frames = tf.cast(end_frames, tf.float32) * 1.0 / 255.0

    pred_frames = model_interpolation(first_frames, end_frames, training=False)
    return pred_frames

def export_model(args):
    input_frames = tf.placeholder(tf.uint8, shape=[None, None, None, 3])
    pred_frames = tf.cast(build_model_test(input_frames)*255, tf.uint8)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    saver.restore(sess, checkpoint_path + args.lr_str + '/interpolate.ckpt-' + str(args.modeli))

    tensor_info_input_frames = tf.saved_model.utils.build_tensor_info(input_frames)
    tensor_info_pred_frames = tf.saved_model.utils.build_tensor_info(pred_frames)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs = {'frame_input': tensor_info_input_frames},
            outputs={'frame_output': tensor_info_pred_frames},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    export_path_base = 'export_models/vfs_base'
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(1)))
    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_images':
                prediction_signature,
        }, strip_default_attrs=True)
    builder.save()
    print('Done exporting!')

def interpolate(args):
    import cv2
    print("reading images")

    img_lists = []
    with open(os.path.join(args.img_dir, 'img_list.txt'), 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            img_lists.append(os.path.join(args.img_dir, line.strip()))

    first_names = img_lists[:-1]
    end_names = img_lists[1:]

    first_frame_list = [cv2.imread(fn)[:,:,::-1]for fn in first_names]
    end_frame_list = [cv2.imread(fn)[:,:,::-1]for fn in end_names]

    # first_frame_list = [cv2.resize(img, (480, 640))for img in first_frame_list]
    # end_frame_list = [cv2.resize(img, (480, 640)) for img in end_frame_list]

    raw_shape = cv2.imread(first_names[0]).shape
    # print(raw_shape)
    first_frame_list = [pad_img(img, pwc_opt.pyr_lvls) for img in first_frame_list]
    end_frame_list = [pad_img(img, pwc_opt.pyr_lvls) for img in end_frame_list]
    # import pdb; pdb.set_trace();
    first_img = np.stack(first_frame_list)
    end_img = np.stack(end_frame_list)

    first_img_p = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    end_img_p = tf.placeholder(tf.float32, shape=[None, None, None, 3])

    vgg_data_dict = np.load(config.TRAIN.vgg19_npy_path, encoding='latin1').item()

    with tf.device('/gpu:0'):
        with tf.name_scope('tower_0'):
            pred_mid_img = build_test_model(first_img_p, end_img_p, vgg_data_dict=vgg_data_dict,
                             reuse=False)

    # params_count(name='interpolation_net', train_only=True)
    # params_count(name='pwcnet', train_only=True)
    # params_count(name='interpolation_net', train_only=False)
    # params_count(name='pwcnet', train_only=False)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    saver.restore(sess, checkpoint_path + args.lr_str + '/interpolate.ckpt-' + str(args.modeli))

    mid_imgs = []

    all_images = []
    all_images.append(first_img[0])
    # import time
    # start = time.time()
    for i in range(first_img.shape[0]):
        mid_img = sess.run(pred_mid_img, feed_dict={first_img_p: np.expand_dims(first_img[i], 0),
                                                    end_img_p: np.expand_dims(end_img[i],0)})
        mid_imgs.append(mid_img*255.0)
    # run_time = time.time() - start
    # log = 'run time: {}\n'.format(run_time/first_img.shape[0])
    # print(log)

    mid_imgs = np.concatenate(mid_imgs, axis=0)

    for i in range(len(end_img)):
        all_images.append(mid_imgs[i])
        all_images.append(end_img[i])

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    with open(os.path.join(args.out_dir, 'img_list.txt'), 'w') as f:
        for i in range(len(all_images)):
            all_images[i] = all_images[i][:raw_shape[0], :raw_shape[1], :]
            cv2.imwrite(os.path.join(args.out_dir, '{}.png'.format(i)), np.array(all_images[i][:,:,::-1]).astype(np.uint8))
            f.write(os.path.join(args.out_dir, '{}.png'.format(i)) + '\n')
            print('write to {}'.format(os.path.join(args.out_dir, '{}.png'.format(i))))

def build_test_model(first_img_p, end_img_p, vgg_data_dict=None, reuse=False, training=False):
    first_img_p = tf.cast(first_img_p, tf.float32) / 255.0
    end_img_p = tf.cast(end_img_p, tf.float32) / 255.0
    assert vgg_data_dict is not None, 'Invalid vgg data dict'
    vgg = Vgg19(vgg_data_dict)
    pred_mid_img = model_interpolation(first_img_p, end_img_p, ctx_net=vgg, reuse=reuse, training=training)
    pred_mid_img = tf.clip_by_value(pred_mid_img, 0., 1.)
    return pred_mid_img

def psnr(args):
    # pwc_model = ModelPWCNet(mode='test', options=nn_opts)

    if args.lr_str:
        lr_str = args.lr_str

    global num_gpu
    global batch_size

    num_gpu = 1
    batch_size = per_gpu_batch_size * num_gpu
    print("reading images")

    traindata_list = config.TEST.data_path
    first_names, mid_names, end_names = [], [], []
    traindata_base = os.path.join(os.path.os.path.dirname(traindata_list), 'sequences')

    with open(traindata_list) as f:
        while True:
            cur_name = f.readline().strip()
            if not cur_name: break
            video_segment_path = os.path.join(traindata_base, cur_name)
            first_names.append(os.path.join(video_segment_path, 'im1.png'))
            mid_names.append(os.path.join(video_segment_path, 'im2.png'))
            end_names.append(os.path.join(video_segment_path, 'im3.png'))

    first_img_p = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    mid_img_p = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    end_img_p = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    pred_mid_img_p = tf.placeholder(tf.float32, shape=[None, None, None, 3])

    with tf.device('/cpu:0'):
        vgg_data_dict = np.load(config.TRAIN.vgg19_npy_path, encoding='latin1').item()
        with tf.device('/gpu:%s' % 0):
            with tf.name_scope('%s_%s' % ('tower', 0)):
                pred_mid_img = build_test_model(first_img_p, end_img_p, vgg_data_dict=vgg_data_dict)
                psnr_batch = tf.image.psnr(pred_mid_img_p, tf.cast(mid_img_p, tf.float32)/255.0, max_val=1.0)
                ssim_batch = tf.image.ssim(pred_mid_img_p, tf.cast(mid_img_p, tf.float32)/255.0, max_val=1.0)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    try:
        saver.restore(sess, checkpoint_path + lr_str + '/interpolate.ckpt-' + str(args.modeli))
    except Exception as ex:
        print(ex)
        return

    raw_shape = cv2.imread(first_names[0]).shape
    psnr_list, ssim_list = [], []
    for i in range(0, len(first_names), batch_size):
        first_batch_names = first_names[i:i+batch_size]
        end_batch_names = end_names[i:i+batch_size]
        mid_batch_names = mid_names[i:i+batch_size]
        first_img_np = [cv2.imread(fn)[:, :, ::-1] for fn in first_batch_names]
        end_img_np = [cv2.imread(fn)[:, :, ::-1] for fn in end_batch_names]
        mid_img_np = [cv2.imread(fn)[:, :, ::-1] for fn in mid_batch_names]
        first_img_np = np.stack(first_img_np)
        end_img_np = np.stack(end_img_np)
        mid_img_np = np.stack(mid_img_np)

        pred_mid_img_np = sess.run(pred_mid_img, feed_dict={first_img_p: first_img_np,
                                                            end_img_p: end_img_np})
        pred_mid_img_np = pred_mid_img_np[:, :raw_shape[0], :raw_shape[1], :]
        [psnr_batch_np, ssim_batch_np] = sess.run([psnr_batch, ssim_batch], feed_dict={pred_mid_img_p: pred_mid_img_np,
                                                                                       mid_img_p: mid_img_np})
        print("iteration: {}, psnr: {}, ssim: {}".format(i, np.mean(psnr_batch_np), np.mean(ssim_batch_np)))
        # import pdb; pdb.set_trace();
        psnr_list.append(psnr_batch_np)
        ssim_list.append(ssim_batch_np)
    psnr_list_np = np.concatenate(psnr_list, axis=0)
    ssim_list_np = np.concatenate(ssim_list, axis=0)
    psnr_v = np.mean(psnr_list_np)
    ssim_v = np.mean(ssim_list_np)

    print('psnr:', psnr_v, 'ssim:', ssim_v)
    with open('psnr_ssim.txt', 'w') as f:
        f.write('psnr:' + str(psnr_v) + 'ssim:' + str(ssim_v) + '\n')
    return

if __name__ == '__main__':
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', help='train, evaluate, test, export, psnr')
    parser.add_argument('--pretrained',type=bool, default=False,help='False, True')
    parser.add_argument('--loss_type', type=str, default='pixel_wise', help='feature_reconstruct, pixel_wise')
    parser.add_argument('--seed', type=int, default=66, help='a random seed')
    parser.add_argument('--lr_str', type=str, default='08-16-13:09 bn:True,opt:Adam,bs:4,ims:256,lr:0.0001,lt:pixel_wise,ks:3,gn:4',
                        help='checkpoint path')
    parser.add_argument('--modeli', type=int, default=-1, help='loaded model version')
    parser.add_argument('--out_dir', type=str, default='./Beanbags_out', help='output path of the resulting video, either as a single file or as a folder')
    parser.add_argument('--img_dir', type=str, default='./middlebury/Beanbags', help='input image path')

    args = parser.parse_args()

    tf.random.set_random_seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == 'train':
        args.pretrained = True
        args.modeli = 96000
        train(args)
    # if args.mode == 'test':
    #     test(args)
    if args.mode == 'export':
        args.modeli = 96000
        export_model(args)
    if args.mode == 'evaluate':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        args.modeli = 96000
        interpolate(args)
    if args.mode == 'psnr':
        args.modeli = 50000
        psnr(args)