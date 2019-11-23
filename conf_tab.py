from easydict import EasyDict as edict

config = edict()
config.TRAIN = edict()
config.TEST = edict()

config.TRAIN.lr_init = 0.0001
config.TRAIN.pwc_lr_init = 0.000001
config.TRAIN.beta1 = 0.9
config.TRAIN.beta2 = 0.999
# config.TRAIN.pretrained_vgg_path = '/xxx/sepconv-tensorflow/pretrained_vgg/vgg_19.ckpt'
config.TRAIN.lr_decay = 0.1
config.TRAIN.checkpoint_path = './checkpoints/interpolation_checkpoints/'
# config.TRAIN.loss_type = 'pixel_wise'
config.TRAIN.loss_type = 'feature_reconstruct'
config.TRAIN.optim_type = 'Adam'
config.TRAIN.per_gpu_batch_size = 4
config.TRAIN.image_input_size = 256
config.TRAIN.batch_norm = True
config.TRAIN.kernel_size = 3

config.TRAIN.data_path = '/xxx/sunyasheng/datasets/Vimeo90K/vimeo_triplet/tri_trainlist.txt'
config.TRAIN.vgg19_npy_path = './pretrained_vgg19/vgg19.npy'

config.TEST.data_path = '/xxx/sunyasheng/datasets/Vimeo90K/vimeo_triplet/tri_testlist.txt'
config.TEST.res_dir = '/xxx/sunyasheng/checkpoints/interpolation_checkpoints/test_res/'
config.DEBUG = False