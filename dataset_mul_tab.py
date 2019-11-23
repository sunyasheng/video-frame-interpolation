import tensorflow as tf
from conf_tab import config
image_input_size = config.TRAIN.image_input_size

def mkdataset(first_names, mid_names, end_names, batch_size=32,
              gpu_ind=0, num_gpu=1, num_parallel=24):
    def _parse_function(first_name, mid_name, end_name):
        first_img_str = tf.read_file(first_name)
        first_img_decoded = tf.image.decode_png(first_img_str, channels=3)
        first_img_decoded.set_shape([None, None, 3])

        mid_img_str = tf.read_file(mid_name)
        mid_img_decoded = tf.image.decode_png(mid_img_str, channels=3)
        mid_img_decoded.set_shape([None, None, 3])

        end_img_str = tf.read_file(end_name)
        end_img_decoded = tf.image.decode_png(end_img_str, channels=3)
        end_img_decoded.set_shape([None, None, 3])

        p = tf.random_uniform([], 0, 1)
        tmp = tf.cond(p > 0.5, lambda: tf.concat([first_img_decoded, mid_img_decoded, end_img_decoded], axis=2),
                      lambda: tf.concat([end_img_decoded, mid_img_decoded, first_img_decoded], axis=2))

        max_w = tf.minimum(tf.shape(tmp)[0], tf.shape(tmp)[1])
        # crop_w = tf.minimum(tf.random_uniform([], minval=20, maxval=26, dtype=tf.int32) * tf.constant(5, tf.int32, []), max_w)
        crop_w = tf.minimum(tf.random_uniform([], minval=250, maxval=256, dtype=tf.int32) * tf.constant(1, tf.int32, []), max_w)

        tmp = tf.random_crop(tmp, [crop_w, crop_w, 3*3])
        tmp = tf.image.random_flip_left_right(tmp)
        tmp = tf.image.random_flip_up_down(tmp)
        tmp = tf.image.resize_bilinear(tf.expand_dims(tmp, 0), [image_input_size, image_input_size])

        tmp = tf.split(tf.squeeze(tmp, 0), num_or_size_splits=3, axis=2)

        return tmp[0], tmp[1], tmp[2]

    dataset = tf.data.Dataset.from_tensor_slices((first_names, mid_names, end_names))
    dataset = dataset.shard(num_gpu, gpu_ind)
    dataset = dataset.repeat(30).shuffle(buffer_size=72000)
    dataset = dataset.apply(tf.contrib.data.map_and_batch(_parse_function, batch_size=batch_size, num_parallel_calls=num_parallel))
    dataset = dataset.prefetch(32)

    iterator = dataset.make_initializable_iterator()
    first_img, second_img, end_img = iterator.get_next()
    return first_img, second_img, end_img, iterator
