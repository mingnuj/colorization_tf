import tensorflow as tf
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import PIL.Image as Image
import time
import math

from load import *
from network import *
from utils import *

BATCH_SIZE = 20
EPOCH = 25

seed = random.randint(0, 2**31 - 1)

tf.set_random_seed(seed)

# input_path = "E:/work/lab/Dataset/placesnet_256/train"
input_path = '/media/iot/Seagate Backup Plus Drive/place365_dataset'
output_path = "./result3"

def train():
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # luminance, ab value
    l_batch, ab_batch, ab_mask_batch, _, steps_per_epoch = image_batch(input_path)
    print("\nl_batch: ", l_batch.shape, "\nab_batch: ", ab_batch, "\nab_mask_batch: ", ab_mask_batch, "\n")
    pred_model = colorization_network(l_batch, ab_mask_batch)

    # 기존 이미지 rgb 변환
    ground_truth = tf.concat([l_batch, ab_batch], axis = -1)
    origin_image = lab_to_rgb(ground_truth)

    # 예측된 이미지 rgb 변환
    # pred_ab_batch, mask = tf.split(pred_model, [2, 1], axis = -1)
    pred_image = tf.concat([l_batch, pred_model], axis = -1)
    output_image = lab_to_rgb(pred_image)

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    train_variable = tf.trainable_variables()

    # loss function : huber loss
    loss = tf.reduce_mean(tf.losses.huber_loss(labels = ab_batch, predictions= pred_model, delta= 1))

    # Adam Optimizer : learning rate = 0.0002, Adam momentum = 0.9
    opt = tf.train.AdamOptimizer(0.0002, 0.9)
    grads = opt.compute_gradients(loss, var_list=train_variable)
    _train_model = opt.apply_gradients(grads)

    # loss를 계속 update 하면서 돌린다.
    ema = tf.train.ExponentialMovingAverage(decay = 0.99)
    update_losses = ema.apply([loss])

    # 최종 cost(loss) 값
    cost = ema.average(loss)

    # training model grouping
    train_model = tf.group(update_losses, incr_global_step, _train_model)

    with tf.name_scope("loss_summary"):
        tf.summary.scalar("loss", cost)
    with tf.name_scope("luminance_summary"):
        tf.summary.image("luminance", l_batch)
    with tf.name_scope("ground_truth"):
        tf.summary.image("ground_truth", origin_image)
    with tf.name_scope("predicted_model"):
        tf.summary.image("model",output_image)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)
    sv = tf.train.Supervisor(logdir = output_path, save_summaries_secs=0, saver = None)

    with sv.managed_session() as sess:
        print("parameter count = ", sess.run(parameter_count))

        max_steps = EPOCH * steps_per_epoch
        print("max_steps = {}".format(max_steps))
        start = time.time()

        progress_freq = 50
        summary_freq = 100
        save_freq = 50

        tf.reset_default_graph()
        tf.Graph().as_default()
        for step in range(max_steps):
            def should(freq):
                return freq > 0 and ((step+1) % freq == 0 or step == max_steps -1)

            options = None
            run_metadata = None

            fetches = {
                "train" : train_model,
                "global_step" : sv.global_step,
                "output": pred_model,
                "input_l":l_batch,
                "ground_truth": origin_image,
                "result_img": output_image
            }

            if should(progress_freq):
                fetches["loss"] = cost

            if should(summary_freq):
                fetches["summary"] = sv.summary_op
                # fetches["summary"] = tf.summary.merge_all()

            result = sess.run(fetches, options = options, run_metadata= run_metadata)

            print(result['input_l'])
            # plt.imshow(result['input_l'][0, :, :, :])
            # plt.show()

            if should(summary_freq):
                print("recording summary")
                sv.summary_writer.add_summary(result["summary"], result["global_step"])

            if should(progress_freq):
                train_epoch = math.ceil(result["global_step"] / steps_per_epoch)
                train_step = (result["global_step"] - 1) % steps_per_epoch + 1
                rate = (step+1) * BATCH_SIZE / (time.time() - start)
                remaining = (max_steps - step) * BATCH_SIZE / rate
                print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (
                    train_epoch, train_step, rate, remaining / 60))
                print("loss", result["loss"])
                # plt.imshow(result['result_img'][0, :, :, :])
                # plt.show()

            if should(save_freq):
                print("saving model")
                saver.save(sess, os.path.join(output_path, "model"), global_step=sv.global_step)
train()
