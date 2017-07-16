import tensorflow as tf
import inputs.batcher as bat
from .flownet2 import FlowNet2
import numpy as np
from extras import ominus
from losses import safe_rtLoss
from hyperparams import *
slim = tf.contrib.slim
# Create a new network
net = FlowNet2()
train_txt='./train.txt'
val_txt='./val.txt'
checkpoint='./checkpoints/FlowNet2/flownet-2.ckpt-0'

(i_t1,i_t2,p_t1,p_t2)=bat.euromav_batch(train_txt,8)#configs from dataset_configs
(i_v1,i_v2,p_v1,p_v2)=bat.euromav_batch(val_txt,8)#configs from dataset_configs
writer_t = tf.summary.FileWriter('./graphs/train', None)
writer_v = tf.summary.FileWriter('./graphs/validation', None)


rt12_g = ominus(p_t2, p_t1)
inputs = {
            'input_a': i_t1,
            'input_b': i_t2,
        }
RT_e=net.euro(inputs)
(rtd, rta) = safe_rtLoss(RT_e, rt12_g)
cost = (rta+rtd)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
with tf.name_scope("RT_loss"):
        odo_loss_sum = tf.summary.scalar("loss", cost)
        rtd_sum = tf.summary.scalar("rtd", rtd)
        rta_sum = tf.summary.scalar("rta", rta)
summary = tf.summary.merge([odo_loss_sum,rtd_sum,rta_sum])

variables_to_restore = slim.get_variables_to_restore(exclude=["f1","f2","beta2_power","beta1_power"]) #these are my last two layers
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
saver = tf.train.Saver(variables_to_restore)

with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    saver.restore(sess, checkpoint)
    for i in range(0,max_iterations):
        # [RT_e_] = sess.run([RT_e])
        [opt, train_loss, summ] = sess.run([optimizer, cost, summary])
        print('iteration:',i,' train loss:',train_loss)
        writer_t.add_summary(summ,i)
        # print(RT_e_[0])