import tensorflow as tf
import inputs.batcher as bat
from .flownet2 import FlowNet2
import numpy as np
from utils import ominus
slim = tf.contrib.slim
# Create a new network
net = FlowNet2()
train_txt='./train.txt'
checkpoint='./checkpoints/FlowNet2/flownet-2.ckpt-0'
(i_t1,i_t2,p_t1,p_t2)=bat.euromav_batch(train_txt,8)#configs from dataset_configs
rt12_g = ominus(p_t2, p_t1)
inputs = {
            'input_a': i_t1,
            'input_b': i_t2,
        }
pred=net.euro(inputs)

variables_to_restore = slim.get_variables_to_restore(exclude=["f1","f2"]) #these are my last two layers
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
saver = tf.train.Saver(variables_to_restore)

with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    saver.restore(sess, checkpoint)

    pred_ = sess.run(pred)
    
    print(np.shape(pred_[0]))