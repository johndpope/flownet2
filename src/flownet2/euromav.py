import tensorflow as tf
import inputs.batcher as bat
from .flownet2 import FlowNet2
import numpy as np
from utils import ominus
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
pred_flow=net.euro(inputs)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    saver.restore(sess, checkpoint)
    pred_flow_ = sess.run(pred_flow)
print(np.shape(pred_flow_[0]))