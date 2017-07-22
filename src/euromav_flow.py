import tensorflow as tf
import inputs.batcher as bat
from flownet2.flownet2 import FlowNet2
from flownet_c.flownet_c import FlowNetC
import numpy as np
from hyperparams import *
from dump2disk import *
from .net import Mode
import uuid
from .flowlib import flow_to_image

batch_size=1
(i_t1,i_t2,p_t1,p_t2)=bat.euromav_batch(train_txt,batch_size)#configs from dataset_configs

x = tf.placeholder("float", shape=[batch_size, height, width,3])
y = tf.placeholder("float", shape=[batch_size, height, width,3])
inputs = {
            'input_a': x,
            'input_b': y,
        }
# Create a new network
net = FlowNet2(mode=Mode.TEST)
checkpoint='./checkpoints/FlowNet2/flownet-2.ckpt-0'
flow=net.flowtest(inputs)
saver = tf.train.Saver()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    saver.restore(sess, checkpoint)
    i1_, i2_ = sess.run([i_t1,i_t2])
    flow_ = sess.run(flow,feed_dict={x: i1_, y: i2_})[0, :, :, :]
    print(np.shape(flow_))           
# 
print "Dumping few visuals" 
dump2disk(vis_dir, 1, i1_,i2_)
#save flow image
unique_name = 'flow-' + str(uuid.uuid4())
flow_img = flow_to_image(flow_)
full_out_path = os.path.join(vis_dir, "flow_"+'{:08}'.format(step)  + '.png')
imsave(full_out_path, flow_img)
