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
from extras import warper
from depth_estimation import *
import lmbspecialops
from cv2 import imread
batch_size=1
##from sun dataset
if sun_dataset:
    input_a_path='./data/samples/img1.jpg'
    input_b_path='./data/samples/img2.jpg' 
    input_a = imread(input_a_path)
    input_b = imread(input_b_path)
    if input_a.max() > 1.0:
        input_a = input_a / 255.0
    if input_b.max() > 1.0:
        input_b = input_b / 255.0
    i1=tf.image.resize_images(input_a, [320, 448])
    i2=tf.image.resize_images(input_b, [320, 448])
    i1=tf.expand_dims(i1, 0)
    i2=tf.expand_dims(i2, 0)
else:### from tf records
    (il_t1,il_t2,ir_t1,ir_t2,p_t1,p_t2)=bat.euromav_batch(train_txt,batch_size)#configs from dataset_configs

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
(i2_warped,occ)=warper(y, flow)
###depth estimation
flow_transposed=tf.transpose(flow[0,:,:,:], perm=[2, 0, 1])#[channels,height,width]
depth=lmbspecialops.flow_to_depth(flow=flow_transposed,intrinsics=intrinsics_1,rotation=rotation,translation=translation,rotation_format='matrix')
depth=tf.transpose(depth, perm=[0,2, 3,1])
depth=tf.tile(depth,[1,1,1,3])


saver = tf.train.Saver()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    saver.restore(sess, checkpoint)
    if not sun_dataset:
        i1_, i2_ = sess.run([il_t1,ir_t1])
        i=3
    else:
        i1_, i2_ = sess.run([i1,i2])
        i=4
    [flow_,i2_warped_,depth_] = sess.run([flow,i2_warped,depth],feed_dict={x: i1_, y: i2_})
    flow_=flow_[0, :, :, :]
#(normalized points) cv2 version
# a=np.array([np.zeros(width),
#             range(0,width),
#             np.ones(width)],dtype=np.float32)
# print(np.shape(a))
# for i in range(1,height):
#     tmp=np.array([i*np.ones(width),
#                   range(0,width),
#                   np.ones(width)],dtype=np.float32)
#     a=np.hstack([a,tmp])
# print(np.shape(a))
# b=a[0:2,:]+np.reshape(flow_,[2,height*width])
# a[0,:]=a[0,:]/height
# a[1,:]=a[1,:]/width
# b[0,:]=b[0,:]/height
# b[1,:]=b[1,:]/width
# depth_=depth_estimation(c1,c2,a,b)
# depth_=np.reshape(depth_[2,:],[1,height,width,1])
# depth_=np.linalg.norm(depth_,axis=3)
# depth_=np.reshape(depth_,[1,height,width,1])
# depth_=np.tile(depth_,[1,1,1,3])
print(np.shape(depth_))
print(depth_)
print('avg:',np.mean(depth_)) 
print('range:',np.min(depth_),np.max(depth_))       
print "Dumping few visuals" 

vis_dir='./vis/flownet2/'
blended_image=0.5*i1_+0.5*i2_warped_
dump2disk(vis_dir, i, i1_,i2_,i2_warped_,depth_,blended_image)
#save flow image
flow_img = flow_to_image(flow_)
full_out_path = os.path.join(vis_dir, "flow_"+'{:08}'.format(i)  + '.png')
imsave(full_out_path, flow_img)

