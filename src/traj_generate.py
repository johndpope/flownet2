import tensorflow as tf
import inputs.batcher as bat
from flownet2.flownet2 import FlowNet2
from flownet_c.flownet_c import FlowNetC
import numpy as np
from extras import ominus,spinning_cursor
from losses import safe_rtLoss,rtLoss
from hyperparams import *
from dump2disk import *
import scipy.io

spinner = spinning_cursor() #just for cool ouput
details()# print all the details

prev_point=np.array([[0],[0],[0],[1]],dtype=np.float32)
gt=np.array([])
pred_traj=np.array([])
error_rtd=np.array([])
error_rta=np.array([])

slim = tf.contrib.slim
# Create a new network
if (archi=="Flownet2"):
	net = FlowNet2()
elif (archi=="Flownetc"):
	net=FlowNetC()
else:
	raise("didnot choose architecture")

variables_to_restore = slim.get_variables_to_restore(exclude=variables_to_exclude) #these are my last two layers
saver = tf.train.Saver(variables_to_restore)

(il_t1,il_t2,ir_t1,ir_t2,p_t1,p_t2)=bat.euromav_batch(train_txt,batch_size,shuffle=False)#configs from dataset_configs
(il_v1,il_v2,ir_v1,ir_v2,p_v1,p_v2)=bat.euromav_batch(val_txt,batch_size,shuffle=False)#configs from dataset_configs

rt12_g = ominus(p_t2, p_t1)

x = tf.placeholder("float", shape=[batch_size, height, width,3])
y = tf.placeholder("float", shape=[batch_size, height, width,3])
inputs = {
			'input_a': x,
			'input_b': y,
		}

RT_e=net.euro(inputs)

(rtd, rta) = rtLoss(RT_e, rt12_g)
cost = rta+(rtd)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
saver1 = tf.train.Saver()
with tf.Session() as sess:
	sess.run(init_op)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	saver1.restore(sess, checkpoint)
	for i in range(0,max_iterations):
		i1_, i2_ = sess.run([il_t1,il_t2])
		[test_loss,rta_,rtd_,RT_e_,rt12_g_] = sess.run([cost,rta,rtd,RT_e,rt12_g],feed_dict={x: i1_, y: i2_})
		# new_point_=np.matmul(RT_e_,prev_point)
		sys.stdout.write('  iteration:%d ,rta loss:%f, rtd loss:%f \n'%(i,rta_,rtd_))
		# print(new_point_,p_t1_[0,0:3,3])
		print(rt12_g_[0,0:3,0:3],RT_e_[0,0:3,0:3])
		sys.stdout.flush()
		if (gt.size==0):
			gt=np.reshape(rt12_g_[0,0:3,0:3],[1,9])
			pred_traj=np.reshape(RT_e_[0,0:3,0:3],[1,9])
			# error_rta=np.reshape(rta_,[1])
			# error_rtd=np.reshape(rtd_,[1])
		else:
			gt=np.vstack([gt,np.reshape(rt12_g_[0,0:3,0:3],[1,9])])
			pred_traj=np.vstack([pred_traj,np.reshape(RT_e_[0,0:3,0:3],[1,9])])

	scipy.io.savemat('onlyrotation.mat',dict(gt_r=gt,pred_r=pred_traj))
