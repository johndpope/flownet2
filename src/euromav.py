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

if Mode=='test':
	writer_test = tf.summary.FileWriter(test_log_dir, None)
else:
	writer_t = tf.summary.FileWriter(train_log_dir, None)
	writer_v = tf.summary.FileWriter(validation_log_dir, None)

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
if (do_zero_motion):
	RT_e=tf.tile(tf.reshape(tf.Variable([[[1,0,0,0],
	 [0,1,0,0],
	 [0,0,1,0],
	 [0,0,0,1]]], dtype=tf.float32),[1,4,4]),[batch_size,1,1])
	(rtd,rta) = rtLoss(RT_e,rt12_g)
	cost = rta+(1.0*rtd)
elif Mode=='train':
	(rtd, rta) = rtLoss(RT_e, rt12_g)
	cost = rta+(1.0*rtd)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
else:
	(rtd, rta) = rtLoss(RT_e, rt12_g)
	cost = rta+(1.0*rtd)
	

with tf.name_scope("RT_loss"):
		odo_loss_sum = tf.summary.scalar("loss", cost)
		rtd_sum = tf.summary.scalar("rtd", rtd)
		rta_sum = tf.summary.scalar("rta", rta)
summary = tf.summary.merge([odo_loss_sum,rtd_sum,rta_sum])
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
saver1 = tf.train.Saver()
with tf.Session() as sess:
	sess.run(init_op)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	if (not do_zero_motion) and (pretrained_flow):
		if Mode=='test':
			saver1.restore(sess, checkpoint)
		else:
			saver.restore(sess, checkpoint)
	for i in range(0,max_iterations):
		if (Mode=='train'):
			# train dataset
			i1_, i2_ = sess.run([il_t1,il_t2])
			[opt, train_loss, summ1,RT_e_,rt12_g_] = sess.run([optimizer, cost, summary,RT_e,rt12_g],feed_dict={x: i1_, y: i2_})
			writer_t.add_summary(summ1,i)
			#validation dataset
			i1_, i2_ = sess.run([il_v1,il_v2])
			[val_loss, summ2] = sess.run([cost, summary],feed_dict={x: i1_, y: i2_})
			sys.stdout.write("\r")
			sys.stdout.write(spinner.next())
			sys.stdout.write('  iteration:%d train loss:%f validation loss:%f'%(i,train_loss,val_loss))
			sys.stdout.flush()
			writer_v.add_summary(summ2,i)
			if (i%10000 ==0):
				save_path = saver1.save(sess, save_dir)
				print("Model saved in file: %s at step %d" % (save_path,i))
				print("predicted T:",RT_e_[0])
				print("ground truth T:",rt12_g_[0])

		elif do_zero_motion:
			# train dataset
			i1_, i2_ = sess.run([i_t1,i_t2])
			[train_loss, summ1] = sess.run([cost, summary],feed_dict={x: i1_, y: i2_})
			writer_t.add_summary(summ1,i)
			#validation dataset
			i1_, i2_ = sess.run([i_v1,i_v2])
			[val_loss, summ2] = sess.run([cost, summary],feed_dict={x: i1_, y: i2_})
			sys.stdout.write("\r")
			sys.stdout.write(spinner.next())
			sys.stdout.write('  iteration:%d zero train loss:%f zero validation loss:%f'%(i,train_loss,val_loss))
			sys.stdout.flush()
			writer_v.add_summary(summ2,i)
		else:
			i1_, i2_ = sess.run([il_t1,il_t2])
			[test_loss, summ1,rta_,rtd_,RT_e_,rt12_g_] = sess.run([cost, summary,rta,rtd,RT_e,rt12_g],feed_dict={x: i1_, y: i2_})
			writer_test.add_summary(summ1,i)
			sys.stdout.write("\r")
			sys.stdout.write(spinner.next())
			# new_point_=np.matmul(RT_e_,prev_point)
			sys.stdout.write('  iteration:%d test loss:%f, rta loss:%f,rtd loss:%f'%(i,test_loss,rta_,rtd_))
			# print(new_point_,p_t1_[0,0:3,3])
			print('pred_=',np.reshape(RT_e_[0,0:3,3],[1,3]),'gt=',np.reshape(rt12_g_[0,0:3,3],[1,3]))
			sys.stdout.flush()
			# if (i%10000 ==0):
			# 	print "Dumping few visuals" 
			# 	dump2disk(vis_dir, 1, i1_,i2_)
	print('optimization finished')
