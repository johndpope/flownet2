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

(i_t1,i_t2,p_t1,p_t2)=bat.euromav_batch(train_txt,batch_size)#configs from dataset_configs
(i_v1,i_v2,p_v1,p_v2)=bat.euromav_batch(val_txt,batch_size)#configs from dataset_configs
writer_t = tf.summary.FileWriter('./graphs/train', None)
writer_v = tf.summary.FileWriter('./graphs/validation', None)
writer_t_z = tf.summary.FileWriter('./graphs/zero_train', None)
writer_v_z = tf.summary.FileWriter('./graphs/zero_validation', None)

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
	(rtd,rta) = safe_rtLoss(RT_e,rt12_g)
	cost = (rta+rtd)
else:
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

saver1 = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init_op)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	if not do_zero_motion:
		saver.restore(sess, checkpoint)
	for i in range(0,max_iterations):
		if not do_zero_motion:
			# train dataset
			i1_, i2_ = sess.run([i_t1,i_t2])
			[opt, train_loss, summ1] = sess.run([optimizer, cost, summary],feed_dict={x: i1_, y: i2_})
			writer_t.add_summary(summ1,i)
			#validation dataset
			i1_, i2_ = sess.run([i_v1,i_v2])
			[val_loss, summ2] = sess.run([cost, summary],feed_dict={x: i1_, y: i2_})
			print('iteration:',i,' train loss:',train_loss,' validation loss:',val_loss)
			writer_v.add_summary(summ2,i)
		else:
			# train dataset
			i1_, i2_ = sess.run([i_t1,i_t2])
			[train_loss, summ1] = sess.run([cost, summary],feed_dict={x: i1_, y: i2_})
			writer_t_z.add_summary(summ1,i)
			#validation dataset
			i1_, i2_ = sess.run([i_v1,i_v2])
			[val_loss, summ2] = sess.run([cost, summary],feed_dict={x: i1_, y: i2_})
			print('iteration:',i,' zero train loss:',train_loss,' zero validation loss:',val_loss)
			writer_v_z.add_summary(summ2,i)
	print('optimization finished')
	save_path = saver1.save(sess, "./checkpoints/last_layer.ckpt")
	print("Model saved in file: %s" % save_path)