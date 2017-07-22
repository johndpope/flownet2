import tensorflow as tf
import inputs.batcher as bat
from flownet2.flownet2 import FlowNet2
from flownet_c.flownet_c import FlowNetC
import numpy as np
from extras import ominus,spinning_cursor
from losses import safe_rtLoss
from hyperparams import *
from dump2disk import *

spinner = spinning_cursor() #just for cool ouput
details()# print all the details

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

(i_t1,i_t2,p_t1,p_t2)=bat.euromav_batch(train_txt,batch_size)#configs from dataset_configs
(i_v1,i_v2,p_v1,p_v2)=bat.euromav_batch(val_txt,batch_size)#configs from dataset_configs

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
elif Mode=='train':
	(rtd, rta) = safe_rtLoss(RT_e, rt12_g)
	cost = (rta+rtd)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
else:
	(rtd, rta) = safe_rtLoss(RT_e, rt12_g)
	cost = (rta+rtd)

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
		saver.restore(sess, checkpoint)
	for i in range(0,max_iterations):
		if (Mode=='train'):
			# train dataset
			i1_, i2_ = sess.run([i_t1,i_t2])
			[opt, train_loss, summ1] = sess.run([optimizer, cost, summary],feed_dict={x: i1_, y: i2_})
			writer_t.add_summary(summ1,i)
			#validation dataset
			i1_, i2_ = sess.run([i_v1,i_v2])
			[val_loss, summ2] = sess.run([cost, summary],feed_dict={x: i1_, y: i2_})
			sys.stdout.write("\r")
			sys.stdout.write(spinner.next())
			sys.stdout.write('  iteration:%d train loss:%f validation loss:%f'%(i,train_loss,val_loss))
			sys.stdout.flush()
			writer_v.add_summary(summ2,i)
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
			i1_, i2_ = sess.run([i_v1,i_v2])
			[test_loss, summ1] = sess.run([cost, summary],feed_dict={x: i1_, y: i2_})
			writer_test.add_summary(summ1,i)
			sys.stdout.write("\r")
			sys.stdout.write(spinner.next())
			sys.stdout.write('  iteration:%d test loss:%f'%(i,test_loss))
			sys.stdout.flush()
		if (i%10000 ==0):
			save_path = saver1.save(sess, save_dir)
			print("Model saved in file: %s at step %d" % (save_path,i))
			print "Dumping few visuals" 
            dump2disk(vis_dir, i, i1_,i2_)
	print('optimization finished')
