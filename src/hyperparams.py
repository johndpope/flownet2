import sys
import tensorflow as tf
import numpy as np
#core parameters
max_iterations=100000
batch_size=8
height=320
width=448
new_resol=True

#control parameters
do_zero_motion=False

pretrained_flow=True

archi='Flownetc'
quater=True
fine_tune=False
do_avgpooling=True
if not do_zero_motion:
	Mode='train'

#read tf records txt files
train_txt='./train.txt'
val_txt='./val.txt'

#camera extrinsics
# intrinsics are in pixels
resolution=[752, 480]
height_old=480
width_old=752
scal_w=1
# float(width)/float(width_old)
scal_h=1
# float(height)/float(height_old)
print('scaling',scal_w,scal_h)
pixel_size=3.75E-3
sensor_size=pixel_size*resolution[0]*resolution[1]
camera_model='pinhole'
distortion_model='radial-tangential'
#camera 0 wrt body_frame
b_c_1=np.array([[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
	   [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
       [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
       [0,0,0,1]],dtype=np.float32)

intrinsics_1= [458.654, 457.296, 367.215, 248.375] #fu, fv, cu, cv
s_intrinsics_1= [458.654*scal_w, 457.296*scal_h, 367.215*scal_w, 248.375*scal_h] #fu, fv, cu, cv
k1=np.array([[intrinsics_1[0],0,intrinsics_1[2]],
				[0,intrinsics_1[1],intrinsics_1[3]],
				[0,0,1]],dtype=np.float32)
scaled_k1=np.array([[s_intrinsics_1[0],0,s_intrinsics_1[2]],
				[0,s_intrinsics_1[1],s_intrinsics_1[3]],
				[0,0,1]],dtype=np.float32)

c1=np.matmul(scaled_k1,b_c_1[0:3,:])

distortion_coefficients_1= np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05],dtype=np.float32)
#camera 1 wrt body_frame11
b_c_2=np.array([[0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556],
       [0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024],
       [-0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038],
       [0,0,0,1]],dtype=np.float32)
intrinsics_2=np.array([457.587, 456.134, 379.999, 255.238],dtype=np.float32) #fu, fv, cu, cv
s_intrinsics_2=np.array([457.587*scal_w, 456.134*scal_h, 379.999*scal_w, 255.238*scal_h],dtype=np.float32) #fu, fv, cu, cv
k2=np.array([[intrinsics_2[0],0,intrinsics_2[2]],
				[0,intrinsics_2[1],intrinsics_2[3]],
				[0,0,1]],dtype=np.float32)
scaled_k2=np.array([[s_intrinsics_2[0],0,s_intrinsics_2[2]],
				[0,s_intrinsics_2[1],s_intrinsics_2[3]],
				[0,0,1]],dtype=np.float32)

c2=np.matmul(scaled_k2,b_c_2[0:3,:])
c2_1= np.matmul(np.linalg.inv(b_c_2),(b_c_1))
rotation=np.array(c2_1[0:3,0:3], dtype=np.float32)
translation=np.reshape(c2_1[0:3,3],[1,3])
distortion_coefficients_2= np.array([-0.28368365,  0.07451284, -0.00010473, -3.55590700e-05])


generate_trajectory=False
if generate_trajectory:
	max_iterations=100
	batch_size=1
	Mode='test'
	train_txt='./records.txt'




name='lr_e_4_'
#log directories
log_dir='./graphs/'
if do_zero_motion:
	train_log_dir=log_dir+'zero_train'
	validation_log_dir=log_dir+'zero_validation'
if Mode=='train':
	dir_='train_'+('2_' if archi=='Flownet2' else 'c_')+('sc_' if not quater else 'q_')+('avg' if do_avgpooling else '')+('_scratch' if not pretrained_flow else '')+('_fine/' if fine_tune else '/')
	train_log_dir=log_dir+name+dir_
	validation_log_dir=log_dir+name+'validation'+('2_' if archi=='Flownet2' else 'c_')+('sc_' if not quater else 'q_')+('avg' if do_avgpooling else '')+('_scratch' if not pretrained_flow else '')+('_fine/' if fine_tune else '/')
if Mode=='test':
	dir_='test_'+('2_' if archi=='Flownet2' else 'c_')+('sc_' if not quater else 'q_')+('avg' if do_avgpooling else '')+('_scratch' if not pretrained_flow else '')+('_fine/' if fine_tune else '/')
	test_log_dir=log_dir+dir_

vis_dir='./vis/'+dir_
#checkpoint name to save
checkpoint_dir='./checkpoints/'
if Mode=='train':
	save_dir=checkpoint_dir+('last_layer_' if not fine_tune else 'fine_tune_')+('flownet2_' if archi=='Flownet2' else 'flownetc_')+('q_' if quater else 'sc_')+('_scratch' if not pretrained_flow else '')+('avg' if do_avgpooling else 'fc')+(name+'/')+('last_layer.ckpt' if not fine_tune else 'fine_tune.ckpt')

#checkpoint to load
if (Mode=='train'): 
	if not fine_tune:
		learning_rate=0.00001
		variables_to_exclude=["f1","f2","beta2_power","beta1_power"]
		checkpoint=checkpoint_dir+('FlowNet2/flownet-2.ckpt-0'if archi=='Flownet2' else 'FlowNetC/flownet-C.ckpt-0')
	else:
		learning_rate=0.0000001
		variables_to_exclude=[]
		checkpoint=checkpoint_dir+'last_layer_'+('flownet2_' if archi=='Flownet2' else 'flownetc_')+('q_' if quater else 'sc_')+('avg/' if do_avgpooling else 'fc/')+'last_layer.ckpt' 
else:
	variables_to_exclude=[]
	# checkpoint=checkpoint_dir+'translation_flownetc_q_fc/'+'last_layer.ckpt'
	# checkpoint=checkpoint_dir+'last_layer_flownetc_q_fconlyrotation/'+'last_layer.ckpt' 
	checkpoint=checkpoint_dir+'last_layer_'+('flownet2_' if archi=='Flownet2' else 'flownetc_')+('q_' if quater else 'sc_')+('avg/' if do_avgpooling else 'fc/')+'last_layer.ckpt' 


def details():
	sys.stdout.write('-'*30)
	sys.stdout.write('\n    %s     %s\n'%(Mode,archi))
	sys.stdout.write('-'*30)
	sys.stdout.write('\n    %s     \n'%('Loading pretrained flow' if pretrained_flow else 'Training from scratch'))
	if pretrained_flow:
		sys.stdout.write('-'*30)
		sys.stdout.write('\n loading checkpoint from directory:    %s     \n'%(checkpoint))
		sys.stdout.write('-'*30)
		sys.stdout.write('\n    %s     \n'%('to fine tune' if fine_tune else 'training only last layer'))
	sys.stdout.write('-'*30)
	sys.stdout.write('\n    %s     \n'%('with avg pooling last layer' if do_avgpooling else 'with fully connected last layer'))
	sys.stdout.write('-'*30)
	sys.stdout.write('\n    %s     \n'%('with quaternion as output' if quater else 'sincos terms as output'))
	if Mode=='train':
		sys.stdout.write('-'*30)
		sys.stdout.write('\nwill save checkpoint to directory:    %s     \n'%(save_dir))
		sys.stdout.write('-'*30)
		sys.stdout.write('\nlearning rate:    %f     \n'%(learning_rate))		
		sys.stdout.write('-'*30)
		sys.stdout.write('\nBatch size:   %d   Maximum Iterations   %d    \n'%(batch_size, max_iterations))		
		sys.stdout.write('-'*30)
	sys.stdout.write('\nImage Height:   %d   Image Width   %d    \n'%(height, width))	
	sys.stdout.write('-'*30)
	sys.stdout.write('\n')
	sys.stdout.flush() 