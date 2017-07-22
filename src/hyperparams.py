import sys
#core parameters
max_iterations=100000
batch_size=8
height=320
width=448

#control parameters
do_zero_motion=False

pretrained_flow=False

archi='Flownetc'
quater=True
fine_tune=False
do_avgpooling=False
if not do_zero_motion:
	Mode='train'

#read tf records txt files
train_txt='./train.txt'
val_txt='./val.txt'

#log directories
log_dir='./graphs/'
if do_zero_motion:
	train_log_dir=log_dir+'zero_train'
	validation_log_dir=log_dir+'zero_validation'
if Mode=='train':
	dir_='train_'+('2_' if archi=='Flownet2' else 'c_')+('sc_' if not quater else 'q_')+('avg' if do_avgpooling else '')+('_scratch' if not pretrained_flow else '')+('_fine/' if fine_tune else '/')
	train_log_dir=log_dir+dir_
	validation_log_dir=log_dir+'validation'+('2_' if archi=='Flownet2' else 'c_')+('sc_' if not quater else 'q_')+('avg' if do_avgpooling else '')+('_scratch' if not pretrained_flow else '')+('_fine/' if fine_tune else '/')
if Mode=='test':
	dir_='test_'+('2_' if archi=='Flownet2' else 'c_')+('sc_' if not quater else 'q_')+('avg' if do_avgpooling else '')+('_scratch' if not pretrained_flow else '')+('_fine/' if fine_tune else '/')
	test_log_dir=log_dir+dir_

vis_dir='./vis/'+dir_
#checkpoint name to save
checkpoint_dir='./checkpoints/'
if Mode=='train':
	save_dir=checkpoint_dir+('last_layer_' if not fine_tune else 'fine_tune_')+('flownet2_' if archi=='Flownet2' else 'flownetc_')+('q_' if quater else 'sc_')+('_scratch' if not pretrained_flow else '')+('avg/' if do_avgpooling else 'fc/')+('last_layer.ckpt' if not fine_tune else 'fine_tune.ckpt')

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
	sys.stdout.write('-'*30)
	sys.stdout.write('\nwill save checkpoint to directory:    %s     \n'%(save_dir))
	if not do_zero_motion:
		sys.stdout.write('-'*30)
		sys.stdout.write('\nlearning rate:    %f     \n'%(learning_rate))		
	sys.stdout.write('-'*30)
	sys.stdout.write('\nImage Height:   %d   Image Width   %d    \n'%(height, width))	
	sys.stdout.write('-'*30)
	sys.stdout.write('\nBatch size:   %d   Maximum Iterations   %d    \n'%(batch_size, max_iterations))		
	sys.stdout.write('-'*30)
	sys.stdout.write('\n')
	sys.stdout.flush() 