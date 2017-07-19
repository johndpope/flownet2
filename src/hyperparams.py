#core parameters
learning_rate=0.00001
max_iterations=100000
batch_size=8
height=320
width=448

#control parameters
do_zero_motion=False

archi='Flownetc'
do_avgpooling=True
quater=True
fine_tune=False

#directories
log_dir='./graphs/'
if do_zero_motion:
	train_log_dir=log_dir+'zero_train'
	validation_log_dir=log_dir+'zero_validation'
else:
	train_log_dir=log_dir+'train_'+('2_' if archi=='Flownet2' else 'c_')+('sc_' if not quater else 'q_')+('avg/' if do_avgpooling else 'fc/')
	validation_log_dir=log_dir+'validation'+('2_' if archi=='Flownet2' else 'c_')+('sc_' if not quater else 'q_')+('avg/' if do_avgpooling else 'fc/')

checkpoint_dir='./checkpoints/'
save_dir=checkpoint_dir+('last_layer_' if not fine_tune else 'fine_tune_')+('flownet2_' if archi=='Flownet2' else 'flownetc_')+('q_' if quater else 'sc_')+('avg/' if do_avgpooling else 'fc/')+('last_layer.ckpt' if not fine_tune else 'fine_tune.ckpt')

#checkpoint to load
if not fine_tune:
	variables_to_exclude=["f1","f2","beta2_power","beta1_power"]
	checkpoint=checkpoint_dir+('FlowNet2/flownet-2.ckpt-0'if archi=='Flownet2' else 'FlowNetC/flownet-C.ckpt-0')
else:
	variables_to_exclude=[]
	checkpoint_dir=checkpoint_dir+('last_layer_' if not fine_tune else 'fine_tune_')+('flownet2_' if archi=='Flownet2' else 'flownetc_')+('q_' if quater else 'sc_')+('avg/' if do_avgpooling else 'fc/')+('last_layer.ckpt' if not fine_tune else 'fine_tune.ckpt')

