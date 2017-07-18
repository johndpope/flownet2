learning_rate=0.000001
max_iterations=100000
batch_size=8
height=320
width=448
do_zero_motion=False
archi="Flownetc"
fine_tune=True
if (fine_tune==True):
	train_lastlayer=False
else:
	train_lastlayer=True