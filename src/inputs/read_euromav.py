import numpy as np
import tensorflow as tf
import os
import sys 

inDir='../../records/'
def read_euromav(filename_queue):
    compress = tf.python_io.TFRecordOptions(compression_type = tf.python_io.TFRecordCompressionType.GZIP)
    reader = tf.TFRecordReader(options=compress)
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'prev_img_l': tf.FixedLenFeature([], tf.string),
            'curr_img_l': tf.FixedLenFeature([], tf.string),
            'prev_img_r': tf.FixedLenFeature([], tf.string),
            'curr_img_r': tf.FixedLenFeature([], tf.string),
            'p1_raw': tf.FixedLenFeature([16], tf.float32),
            'p2_raw': tf.FixedLenFeature([16], tf.float32),
        })
    height=features['height']
    width=features['width']

    prev_img_l = tf.decode_raw(features['prev_img_l'], tf.uint8)
    curr_img_l = tf.decode_raw(features['curr_img_l'], tf.uint8)
    prev_img_r = tf.decode_raw(features['prev_img_r'], tf.uint8)
    curr_img_r = tf.decode_raw(features['curr_img_r'], tf.uint8)
    p1 = features['p1_raw']
    p2 = features['p2_raw']
    # rel_pose = tf.decode_raw(features['rel_pose'], tf.float64)
    
    prev_img_l = tf.reshape(prev_img_l, [480,752],name='prev_img_l')
    curr_img_l = tf.reshape(curr_img_l, [480,752],name='curr_img_l')
    prev_img_r = tf.reshape(prev_img_r, [480,752],name='prev_img_r')
    curr_img_r = tf.reshape(curr_img_r, [480,752],name='curr_img_r')
    # rel_pose = tf.reshape(rel_pose, [4,4],name='rel_pose')
    pose_shape = tf.stack([4, 4])
    p1 = tf.reshape(p1, pose_shape)
    p2 = tf.reshape(p2, pose_shape)
    
    # imu=tf.cast(imu,tf.float32)
    # rel_pose=tf.cast(rel_pose,tf.float32)

    return (height, width, prev_img_l, curr_img_l, prev_img_r, curr_img_r, p1, p2)

if __name__=="__main__":
	nRecords= len([name for name in os.listdir(inDir) if os.path.isfile(os.path.join(inDir, name))])
	print 'found %d records' % nRecords
	records = [inDir+'/sample_'+ str(i)+'.tfrecord' for i in range(1,20)]
	queue = tf.train.string_input_producer(records, shuffle=False)
 	height, width, prev_img_l, curr_img_l,prev_img_r, curr_img_r, p1, p2=read_euromav(queue)
 	BATCH_SIZE=10
	height, width, prev_img_l, curr_img_l,prev_img_r, curr_img_r, p1, p2 = tf.train.batch([height, width, prev_img_l, curr_img_l,prev_img_r, curr_img_r, p1, p2],batch_size=BATCH_SIZE)
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	with tf.Session()  as sess:		
		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		height_, width_, prev_img_l_, curr_img_l_,prev_img_r_, curr_img_r_,p1_,p2_=sess.run([height, width,prev_img_l, curr_img_l,prev_img_r, curr_img_r,p1,p2])
		print(p1_[0])
	coord.request_stop()
	coord.join(threads)