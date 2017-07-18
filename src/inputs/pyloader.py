import tensorflow as tf
import sys
import hyperparams as hyp
from utils import *
sys.path.append('..')
import os

def pyloader(dataset_dir,bs,h,w,off_h,off_w):
    # print "%"*50
    
    # image is H x W x 3
    i_ = np.load("%s/i.npy" % dataset_dir)
    height = int(i_.shape[0])
    width = int(i_.shape[1])
    i = tf.constant(i_)
    i = preprocess_color(i)
    if hyp.dataset_name == 'TOY':
        i, _, _ = topleft_crop(i,h,w,height,width)
    else:
        i, _, _ = bottomcenter_crop(i,h,w,height,width)
    
    # depth is H x W
    d_ = np.load("%s/d.npy" % dataset_dir)
    d = tf.constant(d_)
    d = tf.expand_dims(d,2)
    d = preprocess_depth(d)
    if hyp.dataset_name == 'TOY':
        d, _, _ = topleft_crop(d,h,w,height,width)
    else:
        d, _, _ = bottomcenter_crop(d,h,w,height,width)
     
    # valid is H x W
    v_ = np.load("%s/v.npy" % dataset_dir)
    v = tf.constant(v_)
    v = tf.expand_dims(v,2)
    v = preprocess_valid(v)
    if hyp.dataset_name == 'TOY':
        v, _, _ = topleft_crop(v,h,w,height,width)
    else:
        v, _, _ = bottomcenter_crop(v,h,w,height,width)

    i = tf.expand_dims(i,0,name="i")
    d = tf.expand_dims(d,0,name="d")
    v = tf.expand_dims(v,0,name="v")

    print_shape(i)
    print_shape(d)
    print_shape(v)
    
    return i, d, v
