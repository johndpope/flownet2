import tensorflow as tf
import sys
sys.path.append('..')
import hyperparams as hyp
import batcher as bat
from utils import split_intrinsics

class Inputs:
    def __init__(self):
        #train
        self.i1_g_t = tf.ones((hyp.bs, hyp.h, hyp.w, 1))
        self.i2_g_t = tf.ones((hyp.bs, hyp.h, hyp.w, 1))
        self.p1_g_t = tf.ones((hyp.bs, 4, 4))
        self.p2_g_t = tf.ones((hyp.bs, 4, 4))
        self.off_h_t = tf.ones((hyp.bs))
        self.off_w_t = tf.ones((hyp.bs))
        self.fy_t = tf.ones((hyp.bs))
        self.fx_t = tf.ones((hyp.bs))
        self.y0_t = tf.ones((hyp.bs))
        self.x0_t = tf.ones((hyp.bs))
        #validation
        self.i1_g_v = tf.ones((hyp.bs, hyp.h, hyp.w, 3))
        self.i2_g_v = tf.ones((hyp.bs, hyp.h, hyp.w, 3))
        self.p1_g_v = tf.ones((hyp.bs, 4, 4))
        self.p2_g_v = tf.ones((hyp.bs, 4, 4))
        self.off_h_v = tf.ones((hyp.bs))
        self.off_w_v = tf.ones((hyp.bs))
        self.fy_v = tf.ones((hyp.bs))
        self.fx_v = tf.ones((hyp.bs))
        self.y0_v = tf.ones((hyp.bs))
        self.x0_v = tf.ones((hyp.bs))
        
        (self.i1_g_t, self.i2_g_t,
         self.p1_g_t, self.p2_g_t,
         self.off_h_t, self.off_w_t) = bat.euromav_batch(hyp.dataset_t,
                                                         hyp.bs,hyp.h,hyp.w,
                                                         shuffle=True)

        (self.i1_g_v, self.i2_g_v,
         self.p1_g_v, self.p2_g_v,
         self.off_h_v, self.off_w_v) = bat.euromav_batch(hyp.dataset_v,
                                                         hyp.bs,hyp.h,hyp.w,
                                                         shuffle=True)
        # for svkitti, let's just grab the intrinsics from hyp
        self.fy_t = tf.cast(tf.tile(tf.reshape(hyp.fy,[1]),[hyp.bs]),tf.float32)
        self.fx_t = tf.cast(tf.tile(tf.reshape(hyp.fx,[1]),[hyp.bs]),tf.float32)
        self.y0_t = tf.cast(tf.tile(tf.reshape(hyp.y0,[1]),[hyp.bs]),tf.float32)
        self.x0_t = tf.cast(tf.tile(tf.reshape(hyp.x0,[1]),[hyp.bs]),tf.float32)
        self.fy_v = tf.cast(tf.tile(tf.reshape(hyp.fy,[1]),[hyp.bs]),tf.float32)
        self.fx_v = tf.cast(tf.tile(tf.reshape(hyp.fx,[1]),[hyp.bs]),tf.float32)
        self.y0_v = tf.cast(tf.tile(tf.reshape(hyp.y0,[1]),[hyp.bs]),tf.float32)
        self.x0_v = tf.cast(tf.tile(tf.reshape(hyp.x0,[1]),[hyp.bs]),tf.float32)

