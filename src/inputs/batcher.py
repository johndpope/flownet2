import tensorflow as tf
import sys
sys.path.append('..')
import os
from readers import *
dataset_location='./records/'
# sys.path.append('../writers_readers')
# from read_svkitti_tfrecords import *
def euromav_batch(dataset,bs,shuffle=True,dotrim=False):
    with open(dataset) as f:
        content = f.readlines()
    records = [dataset_location + line.strip() for line in content]
    nRecords = len(records)
    print 'found %d records' % nRecords
    for record in records:
        assert os.path.isfile(record), 'Record at %s was not found' % record

    queue = tf.train.string_input_producer(records, shuffle=shuffle)

    (h,w,i1,i2,p1,p2) = read_and_decode_euromav(queue)

    i1 = tf.cast(i1,tf.float32) * 1./255 - 0.5
    i2 = tf.cast(i2,tf.float32) * 1./255 - 0.5
    i1 = tf.image.resize_images(i1, [320, 448])
    i2 = tf.image.resize_images(i2, [320, 448])
    i1 =tf.tile(i1,[1,1,3])
    i2 =tf.tile(i2,[1,1,3])

    # d1 = d1*v1 # put 0 depth at invalid spots
    # d2 = d2*v2
    # image tensors need to be cropped. we'll do them all at once.
    # allCat = tf.concat(axis=2,values=[i1,i2],
    #                    name="allCat")
    
    # # image tensors need to be cropped. we'll do them all at once.
    # print_shape(allCat)
    off_h=0
    off_w=0
    # allCat = tf.slice(allCat,[off_h,off_w,0],[-1,-1,-1],name="cropped_tensor")
    # # allCat_crop, off_h, off_w = random_crop(allCat,crop_h,crop_w,480,752)
    # print_shape(allCat)
    # i1 = tf.slice(allCat, [0,0,0], [-1,-1,1], name="i1")
    # i2 = tf.slice(allCat, [0,0,1], [-1,-1,1], name="i2")
    
    batch = tf.train.batch([i1,i2,
                            p1,p2],
                           batch_size=bs,
                           dynamic_pad=True)
    return batch

def svkitti_batch(dataset,bs,crop_h,crop_w,shuffle=True,dotrim=False):
    with open(dataset) as f:
        content = f.readlines()
    records = [hyp.dataset_location + line.strip() for line in content]
    nRecords = len(records)
    print 'found %d records' % nRecords
    for record in records:
        assert os.path.isfile(record), 'Record at %s was not found' % record

    queue = tf.train.string_input_producer(records, shuffle=shuffle)

    (h,w,i1,i2,d1,d2,f12,f23,v1,v2,p1,p2,m1,m2) = read_and_decode_svkitti(queue)

    i1 = tf.cast(i1,tf.float32) * 1./255 - 0.5
    i2 = tf.cast(i2,tf.float32) * 1./255 - 0.5
    d1 = tf.cast(d1,tf.float32)
    d2 = tf.cast(d2,tf.float32)
    v1 = tf.cast(v1,tf.float32) # 1 at non-sky pixels
    v2 = tf.cast(v2,tf.float32)
    m1 = tf.cast(m1,tf.float32) * 1./255 # these are stored in [0,255], and 255 means moving.
    m2 = tf.cast(m2,tf.float32) * 1./255
    # d1 = d1*v1 # put 0 depth at invalid spots
    # d2 = d2*v2
    d1 = encode_depth(d1, hyp.depth_encoding) # encode 
    d2 = encode_depth(d2, hyp.depth_encoding)

    # image tensors need to be cropped. we'll do them all at once.
    allCat = tf.concat(axis=2,values=[i1,i2,
                                      d1,d2,
                                      f12,f23,
                                      v1,v2,
                                      m1,m2],
                       name="allCat")
    
    # image tensors need to be cropped. we'll do them all at once.
    print_shape(allCat)
    allCat_crop, off_h, off_w = random_crop(allCat,crop_h,crop_w,h,w)
    print_shape(allCat_crop)
    i1 = tf.slice(allCat_crop, [0,0,0], [-1,-1,3], name="i1")
    i2 = tf.slice(allCat_crop, [0,0,3], [-1,-1,3], name="i2")
    d1 = tf.slice(allCat_crop, [0,0,6], [-1,-1,1], name="d1")
    d2 = tf.slice(allCat_crop, [0,0,7], [-1,-1,1], name="d2")
    f12 = tf.slice(allCat_crop, [0,0,8], [-1,-1,2], name="f12")
    f23 = tf.slice(allCat_crop, [0,0,10], [-1,-1,2], name="f23")
    v1 = tf.slice(allCat_crop, [0,0,12], [-1,-1,1], name="v1")
    v2 = tf.slice(allCat_crop, [0,0,13], [-1,-1,1], name="v2")
    m1 = tf.slice(allCat_crop, [0,0,14], [-1,-1,1], name="m1")
    m2 = tf.slice(allCat_crop, [0,0,15], [-1,-1,1], name="m2")
    
    batch = tf.train.batch([i1,i2,
                            p1,p2,
                            off_h,off_w],
                           batch_size=bs,
                           dynamic_pad=True)
    return batch

def vkitti_batch(dataset,bs,crop_h,crop_w,shuffle=True,dotrim=False):
    with open(dataset) as f:
        content = f.readlines()
    records = [hyp.dataset_location + line.strip() for line in content]
    nRecords = len(records)
    print 'found %d records' % nRecords
    for record in records:
        assert os.path.isfile(record), 'Record at %s was not found' % record

    queue = tf.train.string_input_producer(records, shuffle=shuffle)

    (h,w,i1,i2,s1,s2,d1,d2,f12,f23,v1,v2,p1,p2,m1,m2,
     o1c, o1i, o1b, o1p, o1o,
     o2c, o2i, o2b, o2p, o2o,
     nc1,nc2,car1,car2,
     dets1,feats1,dets2,feats2) = read_and_decode_vkitti_w_everything(queue)

    i1 = tf.cast(i1,tf.float32) * 1./255 - 0.5
    i2 = tf.cast(i2,tf.float32) * 1./255 - 0.5
    s1 = tf.cast(s1,tf.float32)
    s2 = tf.cast(s2,tf.float32)
    car1 = tf.cast(car1,tf.float32)
    car2 = tf.cast(car2,tf.float32)

    d1 = tf.cast(d1,tf.float32)
    d2 = tf.cast(d2,tf.float32)
    v1 = tf.cast(v1,tf.float32)
    v2 = tf.cast(v2,tf.float32)
    m1 = tf.cast(m1,tf.float32) * 1./255 # for some reason these are stored in [0,255]
    m2 = tf.cast(m2,tf.float32) * 1./255

    # image tensors need to be cropped. we'll do them all at once.
    allCat = tf.concat(axis=2,values=[i1,i2,
                          s1,s2,
                          d1,d2,
                          f12,f23,
                          v1,v2,
                          m1,m2,
                          car1, car2],
                       name="allCat")

    # image tensors need to be cropped. we'll do them all at once.

    print_shape(allCat)
    allCat_crop, off_h, off_w = random_crop(allCat,crop_h,crop_w,h,w)
    print_shape(allCat_crop)
    i1 = tf.slice(allCat_crop, [0,0,0], [-1,-1,3], name="i1")
    i2 = tf.slice(allCat_crop, [0,0,3], [-1,-1,3], name="i2")
    s1 = tf.slice(allCat_crop, [0,0,6], [-1,-1,1], name="s1")
    s2 = tf.slice(allCat_crop, [0,0,7], [-1,-1,1], name="s2")
    d1 = tf.slice(allCat_crop, [0,0,8], [-1,-1,1], name="d1")
    d2 = tf.slice(allCat_crop, [0,0,9], [-1,-1,1], name="d2")
    f12 = tf.slice(allCat_crop, [0,0,10], [-1,-1,2], name="f12")
    f23 = tf.slice(allCat_crop, [0,0,12], [-1,-1,2], name="f23")
    v1 = tf.slice(allCat_crop, [0,0,14], [-1,-1,1], name="v1")
    v2 = tf.slice(allCat_crop, [0,0,15], [-1,-1,1], name="v2")
    m1 = tf.slice(allCat_crop, [0,0,16], [-1,-1,1], name="m1")
    m2 = tf.slice(allCat_crop, [0,0,17], [-1,-1,1], name="m2")
    car1 = tf.slice(allCat_crop, [0,0,18],[-1,-1,1], name='car1')
    car2 = tf.slice(allCat_crop, [0,0,19],[-1,-1,1], name='car2')
    # make it go from 0 to nLabels-1, rather than 1 to nLabels
    s1 = s1-1 
    s2 = s2-1
    # cast the seg labels back to int
    s1 = tf.cast(s1, tf.int32)
    s2 = tf.cast(s2, tf.int32)
    car1 = tf.cast(car1, tf.int32)
    car2 = tf.cast(car2, tf.int32)

    #first scale the bboxes
    o1b = tf.cast(tf.cast(o1b, tf.float32)*hyp.scale, tf.int64)
    o2b = tf.cast(tf.cast(o2b, tf.float32)*hyp.scale, tf.int64)
    
    #we need to shift the object bounding boxes over a bit
    #this function is in utils
    o1b = offsetbbox(o1b, off_h, off_w, crop_h, crop_w, dotrim)
    o2b = offsetbbox(o2b, off_h, off_w, crop_h, crop_w, dotrim)

    dets1 = offsetbbox(dets1, off_h, off_w, crop_h, crop_w, dotrim)
    dets2 = offsetbbox(dets2, off_h, off_w, crop_h, crop_w, dotrim)
    
    batch = tf.train.batch([i1,i2,
                            s1,s2,
                            d1,d2,
                            f12,f23,
                            v1,v2,
                            p1,p2,
                            m1,m2,
                            off_h,off_w,
                            o1c,o1i,o1b,o1p,o1o,
                            o2c,o2i,o2b,o2p,o2o,
                            nc1,nc2,car1,car2,
                            dets1,feats1,
                            dets2,feats2],
                           batch_size=bs,
                           dynamic_pad=True)
    return batch


def toy_batch(dataset,bs,h,w,shuffle=True):
    with open(dataset) as f:
        content = f.readlines()
    records = [hyp.dataset_location + '/' + line.strip() for line in content]
    
    nRecords = len(records)
    print 'found %d records' % nRecords
    queue = tf.train.string_input_producer(records, shuffle=shuffle)
    
    height,width,i1,i2,d1,d2,v1,v2,m1,m2 = read_and_decode_toy(queue)

    # some tensors need to be cast to float
    i1 = tf.cast(i1,tf.float32) * 1./255 - 0.5
    i2 = tf.cast(i2,tf.float32) * 1./255 - 0.5
    d1 = tf.cast(d1,tf.float32)
    d2 = tf.cast(d2,tf.float32)
    v1 = 1-tf.cast(v1,tf.float32) # this has ones at INVALID pixels
    v2 = 1-tf.cast(v2,tf.float32)
    m1 = tf.cast(m1,tf.float32) * 1./255 # this is in [0,255]
    m2 = tf.cast(m2,tf.float32) * 1./255

    # we need to clean up the valid mask a bit
    d1_ = tf.expand_dims(d1,0)
    d2_ = tf.expand_dims(d2,0)
    v1_ = tf.expand_dims(v1,0)
    v2_ = tf.expand_dims(v2,0)
    # high pass on depth to find more invalids
    blur_kernel = tf.transpose(tf.constant([[[[1./16,1./8,1./16],
                                              [1./8,1./4,1./8],
                                              [1./16,1./8,1./16]]]],
                                           dtype=tf.float32),perm=[3,2,1,0])
    blurred_d1 = tf.nn.conv2d(d1_, blur_kernel, strides=[1,1,1,1], padding="SAME")
    blurred_d2 = tf.nn.conv2d(d2_, blur_kernel, strides=[1,1,1,1], padding="SAME")
    sharp_d1 = d1_-blurred_d1
    sharp_d2 = d2_-blurred_d2
    zero_kernel = tf.zeros([3,3,1])
    # also erode valid
    v1_ = tf.cast(tf.less(tf.abs(sharp_d1),0.01*tf.ones_like(sharp_d1)),tf.float32)*v1_
    v2_ = tf.cast(tf.less(tf.abs(sharp_d2),0.01*tf.ones_like(sharp_d2)),tf.float32)*v2_
    # v1_ = tf.nn.erosion2d(v1_,zero_kernel,[1,1,1,1],[1,1,1,1], "SAME")
    # v2_ = tf.nn.erosion2d(v2_,zero_kernel,[1,1,1,1],[1,1,1,1], "SAME")
    v1_ = tf.squeeze(v1_,0)
    v2_ = tf.squeeze(v2_,0)

    # image tensors need to be cropped. we'll do them all at once.
    allCat = tf.concat(axis=2,values=[i1,i2,d1,d2,v1_,v2_,m1,m2],name="allCat")
    allCat_crop, off_h, off_w = topleft_crop(allCat,h,w,height,width)
    print_shape(allCat_crop)
    i1 = tf.slice(allCat_crop, [0,0,0], [-1,-1,3], name="i1")
    i2 = tf.slice(allCat_crop, [0,0,3], [-1,-1,3], name="i2")
    d1 = tf.slice(allCat_crop, [0,0,6], [-1,-1,1], name="d1")
    d2 = tf.slice(allCat_crop, [0,0,7], [-1,-1,1], name="d2")
    v1 = tf.slice(allCat_crop, [0,0,8], [-1,-1,1], name="v1")
    v2 = tf.slice(allCat_crop, [0,0,9], [-1,-1,1], name="v2")
    m1 = tf.slice(allCat_crop, [0,0,10], [-1,-1,1], name="m1")
    m2 = tf.slice(allCat_crop, [0,0,11], [-1,-1,1], name="m2")

    # grab a batch
    if shuffle:
        # this will shuffle BEYOND JUST THE BATCH!
        batch = tf.train.shuffle_batch([i1,i2,
                                        d1,d2,
                                        v1,v2,
                                        m1,m2,
                                        off_h,off_w],
                                       batch_size=bs,
                                       min_after_dequeue=100,
                                       capacity=100+3*bs,
                                       num_threads=2)
    else:
        batch = tf.train.batch([i1,i2,
                                d1,d2,
                                v1,v2,
                                m1,m2,
                                off_h,off_w],
                               batch_size=bs)
    return batch


def ycb_batch(dataset,bs,crop_h,crop_w,shuffle=True):
    with open(dataset) as f:
        content = f.readlines()
    records = [hyp.dataset_location + '/' + line.strip() for line in content]        
    nRecords = len(records)
    print 'found %d records' % nRecords
    for record in records:
        assert os.path.isfile(record), 'Record at %s was not found' % record

    queue = tf.train.string_input_producer(records, shuffle=shuffle)

    h,w,i1,i2,d1,d2,v1,v2,m1,m2,p1,p2,k1,k2 = read_and_decode_ycb(queue)

    i1 = tf.cast(i1,tf.float32) * 1./255 - 0.5
    i2 = tf.cast(i2,tf.float32) * 1./255 - 0.5
    d1 = tf.cast(d1,tf.float32)
    d2 = tf.cast(d2,tf.float32)
    v1 = 1-tf.cast(v1,tf.float32)
    v2 = 1-tf.cast(v2,tf.float32)
    m1 = 1-(tf.cast(m1,tf.float32) * 1./255)
    m2 = 1-(tf.cast(m2,tf.float32) * 1./255)

    # image tensors need to be cropped. we'll do them all at once.
    allCat = tf.concat(axis=2,values=[i1,i2,
                          d1,d2,
                          v1,v2,
                          m1,m2],
                       name="allCat")
    print_shape(allCat)
    allCat_crop, off_h, off_w = random_crop(allCat,crop_h,crop_w,h,w)
    print_shape(allCat_crop)
    i1 = tf.slice(allCat_crop, [0,0,0], [-1,-1,3], name="i1")
    i2 = tf.slice(allCat_crop, [0,0,3], [-1,-1,3], name="i2")
    d1 = tf.slice(allCat_crop, [0,0,6], [-1,-1,1], name="d1")
    d2 = tf.slice(allCat_crop, [0,0,7], [-1,-1,1], name="d2")
    v1 = tf.slice(allCat_crop, [0,0,8], [-1,-1,1], name="v1")
    v2 = tf.slice(allCat_crop, [0,0,9], [-1,-1,1], name="v2")
    m1 = tf.slice(allCat_crop, [0,0,10], [-1,-1,1], name="m1")
    m2 = tf.slice(allCat_crop, [0,0,11], [-1,-1,1], name="m2")

    # grab a batch
    if shuffle:
        # this will shuffle BEYOND JUST THE BATCH!
        batch = tf.train.shuffle_batch([i1,i2,
                                        d1,d2,
                                        v1,v2,
                                        m1,m2,
                                        p1,p2,
                                        k1,k2,
                                        off_h,off_w],
                                       batch_size=bs,
                                       min_after_dequeue=100,
                                       capacity=100+3*bs,
                                       num_threads=2)
    else:
        batch = tf.train.batch([i1,i2,
                                d1,d2,
                                v1,v2,
                                m1,m2,
                                p1,p2,
                                k1,k2,
                                off_h,off_w],
                               batch_size=bs)
    return batch



def stoy_batch(dataset,bs,h,w,shuffle=True):
    with open(dataset) as f:
        content = f.readlines()
    records = [hyp.dataset_location + '/' + line.strip() for line in content]
    
    nRecords = len(records)
    print 'found %d records' % nRecords
    queue = tf.train.string_input_producer(records, shuffle=shuffle)
    
    height,width,i1,i2,d1,d2,v1,v2 = read_and_decode_stoy(queue)

    # some tensors need to be cast to float
    i1 = tf.cast(i1,tf.float32) * 1./255 - 0.5
    i2 = tf.cast(i2,tf.float32) * 1./255 - 0.5
    d1 = tf.cast(d1,tf.float32)
    d2 = tf.cast(d2,tf.float32)
    v1 = 1-tf.cast(v1,tf.float32) # this has ones at INVALID pixels
    v2 = 1-tf.cast(v2,tf.float32)

    # image tensors need to be cropped. we'll do them all at once.
    allCat = tf.concat(axis=2,values=[i1,i2,d1,d2,v1,v2],name="allCat")
    allCat_crop, off_h, off_w = bottomcenter_crop(allCat,h,w,height,width)

    if hyp.do_horz_flip_aug:
        allCat_crop = tf.image.random_flip_left_right(allCat_crop)
    if hyp.do_vert_flip_aug:
        allCat_crop = tf.image.random_flip_up_down(allCat_crop)

    print_shape(allCat_crop)
    i1 = tf.slice(allCat_crop, [0,0,0], [-1,-1,3], name="i1")
    i2 = tf.slice(allCat_crop, [0,0,3], [-1,-1,3], name="i2")
    d1 = tf.slice(allCat_crop, [0,0,6], [-1,-1,1], name="d1")
    d2 = tf.slice(allCat_crop, [0,0,7], [-1,-1,1], name="d2")
    v1 = tf.slice(allCat_crop, [0,0,8], [-1,-1,1], name="v1")
    v2 = tf.slice(allCat_crop, [0,0,9], [-1,-1,1], name="v2")

    if hyp.do_photo_aug:
        # the aug scripts expect the images to be in [0,1]
        i1_ = i1 + 0.5
        i2_ = i2 + 0.5
        # contrastMin,contrastMax,brightnessStd,colorMin,colorMax,gammaMin,gammaMax,noiseStd
	photoParam1 = photoAugParam(0.7,1.3,0.2,0.8,1.2,0.7,1.5,0.05)
	photoParam2 = photoAugParam(0.7,1.3,0.2,0.8,1.2,0.7,1.5,0.05)
	i1_ = photoAug(i1_,photoParam1)
	i2_ = photoAug(i2_,photoParam2)
        i1_ = tf.clip_by_value(i1_, 0, 1)
        i2_ = tf.clip_by_value(i2_, 0, 1)
        i1 = i1_ - 0.5
        i2 = i2_ - 0.5
        
    # grab a batch
    if shuffle:
        # this will shuffle BEYOND JUST THE BATCH!
        batch = tf.train.shuffle_batch([i1,i2,
                                        d1,d2,
                                        v1,v2,
                                        off_h,off_w],
                                       batch_size=bs,
                                       min_after_dequeue=100,
                                       capacity=100+3*bs,
                                       num_threads=2)
    else:
        batch = tf.train.batch([i1,i2,
                                d1,d2,
                                v1,v2,
                                off_h,off_w],
                               batch_size=bs)
    return batch

