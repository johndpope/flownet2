import cv2
import numpy as np
import tensorflow as tf
from hyperparams import *
def undistort(i1_,i2_):   
	if not new_resol:
		h=height_old
		w=width_old
	else:
		h=height
		w=width 
	i1_=np.array(i1_[0,:,:,0])
	i2_=np.array(i2_[0,:,:,0])
	newcameramtx1, roi1=cv2.getOptimalNewCameraMatrix(scaled_k1,distortion_coefficients_1,(h,w),1,(h,w))
	# i1_=cv2.undistort(i1_,k1,distortion_coefficients_1,None,newcameramtx1)
	i1_=cv2.undistort(i1_,scaled_k1,distortion_coefficients_1)
	newcameramtx2, roi2=cv2.getOptimalNewCameraMatrix(scaled_k2,distortion_coefficients_2,(h,w),1,(h,w))
	# i2_=cv2.undistort(i2_,k2,distortion_coefficients_2,None,newcameramtx2)
	i2_=cv2.undistort(i2_,scaled_k2,distortion_coefficients_2)
	
	# remove unwanted pixels
	# x1_,y1_,w1,h1 = roi1
	# i1_ = i1_[y1_:y1_+h1, x1_:x1_+w1]
	# x2_,y2_,w2,h2 = roi2
	# i2_ = i2_[y2_:y2_+h2, x2_:x2_+w2]
	# print(x2_,y2_,h2,w2)
	# i1_=cv2.resize(i1_, (height_old,width_old),interpolation = cv2.INTER_CUBIC) 
	# i2_=cv2.resize(i1_, (height_old,width_old),interpolation = cv2.INTER_CUBIC) 

	i1_=np.reshape(i1_,[1,h,w,1])
	i2_=np.reshape(i2_,[1,h,w,1])
	i1_=np.tile(i1_,[1,1,1,3])
	i2_=np.tile(i2_,[1,1,1,3])
	return(i1_,i2_)

def depth_estimation( flow ):
	# c1=np.vstack([c1,[[0,0,0,1]]])
	# c2=np.vstack([c2,[[0,0,0,1]]])
	# p2=np.matmul(np.linalg.inv(c2),(c1))
	# p1=np.eye(4)
	a=np.array([np.zeros(width),
            range(0,width),
            np.ones(width)],dtype=np.float32)
	print(np.shape(a))
	for i in range(1,height):
	    tmp=np.array([i*np.ones(width),
	                  range(0,width),
	                  np.ones(width)],dtype=np.float32)
	    a=np.hstack([a,tmp])
	print(np.shape(a))
	b=a[0:2,:]+np.reshape(flow,[2,height*width])
	a[0,:]=a[0,:]/np.max(a[0,:])
	a[1,:]=a[1,:]/np.max(a[1,:])
	b[0,:]=b[0,:]/np.max(b[0,:])
	b[1,:]=b[1,:]/np.max(b[1,:])

	depth_ = cv2.triangulatePoints(c1[:3],c2[:3], a[:2], b[:2])
	depth_=depth_/depth_[3]
	depth_=np.reshape(depth_[2,:],[1,height,width,1])
	depth_=np.linalg.norm(depth_,axis=3)
	depth_=np.reshape(depth_,[1,height,width,1])
	depth_=np.tile(depth_,[1,1,1,3])
	print(np.shape(depth_))
	print(depth_)
	print('avg:',np.mean(depth_)) 
	print('range:',np.min(depth_),np.max(depth_)) 

	return depth_