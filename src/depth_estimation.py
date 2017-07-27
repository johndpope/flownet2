import cv2
import numpy as np
def depth_estimation( c1,c2, x1, x2 ):
	# c1=np.vstack([c1,[[0,0,0,1]]])
	# c2=np.vstack([c2,[[0,0,0,1]]])
	# p2=np.matmul(np.linalg.inv(c2),(c1))
	# p1=np.eye(4)
	X = cv2.triangulatePoints(c1[:3],c2[:3], x1[:2], x2[:2])
	return X/X[3]