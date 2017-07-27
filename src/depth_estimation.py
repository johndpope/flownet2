import cv2
import numpy as np
def depth_estimation( c1,c2, x1, x2 ):
	X = cv2.triangulatePoints(c1[:3],c2[:3], x1[:2], x2[:2])
	return X/X[3]