import cv2
import numpy as np
def depth_estimation( P2, x1, x2 ):
	P1 = np.eye(4)
	X = cv2.triangulatePoints(P1[:3],P2[:3], x1[:2], x2[:2])
	return X/X[3]