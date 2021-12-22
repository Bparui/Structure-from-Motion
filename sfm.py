import os
import glob
import numpy as np 
import cv2 as cv
from matplotlib import pyplot as plt
import math
import numpy.linalg as alg
from mpl_toolkits.mplot3d import Axes3D


# Declaring parameters
feature_params = dict( maxCorners = 4000,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
lk_params = dict(winSize=(21,21),maxLevel = 2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
pnp_params = dict(distCoeffs =np.zeros((5, 1)),iterationsCount = 100,reprojectionError = 8.0,confidence = 0.99,flags = 1)

# Triangulation 
def triangulate(R1,t1,R2,t2,kp1,kp2,k):
    '''kp1  = kp1.astype(float)
    kp2  = kp2.astype(float)'''
    P1 = np.dot(k,np.hstack((R1,t1)))
    P2 = np.dot(k,np.hstack((R2,t2)))
    kp_1 = kp1.T
    kp_2 = kp2.T
    cloud = cv.triangulatePoints(P1, P2, kp_1,kp_2)
    cloud= cloud /(cloud[3,:]+1e-8)
    return cloud.T[:,:3]

def tracked_keypoint(img1,img2,kp1):
    pts2,st,_ = cv.calcOpticalFlowPyrLK(img1, img2,np.float32(kp1), None, **lk_params)
    #pts2 = pts2[st.ravel()==1]
    #pts1 = kp1[st.ravel()==1]
    return pts2,kp1,st

def relpose(kp1,kp2,k,R_old,t_old):
    F, mask = cv.findFundamentalMat(kp1,kp2,cv.FM_RANSAC,0.6,0.9)
    kp1 = kp1[mask.ravel()==1]
    kp2 = kp2[mask.ravel()==1]
    E = (k.T)@F@k
    points, R_est, t_est, mask_pose = cv.recoverPose(E, kp1,kp2,k)
    t_new = -(np.dot(R_old,t_est)).T
    R_new =R_old@R_est
    return t_new.T,R_new,kp1,kp2

# Loading dataset 
