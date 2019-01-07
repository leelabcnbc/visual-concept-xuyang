from __future__ import division
import pickle
import numpy as np
import cv2
import math
from sys import argv
from copy import *
from ProjectUtils import *
from GetDataPath import *
from testvgg import TestVgg
from feature_extractor import FeatureExtractor

#predictor=FeatureExtractor()
predictor=TestVgg()
img_path=LoadImage(0)

#print(len(predictor.extract_from_paths(original_path)))
num=0
for i in range(len(img_path)):
	img=cv2.imread(img_path[i], cv2.IMREAD_UNCHANGED)
	print(predictor.getprob(img,1))
	print('second')
	feature=predictor.img2feature(img)
	feature[0][0][1]=0
	result=predictor.feature2result(feature,0)
	print(result)
	#print(predictor.getprob(img,0))
	#print(predictor.getprob(img,20))
	break
print num