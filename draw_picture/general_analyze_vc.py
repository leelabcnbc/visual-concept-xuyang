from __future__ import division
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import Image, display
import pickle
import numpy as np
from sys import argv
import sys
from GetDataPath import *
from sys import argv
import math

all_img_vc_avg=np.zeros(100)
for cat in range(20):
	cat=str(cat)
	fname ='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/cat'+str(cat)+'.npz'
	ff=np.load(fname)
	img_vc=ff['vc_score']
	vc_num=len(img_vc[0])
	img_num=len(img_vc)
	#print(img_vc[0])
	img_vc_avg=[]
	for i in range(vc_num):
		img_vc_avg.append(float(np.sum(img_vc[np.where(img_vc[:,i]!=-1),i]))/img_num)
	img_vc_avg=np.asarray(img_vc_avg)
	print('aa')
	print(img_vc_avg)
	myindex=np.argsort(-img_vc_avg)
	for i in range(100):
		bb=int(i*float(vc_num-1)/99)
		theindex=myindex[bb]
		aa=img_vc_avg[theindex]
		all_img_vc_avg[i]+=aa

all_img_vc_avg=all_img_vc_avg/20
	# show curve
x=[i for i in range(100)]
plt.figure()
plt.title('20 class importance measurement curve')
plt.xlabel("Ranked Visual Concepts")
plt.ylabel("drop in the probability of the target class")
plt.plot(x,all_img_vc_avg,'r-')
savedir='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/ImportanceExample/GeneralImportanceCurve.png'
plt.savefig(savedir) 