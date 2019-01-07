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
from GetDataPath import *
from sys import argv
import math

cat=argv[1]
x=[i for i in range(80)]
fname ='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/cat'+str(cat)+'.npz'
ff=np.load(fname)
img_vc=ff['vc_score']
vc_num=len(img_vc[0])
img_num=len(img_vc)
#print(img_vc[0])
img_vc_avg=[]
for i in range(vc_num):
	img_vc_avg.append(float(np.sum(img_vc[np.where(img_vc[:,i]!=-1),i]))/img_num)

img_vc_avg=list(img_vc_avg)
img_vc_avg.sort(reverse=True)

#2vc
fname ='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_combinescore/2vc/cat'+str(cat)+'.npz'
ff=np.load(fname)
dou_img_vc_avg=ff['vc_score']
dou_img_vc_avg=list(dou_img_vc_avg)
dou_img_vc_avg.sort(reverse=True)

fname ='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_combinescore/3vc/cat'+str(cat)+'.npz'
ff=np.load(fname)
trip_img_vc_avg=ff['vc_score']
trip_img_vc_avg=list(trip_img_vc_avg)
trip_img_vc_avg.sort(reverse=True)

fname ='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_combinescore/4vc/cat'+str(cat)+'.npz'
ff=np.load(fname)
quad_img_vc_avg=ff['vc_score']
quad_img_vc_avg=list(quad_img_vc_avg)
quad_img_vc_avg.sort(reverse=True)

plt.figure()
plt.title('importance measurement curve')
plt.xlabel("Ranked Visual Concepts")
plt.ylabel("drop in the probability of the target class")
plt.plot(x,img_vc_avg[:80],'r-',label='singleton VC')
plt.plot(x,dou_img_vc_avg[:80],'b-',label='pair VCs')
plt.plot(x,trip_img_vc_avg[:80],'g-',label='triplet VCs')
plt.plot(x,quad_img_vc_avg[:80],'c',label='quadruplet VC')
savedir='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/ImportanceExample/'+cat+'/combine_curve.png'
plt.savefig(savedir) 