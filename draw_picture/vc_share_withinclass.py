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
fname ='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/cat'+str(cat)+'.npz'
ff=np.load(fname)
img_vc=ff['vc_score']
vc_num=len(img_vc[0])
img_num=len(img_vc)
#print(img_vc[0])

img_vc_avg=[]
for i in range(vc_num):
	img_vc_avg.append(np.average(img_vc[np.where(img_vc[:,i]!=-1),i]))
img_vc_avg=np.asarray(img_vc_avg)
indexsort=np.argsort(img_vc_avg)
for i in range(vc_num):
	img_vc_avg[indexsort[i]]=int(100*(float(i)/len(img_vc_avg)))
print(img_vc_avg)
share_times=[]
for i in range(vc_num):
	share_times.append(len(np.where(img_vc[:,i]!=-1)[0]))

print(share_times)

plt.title("the relationship between shared times within class and importance") 
plt.xlabel('importance')
plt.ylabel('Number of images that have the VC ')
plt.ylim(ymax=max(share_times),ymin=1)
plt.xlim(xmax=100,xmin=1)
plt.plot(img_vc_avg,share_times,'ro')
savedir='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/ImportanceExample/'+cat+'/Importance_share_withinclass_Curve.png'
plt.savefig(savedir) 

