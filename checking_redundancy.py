from sklearn import manifold
import numpy as np
import time
import math
import cv2
import sys
import os
from sys import argv
import pickle
cat=argv[1]

feature_path='/data2/xuyangf/OcclusionProject/NaiveVersion/ApertureImage/train/bubble_image/center_feature/'+cat+'/'


thresh1 = 0.8  # magic number
thresh2 = 0.6
pool3_merge=[i for i in range(20)]
pool4_merge=[i for i in range(20)]

for k in range(0,20):
	ff = np.load(feature_path+str(k)+'.npz')
	pool3_res=np.array(ff['pool3_res'])
	pool4_res=np.array(ff['pool4_res'])
	pool3_center=np.average(pool3_res,0)
	pool4_center=np.average(pool4_res,0)
	pool3_dist = np.sqrt(np.sum((pool3_res - pool3_center)**2, axis=1))
	sort_value = np.sort(pool3_dist)
	pool3_dist_thresh = sort_value[int(thresh1*len(sort_value))]
	pool4_dist = np.sqrt(np.sum((pool4_res - pool4_center)**2, axis=1))
	sort_value = np.sort(pool4_dist)
	pool4_dist_thresh = sort_value[int(thresh1*len(sort_value))]

	for l in range(k+1,20):
		ff = np.load(feature_path+str(l)+'.npz')
		pool3_res=np.array(ff['pool3_res'])
		pool4_res=np.array(ff['pool4_res'])
		pool3_dist = np.sqrt(np.sum((pool3_res - pool3_center)**2, axis=1))
		pool4_dist = np.sqrt(np.sum((pool4_res - pool4_center)**2, axis=1))
		if np.mean(pool3_dist<pool3_dist_thresh) >= thresh2:
			pool3_merge[k]=min(pool3_merge[k],pool3_merge[l])
			pool3_merge[l]=min(pool3_merge[k],pool3_merge[l])
		if np.mean(pool4_dist<pool4_dist_thresh) >= thresh2:
			pool4_merge[k]=min(pool4_merge[k],pool4_merge[l])
			pool4_merge[l]=min(pool4_merge[k],pool4_merge[l])
print(pool3_merge)
print(pool4_merge)
pool3_merge=np.unique(pool3_merge)
pool4_merge=np.unique(pool4_merge)
######### drawing examples
fname='/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/prunL3/dictionary_'+cat+'.pickle'
with open(fname,'rb') as fh:
    assignment, centers, example, __= pickle.load(fh)

ss=int(math.sqrt(example[0].shape[0]/3))

fname ='/data2/xuyangf/OcclusionProject/NaiveVersion/new_vc_score/layer3/cat'+str(cat)+'.npz'
ff=np.load(fname)
img_vc=ff['vc_score']
vc_num=len(img_vc[0])
img_num=len(img_vc)
img_vc_avg=[]
for i in range(vc_num):
    img_vc_avg.append(float(np.sum(img_vc[np.where(img_vc[:,i]!=-1),i]))/img_num)
img_vc_avg=np.asarray(img_vc_avg)
rindexsort=np.argsort(-img_vc_avg)

big_img = np.zeros((5+ss, 5+(ss+5)*len(pool4_merge), 3))


for j in range(len(pool4_merge)):

	rnum=5
	cnum=5+j*(ss+5)
	select_vc=rindexsort[pool4_merge[j]]
	big_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[select_vc][:,0].reshape(ss,ss,3).astype(int)

fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/example/tmpexample/'+str(cat)+'.png'
cv2.imwrite(fname, big_img)