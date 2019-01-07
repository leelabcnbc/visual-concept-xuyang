from __future__ import division
import pickle
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import Image, display
import math
from sys import argv
from copy import *
from ProjectUtils import *
from GetDataPath import *
from testvgg import TestVgg

sys.path.insert(0, './')
sys.path.append('/home/xuyangf/project/ML_deliverables/Siamese_iclr17_tf-master/src/')

import network as vgg
from feature_extractor import FeatureExtractor
from utils import *
# import importlib
# importlib.reload(sys)
reload(sys)  
sys.setdefaultencoding('utf8')

cat=argv[1]
# tight_trick=argv[2]
# count_trick=argv[3]
#loose_tresh=float(argv[4])/10
cluster_num = 256
layer_name = 'pool3'
file_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/feature/feature3/L3Feature'+cat

print('loading data...')

# number of files to read in
file_num = 10
maximg_cnt=20000

fname = file_path+str(0)+'.npz'
ff = np.load(fname)

feat_dim = ff['res'].shape[0]
img_cnt = ff['res'].shape[1]
oldimg_index=0

originimage=[]
feat_set = np.zeros((feat_dim, maximg_cnt*file_num))
feat_set[:,0:img_cnt] = ff['res']

originimage+=list(ff['originpath'])
loc_dim = ff['loc_set'].shape[1]
loc_set = np.zeros((maximg_cnt*file_num, loc_dim))
loc_set[0:img_cnt,:] = ff['loc_set']

img_dim = ff['img_set'].shape[1:]
img_set = np.zeros([maximg_cnt*file_num]+list(img_dim))
img_set[0:img_cnt] = ff['img_set']

oldimg_index+=img_cnt

for ii in range(1,file_num):
    print(ii)
    fname = file_path+str(ii)+'.npz'
    ff = np.load(fname)
    originimage+=list(ff['originpath'])
    img_cnt=ff['res'].shape[1]
    print(img_cnt)
    feat_set[:,oldimg_index:(oldimg_index + img_cnt)] = ff['res']
    loc_set[oldimg_index:(oldimg_index + img_cnt),:] = ff['loc_set']
    img_set[oldimg_index:(oldimg_index + img_cnt)] = ff['img_set']
    oldimg_index+=img_cnt

feat_set=feat_set[:,:oldimg_index]
img_set=img_set[:oldimg_index]
loc_set=loc_set[:oldimg_index,:]


print 'load finish'

# L2 normalization
myfeat_norm = np.sqrt(np.sum(feat_set**2, 0))
feat_set = feat_set/myfeat_norm


basedir='/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/prunL3/dictionary_'+cat+'.pickle'
Allsharedir=[]
for i in range(0,100):

	sharedir='/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/prunL3/dictionary_'+str(i)+'.pickle'
	Allsharedir.append(sharedir)

with open(basedir,'rb') as fh:
    assignment, centers, example, norm = pickle.load(fh)

vc_num=len(centers)
print(vc_num)
print(len(example))
share_times=np.ones(vc_num)

thresh1=0.1
thresh0=[]
for i in range(vc_num):
	target=centers[i]
	index = np.where(assignment==i)[0]
	temp_feat = feat_set[:, index]
	error =np.sqrt(np.sum((temp_feat.T - target)**2, 1))

	dist=error
	sort_value = np.sort(dist)

	dist_thresh = sort_value[int(thresh1*len(sort_value))]
	thresh0.append(dist_thresh)

#test

for i in range(0,100):
	# if str(i)==str(cat):
	# 	continue
	if i==39:
		continue
	fname=Allsharedir[i]
	with open(fname,'rb') as ffh:
		myassignment, mycenters, myexample,myfeat = pickle.load(ffh)
	myfeat=np.asarray(myfeat)
	myvc_num=len(mycenters)
	print(myvc_num)
	print('finish'+str(i))
	for k in range(vc_num):
		target=centers[k]
		for j in range(myvc_num):
			temp_feat=myfeat[j]
			# temp_feat=mycenters[j]
			# l2_norm = np.sqrt(np.sum(temp_feat**2, 0))
			# temp_feat = temp_feat/l2_norm
			center_dist = np.sqrt(np.sum((temp_feat-target)**2, axis=0))
			if center_dist<thresh0[k]:
				print(str(k)+';'+str(i)+':'+str(j))
				print(center_dist)
				print(thresh0[k])
				share_times[k]+=1
				break

print(share_times)

indexsort=np.argsort(share_times)


fname ='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/cat'+str(cat)+'.npz'
ff=np.load(fname)
img_vc=ff['vc_score']
vc_num=len(img_vc[0])
img_num=len(img_vc)
#print(img_vc[0])
img_vc_avg=[]

for i in range(vc_num):
	img_vc_avg.append(float(np.sum(img_vc[np.where(img_vc[:,i]!=-1),i]))/img_num)
# myindexsort=np.argsort(img_vc_avg)
# for i in range(len(img_vc_avg)):
# 	img_vc_avg[myindexsort[i]]=int(100*(float(i)/len(img_vc_avg)))

print(len(img_vc_avg))
print(len(share_times))
print(max(img_vc_avg))
print(min(img_vc_avg))
print(img_vc_avg)
plt.title("the relationship between occur times in different classes and importance") 
plt.xlabel('importance')
plt.ylabel('Number of classes that have the VC ')
plt.ylim(ymax=100,ymin=1)
plt.xlim(xmax=max(img_vc_avg),xmin=min(img_vc_avg))
plt.plot(img_vc_avg,share_times,'ro')
savedir='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/ImportanceExample/'+cat+'/Importance_share_Curve.png'
plt.savefig(savedir) 


ss=int(math.sqrt(example[0].shape[0]/3))
top_img = np.zeros((10+(ss+10)*2, 10+(ss+10)*5, 3))
last_img = np.zeros((10+(ss+10)*2, 10+(ss+10)*5, 3))

for i in range(0,10):
	aa = i//5
	bb = i%5	
	rnum = 10+aa*(ss+10)
	cnum = 10+bb*(ss+10)
	#top10
	top_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[indexsort[vc_num-1-i]][:,0].reshape(ss,ss,3).astype(int)
	print('top information')
	print(share_times[indexsort[vc_num-1-i]])
	print(indexsort[vc_num-1-i])
	print(len(np.where(assignment==(indexsort[vc_num-1-i]))[0]))
	#last 10
	last_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[indexsort[i]][:,0].reshape(ss,ss,3).astype(int)
	print('last information')
	print(share_times[indexsort[i]])

fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/ImportanceExample/'+cat+'/topshare.png'
cv2.imwrite(fname, top_img)

fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/ImportanceExample/'+cat+'/lastshare.png'
cv2.imwrite(fname, last_img)

print('draw finish')
