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
cluster_num = 256
myimage_path=LoadImage(cat)
image_path=[]
for mypath in myimage_path:
    myimg=cv2.imread(mypath, cv2.IMREAD_UNCHANGED)
    if(max(myimg.shape[0],myimg.shape[1])>100):
        image_path.append(mypath)
img_num=len(image_path)
layer_name = 'pool3'
file_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/feature/feature3/L3Feature'+cat
#cluster_file = '/data2/xuyangf/OcclusionProject/NaiveVersion/cluster/clusterL3/vgg16_'+cat+'_K'+str(cluster_num)+'.pickle'
prun_file = '/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/prunL3/dictionary_'+cat+'.pickle'
trip_save_file='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_combinescore/3vc/cat'+cat
quad_save_file='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_combinescore/4vc/cat'+cat

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
print(loc_dim)
loc_set = np.zeros((maximg_cnt*file_num, loc_dim))
loc_set[0:img_cnt,:] = ff['loc_set']

#img_dim = ff['img_set'].shape[1:]
#img_set = np.zeros([maximg_cnt*file_num]+list(img_dim))
#img_set[0:img_cnt] = ff['img_set']

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
    #img_set[oldimg_index:(oldimg_index + img_cnt)] = ff['img_set']
    oldimg_index+=img_cnt

feat_set=feat_set[:,:oldimg_index]
#img_set=img_set[:oldimg_index]
loc_set=loc_set[:oldimg_index,:]

print('all feat_set')
print(feat_set.shape)
print('all img_set')
#print(img_set.shape)
#assert(len(originimage)==len(img_set))

with open(prun_file, 'rb') as fh:
    assignment, centers, _, norm = pickle.load(fh)

print('load finish')

predictor=TestVgg()


# def GetPossDecrease(original_img,occluded_img,category):
# 	originalp=predictor.getprob(original_img,category)
# 	occludedp=predictor.getprob(occluded_img,category)
# 	print('diff')
# 	print(originalp)
# 	print(occludedp)
# 	#0.3 magic number
# 	if(originalp<0.3):
# 		return -1
# 	if(originalp<occludedp):
# 		return -1
# 	return float(originalp-occludedp)/originalp

def GetPossDecrease(original_feature,occluded_feature,category):
	originalp=predictor.feature2result(original_feature,category)
	occludedp=predictor.feature2result(occluded_feature,category)
	print('diff')
	print(originalp)
	print(occludedp)
	#0.3 magic number
	if(originalp<0.3):
		return -1
	# if(originalp<occludedp):
	# 	return -1
	return float(originalp-occludedp)/originalp

def disttresh(input_index,cluster_center):
	thresh1=0.8
	temp_feat=feat_set[:,input_index]
	error = np.sum((temp_feat.T - cluster_center)**2, 1)
	sort_idx = np.argsort(error)
	return input_index[sort_idx[:int(thresh1*len(sort_idx))]]


fname ='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_combinescore/2vc/cat'+str(cat)+'.npz'
ff=np.load(fname)
img_vc_avg=ff['vc_score']
vc_name=ff['vc']
vc_num=len(img_vc_avg)
img_vc_avg=np.array(img_vc_avg)
vc_name=np.array(vc_name)
b=np.argsort(-img_vc_avg)

trip_img_vc_score=[]
trip_img_vc=[]
quad_img_vc_score=[]
quad_img_vc=[]

topvc=b[:25]
for indexk in topvc:
	thename=vc_name[indexk]
	k1=int(thename[0])
	k2=int(thename[1])
	for indexkk in topvc:
		if indexk<=indexkk:
			continue
		thename2=vc_name[indexkk]
		k3=int(thename2[0])
		k4=int(thename2[1])
		myk=[]
		myk.append(k1)
		myk.append(k2)
		myk.append(k3)
		myk.append(k4)
		myk=set(myk)
		myk=list(myk)
		if len(myk)==3:
			trip_img_vc.append(str(myk[0])+'_'+str(myk[1])+'_'+str(myk[2]))
		else:
			quad_img_vc.append(str(myk[0])+'_'+str(myk[1])+'_'+str(myk[2])+'_'+str(myk[3]))
trip_img_vc=list(set(trip_img_vc))
quad_img_vc=list(set(quad_img_vc))


for s in trip_img_vc:
	dot1=s.find('_')
	k1=int(s[:dot1])
	s2=s[dot1+1:]
	dot2=s2.find('_')
	k2=int(s2[:dot2])
	s3=s2[dot2+1:]
	k3=int(s3)

	target1=centers[k1]
	index1=np.where(assignment==k1)[0]
	index1=disttresh(index1,target1)

	target2=centers[k2]
	index2=np.where(assignment==k2)[0]
	index2=disttresh(index2,target2)

	target3=centers[k3]
	index3=np.where(assignment==k3)[0]
	index3=disttresh(index3,target3)
	myscore=[]
	for n in range(0,img_num):
		myindex1=[]
		for i in range(len(index1)):
			if image_path[n]==originimage[index1[i]]:
				myindex1.append(index1[i])
		#myindex=OnlyTheClosest(myindex,target), or other preprocessing method
		myindex2=[]
		for i in range(len(index2)):
			if image_path[n]==originimage[index2[i]]:
				myindex2.append(index2[i])
		myindex3=[]
		for i in range(len(index3)):
			if image_path[n]==originimage[index3[i]]:
				myindex3.append(index3[i])

		if len(myindex1)==0:
			continue
		if len(myindex2)==0:
			continue
		if len(myindex3)==0:
			continue

		myindex=myindex1+myindex2+myindex3
		original_img=cv2.imread(image_path[n], cv2.IMREAD_UNCHANGED)
		original_feature=predictor.img2feature(original_img)
		occluded_feature=deepcopy(original_feature)
		for i in range(len(myindex)):
			fhi=int(loc_set[myindex[i]][7])
			fwi=int(loc_set[myindex[i]][8])
			
			# Gause occlusion
			occluded_feature[0][fhi][fwi]=0
			occluded_feature[0][fhi-1][fwi]=0.25*occluded_feature[0][fhi-1][fwi]
			occluded_feature[0][fhi][fwi-1]=0.25*occluded_feature[0][fhi][fwi-1]
			occluded_feature[0][fhi+1][fwi]=0.25*occluded_feature[0][fhi+1][fwi]
			occluded_feature[0][fhi][fwi+1]=0.25*occluded_feature[0][fhi][fwi+1]

			occluded_feature[0][fhi-1][fwi-1]=0.375*occluded_feature[0][fhi-1][fwi-1]
			occluded_feature[0][fhi+1][fwi-1]=0.375*occluded_feature[0][fhi+1][fwi-1]
			occluded_feature[0][fhi+1][fwi+1]=0.375*occluded_feature[0][fhi+1][fwi+1]
			occluded_feature[0][fhi-1][fwi+1]=0.375*occluded_feature[0][fhi-1][fwi+1]
			
			occluded_feature[0][fhi+2][fwi]=0.625*occluded_feature[0][fhi+2][fwi]
			occluded_feature[0][fhi-2][fwi]=0.625*occluded_feature[0][fhi-2][fwi]
			occluded_feature[0][fhi][fwi-2]=0.625*occluded_feature[0][fhi][fwi-2]
			occluded_feature[0][fhi][fwi+2]=0.625*occluded_feature[0][fhi][fwi+2]

			occluded_feature[0][fhi-1][fwi-2]=0.75*occluded_feature[0][fhi-1][fwi-2]
			occluded_feature[0][fhi-1][fwi+2]=0.75*occluded_feature[0][fhi-1][fwi+2]
			occluded_feature[0][fhi+1][fwi-2]=0.75*occluded_feature[0][fhi+1][fwi-2]
			occluded_feature[0][fhi+1][fwi+2]=0.75*occluded_feature[0][fhi+1][fwi+2]
			occluded_feature[0][fhi-2][fwi-1]=0.75*occluded_feature[0][fhi-2][fwi-1]
			occluded_feature[0][fhi-2][fwi+1]=0.75*occluded_feature[0][fhi-2][fwi+1]
			occluded_feature[0][fhi+2][fwi-1]=0.75*occluded_feature[0][fhi+2][fwi-1]
			occluded_feature[0][fhi+2][fwi+1]=0.75*occluded_feature[0][fhi+2][fwi+1]

			occluded_feature[0][fhi-2][fwi-2]=0.875*occluded_feature[0][fhi-21][fwi-2]
			occluded_feature[0][fhi-2][fwi+2]=0.875*occluded_feature[0][fhi-2][fwi+2]
			occluded_feature[0][fhi+2][fwi-2]=0.875*occluded_feature[0][fhi+2][fwi-2]
			occluded_feature[0][fhi+2][fwi+2]=0.875*occluded_feature[0][fhi+2][fwi+2]				

			# print(hi)
			# print(wi)
			# print(patch_size)
		drop=GetPossDecrease(original_feature,occluded_feature,int(cat))
		if drop!=-1:
			myscore.append(drop)
	trip_img_vc_score.append(float(np.sum(myscore))/img_num)

np.savez(trip_save_file,vc_score=trip_img_vc_score,vc=trip_img_vc)

for s in quad_img_vc:
	dot1=s.find('_')
	k1=int(s[:dot1])
	s2=s[dot1+1:]
	dot2=s2.find('_')
	k2=int(s2[:dot2])
	s3=s2[dot2+1:]
	dot3=s3.find('_')
	k3=int(s3[:dot3])
	s4=s3[dot3+1:]
	k4=int(s4)

	target1=centers[k1]
	index1=np.where(assignment==k1)[0]
	index1=disttresh(index1,target1)

	target2=centers[k2]
	index2=np.where(assignment==k2)[0]
	index2=disttresh(index2,target2)

	target3=centers[k3]
	index3=np.where(assignment==k3)[0]
	index3=disttresh(index3,target3)

	target4=centers[k4]
	index4=np.where(assignment==k4)[0]
	index4=disttresh(index4,target4)
	myscore=[]
	for n in range(0,img_num):
		myindex1=[]
		for i in range(len(index1)):
			if image_path[n]==originimage[index1[i]]:
				myindex1.append(index1[i])
		#myindex=OnlyTheClosest(myindex,target), or other preprocessing method
		myindex2=[]
		for i in range(len(index2)):
			if image_path[n]==originimage[index2[i]]:
				myindex2.append(index2[i])
		myindex3=[]
		for i in range(len(index3)):
			if image_path[n]==originimage[index3[i]]:
				myindex3.append(index3[i])
		myindex4=[]
		for i in range(len(index4)):
			if image_path[n]==originimage[index4[i]]:
				myindex4.append(index4[i])

		if len(myindex1)==0:
			continue
		if len(myindex2)==0:
			continue
		if len(myindex3)==0:
			continue
		if len(myindex4)==0:
			continue
		myindex=myindex1+myindex2+myindex3+myindex4
		original_img=cv2.imread(image_path[n], cv2.IMREAD_UNCHANGED)
		original_feature=predictor.img2feature(original_img)
		occluded_feature=deepcopy(original_feature)
		for i in range(len(myindex)):
			fhi=int(loc_set[myindex[i]][7])
			fwi=int(loc_set[myindex[i]][8])
			
			# Gause occlusion
			occluded_feature[0][fhi][fwi]=0
			occluded_feature[0][fhi-1][fwi]=0.25*occluded_feature[0][fhi-1][fwi]
			occluded_feature[0][fhi][fwi-1]=0.25*occluded_feature[0][fhi][fwi-1]
			occluded_feature[0][fhi+1][fwi]=0.25*occluded_feature[0][fhi+1][fwi]
			occluded_feature[0][fhi][fwi+1]=0.25*occluded_feature[0][fhi][fwi+1]

			occluded_feature[0][fhi-1][fwi-1]=0.375*occluded_feature[0][fhi-1][fwi-1]
			occluded_feature[0][fhi+1][fwi-1]=0.375*occluded_feature[0][fhi+1][fwi-1]
			occluded_feature[0][fhi+1][fwi+1]=0.375*occluded_feature[0][fhi+1][fwi+1]
			occluded_feature[0][fhi-1][fwi+1]=0.375*occluded_feature[0][fhi-1][fwi+1]
			
			occluded_feature[0][fhi+2][fwi]=0.625*occluded_feature[0][fhi+2][fwi]
			occluded_feature[0][fhi-2][fwi]=0.625*occluded_feature[0][fhi-2][fwi]
			occluded_feature[0][fhi][fwi-2]=0.625*occluded_feature[0][fhi][fwi-2]
			occluded_feature[0][fhi][fwi+2]=0.625*occluded_feature[0][fhi][fwi+2]

			occluded_feature[0][fhi-1][fwi-2]=0.75*occluded_feature[0][fhi-1][fwi-2]
			occluded_feature[0][fhi-1][fwi+2]=0.75*occluded_feature[0][fhi-1][fwi+2]
			occluded_feature[0][fhi+1][fwi-2]=0.75*occluded_feature[0][fhi+1][fwi-2]
			occluded_feature[0][fhi+1][fwi+2]=0.75*occluded_feature[0][fhi+1][fwi+2]
			occluded_feature[0][fhi-2][fwi-1]=0.75*occluded_feature[0][fhi-2][fwi-1]
			occluded_feature[0][fhi-2][fwi+1]=0.75*occluded_feature[0][fhi-2][fwi+1]
			occluded_feature[0][fhi+2][fwi-1]=0.75*occluded_feature[0][fhi+2][fwi-1]
			occluded_feature[0][fhi+2][fwi+1]=0.75*occluded_feature[0][fhi+2][fwi+1]

			occluded_feature[0][fhi-2][fwi-2]=0.875*occluded_feature[0][fhi-21][fwi-2]
			occluded_feature[0][fhi-2][fwi+2]=0.875*occluded_feature[0][fhi-2][fwi+2]
			occluded_feature[0][fhi+2][fwi-2]=0.875*occluded_feature[0][fhi+2][fwi-2]
			occluded_feature[0][fhi+2][fwi+2]=0.875*occluded_feature[0][fhi+2][fwi+2]				

			# print(hi)
			# print(wi)
			# print(patch_size)
		drop=GetPossDecrease(original_feature,occluded_feature,int(cat))
		if drop!=-1:
			myscore.append(drop)
	quad_img_vc_score.append(float(np.sum(myscore))/img_num)

np.savez(quad_save_file,vc_score=quad_img_vc_score,vc=quad_img_vc)