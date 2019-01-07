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
import sys
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
mylayer=argv[2]

myimage_path=LoadImage(cat)
# myimage_path=[]
# myimage_path+=[os.path.join('/data2/xuyangf/OcclusionProject/NaiveVersion/PortraitImages/train',s) 
#         for s in os.listdir('/data2/xuyangf/OcclusionProject/NaiveVersion/PortraitImages/train')]
# myimage_path=list(myimage_path)

image_path=[]
for mypath in myimage_path:
    myimg=cv2.imread(mypath, cv2.IMREAD_UNCHANGED)
    if(max(myimg.shape[0],myimg.shape[1])>100):
        image_path.append(mypath)
img_num=len(image_path)
file_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/feature/feature'+mylayer+'/L'+mylayer+'Feature'+cat
prun_file = '/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/prunL'+mylayer+'/dictionary_'+cat+'.pickle'
save_file='/data2/xuyangf/OcclusionProject/NaiveVersion/new_vc_score/layer'+mylayer+'/cat'+cat
if not os.path.exists('/data2/xuyangf/OcclusionProject/NaiveVersion/new_vc_score/layer'+mylayer+'/'):
        os.mkdir('/data2/xuyangf/OcclusionProject/NaiveVersion/new_vc_score/layer'+mylayer+'/')

# file_path='/data2/xuyangf/OcclusionProject/NaiveVersion/feature/Portrait/special_test_'
# prun_file='/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/Portrait/special_test.pickle'
# save_file='/data2/xuyangf/OcclusionProject/NaiveVersion/new_vc_score/Portrait/portrait_score'

print('loading data...')

fname = file_path+str(0)+'.npz'
ff = np.load(fname)

feat_dim = ff['res'].shape[0]
img_cnt = ff['res'].shape[1]
oldimg_index=0

# number of files to read in
file_num = 10
maximg_cnt=img_cnt*3

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
loc_set=loc_set[:oldimg_index,:]

#l2
myfeat_norm = np.sqrt(np.sum(feat_set**2, 0))
feat_set = feat_set/myfeat_norm

with open(prun_file, 'rb') as fh:
    assignment, centers, _,norm = pickle.load(fh)

print('load finish')

predictor=TestVgg(int(mylayer[-1]),'fine_tuned')
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


img_vc=np.zeros((img_num,len(centers)))
patch_size = int(loc_set[0,5] - loc_set[0,3])

for k in range(0,len(centers)):
	target=centers[k]
	index=np.where(assignment==k)[0]
	print('before')
	print(len(index))
	#thresh the vc,only choose top 80% 
	index=disttresh(index,target)
	print('after')
	print(len(index))
	print(index)
	for n in range(0,img_num):
		myindex=[]
		for i in range(len(index)):
			if image_path[n]==originimage[index[i]]:
				myindex.append(index[i])
		#myindex=OnlyTheClosest(myindex,target), or other preprocessing method
		print(len(myindex))
		print('rua')
		if len(myindex)==0:
			img_vc[n][k]=-1
		else:
			original_img=cv2.imread(image_path[n], cv2.IMREAD_UNCHANGED)
			original_feature=predictor.img2feature(original_img)
			thesum=0
			for i in range(len(myindex)):
				fhi=int(loc_set[myindex[i]][7])
				fwi=int(loc_set[myindex[i]][8])
				occluded_feature=deepcopy(original_feature)
				# Gause occlusion
				hiddenlayer_shape=occluded_feature[0].shape
				mymask=generate_gaussian_mask(fhi,fwi,hiddenlayer_shape)
				occluded_feature[0]=occluded_feature[0]*mymask
				# occluded_feature[0][fhi-1][fwi]=0.25*occluded_feature[0][fhi-1][fwi]
				# occluded_feature[0][fhi][fwi-1]=0.25*occluded_feature[0][fhi][fwi-1]
				# occluded_feature[0][fhi+1][fwi]=0.25*occluded_feature[0][fhi+1][fwi]
				# occluded_feature[0][fhi][fwi+1]=0.25*occluded_feature[0][fhi][fwi+1]

				# occluded_feature[0][fhi-1][fwi-1]=0.375*occluded_feature[0][fhi-1][fwi-1]
				# occluded_feature[0][fhi+1][fwi-1]=0.375*occluded_feature[0][fhi+1][fwi-1]
				# occluded_feature[0][fhi+1][fwi+1]=0.375*occluded_feature[0][fhi+1][fwi+1]
				# occluded_feature[0][fhi-1][fwi+1]=0.375*occluded_feature[0][fhi-1][fwi+1]
				
				# occluded_feature[0][fhi+2][fwi]=0.625*occluded_feature[0][fhi+2][fwi]
				# occluded_feature[0][fhi-2][fwi]=0.625*occluded_feature[0][fhi-2][fwi]
				# occluded_feature[0][fhi][fwi-2]=0.625*occluded_feature[0][fhi][fwi-2]
				# occluded_feature[0][fhi][fwi+2]=0.625*occluded_feature[0][fhi][fwi+2]

				# occluded_feature[0][fhi-1][fwi-2]=0.75*occluded_feature[0][fhi-1][fwi-2]
				# occluded_feature[0][fhi-1][fwi+2]=0.75*occluded_feature[0][fhi-1][fwi+2]
				# occluded_feature[0][fhi+1][fwi-2]=0.75*occluded_feature[0][fhi+1][fwi-2]
				# occluded_feature[0][fhi+1][fwi+2]=0.75*occluded_feature[0][fhi+1][fwi+2]
				# occluded_feature[0][fhi-2][fwi-1]=0.75*occluded_feature[0][fhi-2][fwi-1]
				# occluded_feature[0][fhi-2][fwi+1]=0.75*occluded_feature[0][fhi-2][fwi+1]
				# occluded_feature[0][fhi+2][fwi-1]=0.75*occluded_feature[0][fhi+2][fwi-1]
				# occluded_feature[0][fhi+2][fwi+1]=0.75*occluded_feature[0][fhi+2][fwi+1]

				# occluded_feature[0][fhi-2][fwi-2]=0.875*occluded_feature[0][fhi-2][fwi-2]
				# occluded_feature[0][fhi-2][fwi+2]=0.875*occluded_feature[0][fhi-2][fwi+2]
				# occluded_feature[0][fhi+2][fwi-2]=0.875*occluded_feature[0][fhi+2][fwi-2]
				# occluded_feature[0][fhi+2][fwi+2]=0.875*occluded_feature[0][fhi+2][fwi+2]				

				thesum+=GetPossDecrease(original_feature,occluded_feature,int(cat))
			img_vc[n][k]=thesum/len(myindex)
			# if(len(myindex)==14):
			# 	print('yes')
			# 	print(k)
			# 	print(n)
			# 	print(img_vc[n][k])
		print('one finish')
	# print(img_vc.shape)
	# print(img_vc[294:296,0])
	# print(img_vc[283][0])
	# break
np.savez(save_file,vc_score=img_vc)
