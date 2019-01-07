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
mylayer=argv[2]

featDim_set = [64, 128, 256, 512, 512] 

cluster_num = featDim_set[int(mylayer)-1]
if mylayer=='4':
    cluster_num=256

layer_name = 'pool'+mylayer
save_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/valassignment/dictionary_'+cat+'.pickle'
file_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/feature/valfeature/'+cat

Arf_set = [6, 16, 44, 100, 212]
patch_size=Arf_set[int(mylayer)-1]

fname = file_path+str(0)+'.npz'
ff = np.load(fname)

feat_dim = ff['res'].shape[0]
img_cnt = ff['res'].shape[1]
oldimg_index=0

file_num = 10
if cat=='39':
	file_num=8
if cat=='22':
	file_num=9
if cat=='71':
	file_num=9
maximg_cnt=img_cnt*3

originimage=[]
feat_set = np.zeros((feat_dim, maximg_cnt*file_num))
feat_set[:,0:img_cnt] = ff['res']

originimage+=list(ff['originpath'])
loc_dim = ff['loc_set'].shape[1]
loc_set = np.zeros((maximg_cnt*file_num, loc_dim))
loc_set[0:img_cnt,:] = ff['loc_set']

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


# L2 normalization as preprocessing
feat_norm = np.sqrt(np.sum(feat_set**2, 0))
feat_set = feat_set/feat_norm

basedir='/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/prunL3/dictionary_'+cat+'.pickle'
with open(basedir,'rb') as fh:
    assignment, centers, example, norm = pickle.load(fh)
vc_num=len(centers)
print('vc_num: '+str(vc_num))

patch_num=len(originimage)
assignment_new = np.zeros(patch_num)
print('patch_num: '+str(patch_num))
print('patch_num2: '+str(len(feat_set[1])))

for i in range(0,patch_num):
	closest_distance=1000000
	assigned_cluster=-1
	for j in range(0,vc_num):
		target=centers[j]
		temp_feat=feat_set[:,i]
		mydist=np.linalg.norm(target - temp_feat)
		if mydist<closest_distance:
			closest_distance=mydist
			assigned_cluster=j
	assignment_new[i]=assigned_cluster
	# if assigned_cluster==88:
	# 	original_img=cv2.imread(originimage[i], cv2.IMREAD_UNCHANGED)
	# 	oimage,_,__=process_image(original_img, '_',0)
	# 	oimage+=np.array([104., 117., 124.])
	# 	aperture_img=np.zeros((224,224,3)).astype('uint8')
	# 	hi=int(loc_set[i,3])
	# 	wi=int(loc_set[i,4])
	# 	Arf=int(loc_set[i,5])-int(loc_set[i,3])
	# 	aperture_img[hi:hi+Arf,wi:wi+Arf,:]=oimage[hi:hi+Arf,wi:wi+Arf,:]
	# 	fname='/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/valassignment/'+str(i)+'_vc'+str(j)+'.jpeg'
	# 	cv2.imwrite(fname,aperture_img)  

with open(save_path, 'wb') as fh:
    pickle.dump([assignment_new], fh)