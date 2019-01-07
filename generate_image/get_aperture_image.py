from __future__ import division
import pickle
import numpy as np
import cv2
import math
import sys
from sys import argv
from copy import *
from ProjectUtils import *
from GetDataPath import *

sys.path.insert(0, './')
sys.path.append('/home/xuyangf/project/ML_deliverables/Siamese_iclr17_tf-master/src/')

import network as vgg
from feature_extractor import FeatureExtractor
from utils import *
# import importlib
# importlib.reload(sys)
reload(sys)  
sys.setdefaultencoding('utf8')

# now, just get top 10 visual concepts as aperture input 

cat=argv[1]

extractor = FeatureExtractor(which_layer='pool3', which_snapshot=0, from_scratch=False)
print('next extractor')
extractor2 = FeatureExtractor(which_layer='1pool4', which_snapshot=0, from_scratch=False)
print('next extractor2')
cluster_num = 256

# myimage_path=LoadImage(cat)
# image_path=[]
# for mypath in myimage_path:
#     myimg=cv2.imread(mypath, cv2.IMREAD_UNCHANGED)
#     if(max(myimg.shape[0],myimg.shape[1])>100):
#         image_path.append(mypath)
# img_num=len(image_path)
# print('img_num')
# print(img_num)
# layer_name = 'pool3'
# file_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/feature/feature3/L3Feature'+cat
# #cluster_file = '/data2/xuyangf/OcclusionProject/NaiveVersion/cluster/clusterL3/vgg16_'+cat+'_K'+str(cluster_num)+'.pickle'
# prun_file = '/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/prunL3/dictionary_'+cat+'.pickle'

# train_open1_name='/data2/xuyangf/OcclusionProject/NaiveVersion/ApertureImage/train/bubble_image/top10vc'
# train_open1_name2='/data2/xuyangf/OcclusionProject/NaiveVersion/ApertureImage/train/bubble_image/center_top10vc'
# center_feature_name='/data2/xuyangf/OcclusionProject/NaiveVersion/ApertureImage/train/bubble_image/center_feature/'+cat

# if not os.path.exists(center_feature_name+'/'):
#     os.mkdir(center_feature_name)
# print('loading data...')

myimage_path=[]
myimage_path+=[os.path.join('/data2/xuyangf/OcclusionProject/NaiveVersion/PortraitImages/train',s) 
        for s in os.listdir('/data2/xuyangf/OcclusionProject/NaiveVersion/PortraitImages/train')]
myimage_path=list(myimage_path)

image_path=[]
for mypath in myimage_path:
    myimg=cv2.imread(mypath, cv2.IMREAD_UNCHANGED)
    if(max(myimg.shape[0],myimg.shape[1])>100):
        image_path.append(mypath)
img_num=len(image_path)
file_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/feature/Portrait/special_test_'
prun_file='/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/Portrait/special_test.pickle'
train_open1_name='/data2/xuyangf/OcclusionProject/NaiveVersion/ApertureImage/train/bubble_image/top10vc_Portrait'
center_feature_name='/data2/xuyangf/OcclusionProject/NaiveVersion/ApertureImage/train/bubble_image/center_feature/Portrait'



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

#img_dim = ff['img_set'].shape[1:]
#img_set = np.zeros([maximg_cnt*file_num]+list(img_dim))
#img_set[0:img_cnt] = ff['img_set']

oldimg_index+=img_cnt

for ii in range(1,file_num):
    fname = file_path+str(ii)+'.npz'
    ff = np.load(fname)
    originimage+=list(ff['originpath'])
    img_cnt=ff['res'].shape[1]
    feat_set[:,oldimg_index:(oldimg_index + img_cnt)] = ff['res']
    loc_set[oldimg_index:(oldimg_index + img_cnt),:] = ff['loc_set']
    #img_set[oldimg_index:(oldimg_index + img_cnt)] = ff['img_set']
    oldimg_index+=img_cnt

feat_set=feat_set[:,:oldimg_index]
#img_set=img_set[:oldimg_index]
loc_set=loc_set[:oldimg_index,:]

myfeat_norm = np.sqrt(np.sum(feat_set**2, 0))
feat_set = feat_set/myfeat_norm

with open(prun_file, 'rb') as fh:
    assignment, centers, example,norm = pickle.load(fh)

print('load finish')

print('get top 10 vc')
# fname ='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/layer3/cat'+str(cat)+'.npz'
# fname ='/data2/xuyangf/OcclusionProject/NaiveVersion/new_vc_score/layer3/cat'+str(cat)+'.npz'
fname ='/data2/xuyangf/OcclusionProject/NaiveVersion/new_vc_score/Portrait/portrait_score.npz'
ff=np.load(fname)
img_vc=ff['vc_score']
vc_num=len(img_vc[0])
img_num=len(img_vc)
print('img_num')
print(img_num)
#print(img_vc[0])
img_vc_avg=[]
for i in range(vc_num):
    img_vc_avg.append(float(np.sum(img_vc[np.where(img_vc[:,i]!=-1),i]))/img_num)
img_vc_avg=np.asarray(img_vc_avg)
rindexsort=np.argsort(-img_vc_avg)

def disttresh(input_index,cluster_center):
    thresh1=0.5
    temp_feat=feat_set[:,input_index]
    error = np.sum((temp_feat.T - cluster_center)**2, 1)
    sort_idx = np.argsort(error)
    return input_index[sort_idx[:int(thresh1*len(sort_idx))]]

mynum=0
mynum2=0
id=0

for k in rindexsort[:20]:
    pool3_res=[]
    pool4_res=[]

    target=centers[k]
    index=np.where(assignment==k)[0]
    index=disttresh(index,target)
    for n in range(0,img_num):
        myindex=[]
        for i in range(len(index)):
            if image_path[n]==originimage[index[i]]:
                myindex.append(index[i])
        #myindex=OnlyTheClosest(myindex,target), or other preprocessing method
        if len(myindex)==0:
            continue
        original_img=cv2.imread(image_path[n], cv2.IMREAD_UNCHANGED)
        oimage,_,__=process_image(original_img, '_',0)
        oimage+=np.array([104., 117., 124.])
        # aperture_img=np.zeros((224,224,3)).astype('uint8')
        findex=image_path[n].rfind('/')
        ffindex=image_path[n].rfind('.')
        for i in range(len(myindex)):
            aperture_img=np.zeros((224,224,3))
            hi=int(loc_set[myindex[i],3])
            wi=int(loc_set[myindex[i],4])
            Arf=int(loc_set[myindex[i],5])-int(loc_set[myindex[i],3])
            center_x=wi+Arf/2
            center_y=hi+Arf/2
            aperture_img,_=generate_bubble_image(oimage,center_x,center_y,5,44,224)
            # aperture_img[hi:hi+Arf,wi:wi+Arf,:]=oimage[hi:hi+Arf,wi:wi+Arf,:]
            fname=train_open1_name+image_path[n][findex:ffindex]+'_VC_'+str(k)+'img'+'_'+str(i)+'.jpeg'
            cv2.imwrite(fname,aperture_img)
            mynum2+=1
            if center_x<30 or center_x+30>=224 or center_y<30 or center_y+30>=224:
                continue
            center_aperture_img=centerlize_aperture(aperture_img,center_x,center_y)
            # fname=train_open1_name2+image_path[n][findex:-5]+'_VC_'+str(k)+'img'+'_'+str(i)+'.jpeg'
            # cv2.imwrite(fname,center_aperture_img)
            pool3_feature=get_center_feature(center_aperture_img,extractor)
            pool4_feature=get_center_feature(center_aperture_img,extractor2)
            pool3_res.append(pool3_feature)
            pool4_res.append(pool4_feature)
           
            mynum+=1

    np.savez(center_feature_name+'/'+str(id)+'.npz',pool3_res=pool3_res,pool4_res=pool4_res)
    id+=1

print(mynum)
print(mynum2)