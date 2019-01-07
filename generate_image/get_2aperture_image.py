from __future__ import division
import pickle
import numpy as np
import cv2
import math
from sys import argv
from copy import *
from ProjectUtils import *
from GetDataPath import *
from newtestvgg import TestVgg
from itertools import combinations
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

# now, just get top 10 visual concepts as aperture input 

cat=argv[1]
num=int(argv[2])
VGG_MEAN = [103.94, 116.78, 123.94]
cluster_num = 256
topvc=10
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

train_open_multiple_name='/data2/xuyangf/OcclusionProject/NaiveVersion/ApertureImage/train/bubble_image/multiple_top10vc/no_overlap_'+str(num)+'/'

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

myfeat_norm = np.sqrt(np.sum(feat_set**2, 0))
feat_set = feat_set/myfeat_norm

with open(prun_file, 'rb') as fh:
    assignment, centers, _,norm = pickle.load(fh)

print('load finish')

print('get top 10 vc')
# fname ='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/layer3/cat'+str(cat)+'.npz'
fname ='/data2/xuyangf/OcclusionProject/NaiveVersion/new_vc_score/layer3/cat'+str(cat)+'.npz'
ff=np.load(fname)
img_vc=ff['vc_score']
vc_num=len(img_vc[0])
img_num=len(img_vc)
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

def choose_representative(input_index,k):
    cluster_center=centers[k]
    temp_feat=feat_set[:,input_index]
    error = np.sum((temp_feat.T - cluster_center)**2, 1)
    idx=np.argmin(error)
    return input_index[idx]

originimage=np.array(originimage)
for n in range(0,img_num):
    original_img=cv2.imread(image_path[n], cv2.IMREAD_UNCHANGED)
    oimage,_,__=process_image(original_img, '_',0)
    oimage+=np.array([104., 117., 124.])
    # print(image_path[n])
    # print(originimage[0])
    # print(np.where(originimage==image_path[n]))
    myindex=np.where(originimage==image_path[n])[0]
    findex=image_path[n].rfind('/')+1
    top10_index=[]
    # print('myindex')
    # print(myindex)
    for k in range(0,topvc):
        mycluster=int(rindexsort[k])
        index=myindex[np.where(assignment[myindex]==mycluster)]
        print(index)
        #tresh
        index=disttresh(index,centers[mycluster])
        if len(index)==0:
            continue
        index=choose_representative(index,mycluster)
        top10_index.append(index)

    top10_num=len(top10_index)
    print('top10_num')
    print(top10_num)
    if top10_num<num:
        continue
    top10_index=np.array(top10_index)

    combins=[c for c in  combinations(range(top10_num), num)]
    for i in range(0,len(combins)):
        allindex=top10_index[np.array(combins[i])]
        part_top10_aperture_img=np.zeros((224,224,3))
        part_top10_aperture_PDF=np.zeros((224,224,3))

        mask=np.zeros((224,224,3))
        for myid in allindex:
            hi=int(loc_set[myid,3])
            wi=int(loc_set[myid,4])
            Arf=int(loc_set[myid,5])-int(loc_set[myid,3])
            center_x=wi+Arf/2
            center_y=hi+Arf/2
            tmp,tmpPDF=generate_bubble_image(oimage,center_x,center_y,5,44,224)
            part_top10_aperture_PDF=np.maximum(part_top10_aperture_PDF,tmpPDF)
            mask[hi:hi+Arf,wi:wi+Arf,:]=1

        if np.count_nonzero(mask)<num*44*44*3:
            continue

        part_top10_aperture_img=part_top10_aperture_PDF*oimage
        back = VGG_MEAN*(1-part_top10_aperture_PDF )*np.ones((224,224,3))
        part_top10_aperture_img = part_top10_aperture_img + back

        myfname=train_open_multiple_name+image_path[n][findex:-5]+'_combine'+str(num)+'_No_'+str(i)+'.jpeg'
        cv2.imwrite( myfname,part_top10_aperture_img)
