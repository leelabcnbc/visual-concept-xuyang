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
mynetwork=argv[3]
# print(cat)
# print(num)
predictor=TestVgg(3,mynetwork)
VGG_MEAN = [103.94, 116.78, 123.94]

enlarge_dir='/data2/xuyangf/OcclusionProject/NaiveVersion/ApertureImage/val/bubble_image/NewVersion_Enlarge/'+str(num)+'/'
combine_dir='/data2/xuyangf/OcclusionProject/NaiveVersion/ApertureImage/val/bubble_image/NewVersion_Combine/'+str(num)+'/'
no_overlap_combine_dir='/data2/xuyangf/OcclusionProject/NaiveVersion/ApertureImage/val/bubble_image/NewVersion_no_overlap_Combine/'+str(num)+'/'

cluster_num = 256
patch_size=44
topvc=10
myimage_path=LoadImage2(cat)
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

file_path2 = '/data2/xuyangf/OcclusionProject/NaiveVersion/feature/valfeature/'+cat
prun_file2 = '/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/valassignment/dictionary_'+cat+'.pickle'

# print('loading data...')


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
# print(loc_dim)
loc_set = np.zeros((maximg_cnt*file_num, loc_dim))
loc_set[0:img_cnt,:] = ff['loc_set']

#img_dim = ff['img_set'].shape[1:]
#img_set = np.zeros([maximg_cnt*file_num]+list(img_dim))
#img_set[0:img_cnt] = ff['img_set']

oldimg_index+=img_cnt

for ii in range(1,file_num):
    # print(ii)
    fname = file_path+str(ii)+'.npz'
    ff = np.load(fname)
    originimage+=list(ff['originpath'])
    img_cnt=ff['res'].shape[1]
    # print(img_cnt)
    feat_set[:,oldimg_index:(oldimg_index + img_cnt)] = ff['res']
    loc_set[oldimg_index:(oldimg_index + img_cnt),:] = ff['loc_set']
    #img_set[oldimg_index:(oldimg_index + img_cnt)] = ff['img_set']
    oldimg_index+=img_cnt

feat_set=feat_set[:,:oldimg_index]
#img_set=img_set[:oldimg_index]
loc_set=loc_set[:oldimg_index,:]

myfeat_norm = np.sqrt(np.sum(feat_set**2, 0))
feat_set = feat_set/myfeat_norm

# load val image
if cat=='39':
    file_num=8
if cat=='22':
    file_num=9
if cat=='71':
    file_num=9
fname = file_path2+str(0)+'.npz'
ff = np.load(fname)

feat_dim = ff['res'].shape[0]
img_cnt = ff['res'].shape[1]
oldimg_index=0

originimage2=[]
feat_set2 = np.zeros((feat_dim, maximg_cnt*file_num))
feat_set2[:,0:img_cnt] = ff['res']

originimage2+=list(ff['originpath'])
loc_dim = ff['loc_set'].shape[1]
# print(loc_dim)
loc_set2 = np.zeros((maximg_cnt*file_num, loc_dim))
loc_set2[0:img_cnt,:] = ff['loc_set']

oldimg_index+=img_cnt

for ii in range(1,file_num):
    # print(ii)
    fname = file_path2+str(ii)+'.npz'
    ff = np.load(fname)
    originimage2+=list(ff['originpath'])
    img_cnt=ff['res'].shape[1]
    # print(img_cnt)
    feat_set2[:,oldimg_index:(oldimg_index + img_cnt)] = ff['res']
    loc_set2[oldimg_index:(oldimg_index + img_cnt),:] = ff['loc_set']
    #img_set[oldimg_index:(oldimg_index + img_cnt)] = ff['img_set']
    oldimg_index+=img_cnt

feat_set2=feat_set2[:,:oldimg_index]
#img_set=img_set[:oldimg_index]
loc_set2=loc_set2[:oldimg_index,:]
originimage2=np.array(originimage2)

myfeat_norm2 = np.sqrt(np.sum(feat_set2**2, 0))
feat_set2 = feat_set2/myfeat_norm2

with open(prun_file, 'rb') as fh:
    assignment, centers, _,norm = pickle.load(fh)

with open(prun_file2, 'rb') as fh:
    assignment_new = pickle.load(fh)

assignment_new=np.array(assignment_new[0])
# print('load finish')

# print('get top 10 vc')
fname ='/data2/xuyangf/OcclusionProject/NaiveVersion/new_vc_score/layer3/cat'+str(cat)+'.npz'
ff=np.load(fname)
img_vc=ff['vc_score']
vc_num=len(img_vc[0])
#print(img_vc[0])
img_vc_avg=[]
for i in range(vc_num):
    img_vc_avg.append(float(np.sum(img_vc[np.where(img_vc[:,i]!=-1),i]))/img_num)
img_vc_avg=np.asarray(img_vc_avg)
rindexsort=np.argsort(-img_vc_avg)

############################### load data finish ##################

############################### tresh
def getdist(k):
    cluster_center=centers[k]
    input_index=np.where(assignment==k)[0]
    thresh1=0.8
    temp_feat=feat_set[:,input_index]
    error = np.sum((temp_feat.T - cluster_center)**2, 1)
    sort_idx = np.argsort(error)
    return error[sort_idx[int(thresh1*len(sort_idx))]]

mytresh=[]
for i in range(0,vc_num):
    mytresh.append(getdist(i))

def getdist2(k):
    cluster_center=centers[k]
    input_index=np.where(assignment==k)[0]
    thresh1=0.1
    temp_feat=feat_set[:,input_index]
    error = np.sum((temp_feat.T - cluster_center)**2, 1)
    sort_idx = np.argsort(error)
    return error[sort_idx[int(thresh1*len(sort_idx))]]

mytresh2=[]
for i in range(0,vc_num):
    mytresh2.append(getdist2(i))

def disttresh(input_index,k):
    cluster_center=centers[k]
    dist=mytresh[k]
    temp_feat=feat_set2[:,input_index]
    error = np.sum((temp_feat.T - cluster_center)**2, 1)
    idx=np.where(error<dist)[0]
    return input_index[idx]

def disttresh2(input_index,k):
    cluster_center=centers[k]
    dist=mytresh2[k]
    temp_feat=feat_set2[:,input_index]
    error = np.sum((temp_feat.T - cluster_center)**2, 1)
    idx=np.where(error<1.5*dist)[0]
    return input_index[idx]


number=0
enlarge_acc=0
combine_acc=0
# print('imgnum:'+str(img_num))
for n in range(0,img_num):
    original_img=cv2.imread(image_path[n], cv2.IMREAD_UNCHANGED)
    oimage,_,__=process_image(original_img, '_',0)
    oimage+=np.array([104., 117., 124.])
    myindex=np.where(originimage2==image_path[n])[0]
    findex=image_path[n].rfind('/')+1
    top10_index=[]

    for k in range(0,topvc):
        mycluster=int(rindexsort[k])
        index=myindex[np.where(assignment_new[myindex]==mycluster)]
        #tresh
        index=disttresh(index,mycluster)
        index=disttresh2(index,mycluster)
        # print('test: '+str(mycluster))
        # print(index)
        if len(index)==0:
            continue

        target = centers[mycluster]
        tempFeat = feat_set2[:,index]
        error = np.sum((tempFeat.T - target)**2, 1)
        sort_idx = np.argsort(error)
        top10_index.append(index[sort_idx[0]])

    if len(top10_index)<num:
        continue
    top10_num=len(top10_index)
    top10_index=np.array(top10_index)
    
    all_single_prediction=[]
    for myid in top10_index:
        myfname=enlarge_dir+image_path[n][findex:-5]+'_enlarge_'+str(num)+'_vc_'+str(myid)+'.jpeg'
        _,__,prediction,prob=predictor.getacc(myfname,int(cat))
        all_single_prediction.append(prediction)
    all_single_prediction=np.array(all_single_prediction)

    combins=[c for c in  combinations(range(top10_num), num)]
    for i in range(0,len(combins)):
        part_top10_aperture_PDF=np.zeros((224,224,3))
        mask=np.zeros((224,224,3))
        allindex=top10_index[np.array(combins[i])]
        for myid in allindex:
            hi=int(loc_set2[myid,3])
            wi=int(loc_set2[myid,4])
            Arf=int(loc_set2[myid,5])-int(loc_set2[myid,3])
            center_x=wi+Arf/2
            center_y=hi+Arf/2
            tmp,tmpPDF=generate_bubble_image(oimage,center_x,center_y,5,44,224)
            part_top10_aperture_PDF=np.maximum(part_top10_aperture_PDF,tmpPDF)
            mask[hi:hi+Arf,wi:wi+Arf,:]=1
        if np.count_nonzero(mask)<num*44*44*3:
            continue
        myfname=no_overlap_combine_dir+image_path[n][findex:-5]+'_combine'+str(num)+'_no_'+str(i)+'.jpeg'
        _,__,prediction,prob=predictor.getacc(myfname,int(cat))

        allprediction=all_single_prediction[np.array(combins[i])]

        if len(np.where(allprediction==int(cat))[0])>0:
            enlarge_acc+=1
        if prediction==int(cat):
            combine_acc+=1
        number+=1

save_file='/data2/xuyangf/OcclusionProject/NaiveVersion/tmptest/New_VC_Combination_Effect/'+mynetwork+'/'+str(num)+'/'+cat
np.savez(save_file,enlarge_acc=enlarge_acc,combine_acc=combine_acc,number=number)
print('success finish cat'+str(cat)+' num'+str(num)+' network'+ str(mynetwork))