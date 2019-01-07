from __future__ import division
import pickle
import numpy as np
import cv2
import math
from sys import argv
from copy import *
from ProjectUtils import *
from GetDataPath import *
from itertools import combinations

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

val_open1_name='/data2/xuyangf/OcclusionProject/NaiveVersion/ApertureImage/val/bubble_image/open1/'
val_opentop10_name='/data2/xuyangf/OcclusionProject/NaiveVersion/ApertureImage/val/bubble_image/opentop10/'
# val_open1_enlarge_name='/data2/xuyangf/OcclusionProject/NaiveVersion/ApertureImage/val/enlarge_open1vc/'
# val_part_opentop10_name='/data2/xuyangf/OcclusionProject/NaiveVersion/ApertureImage/val/part_opentop10/'
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
print(loc_dim)
loc_set2 = np.zeros((maximg_cnt*file_num, loc_dim))
loc_set2[0:img_cnt,:] = ff['loc_set']

oldimg_index+=img_cnt

for ii in range(1,file_num):
    print(ii)
    fname = file_path2+str(ii)+'.npz'
    ff = np.load(fname)
    originimage2+=list(ff['originpath'])
    img_cnt=ff['res'].shape[1]
    print(img_cnt)
    feat_set2[:,oldimg_index:(oldimg_index + img_cnt)] = ff['res']
    loc_set2[oldimg_index:(oldimg_index + img_cnt),:] = ff['loc_set']
    #img_set[oldimg_index:(oldimg_index + img_cnt)] = ff['img_set']
    oldimg_index+=img_cnt

feat_set2=feat_set2[:,:oldimg_index]
#img_set=img_set[:oldimg_index]
loc_set2=loc_set2[:oldimg_index,:]

myfeat_norm2 = np.sqrt(np.sum(feat_set2**2, 0))
feat_set2 = feat_set2/myfeat_norm2

with open(prun_file, 'rb') as fh:
    assignment, centers, _,norm = pickle.load(fh)

with open(prun_file2, 'rb') as fh:
    assignment_new = pickle.load(fh)

assignment_new=np.array(assignment_new[0])
print('load finish')

print('get top 10 vc')
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

#tresh
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

originimage2=np.array(originimage2)

VGG_MEAN = [103.94, 116.78, 123.94]
print('imgnum:'+str(img_num))
for n in range(0,img_num):
    print('start'+str(n))
    print(image_path[n])
    original_img=cv2.imread(image_path[n], cv2.IMREAD_UNCHANGED)
    oimage,_,__=process_image(original_img, '_',0)
    oimage+=np.array([104., 117., 124.])
    myindex=np.where(originimage2==image_path[n])[0]
    top10_aperture_img=np.zeros((224,224,3))
    top10_aperture_PDF=np.zeros((224,224,3))
    findex=image_path[n].rfind('/')
    top10_index=[]
    for k in range(0,topvc):
        mycluster=int(rindexsort[k])
        index=myindex[np.where(assignment_new[myindex]==mycluster)]
        # enlarge_aperture_img=[]
        # for i in range(2,9):
        #     enlarge_aperture_img.append(np.zeros((224,224,3)).astype('uint8'))
        #tresh
        index=disttresh(index,mycluster)
        index=disttresh2(index,mycluster)
        print('test: '+str(mycluster))
        print(index)
        if len(index)==0:
            continue
        top10_index.append(index)
        for myid in index:  
            #create open1
            aperture_img=np.zeros((224,224,3))
            hi=int(loc_set2[myid,3])
            wi=int(loc_set2[myid,4])
            Arf=int(loc_set2[myid,5])-int(loc_set2[myid,3])
            # aperture_img[hi:hi+Arf,wi:wi+Arf,:]=oimage[hi:hi+Arf,wi:wi+Arf,:]
            center_x=wi+Arf/2
            center_y=hi+Arf/2
            aperture_img,tmpPDF=generate_bubble_image(oimage,center_x,center_y,5,44,224)
            fname=val_open1_name+image_path[n][findex:-5]+'_VC'+str(k)+'_'+str(myid)+'.jpeg'
            cv2.imwrite(fname,aperture_img)

            top10_aperture_PDF=np.maximum(top10_aperture_PDF,tmpPDF)
            # top10_aperture_img[hi:hi+Arf,wi:wi+Arf,:]=oimage[hi:hi+Arf,wi:wi+Arf,:]

            #create enlarge open1
            # for i in range(2,6):
            #     total_area=i*patch_size*patch_size
            #     area_size=int(np.sqrt(total_area))
            #     extend_size=area_size/2-patch_size/2
            #     new_hi=int(hi - extend_size)
            #     new_wi=wi- extend_size
            #     new_plus_hi=new_hi+area_size
            #     new_plus_wi=new_wi+area_size
            #     new_wi=int(max(0,new_wi))
            #     new_hi=int(max(0,new_hi))
            #     new_plus_wi=int(min(224,new_plus_wi))
            #     new_plus_hi=int(min(224,new_plus_hi))
            #     enlarge_aperture_img[i-2][new_hi:new_plus_hi,new_wi:new_plus_wi,:]=oimage[new_hi:new_plus_hi,new_wi:new_plus_wi,:]
            # for i in range(6,9):
            #     total_area=(i-4)*224*224/10
            #     area_size=int(np.sqrt(total_area))
            #     extend_size=area_size/2-patch_size/2
            #     new_hi=int(hi - extend_size)
            #     new_wi=wi- extend_size
            #     new_plus_hi=new_hi+area_size
            #     new_plus_wi=new_wi+area_size
            #     new_wi=int(max(0,new_wi))
            #     new_hi=int(max(0,new_hi))
            #     new_plus_wi=int(min(224,new_plus_wi))
            #     new_plus_hi=int(min(224,new_plus_hi))
            #     enlarge_aperture_img[i-2][new_hi:new_plus_hi,new_wi:new_plus_wi,:]=oimage[new_hi:new_plus_hi,new_wi:new_plus_wi,:]

        # for i in range(0,7):
        #     fname=val_open1_enlarge_name+str(i)+'/'+image_path[n][findex:-5]+'_VC'+str(k)+'.jpeg'
        #     cv2.imwrite(fname,enlarge_aperture_img[i])

            
    top10_num=len(top10_index)
    if top10_num==0:
        continue
    fname=val_opentop10_name+image_path[n][findex:-5]+'.jpeg'
    top10_aperture_img=top10_aperture_PDF*oimage
    back = VGG_MEAN*(1-top10_aperture_PDF )*np.ones((224,224,3))
    top10_aperture_img = top10_aperture_img + back
    cv2.imwrite(fname,top10_aperture_img)

    # start to create part top10
    # top10_index=np.array(top10_index)
    # if top10_num>=2:
    #     combins=[c for c in  combinations(range(top10_num), 2)]
    #     for i in range(0,len(combins)):
    #         part_top10_aperture_img=np.zeros((224,224,3)).astype('uint8')
    #         allindex=top10_index[np.array(combins[i])]
    #         allindex=np.concatenate(allindex)
    #         for myid in allindex:
    #             hi=int(loc_set2[myid,3])
    #             wi=int(loc_set2[myid,4])
    #             Arf=int(loc_set2[myid,5])-int(loc_set2[myid,3])
    #             part_top10_aperture_img[hi:hi+Arf,wi:wi+Arf,:]=oimage[hi:hi+Arf,wi:wi+Arf,:]
    #         fname=val_part_opentop10_name+str(0)+'/'+image_path[n][findex:-5]+'_'+str(i)+'.jpeg'
    #         cv2.imwrite(fname,part_top10_aperture_img)
    # if top10_num>=3:
    #     combins=[c for c in  combinations(range(top10_num), 3)]
    #     for i in range(0,len(combins)):
    #         part_top10_aperture_img=np.zeros((224,224,3)).astype('uint8')
    #         allindex=top10_index[np.array(combins[i])]
    #         allindex=np.concatenate(allindex)
    #         for myid in allindex:
    #             hi=int(loc_set2[myid,3])
    #             wi=int(loc_set2[myid,4])
    #             Arf=int(loc_set2[myid,5])-int(loc_set2[myid,3])
    #             part_top10_aperture_img[hi:hi+Arf,wi:wi+Arf,:]=oimage[hi:hi+Arf,wi:wi+Arf,:]
    #         fname=val_part_opentop10_name+str(1)+'/'+image_path[n][findex:-5]+'_'+str(i)+'.jpeg'
    #         cv2.imwrite(fname,part_top10_aperture_img)
    # if top10_num>=4:
    #     combins=[c for c in  combinations(range(top10_num), 4)]
    #     for i in range(0,len(combins)):
    #         part_top10_aperture_img=np.zeros((224,224,3)).astype('uint8')
    #         allindex=top10_index[np.array(combins[i])]
    #         allindex=np.concatenate(allindex)
    #         for myid in allindex:
    #             hi=int(loc_set2[myid,3])
    #             wi=int(loc_set2[myid,4])
    #             Arf=int(loc_set2[myid,5])-int(loc_set2[myid,3])
    #             part_top10_aperture_img[hi:hi+Arf,wi:wi+Arf,:]=oimage[hi:hi+Arf,wi:wi+Arf,:]
    #         fname=val_part_opentop10_name+str(2)+'/'+image_path[n][findex:-5]+'_'+str(i)+'.jpeg'
    #         cv2.imwrite(fname,part_top10_aperture_img)
    # if top10_num>=5:
    #     combins=[c for c in  combinations(range(top10_num), 5)]
    #     for i in range(0,len(combins)):
    #         part_top10_aperture_img=np.zeros((224,224,3)).astype('uint8')
    #         allindex=top10_index[np.array(combins[i])]
    #         allindex=np.concatenate(allindex)
    #         for myid in allindex:
    #             hi=int(loc_set2[myid,3])
    #             wi=int(loc_set2[myid,4])
    #             Arf=int(loc_set2[myid,5])-int(loc_set2[myid,3])
    #             part_top10_aperture_img[hi:hi+Arf,wi:wi+Arf,:]=oimage[hi:hi+Arf,wi:wi+Arf,:]
    #         fname=val_part_opentop10_name+str(3)+'/'+image_path[n][findex:-5]+'_'+str(i)+'.jpeg'
    #         cv2.imwrite(fname,part_top10_aperture_img)

