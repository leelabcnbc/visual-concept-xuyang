#!/export/home/zzhang99/.linuxbrew/bin/python
from __future__ import division
import cv2
import numpy as np
import pickle,os
import time,math
from sklearn.cluster import KMeans
from sys import argv
from enum import Enum
import sys
from utils import *
cat=argv[1]
mylayer=argv[2]
mod=argv[3]

class featDim_set(Enum):
    pool1=64
    pool2=128
    pool3=256
    pool4=512
    pool5=512
    conv2_1=128
    conv2_2=128
    conv3_1=256
    conv3_2=256
    conv3_3=256
    conv4_1=512
    conv4_2=512
    conv4_3=512

cluster_num = featDim_set[mylayer].value
if mylayer[4]=='4':
    cluster_num=256
# Chen Liu Project
# cluster_num=128

# save_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/cluster/clusterL'+mylayer+'/vgg16_'+cat+'_K'+str(cluster_num)+'.pickle'
# file_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/feature/feature'+mylayer+'/L'+mylayer+'Feature'+cat

if mod=='0':
    save_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/cluster/clusterL'+mylayer+'/vgg16_'+cat+'_K'+str(cluster_num)+'.pickle'
    file_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/feature/feature'+mylayer+'/L'+mylayer+'Feature'+cat
    if not os.path.exists('/data2/xuyangf/OcclusionProject/NaiveVersion/cluster/clusterL'+mylayer+'/'):
        os.mkdir('/data2/xuyangf/OcclusionProject/NaiveVersion/cluster/clusterL'+mylayer)

if mod=='1':
    save_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/cluster/Portrait/special_test.pickle'
    file_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/feature/Portrait/special_test_'
class Arf_set(Enum):
    pool1=6
    pool2=16
    pool3=44
    pool4=100
    pool5=212
    conv2_1=10
    conv2_2=14
    conv3_1=24
    conv3_2=32
    conv3_3=40
    conv4_1=60
    conv4_2=76
    conv4_3=92
patch_size=Arf_set[mylayer].value

fname = file_path+str(0)+'.npz'
ff = np.load(fname)

feat_dim = ff['res'].shape[0]
img_cnt = ff['res'].shape[1]
oldimg_index=0

# number of files to read in
# number of files to read in
file_num = 10
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


print('Start K-means...')
_s = time.time()
km = KMeans(n_clusters=cluster_num, init='k-means++', random_state=99, n_jobs=1)
assignment = km.fit_predict(feat_set.T)
centers = km.cluster_centers_
_e = time.time()
print('K-means running time: {0}'.format((_e-_s)/60))

with open(save_path, 'wb') as fh:
    pickle.dump([assignment, centers], fh)
    
# the num of images for each cluster
num = 100
print('save top {0} images for each cluster'.format(num))
example = [None for nn in range(cluster_num)]

for k in range(cluster_num):
    target = centers[k]
    index = np.where(assignment == k)[0]
    num = min(num, len(index))
    
    tempFeat = feat_set[:,index]
    error = np.sum((tempFeat.T - target)**2, 1)
    sort_idx = np.argsort(error)
    patch_set = np.zeros(((patch_size**2)*3, num)).astype('uint8')
    for idx in range(num):
        patchindex=index[sort_idx[idx]]

        oimage=cv2.imread(originimage[patchindex], cv2.IMREAD_UNCHANGED)
        # oimage,_,__=process_image(oimage, '_',0)
        oimage,_,__=process_image2(oimage)
        oimage+=np.array([104., 117., 124.])
        hi=int(loc_set[patchindex,3])
        wi=int(loc_set[patchindex,4])
        Arf=int(loc_set[patchindex,5])-int(loc_set[patchindex,3])
        patch = oimage[hi:hi + Arf, wi:wi + Arf, :]
        patch_set[:,idx] = patch.flatten()
        
    example[k] = np.copy(patch_set)
    if k%20 == 0:
        print(k)
        

with open(save_path, 'wb') as fh:
    pickle.dump([assignment, centers,example], fh)
