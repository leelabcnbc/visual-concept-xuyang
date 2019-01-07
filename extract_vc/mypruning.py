from __future__ import division
import cv2
import pickle,os
import numpy as np
from sys import argv
from GetDataPath import *
from utils import process_image
from collections import Counter
from enum import Enum

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

# file_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/feature/feature'+mylayer+'/L'+mylayer+'Feature'+cat
# cluster_file='/data2/xuyangf/OcclusionProject/NaiveVersion/cluster/clusterL'+mylayer+'/vgg16_'+cat+'_K'+str(cluster_num)+'.pickle'
# save_path1 = '/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/prunL'+mylayer+'/dictionary_'+cat+'.pickle'
#file_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/feature/feature3/L3Feature'+cat
#cluster_file = '/data2/xuyangf/OcclusionProject/NaiveVersion/cluster/clusterL3/vgg16_'+cat+'_K'+str(cluster_num)+'.pickle'
#save_path1 = '/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/prunL3/dictionary_'+cat+'.pickle'
# file_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/feature/feature'+mylayer+'/L'+mylayer+'Feature'+cat
# cluster_file='/data2/xuyangf/OcclusionProject/NaiveVersion/cluster/clusterL'+mylayer+'/vgg16_'+cat+'_K'+str(cluster_num)+'.pickle'
# save_path1 = '/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/prunL'+mylayer+'/dictionary_'+cat+'.pickle'
if mod=='0':
    file_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/feature/feature'+mylayer+'/L'+mylayer+'Feature'+cat
    cluster_file='/data2/xuyangf/OcclusionProject/NaiveVersion/cluster/clusterL'+mylayer+'/vgg16_'+cat+'_K'+str(cluster_num)+'.pickle'
    save_path1 = '/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/prunL'+mylayer+'/dictionary_'+cat+'.pickle'
    if not os.path.exists('/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/prunL'+mylayer+'/'):
        os.mkdir('/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/prunL'+mylayer)

if mod=='1':
    cluster_file='/data2/xuyangf/OcclusionProject/NaiveVersion/cluster/Portrait/special_test.pickle'
    file_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/feature/Portrait/special_test_'
    save_path1 = '/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/Portrait/special_test.pickle'
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


with open(cluster_file, 'rb') as fh:
    assignment, centers,example= pickle.load(fh)

print 'load finish'

# L2 normalization
feat_norm = np.sqrt(np.sum(feat_set**2, 0))
myfeat_set=feat_set
feat_set = feat_set/feat_norm

#delete loose cluster stage1
K_new=0
centers_new=[]
meandist=[]
for i in range(len(centers)):
    target=centers[i]
    index = np.where(assignment==i)[0]
    temp_feat = feat_set[:, index]
    dist = np.sqrt(np.sum((temp_feat - target.reshape(-1,1))**2, axis=0))
    meandist.append(np.mean(dist))
meandist=np.asarray(meandist)
meandist=np.sort(meandist)
print(int(0.8*len(meandist)))
loose_tresh=meandist[int(0.8*len(meandist))]
print(loose_tresh)
for i in range(len(centers)):
    target=centers[i]    
    index = np.where(assignment==i)[0]
    temp_feat = feat_set[:, index]
    dist = np.sqrt(np.sum((temp_feat - target.reshape(-1,1))**2, axis=0))
    if np.mean(dist)>loose_tresh:
        assignment[index]=-1
    else:
        assignment[index]=K_new
        centers_new.append(centers[i])
        K_new+=1

K=K_new
centers=np.asarray(centers_new)

mylen=len(assignment)
myindent=np.where(assignment>-1)[0]
assignment=assignment[myindent]
feat_set=feat_set[:,myindent]
loc_set=loc_set[myindent,:]
originimage=np.asarray(originimage)
originimage=originimage[myindent]
#img_set=img_set[myindent]

print('compute metric...')
# decide the rank of clusters
count = np.bincount(assignment, minlength=K)

print(count)
print(K)
# based on centers
pw_cen = np.zeros((K,K))
for k in range(K):
    for m in range(K):
        pw_cen[k,m] = np.linalg.norm(centers[k]-centers[m])


# based on data points
pw_all = np.zeros((K,K))
for k in range(K):
    target = centers[k]
    for m in range(K):
        index = np.where(assignment==m)[0]
        temp_feat = feat_set[:, index]
        dist = np.sqrt(np.sum((temp_feat - target.reshape(-1,1))**2, axis=0))
        sort_value = np.sort(dist)
        pw_all[k, m] = np.mean(sort_value[0:int(0.95*len(sort_value))])



list = np.zeros(K)
for k in range(K):
    rec = np.zeros(K)
    for m in range(K):
        if m != k:
            rec[m] = (pw_all[m,m] + pw_all[k,k])/pw_cen[m,k]
        
    list[k] = np.max(rec)


# the lower the better
bbb = np.argsort(list)
aaa = np.sort(list)
sort_list = np.stack([aaa,bbb])

# the higher the better
count_norm = count/np.sum(count)
bbb = np.argsort(count_norm)[::-1]
aaa = np.sort(count_norm)[::-1]
sort_count_norm = np.stack([aaa,bbb])

# give big penalty if cluster number is too small
penalty = 100*(count<=100)+100*(count<=50)
#add by xuyang penalty is a magic number

# combine the above metrics, the lower the better
com = list - K*count_norm + penalty
bbb = np.argsort(com)
aaa = np.sort(com)
sort_com = np.stack([aaa,bbb])


print('greedy pruning...')
sort_cls = sort_com[1].astype(int)
rec = np.ones(K)
thresh1 = 0.8  # magic number
thresh2 = 0.3  # magic number

prune = np.zeros((3,0))
prune_res = []

while np.sum(rec) > 0:
    temp = np.zeros((3,0))
    idx = np.where(rec==1)[0][0]
    cls = sort_cls[idx]
    
    target = centers[cls]
    index = np.where(assignment==cls)[0]
    temp_feat = feat_set[:, index]
    dist = np.sqrt(np.sum((temp_feat - target.reshape(-1,1))**2, axis=0))
    sort_value = np.sort(dist)
    dist_thresh = sort_value[int(thresh1*len(sort_value))]
    # add by xuyang # a far distance from the center
    rec[idx] = 0
    for n in range(idx+1,K):
        if rec[n] == 1:
            index = np.where(assignment==sort_cls[n])[0]
            temp_feat = feat_set[:, index]
            dist = np.sqrt(np.sum((temp_feat - target.reshape(-1,1))**2, axis=0))
            if np.mean(dist<dist_thresh) >= thresh2:
                temp = np.column_stack([temp, np.array([n,sort_cls[n],np.mean(dist<dist_thresh)]).reshape(-1,1)])
                rec[n] = 0
                
                
    print('{0}, {1}, {2}'.format(idx, cls, temp.shape[1]))
    prune = np.column_stack([prune, np.array([idx, cls, temp.shape[1]]).reshape(-1,1)])
    prune_res.append(temp)


print('update new dictionary...')
K_new = prune.shape[1]
print('new K is : {0}'.format(K_new))

pruning_table=[None for nn in range(K_new)]

centers_new = np.zeros((K_new, centers.shape[1]))
assignment_new = np.zeros_like(assignment)
for k in range(K_new):
    if prune_res[k].size == 0:
        temp = np.array([prune[1,k]])
    else:
        temp = np.append(prune[1,k], prune_res[k][1,:])
    
    temp=temp.astype(int)
    pruning_table[k]=temp
    weight = count[temp]
    weight = weight/sum(weight)
    temp_cen = centers[temp]
    centers_new[k] = np.dot(temp_cen.T, weight.reshape(-1,1)).squeeze()
    for i in range(len(temp)):
        assignment_new[assignment==temp[i]] = k


K = K_new
centers = centers_new
assignment = assignment_new

#delete cluster that only occur in one picture
K_new=0
centers_new=[]
count = np.bincount(assignment, minlength=K)

for i in range(len(centers)):    
    index = np.where(assignment==i)[0]
    print('cluster image num'+str(i))
    print(Counter(originimage[index]).most_common(1))
    #print(list(Counter(originimage[index]).most_common(1))[0][1])
    if int(Counter(originimage[index]).most_common(1)[0][1])>len(index)/2:
        assignment[index]=-1
    else:
        assignment[index]=K_new
        centers_new.append(centers[i])
        K_new+=1
K=K_new
centers=np.asarray(centers_new)

#delete loose cluster stage2
# K_new=0
# centers_new=[]
# meandist=[]
# for i in range(len(centers)):
#     index = np.where(assignment==i)[0]
#     temp_feat = feat_set[:, index]
#     dist = np.sqrt(np.sum((temp_feat - target.reshape(-1,1))**2, axis=0))
#     meandist.append(np.mean(dist))

# for i in range(len(centers)):    
#     index = np.where(assignment==i)[0]
#     temp_feat = feat_set[:, index]
#     dist = np.sqrt(np.sum((temp_feat - target.reshape(-1,1))**2, axis=0))
#     loose_tresh=np.sort(meandist)[0.8*len(meandist)]
#     if np.mean(dist)>loose_tresh:
#         assignment[index]=-1
#     else:
#         assignment[index]=K_new
#         centers_new.append(centers[i])
#         K_new+=1
# K=K_new
# centers=np.asarray(centers_new)

#evaluate

meandist=[]
for i in range(len(centers)):    
    index = np.where(assignment==i)[0]
    print(len(index))
    target=centers[i]
    temp_feat = feat_set[:, index]
    dist = np.sqrt(np.sum((temp_feat - target.reshape(-1,1))**2, axis=0))
    meandist.append(np.mean(dist))

print(meandist)
print('evaluate cluster')
print np.mean(meandist)

savefeat=[]
# the num of images for each cluster
number = 100
print('save top {0} images for each cluster'.format(number))
example = [None for nn in range(K)]
patch_size = int(loc_set[0,5] - loc_set[0,3])
for k in range(K):
    target = centers[k]
    index = np.where(assignment == k)[0]
    num = min(number, len(index))
    
    tempFeat = feat_set[:,index]
    error = np.sum((tempFeat.T - target)**2, 1)
    sort_idx = np.argsort(error)
    patch_set = np.zeros(((patch_size**2)*3, num)).astype('uint8')
    print 'cluster '+str(k)+'start'
    for idx in range(num):
        patchindex=index[sort_idx[idx]]

        oimage=cv2.imread(originimage[patchindex], cv2.IMREAD_UNCHANGED)
        oimage,_,__=process_image(oimage, '_',0)
        oimage+=np.array([104., 117., 124.])
        hi=int(loc_set[patchindex,3])
        wi=int(loc_set[patchindex,4])
        Arf=int(loc_set[patchindex,5])-int(loc_set[patchindex,3])

        patch = oimage[hi:hi + Arf, wi:wi + Arf, :]
        if idx==0:
            savefeat.append(feat_set[:,patchindex])
            print(feat_set[:,patchindex][0])

        patch_set[:,idx] = patch.flatten()
        
    example[k] = np.copy(patch_set)

print(K)

assignment_new= np.zeros(mylen)
for i in range(0,mylen):
    assignment_new[i]=-1
assignment_new[myindent]=assignment
assignment=assignment_new
save_path = save_path1
with open(save_path, 'wb') as fh:
    pickle.dump([assignment, centers, example,savefeat], fh)

