#!/export/home/zzhang99/.linuxbrew/bin/python
import cv2,os,glob,pickle,sys
import numpy as np
import tensorflow as tf
from sys import argv
from enum import Enum
sys.path.insert(0, './')
sys.path.append('/home/xuyangf/project/ML_deliverables/Siamese_iclr17_tf-master/src/')
from tensorflow.python.client import timeline
from datetime import datetime
from utils import *
from global_variables import *
import cv2
import json
import sys
from copy import *
from feature_extractor import FeatureExtractor
from GetDataPath import *

cat=argv[1]
extract_layer=argv[2]
mod=argv[3]

def enum(**enums):
    return type('Enum', (), enums)

class Apad_set(Enum):
    pool1=2
    pool2=6
    pool3=18
    pool4=42
    pool5=90
    conv2_1=4
    conv2_2=6
    conv3_1=10
    conv3_2=14
    conv3_3=18
    conv4_1=26
    conv4_2=34
    conv4_3=42

class Astride_set(Enum):
    pool1=2
    pool2=4
    pool3=8
    pool4=16
    pool5=32
    conv2_1=2
    conv2_2=2
    conv3_1=4
    conv3_2=4
    conv3_3=4  
    conv4_1=8
    conv4_2=8
    conv4_3=8

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


Arf=float(Arf_set[extract_layer].value)
Apad=float(Apad_set[extract_layer].value)
Astride=float(Astride_set[extract_layer].value)
offset=np.ceil(Apad/Astride)
print('offset')
print(offset)
scale_size = 224

if mod=='0':
    savepath = '/data2/xuyangf/OcclusionProject/NaiveVersion/feature/feature'+extract_layer+'/L'+extract_layer+'Feature'+cat
    if not os.path.exists('/data2/xuyangf/OcclusionProject/NaiveVersion/feature/feature'+extract_layer+'/'):
        os.mkdir('/data2/xuyangf/OcclusionProject/NaiveVersion/feature/feature'+extract_layer)
    myimage_path=LoadImage(cat)

if mod=='1':
    savepath= '/data2/xuyangf/OcclusionProject/NaiveVersion/feature/Portrait/special_test_'
    myimage_path=[]
    myimage_path+=[os.path.join('/data2/xuyangf/OcclusionProject/NaiveVersion/PortraitImages/train',s) 
            for s in os.listdir('/data2/xuyangf/OcclusionProject/NaiveVersion/PortraitImages/train')]
    myimage_path=list(myimage_path)


tf.logging.set_verbosity(tf.logging.INFO)


# myimage_path=LoadImage2(cat)
# for Chen Liu Project


# image_path=[]
# for i in range(0,100):
#     catimage_path=LoadImage2(i)
#     catimage_path=list(catimage_path)
#     image_path+=catimage_path[:5]

# print(image_path)

image_path=[]
for mypath in myimage_path:
    myimg=cv2.imread(mypath, cv2.IMREAD_UNCHANGED)
    if(max(myimg.shape[0],myimg.shape[1])>100):
        image_path.append(mypath)

print(len(image_path))

mylayer=extract_layer
if len(extract_layer)>5:
    mylayer=extract_layer[:5]+'/'+extract_layer

extractor = FeatureExtractor(which_layer=mylayer, which_snapshot=0, from_scratch=False)


print('mytest: '+str(len(image_path)))
batch_num = 10
# batch_num=50
batch_size = int((len(image_path)-1)/10)+1
# batch_size=10
if len(image_path)==100:
    batch_size=10
print('batch_num: {0}'.format(batch_num))
check_num = 1  # save 1 batch to one file

step = int(0)
res=[]
irec = []
img_rec = []
originpath=[]
imageindex=[]
ggtmp=[]
for i in range (0,batch_num):
    print('batch :' + str(i))
    numwi=[]
    numhi=[]
    featurenumwi=[]
    featurenumhi=[]

    curr_paths = image_path[i * batch_size:(i + 1) * batch_size]
    features, images, blanks = extractor.extract_from_paths(curr_paths)
    images+=np.array([104., 117., 124.])
    tmp=features
    height, width = tmp.shape[1:3]

    # remove offset patches
    for j in range(0,len(curr_paths)):
        woffset=0
        hoffset=0
        if(blanks[j][0]>blanks[j][1]):
            woffset=int(np.ceil((np.array(blanks[j][0]).astype(float)+Apad)/Astride))
            hoffset=int(offset)
        else:
            hoffset=int(np.ceil((np.array(blanks[j][1]).astype(float)+Apad)/Astride))
            woffset=int(offset)

        print('batch'+ str(i)+'image'+str(j) )
        print(blanks[j])
        print(woffset)
        print(hoffset)
        #break
        print('j ='+str(j))
        temp = tmp[j:j+1, hoffset:height - hoffset, woffset:width - woffset, :]

        ntmp = np.transpose(temp, (3, 0, 1, 2))
        gtmp = ntmp.reshape(ntmp.shape[0], -1)
        ggtmp.append(deepcopy(gtmp))
        print(gtmp.shape)

        imageindex+=[j for ixx in range(gtmp.shape[1])]
        originpath+=[curr_paths[j] for ixx in range(gtmp.shape[1])]
        for ixx in range(gtmp.shape[1]):
            hhi,wwi=np.unravel_index(ixx, (height - 2 * hoffset, width - 2 * woffset))
            featurenumhi.append(hhi+hoffset)
            featurenumwi.append(wwi+woffset)
            phi = Astride * (hhi + hoffset) - Apad
            pwi = Astride * (wwi + woffset) - Apad
            numhi.append(phi)
            numwi.append(pwi)
            #numhi.append(hhi+hoffset)
            #numwi.append(wwi+woffset)
            if ixx==0:
                print(phi)
                print(pwi)
    # print(ggtmp)
    # ggtmp=np.array(ggtmp)
    # print(ggtmp.shape)

    # ggtmp=np.transpose(ggtmp,(1,0,2))
    # ggtmp=ggtmp.reshape(ggtmp.shape[0],-1)
    ggtmp=np.concatenate(ggtmp,axis=1)
    print(ggtmp.shape)
    res.append(deepcopy(ggtmp))
#numpy lie he bin de dao ggtemp
    irec += [i for ixx in range(ggtmp.shape[1])]

    if (i + 1) % check_num == 0 or i == batch_num - 1:
        print('output file {0}'.format(i / check_num))

        res = np.array(res)
        res = np.transpose(res, (1, 0, 2))
        itotal = res.shape[1]


        res = res.reshape(res.shape[0], -1)
        #break
        # should also save the loc_set
        loc_set = []
        img_set = []
        irec = np.array(irec)
        #batch_size_f = batch_size * (height - 2 * offset) * (width - 2 * offset)
        num=0
        for rr in range(res.shape[1]):
            ni=imageindex[rr]

            hi=numhi[rr]
            wi=numwi[rr]
            fhi=featurenumhi[rr]
            fwi=featurenumwi[rr]


            #img = img_rec[0][ni][hi:hi + Arf, wi:wi + Arf, :]
            # img_set.append(img)

            loc_set.append([i // check_num, irec[rr], ni, hi, wi, hi + Arf, wi + Arf,fhi,fwi])
            #if rr == rand_idx[50]:
            # print(loc_set)
        np.savez(savepath + str(i // check_num), res=np.asarray(res), loc_set=np.asarray(loc_set),
            originpath=np.asarray(originpath))
        res = []
        irec = []
        img_rec = []
        imgindex=[]
        originpath=[]
        ggtmp=[]
        imageindex=[]


print('all finish')