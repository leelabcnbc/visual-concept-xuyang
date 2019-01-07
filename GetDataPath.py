import json
import sys
import os

datapath=[]
datapath.append('/data2/xuyangf/OcclusionProject/NaiveVersion/CroppedImage')

catname=[]
for i in range(len(datapath)):
	cname=[]
	for s in os.listdir(datapath[i]):
		ss=s[:9]
		if ss not in cname:
			cname.append(ss)
	catname.append(cname)

def LoadImage(cat):
	image_path=[]
	for i in range(len(datapath)):
		print(len(catname[i]))
		print(catname[i][int(cat)])
		image_path+=[os.path.join(datapath[i],s) 
		for s in os.listdir(datapath[i]) 
		if ( catname[i][int(cat)] in s ) ]
	return image_path

datapath2=[]
datapath2.append('/data2/haow3/data/imagenet/dataset/val_crop_0')
def LoadImage2(cat):
	image_path=[]
	for i in range(len(datapath2)):
		image_path+=[os.path.join(datapath2[i],s) 
		for s in os.listdir(datapath2[i]) 
		if ( catname[i][int(cat)] in s ) ]
	return image_path

