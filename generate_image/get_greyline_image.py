from __future__ import division
import pickle,os
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
# size=argv[2]
# x_number=argv[3]
# y_number=argv[4]
Occlusion_level=argv[2]
frequency=argv[3]

VGG_MEAN = [104., 117., 124.]

myimage_path=LoadImage2(cat)
image_path=[]
for mypath in myimage_path:
	myimg=cv2.imread(mypath, cv2.IMREAD_UNCHANGED)
	if(max(myimg.shape[0],myimg.shape[1])>100):
		image_path.append(mypath)
img_num=len(image_path)

# val_greyline_name='/data2/xuyangf/OcclusionProject/NaiveVersion/ApertureImage/val/bubble_image/greyline_images/greyline_'+size+'_'+x_number+'_'+y_number+'/'
val_greyline_name='/data2/xuyangf/OcclusionProject/NaiveVersion/ApertureImage/val/bubble_image/greyline_images/newgreyline_'+Occlusion_level+'_'+frequency+'/'

if not os.path.exists(val_greyline_name):
	os.mkdir(val_greyline_name)

# x_number=int(x_number)
# y_number=int(y_number)
# size=int(size)

size=(112*float(Occlusion_level)/100)/float(frequency)
x_number=int(frequency)*2
y_number=0

for n in range(0,img_num):
	original_img=cv2.imread(image_path[n], cv2.IMREAD_UNCHANGED)
	oimage,_,__=process_image(original_img, '_',0)
	oimage+=np.array([104., 117., 124.])
	# center=224/(x_number+1)
	center=224/x_number
	if x_number>0:
		for i in range(x_number):
			tmpimg,tmpPDF=generate_greyline_image(oimage,'x',center*(i+1)-size/2,3,size,224)
			part_top10_aperture_PDF=tmpPDF
			part_top10_aperture_img=(1-part_top10_aperture_PDF )*oimage
			back = VGG_MEAN*part_top10_aperture_PDF*np.ones((224,224,3))
			oimage = part_top10_aperture_img + back

	center=224/(y_number+1)
	if y_number>0:
		for i in range(y_number):
			tmpimg,tmpPDF=generate_greyline_image(oimage,'y',center*(i+1),5,size,224)
			part_top10_aperture_PDF=tmpPDF
			part_top10_aperture_img=(1-part_top10_aperture_PDF )*oimage
			back = VGG_MEAN*part_top10_aperture_PDF*np.ones((224,224,3))
			oimage = part_top10_aperture_img + back
	findex=image_path[n].rfind('/')+1
	ffindex=image_path[n].rfind('.')
	fname=val_greyline_name+image_path[n][findex:ffindex]+'.jpeg'
	cv2.imwrite(fname,oimage)