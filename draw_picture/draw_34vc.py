from __future__ import division
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import Image, display
import pickle
import numpy as np
from sys import argv
from GetDataPath import *
from sys import argv
import math

cat=argv[1]

fname='/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/prunL3/dictionary_'+cat+'.pickle'
with open(fname,'rb') as fh:
    assignment, centers, example,_ = pickle.load(fh)

fname ='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_combinescore/4vc/cat'+str(cat)+'.npz'
ff=np.load(fname)
img_vc_avg=ff['vc_score']
vc_name=ff['vc']
vc_num=len(img_vc_avg)
print(vc_num)
img_vc_avg=np.array(img_vc_avg)
vc_name=np.array(vc_name)
indexsort=np.argsort(img_vc_avg)

ss=44
top_img = np.zeros((3+(ss+3)*4, 20+(ss+20)*4, 3))
middle_img = np.zeros((3+(ss+3)*4, 20+(ss+20)*4, 3))
last_img = np.zeros((3+(ss+3)*4, 20+(ss+20)*4, 3))

for i in range(0,4):
	aa = i//4
	bb = i%4	
	rnum = 3+aa*(ss+3)
	cnum = 20+bb*(ss+20)
	#top10
	myindex=indexsort[vc_num-1-i]
	s=vc_name[myindex]
	dot1=s.find('_')
	k1=int(s[:dot1])
	myindex=k1
	top_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[myindex][:,0].reshape(ss,ss,3).astype(int)
	print('top information')
	print(indexsort[vc_num-1-i])
	print(len(np.where(assignment==(indexsort[vc_num-1-i]))[0]))
	#middle
	# random_index=np.random.randint(100,200)
	# middle_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[indexsort[random_index]][:,0].reshape(ss,ss,3).astype(int)
	#last 10
	myindex=indexsort[i]
	s=vc_name[myindex]
	dot1=s.find('_')
	k1=int(s[:dot1])
	myindex=k1
	last_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[myindex][:,0].reshape(ss,ss,3).astype(int)

for i in range(0,4):
	aa = i//4+1
	bb = i%4	
	rnum = 3+aa*(ss+3)
	cnum = 20+bb*(ss+20)
	#top10
	myindex=indexsort[vc_num-1-i]
	s=vc_name[myindex]
	dot1=s.find('_')
	k1=int(s[:dot1])
	s2=s[dot1+1:]
	dot2=s2.find('_')
	k2=int(s2[:dot2])
	myindex=k2
	top_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[myindex][:,0].reshape(ss,ss,3).astype(int)
	print('top information')
	print(indexsort[vc_num-1-i])
	print(len(np.where(assignment==(indexsort[vc_num-1-i]))[0]))
	#middle
	# random_index=np.random.randint(100,200)
	# middle_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[indexsort[random_index]][:,0].reshape(ss,ss,3).astype(int)
	#last 10
	myindex=indexsort[i]
	s=vc_name[myindex]
	dot1=s.find('_')
	k1=int(s[:dot1])
	s2=s[dot1+1:]
	dot2=s2.find('_')
	k2=int(s2[:dot2])
	myindex=k2
	last_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[myindex][:,0].reshape(ss,ss,3).astype(int)

for i in range(0,4):
	aa = i//4+2
	bb = i%4	
	rnum = 3+aa*(ss+3)
	cnum = 20+bb*(ss+20)
	#top10
	myindex=indexsort[vc_num-1-i]
	s=vc_name[myindex]
	dot1=s.find('_')
	k1=int(s[:dot1])
	s2=s[dot1+1:]
	dot2=s2.find('_')
	k2=int(s2[:dot2])
	s3=s2[dot2+1:]
	dot3=s3.find('_')
	k3=int(s3[:dot3])
	myindex=k3
	top_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[myindex][:,0].reshape(ss,ss,3).astype(int)
	print('top information')
	print(indexsort[vc_num-1-i])
	print(len(np.where(assignment==(indexsort[vc_num-1-i]))[0]))
	#middle
	# random_index=np.random.randint(100,200)
	# middle_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[indexsort[random_index]][:,0].reshape(ss,ss,3).astype(int)
	#last 10
	myindex=indexsort[i]
	s=vc_name[myindex]
	dot1=s.find('_')
	k1=int(s[:dot1])
	s2=s[dot1+1:]
	dot2=s2.find('_')
	k2=int(s2[:dot2])
	s3=s2[dot2+1:]
	dot3=s3.find('_')
	k3=int(s3[:dot3])
	myindex=k3
	last_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[myindex][:,0].reshape(ss,ss,3).astype(int)

for i in range(0,4):
	aa = i//4+3
	bb = i%4	
	rnum = 3+aa*(ss+3)
	cnum = 20+bb*(ss+20)
	#top10
	myindex=indexsort[vc_num-1-i]
	s=vc_name[myindex]
	dot1=s.find('_')
	k1=int(s[:dot1])
	s2=s[dot1+1:]
	dot2=s2.find('_')
	k2=int(s2[:dot2])
	s3=s2[dot2+1:]
	dot3=s3.find('_')
	k3=int(s3[:dot3])
	s4=s3[dot3+1:]
	k4=int(s4)
	myindex=k4
	top_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[myindex][:,0].reshape(ss,ss,3).astype(int)
	print('top information')
	print(indexsort[vc_num-1-i])
	print(len(np.where(assignment==(indexsort[vc_num-1-i]))[0]))
	#middle
	# random_index=np.random.randint(100,200)
	# middle_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[indexsort[random_index]][:,0].reshape(ss,ss,3).astype(int)
	#last 10
	myindex=indexsort[i]
	s=vc_name[myindex]
	dot1=s.find('_')
	k1=int(s[:dot1])
	s2=s[dot1+1:]
	dot2=s2.find('_')
	k2=int(s2[:dot2])
	s3=s2[dot2+1:]
	dot3=s3.find('_')
	k3=int(s3[:dot3])
	s4=s3[dot3+1:]
	k4=int(s4)
	myindex=k4
	last_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[myindex][:,0].reshape(ss,ss,3).astype(int)

fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/ImportanceExample/'+cat+'/4vc_top.png'
cv2.imwrite(fname, top_img)
# fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/ImportanceExample/'+cat+'/middle.png'
# cv2.imwrite(fname, middle_img)
fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/ImportanceExample/'+cat+'/4vc_last.png'
cv2.imwrite(fname, last_img)

print('draw finish')
