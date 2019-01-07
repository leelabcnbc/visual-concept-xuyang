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
mylayer=argv[2]
fname='/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/prunL'+mylayer+'/dictionary_'+cat+'.pickle'
with open(fname,'rb') as fh:
    assignment, centers, example,_ = pickle.load(fh)


fname ='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/layer'+mylayer+'/cat'+str(cat)+'.npz'
ff=np.load(fname)
img_vc=ff['vc_score']
vc_num=len(img_vc[0])
img_num=len(img_vc)
#print(img_vc[0])
img_vc_avg=[]
for i in range(vc_num):
	img_vc_avg.append(float(np.sum(img_vc[np.where(img_vc[:,i]!=-1),i]))/img_num)

#img_vc_avg=[int(1000*i) for i in img_vc_avg]
# ffname ='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_acc_score/cat'+str(cat)+'.npz'
# fff=np.load(ffname)
# img_acc_vc=fff['vc_acc_score']
# img_acc_vc_avg=[]
# for i in range(vc_num):
# 	img_acc_vc_avg.append(np.average(img_acc_vc[np.where(img_vc[:,i]!=0),i]))
# img_acc_vc_avg=np.asarray(img_acc_vc_avg)

indexsort=np.argsort(img_vc_avg)
img_vc_avg=np.asarray(img_vc_avg)
rindexsort=np.argsort(-img_vc_avg)
print(rindexsort)
# img_acc_vc_avg=img_acc_vc_avg[rindexsort]

x=[i for i in range(len(img_vc_avg))]
img_vc_avg=list(img_vc_avg)
img_vc_avg.sort(reverse=True)


# show curve
plt.figure()
plt.title('importance measurement curve')
plt.xlabel("Ranked Visual Concepts")
plt.ylabel("drop in the probability of the target class")
plt.plot(x,img_vc_avg,'r-')
#plt.plot(x,img_acc_vc_avg,'b-')
savedir='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/layer'+mylayer+'/ImportanceExample/'+cat+'/ImportanceCurve.png'
plt.savefig(savedir) 

img_vc_avg=img_vc_avg.sort()
#show share curve
# plt.title("the relationship between shared times and importance") 
# plt.xlim(xmax=int(img_vc_avg[0])+1,xmin=int(img_vc_avg[vc_num-1])-1)
# plt.ylim(ymax=100,ymin=0)
# plt.plot(img_vc_avg,share_times,'ro')
# savedir='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/ImportanceExample/'+cat+'/Importance_share_Curve.png'
# plt.savefig(savedir) 

#show images

ss=int(math.sqrt(example[0].shape[0]/3))
row=int(vc_num/5)+1
top_img = np.zeros((10+(ss+10)*row, 10+(ss+10)*5, 3))


for i in range(0,vc_num):
	aa = i//5
	bb = i%5	
	rnum = 10+aa*(ss+10)
	cnum = 10+bb*(ss+10)
	#top10
	top_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[indexsort[vc_num-1-i]][:,0].reshape(ss,ss,3).astype(int)
	print('top information')
	print(indexsort[vc_num-1-i])
	print(len(np.where(assignment==(indexsort[vc_num-1-i]))[0]))

fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/layer'+mylayer+'/ImportanceExample/'+cat+'/all.png'
cv2.imwrite(fname, top_img)

ss=int(math.sqrt(example[0].shape[0]/3))
top_img = np.zeros((10+(ss+10)*2, 10+(ss+10)*5, 3))
middle_img = np.zeros((10+(ss+10)*2, 10+(ss+10)*5, 3))
last_img = np.zeros((10+(ss+10)*2, 10+(ss+10)*5, 3))

for i in range(0,10):
	aa = i//5
	bb = i%5	
	rnum = 10+aa*(ss+10)
	cnum = 10+bb*(ss+10)
	#top10
	top_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[indexsort[vc_num-1-i]][:,0].reshape(ss,ss,3).astype(int)
	print('top information')
	print(indexsort[vc_num-1-i])
	print(len(np.where(assignment==(indexsort[vc_num-1-i]))[0]))
	#middle
	random_index=np.random.randint(20,vc_num-20)
	middle_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[indexsort[random_index]][:,0].reshape(ss,ss,3).astype(int)
	#last 10
	last_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[indexsort[i]][:,0].reshape(ss,ss,3).astype(int)

fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/layer'+mylayer+'/ImportanceExample/'+cat+'/top.png'
cv2.imwrite(fname, top_img)
fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/layer'+mylayer+'/ImportanceExample/'+cat+'/middle.png'
cv2.imwrite(fname, middle_img)
fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/layer'+mylayer+'/ImportanceExample/'+cat+'/last.png'
cv2.imwrite(fname, last_img)

print('draw finish')



#share part

