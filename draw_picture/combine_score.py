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

filename='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/cat'+str(cat)+'.npz'
fff=np.load(filename)
myavg=fff['vc_score']
myavg=np.array(myavg)
single_vc_num=len(myavg[0])
my_vc_avg=[]
for i in range(single_vc_num):
	my_vc_avg.append(float(np.sum(myavg[np.where(myavg[:,i]!=-1),i]))/single_vc_num)
my_vc_avg=np.asarray(my_vc_avg)

b=np.argsort(-my_vc_avg)
indexmap=np.zeros(single_vc_num)
for i in range(single_vc_num):
	indexmap[b[i]]=i

fname ='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_combinescore/2vc/cat'+str(cat)+'.npz'
ff=np.load(fname)
img_vc_avg=ff['vc_score']
print('my')
print(len(img_vc_avg))
vc_name=ff['vc']
vc_num=len(img_vc_avg)
print(vc_num)
print('now')
img_vc_avg=np.array(img_vc_avg)
vc_name=np.array(vc_name)
indexsort=np.argsort(img_vc_avg)
img_vc_avg=np.asarray(img_vc_avg)
rindexsort=np.argsort(-img_vc_avg)

big_img = np.zeros((10*10,10*single_vc_num,3))
for i in range(vc_num):
	myindex=indexsort[vc_num-1-i]
	x=int(vc_name[myindex][0])
	y=int(vc_name[myindex][1])
	x=int(indexmap[x])
	y=int(indexmap[y])
	#big_img[10:20,10:20,0]=10
	big_img[10*x:10*(x+1),10*y:10*(y+1),:]=[255-int(float(i*255)/vc_num),0,int(float(i*255)/vc_num)]
	#big_img[10*x:10*(x+1),10*y:10*(y+1),1]=0
	#big_img[10*x:10*(x+1),10*y:10*(y+1),:]=int(float(i*255)/vc_num)

	if y<10:
		big_img[10*y:10*(y+1),10*x:10*(x+1),:]=[255-int(float(i*255)/vc_num),0,int(float(i*255)/vc_num)]

savename = '/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/ImportanceExample/'+cat+'/2vc_colored_table.png'
cv2.imwrite(savename, big_img)
print('colored image draw finish')

print(indexsort.shape)
img_vc_avg=list(img_vc_avg)
x=[i for i in range(len(img_vc_avg))]
img_vc_avg.sort(reverse=True)

ffname ='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_acc_score/2vc_cat'+str(cat)+'.npz'
fff=np.load(ffname)
img_acc_vc=fff['vc_score']
print('my')
print(len(img_acc_vc))
img_acc_vc=np.asarray(img_acc_vc)
img_acc_vc=img_acc_vc[rindexsort]

# show curve
plt.figure()  
plt.plot(x,img_vc_avg,'r-')  
plt.plot(x,img_acc_vc,'b-')
savedir='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/ImportanceExample/'+cat+'/2vc_ImportanceCurve.png'
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
top_img = np.zeros((3+(ss+3)*2, 20+(ss+20)*4, 3))
middle_img = np.zeros((3+(ss+3)*2, 20+(ss+20)*4, 3))
last_img = np.zeros((3+(ss+3)*2, 20+(ss+20)*4, 3))

for i in range(0,4):
	aa = i//4
	bb = i%4	
	rnum = 3+aa*(ss+3)
	cnum = 20+bb*(ss+20)
	#top10
	myindex=indexsort[vc_num-1-i]
	myindex=int(vc_name[myindex][0])
	top_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[myindex][:,0].reshape(ss,ss,3).astype(int)
	print('top information')
	print(indexsort[vc_num-1-i])
	print(len(np.where(assignment==(indexsort[vc_num-1-i]))[0]))
	#middle
	# random_index=np.random.randint(100,200)
	# middle_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[indexsort[random_index]][:,0].reshape(ss,ss,3).astype(int)
	#last 10
	myindex=indexsort[i]
	myindex=int(vc_name[myindex][0])
	last_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[myindex][:,0].reshape(ss,ss,3).astype(int)

for i in range(0,4):
	aa = i//4+1
	bb = i%4	
	rnum = 3+aa*(ss+3)
	cnum = 20+bb*(ss+20)
	#top10
	myindex=indexsort[vc_num-1-i]
	myindex=int(vc_name[myindex][1])
	top_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[myindex][:,1].reshape(ss,ss,3).astype(int)
	print('top information')
	print(indexsort[vc_num-1-i])
	print(len(np.where(assignment==(indexsort[vc_num-1-i]))[0]))
	#middle
	# random_index=np.random.randint(100,200)
	# middle_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[indexsort[random_index]][:,0].reshape(ss,ss,3).astype(int)
	#last 10
	myindex=indexsort[i]
	myindex=int(vc_name[myindex][1])
	last_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[myindex][:,1].reshape(ss,ss,3).astype(int)


fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/ImportanceExample/'+cat+'/2vc_top.png'
cv2.imwrite(fname, top_img)
# fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/ImportanceExample/'+cat+'/middle.png'
# cv2.imwrite(fname, middle_img)
fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/ImportanceExample/'+cat+'/2vc_last.png'
cv2.imwrite(fname, last_img)

print('draw finish')

