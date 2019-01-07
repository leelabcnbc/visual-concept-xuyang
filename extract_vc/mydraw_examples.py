import pickle
import numpy as np
import cv2
import math
from sys import argv

cat=argv[1]
fname='/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/prunL3/dictionary_'+cat+'.pickle'
with open(fname,'rb') as fh:
    assignment, centers, example, __= pickle.load(fh)

ss=int(math.sqrt(example[0].shape[0]/3))
print ss

# cluster_num=len(centers)
# print(cluster_num)

# fname ='/data2/xuyangf/OcclusionProject/NaiveVersion/new_vc_score/Portrait/portrait_score.npz'
# ff=np.load(fname)
# img_vc=ff['vc_score']
# vc_num=len(img_vc[0])
# img_num=len(img_vc)
# #print(img_vc[0])
# img_vc_avg=[]
# print(img_vc)

# for i in range(vc_num):
#     img_vc_avg.append(float(np.sum(img_vc[np.where(img_vc[:,i]!=-1),i]))/img_num)
# img_vc_avg=np.asarray(img_vc_avg)
# rindexsort=np.argsort(-img_vc_avg)


# print(img_vc_avg)
# # rows=int((cluster_num-1)/20)+1

big_img = np.zeros((5+(ss+5)*8, 5+(ss+5)*20, 3))

# selected_vc=[7,23,27,28,37,38,42,44,47,52,56,58,62,70,75,78,87,109,110,118]
for i in range(8):
	for j in range(20):

		rnum=5+i*(ss+5)
		cnum=5+j*(ss+5)
		# select_vc=rindexsort[i*20+j]
		big_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[i*20+j][:,0].reshape(ss,ss,3).astype(int)

fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/example/tmpexample/pool3_example.png'
cv2.imwrite(fname, big_img)

####################
fname='/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/prunL4/dictionary_'+cat+'.pickle'
with open(fname,'rb') as fh:
    assignment, centers, example, __= pickle.load(fh)

ss=int(math.sqrt(example[0].shape[0]/3))
print ss
big_img = np.zeros((5+(ss+5)*8, 5+(ss+5)*20, 3))

# selected_vc=[7,23,27,28,37,38,42,44,47,52,56,58,62,70,75,78,87,109,110,118]
for i in range(8):
	for j in range(20):

		rnum=5+i*(ss+5)
		cnum=5+j*(ss+5)
		# select_vc=rindexsort[i*20+j]
		big_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[i*20+j][:,0].reshape(ss,ss,3).astype(int)

fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/example/tmpexample/pool4_example.png'
cv2.imwrite(fname, big_img)
################
fname='/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/prunLpool2/dictionary_'+cat+'.pickle'
with open(fname,'rb') as fh:
    assignment, centers, example, __= pickle.load(fh)

ss=int(math.sqrt(example[0].shape[0]/3))
print ss
big_img = np.zeros((5+(ss+5)*5, 5+(ss+5)*20, 3))

# selected_vc=[7,23,27,28,37,38,42,44,47,52,56,58,62,70,75,78,87,109,110,118]
for i in range(5):
	for j in range(20):

		rnum=5+i*(ss+5)
		cnum=5+j*(ss+5)
		# select_vc=rindexsort[i*20+j]
		big_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[i*20+j][:,0].reshape(ss,ss,3).astype(int)

fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/example/tmpexample/pool2_example.png'
cv2.imwrite(fname, big_img)



# for ii in range(len(example)):
#     big_img = np.zeros((10+(ss+10)*4, 10+(ss+10)*5, 3))
#     for iis in range(20):
#         if iis >= example[ii].shape[1]:
#             continue
#         aa = iis//5
#         bb = iis%5
#         rnum = 10+aa*(ss+10)
#         cnum = 10+bb*(ss+10)
#         big_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[ii][:,iis].reshape(ss,ss,3).astype(int)

#     fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/example/example'+cat+'/'+ str(ii) + '.png'
#     cv2.imwrite(fname, big_img)
#     print 'draw finish'

