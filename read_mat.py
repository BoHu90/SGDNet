import scipy.io
import os

#root = '/home/dataset/AADB'
#mat_l = scipy.io.loadmat(os.path.join(root, 'AADBinfo.mat'))
#print(mat_l)
#'testNameList', 'testScore', 'trainNameList', 'trainScore'

# print(mat['testNameList'][0][1])
# print(mat['testScore'][0][1])
# print(mat['trainNameList'][0][1])
# print(mat['trainScore'][0][1])
# print(mat.keys())

#mat = scipy.io.loadmat(os.path.join(root, 'attMat.mat'))
#'C', 'ans', 'attID', 'attributeNames', 'dataset', 'fid', 'filename', 'imgLabel', 'lineID', 'numImage', 'phaseID', 'phaseNames', 'tline'
# print(mat['C'][0][0])#['farm1_297_19996731909_b41049920b_b.jpg']
# print(len(mat['C']))
#print(mat['ans'])#[[0]]
# print(mat['attID'])#[[12]]
# print(mat['attributeNames'].shape)#(1,12)
# print(mat['dataset'].shape)
# print(mat['fid'])
# print(mat['filename'])
# print(mat['imgLabel'])
# print(mat['lineID'])
# print(mat['numImage'])
# print(mat['phaseID'])
# print(mat['phaseNames'])
# print(mat['tline'])
# print(mat)

# coding:utf-8
aa = []

with open("/home/dataset/AADB/imgListTestNewRegression_score.txt", "r") as f:
    for line in f.readlines():
        temp = line.strip().split()
        #data = data.split(' ')
        print(temp)
        break

