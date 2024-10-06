import torch.utils.data as data
from PIL import Image
import os
import os.path
import scipy.io
import numpy as np
import csv
from openpyxl import load_workbook
import pandas as pd
import random
import torchvision


class LIVEFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):

        refpath = os.path.join(root, 'refimgs')
        refname = getFileName(refpath, '.bmp')

        jp2kroot = os.path.join(root, 'jp2k')
        jp2kname = self.getDistortionTypeFileName(jp2kroot, 227)

        jpegroot = os.path.join(root, 'jpeg')
        jpegname = self.getDistortionTypeFileName(jpegroot, 233)

        wnroot = os.path.join(root, 'wn')
        wnname = self.getDistortionTypeFileName(wnroot, 174)

        gblurroot = os.path.join(root, 'gblur')
        gblurname = self.getDistortionTypeFileName(gblurroot, 174)

        fastfadingroot = os.path.join(root, 'fastfading')
        fastfadingname = self.getDistortionTypeFileName(fastfadingroot, 174)  # 获得没一张图像的路径
        # [0:227] [227:460] [460:634] [634:808] [808:982] 分别为五种失真类型的索引，可单独训练每一种失真类型
        imgpath = jp2kname + jpegname + wnname + gblurname + fastfadingname
        imgpath = imgpath[227:460]
        dmos = scipy.io.loadmat(os.path.join(root, 'dmos_realigned.mat'))
        labels = dmos['dmos_new'][:, 227:460].astype(np.float32)
        orgs = dmos['orgs'][:, 227:460]
        refnames_all = scipy.io.loadmat(os.path.join(root, 'refnames_all.mat'))
        refnames_all = refnames_all['refnames_all'][:, 227:460]
        sample = []

        for i in range(0, len(index)):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = train_sel * ~orgs.astype(np.bool_)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[1].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((imgpath[item], labels[0][item]))
        print(len(sample))
        print(sample)
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

    def getDistortionTypeFileName(self, path, num):
        filename = []
        index = 1
        for i in range(0, num):
            name = '%s%s%s' % ('img', str(index), '.bmp')
            filename.append(os.path.join(path, name))
            index = index + 1
        return filename


class TID2013Folder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        refpath = os.path.join(root, 'reference_images')
        refname = getTIDFileName(refpath, '.bmp.BMP')
        txtpath = os.path.join(root, 'mos_with_names.txt')
        fh = open(txtpath, 'r')
        imgnames = []

        target = []
        refnames_all = []
        refnames_all2 = []
        for line in fh:
            line = line.split('\n')
            words = line[0].split()
            imgnames.append((words[1]))
            target.append(words[0])
            ref_temp = words[1].split("_")
            refnames_all.append(ref_temp[0][1:])
            refnames_all2.append(ref_temp[0] + "_" + ref_temp[1])

        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)
        sample = []
        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)  # 返回参考图像对应的失真图的全部索引
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(1):
                    sample.append((os.path.join(root, 'distorted_images', imgnames[item]), labels[item]))

        # 0:5 表示每个参考图对应的第一种失真类型 单独训练每一种失真类型  下面的分段表示没一种失真类型，一种24种
        # 0:5  5:10  10:15  15:20 20:25 25:30 30:35 35:40 40:45 45:50 50:55
        # 55:60 60:65  65:70  70:75  75:80  80:85  85:90  90:95  95:100  100:105
        # 105:110  110:115  115:120
        sample = np.array(sample)
        print(sample.shape)
        imgnames2 = sample[:, 0].reshape(len(index), 120)[:, 115:120]
        labels2 = sample[:, 1].reshape(len(index), 120)[:, 115:120]
        imgnames2 = imgnames2.reshape(imgnames2.size, ).tolist()
        labels2 = labels2.reshape(labels2.size, ).tolist()
        temp1 = []
        for item in imgnames2:
            for _ in range(patch_num):
                temp1.append(item)
        temp2 = []
        for item in labels2:
            for _ in range(patch_num):
                temp2.append(item)
        sample = np.c_[temp1, temp2]
        print(sample)
        print(sample.shape)
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename


def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


if __name__ == '__main__':
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
    ])

    for i in range(1):
        sel_num = list(range(0, 25))
        random.shuffle(sel_num)
        train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
        test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]

        # data = MDFolder(root='D:\PythonProject\hyperIQA\data\MD_Uniform_Database',
        #                 file_dir="D:\PythonProject\hyperIQA\data\\name_mos_reference_uniform1119.csv", index=train_index,
        #                 transform=transforms, patch_num=25, istrain=True, i=i)
        # data2 = MDFolder(root='D:\PythonProject\hyperIQA\data\MD_Uniform_Database',
        #                 file_dir="D:\PythonProject\hyperIQA\data\\name_mos_reference_uniform1119.csv", index=test_index,
        #                 transform=transforms, patch_num=25, istrain=False, i=i)
        # data = TID2013Folder(root="D:\PythonProject\\tid2013", index=test_index, transform=transforms, patch_num=1)
        data = LIVEFolder(root="D:\PythonProject\live", index=[1, 2], transform=transforms, patch_num=2)
        # print(data[24][0].show(), data[24][1])
    pass
