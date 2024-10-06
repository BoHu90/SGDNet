import torch.utils.data as data
from PIL import Image
import os
import os.path
import scipy.io
import numpy as np
import csv
from openpyxl import load_workbook
import time
import pandas as pd

class AVAFolder(data.Dataset):

    def __init__(self,istrain, root):

        images_inf = pd.read_csv('my_train_and_test.csv')
        # Index([ 'MOS', 'image_name', 'MLS','set_new', 'width', 'height', 'ID', 'threshold', 'mos_float'],
        sample = []
        if istrain:
            for i, item in enumerate(list(images_inf['set_new'])):
                if item == 'train':
                    lists = (images_inf['1'][i], images_inf['2'][i], images_inf['3'][i], images_inf['4'][i], images_inf['5'][i],
                            images_inf['6'][i], images_inf['7'][i], images_inf['8'][i], images_inf['9'][i], images_inf['10'][i])
                    sample.append((os.path.join(root,images_inf['image_name'][i]), lists,
                                  np.array(images_inf['mos_float'][i]).astype(np.float32)))
                # if i == 10000:
                #     break
            print('train数据：', len(sample))
        else:
            for i, item in enumerate(list(images_inf['set_new'])):
                if item == 'test':
                    lists = (
                    images_inf['1'][i], images_inf['2'][i], images_inf['3'][i], images_inf['4'][i], images_inf['5'][i],
                    images_inf['6'][i], images_inf['7'][i], images_inf['8'][i], images_inf['9'][i], images_inf['10'][i])
                    sample.append((os.path.join(root, images_inf['image_name'][i]), lists,
                                  np.array(images_inf['mos_float'][i]).astype(np.float32)))
                # if i == 10000:
                #     break
            print('test数据：', len(sample))
        # for i, item in enumerate(index):
        #     for aug in range(patch_num):
        #         sample.append((os.path.join(root, item[0]),item[1], np.array(item[2]).astype(np.float32)))

        self.samples = sample

    def __getitem__(self, index):
        #num = list(range(0,len(index)))
        path,p_tar, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, p_tar, target

    def __len__(self):
        length = len(self.samples)
        return length

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def main():
    path = '/data/niexixi/IAA_dataset/AVA/images'
    istrain=True
    data = AVAFolder(istrain=istrain, root=path)
    istrain = False
    data = AVAFolder(istrain=istrain, root=path)

if __name__ == '__main__':
    main()



