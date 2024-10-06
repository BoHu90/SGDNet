import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import math
import argparse
import random
import modl2,assessment
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import time
from PIL import Image

parser = argparse.ArgumentParser(description="Deep Multi-Patch Hierarchical Network")
parser.add_argument("-e","--epochs",type = int, default = 2600)
parser.add_argument("-se","--start_epoch",type = int, default = 0)
parser.add_argument("-b","--batchsize",type = int, default = 2)
parser.add_argument("-s","--imagesize",type = int, default = 256)
parser.add_argument("-l","--learning_rate", type = float, default = 0.0001)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()

#Hyper Parameters
METHOD = "FCT2"
SAMPLE_DIR = "test_samples"
EXPDIR = "FCT2_test_res"
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
GPU = args.gpu
BATCH_SIZE = args.batchsize
IMAGE_SIZE = args.imagesize

def save_images(images, name):
    filename = './test_results/' + EXPDIR + "/" + name
    torchvision.utils.save_image(images, filename)


def main():
    depths = [2, 2, 2, 2, 2, 2, 2, 2, 2]#768 384
    encoder_lv1 = modl2.Uformer(img_size=768, embed_dim=16, depths=depths,
                                win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True,
                                shift_flag=False, feature=0)
    encoder_lv2 = modl2.Uformer(img_size=384, embed_dim=16, depths=depths,
                                win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True,
                                shift_flag=False, feature=3)
    encoder_lv3 = modl2.Uformer(img_size=384, embed_dim=16, depths=depths,
                                win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True,
                                shift_flag=False, feature=4)

    if os.path.exists(str('./' + METHOD + "/encoder_lv1.pkl")):
        encoder_lv1.load_state_dict(torch.load(str('./' + METHOD + "/encoder_lv1.pkl")))
        print("load encoder_lv1 success")
    if os.path.exists(str('./' + METHOD + "/encoder_lv2.pkl")):
        encoder_lv2.load_state_dict(torch.load(str('./' + METHOD + "/encoder_lv2.pkl")))
        print("load encoder_lv2 success")
    if os.path.exists(str('./' + METHOD + "/encoder_lv3.pkl")):
        encoder_lv3.load_state_dict(torch.load(str('./' + METHOD + "/encoder_lv3.pkl")))
        print("load encoder_lv3 success")
    encoder_lv1.cuda(GPU)  # apply直接把数据放入函数进行操作
    encoder_lv2.cuda(GPU)
    encoder_lv3.cuda(GPU)

    if os.path.exists('./test_results/' + EXPDIR) == False:
        os.system('mkdir ./test_results/' + EXPDIR)
            
    iteration = 0.0
    test_time = 0.0
    for images_name in os.listdir(SAMPLE_DIR):
        with torch.no_grad():
            images_lv1 = transforms.ToTensor()(Image.open(SAMPLE_DIR + '/' + images_name).convert('RGB'))
            images_lv1 = Variable(images_lv1 - 0.5).unsqueeze(0).cuda(GPU)
            start = time.time()
            print(images_lv1.size())
            # ZeroPad = nn.ZeroPad2d(padding=(0,0,48, 0))
            # images_lv1 = ZeroPad(images_lv1)

            H = images_lv1.size(2)
            W = images_lv1.size(3)

            images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
            images_lv2_2 = images_lv1[:,:,int(H/2):H,:]
            images_lv3_1 = images_lv2_1[:,:,:,0:int(W/2)]
            images_lv3_2 = images_lv2_1[:,:,:,int(W/2):W]
            images_lv3_3 = images_lv2_2[:,:,:,0:int(W/2)]
            images_lv3_4 = images_lv2_2[:,:,:,int(W/2):W]

            # print(images_lv1.size())
            f31, feature_lv3_1 = encoder_lv3(images_lv3_1)
            f32, feature_lv3_2 = encoder_lv3(images_lv3_2)
            f33, feature_lv3_3 = encoder_lv3(images_lv3_3)
            f34, feature_lv3_4 = encoder_lv3(images_lv3_4)
            feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 1)
            feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 1)

            # feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
            residual_lv3_top = torch.cat((f31, f32), 3)
            residual_lv3_bot = torch.cat((f33, f34), 3)

            f21, feature_lv2_1 = encoder_lv2(x=images_lv2_1 + residual_lv3_top, feature=feature_lv3_top)
            # print((images_lv2_1 + residual_lv3_top).size())
            f22, feature_lv2_2 = encoder_lv2(x=images_lv2_2 + residual_lv3_bot, feature=feature_lv3_bot)
            feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 1)
            residual_lv2 = torch.cat((f21, f22), 2)

            # print('f31',f31.size())
            # print('f22',f22.size())
            # print('2*f22',residual_lv2.size())
            # print(images_lv1.size())
            # break
            deblur_image, feature_lv1 = encoder_lv1(images_lv1 + residual_lv2, feature=feature_lv2)

            stop = time.time()
            test_time += stop-start
            print('RunTime:%.4f'%(stop-start), '  Average Runtime:%.4f'%(test_time/(iteration+1)))
            # psnr = assessment.psnr(deblur_image, gt).item()
            # ssim = assessment.ssim(deblur_image, gt).item()
            # psnr += psnr
            # ssim += ssim
            # print("psnr:", psnr / (iteration + 2), "ssim:", ssim / (iteration + 2))
            save_images(deblur_image.data + 0.5, images_name)
            iteration += 1
            
if __name__ == '__main__':
    main()