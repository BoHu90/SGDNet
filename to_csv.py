# '''
# 通过创建writer对象写入（每次写入一行）
# '''
# import csv
#
# header = ['name', 'score']
# names = ['DI4A1127_AGUTIR.PNG', 'DI4A1236_AGUTIR.PNG', 'DI4A1117_AGUTIR.PNG', 'DI4A1110_AGUTIR.PNG', 'DI4A1141_AGUTIR.PNG', 'DI4A1115_AGUTIR.PNG', 'DI4A1151_AGUTIR.PNG', 'DI4A1258_AGUTIR.PNG', 'DI4A1156_AGUTIR.PNG', 'DI4A1257_AGUTIR.PNG', 'DI4A1102_AGUTIR.PNG', 'DI4A1117_MPRNet.png', 'DI4A1156_MPRNet.png', 'DI4A1258_MPRNet.png', 'DI4A1127_Stripformer.png', 'DI4A1156_Stripformer.png', 'DI4A1141_MPRNet.png', 'DI4A1115_Stripformer.png', 'DI4A1236_Stripformer.png', 'DI4A1236_MPRNet.png', 'DI4A1127_MPRNet.png', 'DI4A1141_Stripformer.png', 'DI4A1257_MPRNet.png', 'DI4A1102_MPRNet.png', 'DI4A1117_Stripformer.png', 'DI4A1258_Stripformer.png', 'DI4A1102_Stripformer.png', 'DI4A1257_Stripformer.png', 'DI4A1151_MPRNet.png', 'DI4A1151_Stripformer.png', 'DI4A1115_MPRNet.png', 'DI4A1110_MPRNet.png', 'DI4A1110_Stripformer.png', 'DI4A1151_BANet.JPG', 'DI4A1127_BANet.JPG', 'DI4A1156.JPG', 'DI4A1127_HINet.JPG', 'DI4A1102_BANet.JPG', 'DI4A1102_HINet.JPG', 'DI4A1117.JPG', 'DI4A1236_BANet.JPG', 'DI4A1115_BANet.JPG', 'DI4A1127.JPG', 'DI4A1115.JPG', 'DI4A1102.JPG', 'DI4A1258_BANet.JPG', 'DI4A1110_BANet.JPG', 'DI4A1156_HINet.JPG', 'DI4A1141_HINet.JPG', 'DI4A1258_HINet.JPG', 'DI4A1115_HINet.JPG', 'DI4A1151_HINet.JPG', 'DI4A1258.JPG', 'DI4A1141_BANet.JPG', 'DI4A1257_HINet.JPG', 'DI4A1110.JPG', 'DI4A1110_HINet.JPG', 'DI4A1257_BANet.JPG', 'DI4A1117_BANet.JPG', 'DI4A1151.JPG', 'DI4A1257.JPG', 'DI4A1117_HINet.JPG', 'DI4A1236_HINet.JPG', 'DI4A1141.JPG', 'DI4A1156_BANet.JPG', 'DI4A1236.JPG']
# score = [38.64810867, 55.33884735, 24.91416512, 57.31817169, 27.02234592, 61.07740936,
#  38.65398865, 46.12606583, 46.94070587, 42.95021858, 62.4002636,  25.15277901,
#  48.56538239, 39.39150124, 36.90391617, 54.85911751, 26.19396,    55.77140198,
#  50.01274719, 46.25812759, 34.67755165, 24.31949043, 37.37019081, 60.21835442,
#  23.98877373, 42.52428131, 65.17762032, 42.47890549, 28.92038231, 32.67913036,
#  59.02633438, 50.70175781, 54.4684021,  25.89820213, 32.56190166, 21.37547874,
#  31.65770016, 58.41693573, 59.5116951,  22.73349476, 35.41193695, 50.1440609,
#  23.40338116, 25.68342323, 38.06956787, 29.6411993,  40.44827385, 30.19777966,
#  23.32536221, 28.14806881, 54.07668648, 25.88149338, 16.31525011, 24.8761013,
#  45.12774277, 32.35221767, 43.67839508, 41.7555294,  20.1278616,  19.16081772,
#  31.78387947, 22.84820976, 38.06319809, 22.7588728,  24.76123543, 22.93504848]
#
# with open('deblurred-image.csv','w',encoding='utf-8') as file:
#     #创建writer对象
#     writer = csv.writer(file)
#     #写表头
#     writer.writerow(header)
#     #遍历列表，将每一行写入csv
#     for i,item in enumerate(names):
#         lists = []
#         lists.append(item)
#         lists.append(score[i])
#         writer.writerow(lists)
#
# print('finish')

# -*- coding:utf-8 -*-
import os
from PIL import Image
import pandas as pd

# file_list = []
# width_list = []
# height_list = []
# root_path = '/home/dataset/PARA/imgs'
# suffix = ['.jpg', '.png']
#
# print('began')
# for dirpath, dirnames, files in os.walk(root_path):
#     for file in files:
#         file_path = os.path.join(dirpath, file)
#         for suf in suffix:
#             if file.endswith(suf):
#                 img = Image.open(file_path)
#                 file_list.append(file)
#                 width_list.append(img.size[0])
#                 height_list.append(img.size[1])
#
# content_dict = {
#     'dir_name': file_list,
#     'width': width_list,
#     'height': height_list
# }
#
# df = pd.DataFrame(content_dict)
# csv_path = '/home/dataset/PARA/image_size.csv'
# df.to_csv(csv_path, encoding='utf_8_sig')
# print('finished')

'''
读txt文件
'''
# data = []
# with open('/home/dataset/AADB/imgListTestNewRegression_score.txt', 'r',encoding='utf8') as f:
# 	for i in f:
# 		data.append([j for j in i.split()])
#
# print(data)
# print(data[0][0])

'''
读csV文件
'''
images_inf = pd.read_csv('/home/dataset/PARA/PARA-GiaaTrain.csv')
#print(images_inf.keys())
for i in range(0,3):
    lists = (
        images_inf['aestheticScore_1.0'][i], images_inf['aestheticScore_1.5'][i], images_inf['aestheticScore_2.0'][i],
        images_inf['aestheticScore_2.5'][i], images_inf['aestheticScore_3.0'][i],
        images_inf['aestheticScore_3.5'][i], images_inf['aestheticScore_4.0'][i], images_inf['aestheticScore_4.5'][i],
        images_inf['aestheticScore_5.0'][i])
    print(lists)