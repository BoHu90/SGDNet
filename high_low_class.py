'''
将livec和konIQ数据集划分为高质量和低质量两个分布
---liveC已经完成
---konIQ已经完成
'''
import csv
import scipy.io
import numpy as np


def liveC_divide():
    imgpath = scipy.io.loadmat('/usr/dataset/ChallengeDB_release/Data/AllImages_release.mat')
    imgpath = imgpath['AllImages_release']
    imgpath = list(imgpath[7:1169])
    mos = scipy.io.loadmat('/usr/dataset/ChallengeDB_release/Data/AllMOS_release.mat')
    labels = mos['AllMOS_release'].astype(np.float32)
    labels = labels[0][7:1169]

    good_path = '/usr/dataset/ChallengeDB_release/Data/good.csv'
    bad_path = '/usr/dataset/ChallengeDB_release/Data/bad.csv'
    header = ['imgpath', 'mos']

    good_fp = open(good_path, 'w')
    bad_fp = open(bad_path, 'w')

    writer_good = csv.writer(good_fp)
    writer_bad = csv.writer(bad_fp)

    writer_good.writerow(header)
    writer_bad.writerow(header)

    for i, item in enumerate(imgpath):
        if labels[i] >= 44.505 and labels[i] <= 93:  # good quality
            writer_good.writerow((item[0][0], labels[i]))
        elif labels[i] >= 3.42 and labels[i] < 44.505:  # bad quality
            writer_bad.writerow((item[0][0], labels[i]))
    print('finished liveC divide')

def konIQ_divide():
    imgname = []
    mos_all = []
    csv_file = '/usr/dataset/koniq10k/koniq10k_scores_and_distributions.csv'
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            imgname.append(row['image_name'])
            mos = np.array(float(row['MOS_zscore'])).astype(np.float32)
            mos_all.append(mos)

    good_path = '/usr/dataset/koniq10k/good.csv'
    bad_path = '/usr/dataset/koniq10k/bad.csv'
    header = ['imgpath', 'mos']

    good_fp = open(good_path, 'w')
    bad_fp = open(bad_path, 'w')

    writer_good = csv.writer(good_fp)
    writer_bad = csv.writer(bad_fp)

    writer_good.writerow(header)
    writer_bad.writerow(header)

    for i,item in enumerate(imgname):
        if mos_all[i] > 49.5 and mos_all[i] <= 100:  # good
            writer_good.writerow((item, mos_all[i]))
        elif mos_all[i] >= 1 and mos_all[i] <= 49.5:#bad
            writer_bad.writerow((item, mos_all[i]))
    print('finished koniq divide')


def main():
    #liveC_divide()
    konIQ_divide()


if __name__ == '__main__':
    main()



