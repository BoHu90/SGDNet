import argparse
import random
import numpy as np
from Cross_IQASolver import IQASolver
import time
import os
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# 获取当前文件的绝对路径
Absolute_path = os.getcwd()
# 获取当前时间
time_now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
# 定义日志路径
log_dir = './Cross_logs'
try:
    os.mkdir(log_dir)
except OSError:
    pass
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    filename=os.path.join(log_dir, "{}.log".format(time_now)),
                    filemode='w')
if os.path.exists(log_dir):
    logging.info("Created log_dir directory:" + str(log_dir))


def main(config):
    folder_path = {
        'live': '/home/wsj/wangshuaijian/datasets/live',
        'csiq': '/home/wsj/wangshuaijian/datasets/csiq',
        'tid2013': '/home/wsj/wangshuaijian/datasets/tid2013',
        'livec': '/home/wsj/wangshuaijian/datasets/ChallengeDB_release',
        'koniq-10k': '/home/wsj/wangshuaijian/datasets/koniq10k',
        'bid': '/home/wsj/wangshuaijian/datasets/BID',
    }

    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'livec': list(range(0, 1162)),
        'koniq-10k': list(range(0, 10073)),
        'bid': list(range(0, 586)),
    }

    srcc_all = np.zeros(config.train_test_num, dtype=np.float32)
    plcc_all = np.zeros(config.train_test_num, dtype=np.float32)
    krocc_all = np.zeros(config.train_test_num, dtype=np.float32)
    RMSE_all = np.zeros(config.train_test_num, dtype=np.float32)

    logging.info('Training on %s dataset for %d rounds...' % (config.train_dataset, config.train_test_num))
    logging.info('Testing on %s dataset for %d rounds...' % (config.test_dataset, config.train_test_num))
    for i in range(config.train_test_num):
        print('Round %d' % (i + 1))
        # 定义保存模型的路径
        checkpoint_dir = os.path.join(Absolute_path, "Cross_database", time_now)
        try:
            os.makedirs(checkpoint_dir)
            logging.info('Created checkpoint directory:' + str(checkpoint_dir))
        except OSError:
            pass

        # Randomly select 80% images for training and the rest for testing
        # random.shuffle(sel_num)
        # train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
        # print(train_index)
        # test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]
        # print(test_index)
        train_index = img_num[config.train_dataset]
        test_index = img_num[config.test_dataset]
        solver = IQASolver(config, folder_path[config.train_dataset], folder_path[config.test_dataset], train_index,
                           test_index)
        srcc_all[i], plcc_all[i], krocc_all[i], RMSE_all[i] = solver.train(i, checkpoint_dir, config)

    # print(srcc_all)
    # print(plcc_all)
    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)
    krocc_med = np.median(krocc_all)
    rmse_med = np.median(RMSE_all)
    srcc_mean = np.mean(srcc_all)
    plcc_mean = np.mean(plcc_all)
    krocc_mean = np.mean(krocc_all)
    rmse_mean = np.mean(RMSE_all)
    print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f,\tmedian KROCC %4.4f,\tmedian RMSE %4.4f' % (
        srcc_med, plcc_med, krocc_med, rmse_med))
    logging.info('Testing median SRCC %4.4f,\tmedian PLCC %4.4f,\tmedian KROCC %4.4f,\tmedian RMSE %4.4f' % (
        srcc_med, plcc_med, krocc_med, rmse_med))
    logging.info('Testing mean SRCC %4.4f,\tmean PLCC %4.4f,\tmean KROCC %4.4f,\tmean RMSE %4.4f' % (
        srcc_mean, plcc_mean, krocc_mean, rmse_mean))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', dest='train_dataset', type=str, default='live',
                        help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013')
    parser.add_argument('--test_dataset', dest='test_dataset', type=str, default='bid',
                        help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=5,
                        help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=5,
                        help='Number of sample patches from testing image')
    parser.add_argument('--pvt_lr', dest='pvt_lr', type=float, default=1e-5, help='pvt_lr rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--lr2', dest='lr2', type=float, default=1e-4, help='Learning rate ratio for add network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=32, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=384,
                        help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=1, help='Train-test times')
    parser.add_argument('--num_rcabs_per_group', dest='num_rcabs_per_group', type=int, default=3,
                        help='num_rcabs_per_group')
    parser.add_argument('--num_residual_groups', dest='num_residual_groups', type=int, default=1,
                        help='num_residual_groups')
    parser.add_argument('--num_rpabs_per_group', dest='num_rpabs_per_group', type=int, default=3,
                        help='num_rcabs_per_group')
    parser.add_argument('--num_residual_rpab_groups', dest='num_residual_rpab_groups', type=int, default=1,
                        help='num_residual_groups')
    parser.add_argument('--pvt_path', dest='pvt_path', type=str, default='./Premodel/pvt_v2_b3.pth', help='pvt_path')
    config = parser.parse_args()
    main(config)
