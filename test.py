import argparse

import torch
from scipy import stats
import numpy as np
from models_cnn import Net, pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
from models_cnn import *
import data_loader
import os
import time
import logging
from edm_loss import EDMLoss_train
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class IQASolver(object):
    """Solver for training and testing hyperIQA"""

    def __init__(self, config, path):

        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.root = config.dataset
        pvt = None
        # 加载PVT 预训练权重
        if config.pvt_path == './Premodel/pvt_v2_b1.pth':
            pvt = pvt_v2_b1()
        elif config.pvt_path == './Premodel/pvt_v2_b2.pth':
            pvt = pvt_v2_b2()
        elif config.pvt_path == './Premodel/pvt_v2_b3.pth':
            pvt = pvt_v2_b3()
        elif config.pvt_path == './Premodel/pvt_v2_b4.pth':
            pvt = pvt_v2_b4()
        elif config.pvt_path == './Premodel/pvt_v2_b5.pth':
            pvt = pvt_v2_b5()
        checkpoint = config.pvt_path
        pvt.load_state_dict(torch.load(checkpoint))
        self.model = Net(root=config.dataset, pvt=pvt, dims=[64, 128, 320, 512], config=config).cuda()
        if self.root == 'ava':
            self.loss = EDMLoss_train().cuda()
        else:
            self.loss = torch.nn.L1Loss(reduction='mean').cuda()
        self.pvt_lr = config.pvt_lr
        self.lr2 = config.lr2  # 添加的网络的学习率
        self.weight_decay = config.weight_decay

        Pvt_params = list(map(id, self.model.Pvt.parameters()))
        base_params = filter(lambda p: id(p) not in Pvt_params, self.model.parameters())
        self.optimizer = torch.optim.AdamW([
            {'params': self.model.Pvt.parameters()},
            {'params': base_params, 'lr': self.lr2}], lr=self.pvt_lr, weight_decay=self.weight_decay)

        test_idx = []
        test_loader = data_loader.DataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num,
                                             istrain=False)
        self.test_data = test_loader.get_data()
        self.test = self.test

    def train(self, i, checkpoint_dir, config):
        """Training"""
        logging.info(f'''Starting training:
            Epochs:          {config.epochs}
            Batch size:      {config.batch_size}
            Checkpoints:     {checkpoint_dir}
            patch_size:      {config.patch_size}
            train_patch_num：{config.train_patch_num}
            test_patch_num:  {config.test_patch_num}
            pvt_path:        {config.pvt_path}
            num_rcabs_per_group: {config.num_rcabs_per_group}
            num_residual_groups: {config.num_residual_groups}
            num_rpabs_per_group: {config.num_rpabs_per_group}
            num_residual_rpab_groups: {config.num_residual_rpab_groups}
        ''')
        #print(self.model)
        total = sum([param.nelement() for param in self.model.parameters()])
        #print("Number of parameter: %.2fM" % (total / 1e6))

        best_srcc = -2
        best_plcc = -2
        print('Epoch\t\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tTest_KRCC\tTest_RMSE')
        logging.info("Epoch\t\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tTest_KRCC\tTest_RMSE")
        for epoch in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            if self.root == 'ava':
                for img,p_tr, label in self.train_data:
                    img = torch.as_tensor(img.cuda())
                    p_tr = torch.stack(p_tr).transpose(0, 1).contiguous()
                    p_tr = p_tr.cuda()
                    label = torch.as_tensor(label.cuda())
                    # Quality prediction
                    p, pred = self.model(img)

                    pred_scores = pred_scores + pred.cpu().tolist()
                    gt_scores = gt_scores + label.cpu().tolist()

                    # print('*****')
                    # print(p_tr.detach().shape) #shape=[batch_size,10]
                    # print(p.squeeze().shape)
                    # print('*****')
                    loss = self.loss(p_tr.detach(), p.squeeze())
                    epoch_loss.append(loss.item())
                    # 优化器
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            else:
                for img, label in self.train_data:
                    img = torch.as_tensor(img.cuda())
                    label = torch.as_tensor(label.cuda())

                    # Quality prediction
                    pred = self.model(img)  # 预测结果

                    pred_scores = pred_scores + pred.cpu().tolist()
                    gt_scores = gt_scores + label.cpu().tolist()

                    loss = self.loss(pred.squeeze(), label.float().detach())
                    epoch_loss.append(loss.item())

                    # 优化器
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

            # 保存最后一个epoch的模型
            final_model_name = os.path.join(checkpoint_dir, 'final_model.pth')
            torch.save({
                'final_model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, final_model_name)
            # logging.info(f'Checkpoint final model saved! epoch: {epoch + 1} ')
            # logging.info('-' * 88)

            #  测试开始
            self.model.eval()
            test_srcc, test_plcc, test_krocc, test_RMSE = self.test(self.test_data)

            print('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (epoch + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc, test_krocc,
                   test_RMSE))
            logging.info('  %d\t\t%4.3f\t\t%4.5f\t\t%4.5f\t\t%4.5f\t\t%4.5f\t\t%4.5f' %
                         (epoch + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc, test_krocc,
                          test_RMSE))

            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
                best_krcc = test_krocc
                best_rmse = test_RMSE
                best_epoch = epoch
                # 保存测试效果最好的模型
                best_model_name = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save({
                    'best_model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, best_model_name)
                # logging.info(f'Checkpoint best model saved! epoch: {epoch + 1}')
        print('Best test SRCC %f, PLCC %f, KRCC %f, RMSE %f, epoch %d' % (
        best_srcc, best_plcc, best_krcc, best_rmse, int(best_epoch + 1)))
        logging.info('Best test SRCC %f, PLCC %f, KRCC %f, RMSE %f, epoch %d' % (
        best_srcc, best_plcc, best_krcc, best_rmse, int(best_epoch + 1)))
        logging.info(
            '第{}次训练结束********************************************************************************'.format(
                i + 1))
        return best_srcc, best_plcc, best_krcc, best_rmse

    def test(self, data):
        """Testing"""
        pred_scores = []
        gt_scores = []
        with torch.no_grad():
            if self.root == 'ava':
                for img, p_tr, label in data:
                    # Data.
                    img = torch.as_tensor(img.cuda())
                    label = torch.as_tensor(label.cuda())
                    p, pred = self.model(img)
                    pred_scores.append(float(pred.item()))
                    gt_scores = gt_scores + label.cpu().tolist()
                # print('pred_scores.length:', len(pred_scores))
                # print('test_patch_num:', self.test_patch_num)
                pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
                gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
                test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
                test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
                test_krocc, _ = stats.stats.kendalltau(pred_scores, gt_scores)
                test_RMSE = np.sqrt(((pred_scores - gt_scores) ** 2).mean())
            else:
                for img in data:
                    # Data.
                    img = torch.as_tensor(img.cuda())
                    # label = torch.as_tensor(label.cuda())
                    pred = self.model(img)
                    pred_scores.append(float(pred.item()))
                    # gt_scores = gt_scores + label.cpu().tolist()
                # print('pred_scores.length:', len(pred_scores))
                # print('test_patch_num:', self.test_patch_num)
                pred_scores_mean = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
                # gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
                # test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
                # test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
                # test_krocc, _ = stats.stats.kendalltau(pred_scores, gt_scores)
                # test_RMSE = np.sqrt(((pred_scores - gt_scores) ** 2).mean())

        #self.model.train()
        return pred_scores, pred_scores_mean

def main(config):
    solver = IQASolver(config, path='tests')
    model = solver.model
    checkpoint_net = torch.load('/usr/zhengjia/SGDNet/best_model.pth')
    model.load_state_dict(checkpoint_net['best_model'])
    solver.optimizer.load_state_dict(checkpoint_net['optimizer'])
    pred_scores, pred_scores_mean = solver.test(solver.test_data)

    print(pred_scores)
    print('-------------------------------------------------------------------------------')
    print(pred_scores_mean)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='tests',
                        help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013|tests')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=10,
                        help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=10,
                        help='Number of sample patches from testing image')
    parser.add_argument('--pvt_lr', dest='pvt_lr', type=float, default=1e-5, help='pvt_lr rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--lr2', dest='lr2', type=float, default=1e-4, help='Learning rate ratio for add network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=30, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=384,
                        help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=7, help='Train-test times')
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


