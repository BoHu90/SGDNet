import torch
from scipy import stats
import numpy as np
from models_cnn import Net, pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
from models_cnn import *
import data_loader
import os
import time
import logging

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

class IQASolver(object):
    """Solver for training and testing hyperIQA"""

    def __init__(self, config, path, train_idx, test_idx):

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
        #self.model = Net(root=config.dataset, pvt=pvt, dims=[64, 128, 320, 512], config=config).cuda()
        model = Net(root=config.dataset, pvt=pvt, dims=[64, 128, 320, 512], config=config)
        self.model = torch.nn.DataParallel(model, device_ids=[6, 7])  # 多gpu并行计算
        self.model = self.model.to(device)
        self.loss = torch.nn.L1Loss(reduction='mean').to(device)
        self.pvt_lr = config.pvt_lr
        self.lr2 = config.lr2  # 添加的网络的学习率
        self.weight_decay = config.weight_decay

        Pvt_params = list(map(id, self.model.module.Pvt.parameters()))
        base_params = filter(lambda p: id(p) not in Pvt_params, self.model.module.parameters())
        self.optimizer = torch.optim.AdamW([
            {'params': self.model.module.Pvt.parameters()},
            {'params': base_params, 'lr': self.lr2}], lr=self.pvt_lr, weight_decay=self.weight_decay)

        train_loader = data_loader.DataLoader(config.dataset, path, train_idx, config.patch_size,
                                              config.train_patch_num, batch_size=config.batch_size, istrain=True)
        test_loader = data_loader.DataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num,
                                             istrain=False)
        self.train_data = train_loader.get_data()
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
        total = sum([param.nelement() for param in self.model.module.parameters()])
        #print("Number of parameter: %.2fM" % (total / 1e6))

        best_srcc = -2
        best_plcc = -2
        print('Epoch\t\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tTest_KRCC\tTest_RMSE')
        logging.info("Epoch\t\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tTest_KRCC\tTest_RMSE")

        for epoch in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            for img, label in self.train_data:
                img = torch.as_tensor(img.to(device))
                label = torch.as_tensor(label.to(device))

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
            final_model_name = os.path.join(checkpoint_dir, 'koniq-model.pth'.format(epoch+1))
            torch.save({
                'final_model': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': epoch,
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
                best_model_name = os.path.join(checkpoint_dir, 'koniq_best_model.pth')
                torch.save({
                    'best_model': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch,
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
            for img, label in data:
                # Data.
                img = torch.as_tensor(img.to(device))
                label = torch.as_tensor(label.to(device))
                pred = self.model(img)
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

        self.model.train()
        return test_srcc, test_plcc, test_krocc, test_RMSE
