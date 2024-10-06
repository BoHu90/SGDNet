import torch
import torch.nn as nn
from torch.autograd import Variable


class EDMLoss_train(nn.Module):
    """EMD loss with label smoothing regularizer.Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
    - num_classes (int): number of classes.
    - epsilon (float): weight.
    """
    def __init__(self,num_classes=10):
        super(EDMLoss_train, self).__init__()
        self.num_classes = num_classes
        # self.epsilon = epsilon
    def forward(self, p_target: Variable, p_estimate: Variable):
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]
        # targets = (1 - self.epsilon) * p_target + self.epsilon / self.num_classes
        p_target = p_target.float()
        p_target /= p_target.sum(dim=1).reshape(-1, 1)#reshape(-1,1)转换成1列
        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))
        return samplewise_emd.mean()

class EDMLoss_test(nn.Module):

    def __init__(self):
        super(EDMLoss_test, self).__init__()

    def forward(self, p_target: Variable, p_estimate: Variable):
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]
        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.abs(cdf_diff)
        return samplewise_emd.mean()