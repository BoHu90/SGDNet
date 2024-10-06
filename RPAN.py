from torch import nn
import torch

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        y = self.sigmoid(y)
        return x * y

# Residual Channel Attention Block (RCAB)
class RPAB(nn.Module):
    def __init__(self, kernel_size, n_feat, bn=False):
        super(RPAB, self).__init__()
        # 原来为1 *1
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, bias=True, padding=1),
            nn.BatchNorm2d(n_feat) if bn else nn.Sequential(),
            nn.PReLU(),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, bias=True, padding=1),
            nn.BatchNorm2d(n_feat) if bn else nn.Sequential(),
            SpatialAttention(kernel_size=kernel_size)
        )

    def forward(self, x):
        return x + self.body(x)


class SPAResidualGroup(nn.Module):
    def __init__(self, kernel_size, n_feat, num_rpabs_per_group, bn=False):
        super(SPAResidualGroup, self).__init__()
        modules_body = [RPAB(kernel_size, n_feat, bn=bn) for _ in range(num_rpabs_per_group)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=True))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        return x + self.body(x)
