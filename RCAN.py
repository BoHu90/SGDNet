from torch import nn


# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction, bias):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=False),
            nn.PReLU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv(self.avg_pool(x))
        max_out = self.conv(self.max_pool(x))
        y = avg_out + max_out
        y = self.sigmoid(y)
        return x * y


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, n_feat, reduction=16, bias_ca=False, bn=False):
        super(RCAB, self).__init__()
        # 原来为1 *1
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, bias=True, padding=1),
            nn.BatchNorm2d(n_feat) if bn else nn.Sequential(),
            nn.PReLU(),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, bias=True, padding=1),
            nn.BatchNorm2d(n_feat) if bn else nn.Sequential(),
            CALayer(n_feat, reduction, bias_ca)
        )

    def forward(self, x):
        return x + self.body(x)


class ResidualGroup(nn.Module):
    def __init__(self, n_feat, num_rcabs_per_group, reduction, bn=False):
        super(ResidualGroup, self).__init__()
        modules_body = [RCAB(n_feat, reduction, bn=bn) for _ in range(num_rcabs_per_group)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=True))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        return x + self.body(x)
