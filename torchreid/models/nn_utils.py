from __future__ import absolute_import
from __future__ import division

import torch
from torch.nn import Softmax
from torch import nn
from torch.nn import functional as F


def init_bn(bn_struct):
    nn.init.constant_(bn_struct.weight, 1.0)
    nn.init.constant_(bn_struct.bias, 0.0)


def init_struct(nn_struct):
    if not isinstance(nn_struct, nn.Module):
        return KeyError("Only nn.Module can be initialized!")
    for m in nn_struct.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def init_conv(layer):
    if not (isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Conv2d)):
        return KeyError("Only nn.Conv1d or nn.Conv2d can be initialized!")

    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity='relu')
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)


def init_conv2d(layer):
    if not isinstance(layer, nn.Conv2d):
        return KeyError("Only nn.Conv2d can be initialized!")

    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity='relu')
    # nn.init.xavier_uniform_(layer.weight, gain=1)       #20200421_3
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)


def init_fc(layer):
    if not isinstance(layer, nn.Linear):
        return KeyError("Only nn.Linear can be initialized!")
    nn.init.normal_(layer.weight, 0, 0.01)
    if m.bias is not None:
        nn.init.constant_(layer.bias, 0)


class DimReduceLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DimReduceLayer, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(in_channels,
                      out_channels,
                      1,
                      stride=1,
                      padding=0,
                      bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Conv_Bn_Relu(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=0, norm_cfg=True, activation_cfg=True):
        super(Conv_Bn_Relu, self).__init__()
        if norm_cfg:
            self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False)
        else:
            self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)
        self.norm_cfg = norm_cfg
        self.activation_cfg = activation_cfg

        self.init_params()

    def init_params(self):
        init_bn(self.bn)
        init_conv2d(self.conv)

    def forward(self, x):
        y = self.conv(x)

        if self.norm_cfg:
            y = self.bn(y)

        if self.activation_cfg:
            y = F.relu(y)
        
        return y


class OFPenalty(nn.Module):
    def __init__(self, of_beta):
        super(OFPenalty, self).__init__()

        self.beta = of_beta
        # self.softmax = Softmax(dim=-1)

    def dominant_eigenvalue(self, A):
        B, N, _ = A.size()
        x = torch.randn(B, N, 1, device='cuda')

        for _ in range(1):
            x = torch.bmm(A, x)
        # x: 'B x N x 1'
        numerator = torch.bmm(
            torch.bmm(A, x).view(B, 1, N),
            x
        ).squeeze()
        denominator = (torch.norm(x.view(B, N), p=2, dim=1) ** 2).squeeze()

        return numerator / denominator

    def get_singular_values(self, A):
        AAT = torch.bmm(A, A.permute(0, 2, 1)) # C*S*S*C=C*C
        # AAT = self.softmax(AAT)
        B, N, _ = AAT.size()
        largest = self.dominant_eigenvalue(AAT)
        I = torch.eye(N, device='cuda').expand(B, N, N)  # noqa
        I = I * largest.view(B, 1, 1).repeat(1, N, N)  # noqa
        tmp = self.dominant_eigenvalue(AAT - I)
        return tmp + largest, largest

    def apply_penalty(self, x):
        batches, channels, height, width = x.size()
        W = x.view(batches, channels, -1)
        smallest, largest = self.get_singular_values(W)
        singular_penalty = (largest - smallest) * self.beta

        singular_penalty *= 0.01

        return singular_penalty.sum() / (x.size(0) / 32.)  # Quirk: normalize to 32-batch case
        
    def forward(self, reg_feats):
        singular_penalty = sum([self.apply_penalty(x) for x in reg_feats])

        return singular_penalty