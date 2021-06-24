from __future__ import absolute_import
from __future__ import division

__all__ = ['fpb']

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import resnet50, resnet101, Bottleneck
from .nn_utils import *
from .pc import *

class FPNModule(nn.Module):
    def __init__(self, num_layers, num_channels):
        super(FPNModule, self).__init__()
        self.num_layers = num_layers
        self.num_channels = num_channels 
        self.eps = 0.0001

        self.convs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for _ in range(4):
            conv = Conv_Bn_Relu(self.num_channels, self.num_channels, k=3, p=1)
            self.convs.append(conv)

        for _ in range(2):
            downsample = nn.Sequential(nn.Conv2d(self.num_channels,  self.num_channels, 1, bias=False), nn.BatchNorm2d(self.num_channels), nn.ReLU(inplace=True))
            self.downsamples.append(downsample)

        self.pc1 = PC_Module(self.num_channels, dropout=True)
    
        self._init_params()

    def _init_params(self):
        for downsample in self.downsamples:
            init_struct(downsample)
        
        return

    def forward(self, x):

        reg_feats = []

        y = x
        x_clone = []
        for t in x:
            x_clone.append(t.clone())

        reg_feats.append(y[0])
        y[0] = self.convs[0](y[0])  
        reg_feat = self.pc1(y[1])   
        reg_feats.append(reg_feat)
        y[1] = self.convs[1](reg_feat+F.interpolate(y[0], scale_factor=2, mode='nearest'))    
        y[1] = self.convs[2](y[1])+self.downsamples[0](x_clone[1]) 
        y[0] = self.convs[3](y[0]+F.max_pool2d(y[1], kernel_size=2))+self.downsamples[1](x_clone[0])  

        return y, reg_feats


class FPN(nn.Module):
    def __init__(self, num_layers, in_channels):
        super(FPN, self).__init__()

        self.num_layers = num_layers
        self.in_channels = in_channels
        self.num_neck_channel = 256

        self.lateral_convs = nn.ModuleList()
        for i in range(1, self.num_layers+1):
            conv = Conv_Bn_Relu(in_channels[i], self.num_neck_channel, k=1)
            self.lateral_convs.append(conv)        
    
        self.fpn_module1 = FPNModule(self.num_layers, self.num_neck_channel)

        self.conv = Conv_Bn_Relu(self.num_neck_channel, self.in_channels[1], k=1, activation_cfg=False)
        self.relu = nn.ReLU(inplace=True)

        self._init_params()

    def _init_params(self):

        return

    def forward(self, x):
        y = []
        for i in range(self.num_layers):
            y.append(self.lateral_convs[i](x[i+1]))
        
        y, reg_feat = self.fpn_module1(y)

        y = self.conv(y[0])#
        y = self.relu(y)

        return y, reg_feat        


class FPB(nn.Module):
    def __init__(self, num_classes, loss=None, **kwargs):
        super(FPB, self).__init__()

        resnet_ = resnet50(pretrained=True)

        self.num_parts = 3
        self.branch_layers = 2
        self.loss = loss

        self.layer0 = nn.Sequential(resnet_.conv1, resnet_.bn1, resnet_.relu,
                                    resnet_.maxpool)

        self.layer1 = resnet_.layer1
        self.layer2 = resnet_.layer2
        self.pc1 = PC_Module(512, dropout=True)
        self.layer3 = resnet_.layer3

        layer4 = nn.Sequential(
            Bottleneck(1024,
                       512,
                       downsample=nn.Sequential(
                           nn.Conv2d(1024, 2048, 1, bias=False),
                           nn.BatchNorm2d(2048))), Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        layer4.load_state_dict(resnet_.layer4.state_dict())

        self.layer4 = layer4

        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn_l4 = nn.BatchNorm2d(2048)
        self.classifier_l4 = nn.Linear(2048, num_classes)

        self.in_channels = [2048, 1024, 512, 256]
        self.neck = FPN(self.branch_layers,  self.in_channels)

        self.part_pools = nn.AdaptiveAvgPool2d((self.num_parts, 1))
        self.dim_reds = DimReduceLayer(self.in_channels[1], 256)
        self.classifiers = nn.ModuleList([nn.Linear(256, num_classes) for _ in range(self.num_parts)])

        self._init_params()

    def _init_params(self):
        init_bn(self.bn_l4)

        nn.init.normal_(self.classifier_l4.weight, 0, 0.01)
        if self.classifier_l4.bias is not None:
            nn.init.constant_(self.classifier_l4.bias, 0)

        init_struct(self.dim_reds)
        for c in self.classifiers:
            nn.init.normal_(c.weight, 0, 0.01)
            if c.bias is not None:
                nn.init.constant_(c.bias, 0)

    def featuremaps(self, x):
        fs = []
        y_l0 = self.layer0(x) 
        y_l1 = self.layer1(y_l0) 
        y_l2 = self.layer2(y_l1)
        y_l2_1 = self.pc1(y_l2)  
        y_l3 = self.layer3(y_l2_1) 
        y_l4 = self.layer4(y_l3) 
     
        fs.append(y_l4) 
        fs.append(y_l3) 
        fs.append(y_l2) 
        fs.append(y_l1) 

        return fs, y_l2_1

    def cross_ofp(self, x):
        x[1] = F.max_pool2d(x[1], kernel_size=2)

        y = torch.cat(x, 1)
        return y


    def forward(self, x):
        bs = x.size(0)
        reg_feat_re = []

        fs, y_l2_1 = self.featuremaps(x)

        f_branch, reg_feats = self.neck(fs) 

        f_l4_train = self.global_avgpool(fs[0]) 

        f_parts = self.part_pools(f_branch) 

        f_l4 = self.bn_l4(f_l4_train).view(bs, -1) 

        if not self.training:
            f = []
            f.append(F.normalize(f_l4, p=2, dim=1))
            f.append(F.normalize(f_parts, p=2, dim=1).view(bs, -1))

            f = torch.cat(f, 1)

            return f
        
        y = []

        y_l4 = self.classifier_l4(f_l4)
        y.append(y_l4)

        f_short = self.dim_reds(f_parts) 
        for j in range(self.num_parts):
            f_j = f_short[:, :, j, :].view(bs, -1)
            y_j = self.classifiers[j](f_j)

            y.append(y_j)

        reg_feat_re.append(self.cross_ofp(reg_feats)) 

        if self.loss == 'softmax':
            return y
        elif self.loss == 'engine_FPB':
            f = []
            f.append(F.normalize(f_l4_train, p=2, dim=1).view(bs, -1)) 
            f.append(F.normalize(f_parts, p=2, dim=1).view(bs, -1))

            f = torch.cat(f, 1)

            return y, f, reg_feat_re
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


def fpb(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FPB(num_classes=num_classes, loss=loss, **kwargs)

    return model



        

