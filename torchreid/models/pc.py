###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import math
import torch
from torch.nn import Module, Conv2d, Parameter, Softmax, Conv1d
import torch.nn as nn
from torch.nn import functional as F
from .nn_utils import *

import logging

class SE(Module):

    def __init__(self, in_dim, reduction_rate=8):
        super(SE, self).__init__()
        self.conv1 = Conv_Bn_Relu(in_dim, in_dim//reduction_rate, 1)
        self.conv2 = Conv_Bn_Relu(in_dim//reduction_rate, in_dim, 1)
        # self.conv1 = nn.Conv2d(in_dim, in_dim//reduction_rate, 1, stride=1)
        # self.conv2 = nn.Conv2d(in_dim//reduction_rate, in_dim, 1, stride=1)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # init_conv2d(self.conv1)
        # init_conv2d(self.conv2)

    def forward(self, x):
        bs, c, height, width = x.size()
        a = self.gap(x)
        a = self.conv1(a)       #20200418_1
        a = self.conv2(a).view(bs, -1, 1, 1)    #20200418_1
        # a = F.relu(self.conv1(a))       #20200418_3
        # a = F.sigmoid(self.conv2(a)).view(bs, -1, 1, 1)     #20200418_3
        
        y = x*a
        return y

class SE_Module(Module):
    def __init__(self, in_dim, reduction_rate=8):
        super(SE_Module, self).__init__()
        self.se = SE(in_dim, reduction_rate)

    def forward(self, x):
        y = x+self.se(x)

        return y


class PAM_SE_Module(Module):
    def __init__(self, in_dim, dropout=False):
        super(PAM_SE_Module, self).__init__()
        self.se = SE(in_dim)
        self.pam = PAM(in_dim, dropout)

    def forward(self, x):
        y = x + self.se(x) + self.pam(x)    #20200418_1
        # y = self.pam(x)     #20200418_2
        # y = self.se(y)      #20200418_2

        return y


class PAM_Multi_Head_Module(Module):
    """ Position attention module with multiple heads"""

    def __init__(self, in_dim, n_heads=8, dropout=False):
        super(PAM_Multi_Head_Module, self).__init__()

        self.in_dim = in_dim
        self.n_heads = n_heads
        self.head_dim = in_dim // n_heads

        self.fc_q = nn.Linear(in_dim, in_dim)
        self.fc_k = nn.Linear(in_dim, in_dim)
        self.fc_v = nn.Linear(in_dim, in_dim)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

        self.gamma = Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(in_dim)

        if dropout:
            self.dropout_attention = nn.Dropout(0.2)
            self.dropout_out = nn.Dropout(0.2)
            self.is_dropout = True
        else:
            self.is_dropout = False

        self._init_param()

    def _init_param(self):
        init_fc(self.fc_q)
        init_fc(self.fc_k)
        init_fc(self.fc_v)

        init_bn(self.bn)


    def forward(self, x):
        m_batchsize, C, height, width = x.size()

        input = x.view(m_batchsize, C, -1).permute(0, 2, 1)

        Q = self.fc_q(input)
        K = self.fc_k(input)
        V = self.fc_v(input)

        Q = Q.view(m_batchsize, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(m_batchsize, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(m_batchsize, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))/self.scale.to(x.device)

        attention = torch.softmax(energy, dim=-1)
        if self.is_dropout and self.training:
            output = torch.matmul(self.dropout_attention(attention), V)
        else:
            output = torch.matmul(attention, V)

        output = output.permute(0, 1, 3, 2).contiguous()
        output = output.view(m_batchsize, C, height, width)

        output = self.bn(self.gamma*output)

        if self.is_dropout and self.training:
            output = x+self.dropout_out(output)
        else:
            output = x+output

        return output


class Co_Pam_Module(Module):
    def __init__(self, in_dim_x, in_dim_y, dropout=False, num_heads=8):
        super(Co_Pam_Module, self).__init__()
        self.pam = Co_Pam(in_dim_x, in_dim_y, dropout, num_heads)

    def forward(self, x, y):
        out = self.pam(x, y)
        out = x+out

        return out


class Co_Pam(Module):
    def __init__(self, in_dim_x, in_dim_y, dropout=False, num_heads=8):
        super(Co_Pam, self).__init__()
        self.channel_in_x = in_dim_x
        self.channel_in_y = in_dim_y
        self.num_heads = num_heads

        self.query_conv = Conv2d(in_channels=self.channel_in_y, out_channels=self.channel_in_y//self.num_heads, kernel_size=1)
        self.key_conv = Conv2d(in_channels=self.channel_in_y, out_channels=self.channel_in_y//self.num_heads, kernel_size=1)
        self.value_conv = Conv2d(in_channels=self.channel_in_x, out_channels=self.channel_in_x, kernel_size=1)

        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(self.channel_in_x)

        if dropout:
            self.dropout_attention = nn.Dropout(0.2)
            self.dropout_out = nn.Dropout(0.2) # 20191216_3
            self.is_dropout = True
        else:
            self.is_dropout = False

        self._init_param()

    def _init_param(self):
        # init_conv2d(self.key_conv)
        # init_conv2d(self.value_conv)
        # init_conv2d(self.query_conv)
        init_conv2d(self.key_conv)
        init_conv2d(self.value_conv)
        init_conv2d(self.query_conv)

        init_bn(self.bn)

    def forward(self, x, y):
        m_batchsize, C_x, height_x, width_x = x.size()
        _, C_y, height_y, width_y = y.size()

        size_x = height_x * width_x
        size_y = height_y * width_y
        scale_factor = size_x/size_y

        proj_query = self.query_conv(y).view(m_batchsize, -1, size_y).permute(0, 2, 1)
        proj_key = self.key_conv(y).view(m_batchsize, -1, size_y)

        proj_value = self.value_conv(x)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy).view(m_batchsize, 1, size_y, size_y)
        attention = F.interpolate(attention, scale_factor=scale_factor, mode='nearest').view(m_batchsize, size_x, size_x)
        attention = attention.permute(0, 2, 1)

        x_view = proj_value.view(m_batchsize, -1, size_x)

        if self.is_dropout and self.training:
            out = torch.bmm(x_view, self.dropout_attention(attention))
        else:
            out = torch.bmm(x_view, attention)

        attention_mask = out.view(m_batchsize, C_x, height_x, width_x)

        out = self.gamma * attention_mask

        out = self.bn(out)

        if self.is_dropout and self.training:
            out = self.dropout_out(out) # 20191215_1
        else:
            out = out

        return out


class PAM_Module(Module):
    def __init__(self, in_dim, dropout=False, light_weight=False):
        super(PAM_Module, self).__init__()
        self.pam = PAM(in_dim, dropout, light_weight)

    def forward(self, x):
        y = self.pam(x)
        y = y + x

        return y


class PAM(Module):
    """ Position attention module"""
    # Ref from SAGAN

    def __init__(self, in_dim, dropout=False, light_weight=False):
    # def __init__(self, in_dim, dropout=True):   #20191217_2
        super(PAM, self).__init__()
        self.channel_in = in_dim
        self.light_weight = light_weight


        if self.light_weight:
            t = int(abs(math.log(self.channel_in, 2)+1)/2)
            self.k = t if t % 2 else t+1

            self.query_conv = Conv1d(in_channels=1, out_channels=1, kernel_size=self.k, padding=int(self.k/2), bias=False)
            self.key_conv = Conv1d(in_channels=1, out_channels=1, kernel_size=self.k, padding=int(self.k/2), bias=False)
            # self.value_conv = Conv1d(in_channels=1, out_channels=1, kernel_size=self.k, padding=int(self.k/2), bias=False)
        else:
            self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
            self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)

        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1) #20191212_3
        # self.res_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1) #20191215_2

        self.gamma = Parameter(torch.zeros(1))
        # self.gamma = Parameter(torch.zeros(1)) 20191212_2

        self.softmax = Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(in_dim)

        if dropout:
            # self.dropout = nn.Dropout(0.3) # 20191215_1
            self.dropout_attention = nn.Dropout(0.2)
            self.dropout_out = nn.Dropout(0.2) # 20191216_3
            self.is_dropout = True
        else:
            self.is_dropout = False

        # self.scale = torch.sqrt(torch.FloatTensor([in_dim // 8]))   #20191212_4
        self._init_param()

    def _init_param(self):
        # init_conv2d(self.key_conv)
        # init_conv2d(self.value_conv)
        # init_conv2d(self.query_conv)
        if self.light_weight:
            init_conv(self.query_conv)
            init_conv(self.key_conv)
        else:            
            init_conv2d(self.query_conv)
            init_conv2d(self.key_conv)

        init_conv2d(self.value_conv)

        init_bn(self.bn)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()

        if self.light_weight:
            x_proj = x.view(m_batchsize, C, -1).permute(0, 2, 1).contiguous().view(-1, 1, C)
            proj_query = self.query_conv(x_proj).view(m_batchsize, -1, C)
            proj_key = self.key_conv(x_proj).view(m_batchsize, -1, C).permute(0, 2, 1)
        else:
            proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
            proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)        
        
        proj_value = self.value_conv(x) #20191212_3
        # proj_value = self.value_conv(x_proj).view(m_batchsize, -1, C).permute(0, 2, 1) #20200114_1
        
        energy = torch.bmm(proj_query, proj_key)    #/self.scale.to(x.device)
        # energy = torch.bmm(proj_query, proj_key)/self.scale.to(x.device)    #20191212_4
        attention = self.softmax(energy)
       
        # x_view = x.view(m_batchsize, -1, width * height)
        x_view = proj_value.view(m_batchsize, -1, width * height)
        # x_view = proj_value #20200114_1

                # out = torch.bmm(self.dropout(x_view), self.dropout(attention.permute(0, 2, 1))) # 20191215_1

        # if self.is_dropout:
        #     out = torch.bmm(x_view, self.dropout(attention.permute(0, 2, 1))) #20191215_8
        # else:
        if self.is_dropout and self.training:
            # out = torch.bmm(x_view, self.dropout_attention(attention))
            out = torch.bmm(x_view, self.dropout_attention(attention.permute(0, 2, 1)))
        else:
            # out = torch.bmm(x_view, attention)
            out = torch.bmm(x_view, attention.permute(0, 2, 1))

        # out = torch.bmm(self.dropout(x_view), attention.permute(0, 2, 1)) #20191215_9
        attention_mask = out.view(m_batchsize, C, height, width)

        out = self.gamma * attention_mask
        # out = attention_mask    #20191212_2
        # out = attention_mask*0.01 #20191215_2
        # out = self.res_conv(attention_mask) #20191215_4

        out = self.bn(out) #20191215_3
        # out = self.res_conv(out) #20191216_1        
        # out = self.bn(out+x) #20191215_5

        if self.is_dropout and self.training:
            out = self.dropout_out(out) # 20191215_1
        else:
            out = out

        return out


class Co_CAM_Module(Module):
    def __init__(self, in_dim_x, in_dim_y):
        super(Co_CAM_Module, self).__init__()
        self.cam = Co_CAM(in_dim_x, in_dim_y)

    def forward(self, x, y):
        out = self.cam(x, y)
        out = x+out

        return out
        

class Co_CAM(Module):
    def __init__(self, in_dim_x, in_dim_y):
        super(Co_CAM, self).__init__()
        self.channel_in_x = in_dim_x
        self.channel_in_y = in_dim_y

        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(in_dim_x)
        self._init_param()

    def _init_param(self):
        init_bn(self.bn)

    def forward(self, x, y):
        m_batchsize, C_x, height_x, width_x = x.size()
        _, C_y, height_y, width_y = y.size()

        scale_factor = C_x/C_y

        proj_query = y.view(m_batchsize, C_y, -1)
        proj_key = y.view(m_batchsize, C_y, -1).permute(0, 2, 1)
        proj_value = x.view(m_batchsize, C_x, -1)
        energy = torch.bmm(proj_query, proj_key)
        max_energy_0 = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
        energy_new = max_energy_0-energy
        attention = self.softmax(energy_new).view(m_batchsize, 1, C_y, C_y)
        attention = F.interpolate(attention, scale_factor=scale_factor, mode='nearest').view(m_batchsize, C_x, C_x)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C_x, height_x, width_x)

        gamma = self.gamma.to(out.device)
        out = gamma * out
        out = self.bn(out)

        return out


class CAM_Module(Module):
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.cam = CAM(in_dim)

    def forward(self, x):
        y = self.cam(x)
        y = y+x

        return y


class CAM(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM, self).__init__()
        self.channel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        # self.gamma = Parameter(torch.zeros(1))    #20191212_2

        self.softmax = Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(in_dim)
        self._init_param()

    def _init_param(self):
        init_bn(self.bn)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        max_energy_0 = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
        energy_new = max_energy_0 - energy
        attention = self.softmax(energy_new)

        out = torch.bmm(attention, proj_query)
        out = out.view(m_batchsize, C, height, width)

        gamma = self.gamma.to(out.device)
        out = gamma * out 
        # gamma = self.gamma.to(out.device) #20191212_2
        # out = gamma * out     #20191212_2
        # out = 0.01*out #20191215_2

        # out = self.bn(out) #20191215_3
        out = self.bn(out)
        # out = out + x
        # out = self.bn(out+x) #20191215_5
        return out


class Co_PC_Module(Module):
    def __init__(self, in_dim_x, in_dim_y, dropout=False):
        super(Co_PC_Module, self).__init__()
        self.pam = Co_Pam_Module(in_dim_x, in_dim_y, dropout)
        self.cam = Co_CAM_Module(in_dim_x, in_dim_y)

    def forward(self, x, y):
        out = self.pam(x, y)
        out = self.cam(out, y)

        return out


class PC_Module(nn.Module):

    def __init__(self, in_dim, dropout=False, light_weight=False):
        super().__init__()
        self.in_channel = in_dim
        # self.pam = PAM_Module(in_dim)
        self.pam = PAM_Module(in_dim, dropout, light_weight)  #20191216_4
        self.cam = CAM_Module(in_dim)
        # self.pam = PAM(in_dim, dropout)     #20200421
        # self.cam = CAM(in_dim)      #20200421
        # self.cam = CAM_Module(in_dim) 20191212_1

    def forward(self, x):
        out = self.pam(x)
        out = self.cam(out)
        # out = x + self.pam(x) + self.cam(x)     #20200421
        # out = self.cam(out) 20191212_1
        return out