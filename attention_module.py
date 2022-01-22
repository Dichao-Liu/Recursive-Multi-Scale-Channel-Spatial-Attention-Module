#####################
# Some functions of this script are adapted from the CBAM.
# Woo, Sanghyun, et al.
# "Cbam: Convolutional block attention module."
# Proceedings of the European conference on computer vision (ECCV). 2018.
# https://github.com/Jongchan/attention-module
####################

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Channel_Wise_Attention(nn.Module):
    def __init__(self, gate_channels, pool_types=['avg', 'max']):
        super(Channel_Wise_Attention, self).__init__()
        self.gate_channels = gate_channels


        self.mlp = nn.Sequential(
                Flatten(),
                nn.Linear(gate_channels, gate_channels // 8),
                nn.ReLU(),
                nn.Linear(gate_channels // 8, gate_channels)
            )
        self.mlp2 = nn.Sequential(
                Flatten(),
                nn.Linear(gate_channels, gate_channels // 16),
                nn.ReLU(),
                nn.Linear(gate_channels // 16, gate_channels)
                )
        self.mlp3 = nn.Sequential(
                Flatten(),
                nn.Linear(gate_channels, gate_channels // 32),
                nn.ReLU(),
                nn.Linear(gate_channels // 32, gate_channels)
            )


        self.pool_types = pool_types
    def forward(self, x):

        channel_att_sum = None
        for pool_type in self.pool_types:
                if pool_type=='avg':
                    avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                    channel_att_raw = self.mlp( avg_pool ) +self.mlp2( avg_pool ) +self.mlp3( avg_pool )
                elif pool_type=='max':
                    max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                    channel_att_raw = self.mlp( max_pool ) + self.mlp2( max_pool ) + self.mlp3( max_pool )

                if channel_att_sum is None:
                    channel_att_sum = channel_att_raw
                else:
                    channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scale


class ChannelPooling(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class Spatial_Wise_Attention(nn.Module):
    def __init__(self):
        super(Spatial_Wise_Attention, self).__init__()
        self.compress = ChannelPooling()

        self.spatial = BasicConv(2, 1, 7, stride=1, padding=(7-1) // 2, relu=False)
        self.spatial2 = BasicConv(2, 1, 5, stride=1, padding=(5 - 1) // 2, relu=False)
        self.spatial3 = BasicConv(2, 1, 3, stride=1, padding=(3 - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress) + self.spatial2(x_compress) + self.spatial3(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale


class RMCSAM(nn.Module):
    def __init__(self, gate_channels, pool_types=['avg', 'max']):
        super(RMCSAM, self).__init__()

        self.Channel_Wise_Attention = Channel_Wise_Attention(gate_channels, pool_types)
        self.Spatial_Wise_Attention = Spatial_Wise_Attention()

    def forward(self, x):
        res = x
        x_out = self.Channel_Wise_Attention(x)
        x_out_2 = self.Spatial_Wise_Attention(res)

        xb = x_out + x_out_2
        resb = xb
        x_outb = self.Channel_Wise_Attention(xb)
        x_out_2b = self.Spatial_Wise_Attention(resb)

        xc = x_outb + x_out_2b
        resc = xc
        x_outc = self.Channel_Wise_Attention(xc)
        x_out_2c = self.Spatial_Wise_Attention(resc)

        return x_outc + x_out_2c
