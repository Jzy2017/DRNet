# full assembly of the sub-parts to form the complete net

# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
from common import *
from operations import *
from torch.autograd import Variable
from genotypes import UPSAMPLE_PRIMITIVE
from genotypes import SKIPUP_PRIMITIVE
from genotypes import SKIPDOWN_PRIMITIVE
from genotypes import Genotype
from genotypes import Select

class MixedUp(nn.Module):
  def __init__(self, C_in, C_out, stride=2):
    super(MixedUp, self).__init__()
    self._ops = nn.ModuleList()

    for primitive in UPSAMPLE_PRIMITIVE:
        op = UPSAMPLE_OPS[primitive](C_in, C_out, stride)
        self._ops.append(op)
  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))
#把数据x丢进所有操作，加权，权重为softmax后的α

class UpsampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpsampleBlock, self).__init__()
        self._ops = nn.ModuleList()     #上采样cell的操作
        op = MixedUp(C_in=in_channel,C_out=out_channel)
        self._ops.append(op)

    def forward(self, data, weights):
        s0 = data
        states = [s0]
        offset = 0
        s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
        states.append(s)
        return torch.cat(states[-1:], dim=1)

class Dilconv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(Dilconv, self).__init__()
    self.op = nn.Sequential(
        nn.Conv2d(C_in, C_out, 3, padding=1),
        nn.BatchNorm2d(C_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(C_out, C_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(C_out),
        nn.ReLU(inplace=True))

  def forward(self, x):
    return self.op(x)

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv1, self).__init__()
        self.conv = nn.Sequential(
            # double_conv(in_ch, out_ch),
            Dilconv(in_ch, out_ch, 3, 1, 1, 1, affine=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv2, self).__init__()
        self.conv = nn.Sequential(
            # double_conv(in_ch, out_ch),
            Dilconv(in_ch, out_ch, 3, 1, 2, 2, affine=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv3, self).__init__()
        self.conv = nn.Sequential(
            # double_conv(in_ch, out_ch),
            Dilconv(in_ch, out_ch, 3, 1, 3, 3, affine=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class down1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down1, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            # double_conv(in_ch, out_ch),
            Dilconv(in_ch, out_ch, 3, 1, 1, 1, affine=True)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class down2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down2, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            # double_conv(in_ch, out_ch),
            Dilconv(in_ch, out_ch, 3, 1, 2, 2, affine=True)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class down3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down3, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            # double_conv(in_ch, out_ch),
            Dilconv(in_ch, out_ch, 3, 1, 3, 3, affine=True)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()

        #  but my machine do not have enough memory to handle all those weights
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = UpsampleBlock(in_ch, in_ch)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2, weights_up):
        x1 = self.up(x1, weights_up)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class DownsampleBlock(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3,bias=True,pad='reflection',act_fun='LeakyReLU',downsample_mode='stride'):
        super(DownsampleBlock, self).__init__()
        self.op = nn.Sequential(
            down_conv(in_f=in_channel, out_f=out_channel, kernel_size=kernel_size, stride=2, bias=bias, pad=pad,
                 downsample_mode=downsample_mode),
            bn(num_features=out_channel),
            act(act_fun=act_fun)
        )

    def forward(self, data):
        return self.op(data)

class down0_1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down0_1, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            bn(num_features=out_ch),
            act(act_fun='LeakyReLU')
        )

    def forward(self, x):
        x = self.op(x)
        return x

class skip_up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(skip_up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class up1_0(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up1_0, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels=6, n_classes=3,
                 skip_index=[[0, 0, 0, 0],
                             [0, 0, 0, 0]]):

        super(UNet, self).__init__()
        # 选择skip
        self.skip_index = skip_index
        # 初始化α
        self._initialize_alphas()

        self.iteration = 3

        self.inc1 = inconv1(n_channels, 32)
        self.inc2 = inconv2(n_channels, 32)
        self.inc3 = inconv3(n_channels, 32)

        self.down1_1 = down1(32, 64)
        self.down1_2 = down1(64, 128)
        self.down1_3 = down1(128, 256)
        self.down1_4 = down1(256, 256)

        self.down2_1 = down2(32, 64)
        self.down2_2 = down2(64, 128)
        self.down2_3 = down2(128, 256)
        self.down2_4 = down2(256, 256)

        self.down3_1 = down3(32, 64)
        self.down3_2 = down3(64, 128)
        self.down3_3 = down3(128, 256)
        self.down3_4 = down3(256, 256)

        self.up1 = up(1536, 384)
        self.up2 = up(768, 192)
        self.up3 = up(384, 96)
        self.up4 = up(192, 48)

        self.skip_down_0_1 = down0_1(in_ch=n_channels, out_ch=48)
        self.skip_down_1_2 = DownsampleBlock(in_channel=96, out_channel=96)
        self.skip_down_2_3 = DownsampleBlock(in_channel=192, out_channel=192)
        self.skip_down_3_4 = DownsampleBlock(in_channel=384, out_channel=384)

        self.skip_up_4_3 = skip_up(768, 192)
        self.skip_up_3_2 = skip_up(384, 96)
        self.skip_up_2_1 = skip_up(192, 48)
        self.skip_up_1_0 = up1_0(96, n_classes)

        self.outc = outconv(48, n_classes)

    def forward(self, x):
        # 上采样权重
        weights_up = F.softmax(self.alphas_upsample, dim=-1)
        input = x
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            self.x1_1 = self.inc1(x)
            self.x1_2 = self.down1_1(self.x1_1)
            self.x1_3 = self.down1_2(self.x1_2)
            self.x1_4 = self.down1_3(self.x1_3)
            self.x1_5 = self.down1_4(self.x1_4)

            self.x2_1 = self.inc2(x)
            self.x2_2 = self.down2_1(self.x2_1)
            self.x2_3 = self.down2_2(self.x2_2)
            self.x2_4 = self.down2_3(self.x2_3)
            self.x2_5 = self.down2_4(self.x2_4)

            self.x3_1 = self.inc3(x)
            self.x3_2 = self.down3_1(self.x3_1)
            self.x3_3 = self.down3_2(self.x3_2)
            self.x3_4 = self.down3_3(self.x3_3)
            self.x3_5 = self.down3_4(self.x3_4)

            self.x1 = torch.cat([self.x1_1, self.x2_1, self.x3_1], dim=1)
            self.x2 = torch.cat([self.x1_2, self.x2_2, self.x3_2], dim=1)
            self.x3 = torch.cat([self.x1_3, self.x2_3, self.x3_3], dim=1)
            self.x4 = torch.cat([self.x1_4, self.x2_4, self.x3_4], dim=1)
            self.x5 = torch.cat([self.x1_5, self.x2_5, self.x3_5], dim=1)

            self.x6 = self.up1(self.x5, self.x4, weights_up)
            if self.skip_index[0][0]:
                self.x6 = self.x6 + self.alphas_skipdown[0] * self.skip_down_3_4(self.x3)

            self.x7 = self.up2(self.x6, self.x3, weights_up)
            if self.skip_index[0][1]:
                self.x7 = self.x7 + self.alphas_skipdown[1] * self.skip_down_2_3(self.x2)
            if self.skip_index[1][0]:
                self.x7 = self.x7 + self.alphas_skipup[0] * self.skip_up_4_3(self.x4)

            self.x8 = self.up3(self.x7, self.x2, weights_up)
            if self.skip_index[0][2]:
                self.x8 = self.x8 + self.alphas_skipdown[2] * self.skip_down_1_2(self.x1)
            if self.skip_index[1][1]:
                self.x8 = self.x8 + self.alphas_skipup[1] * self.skip_up_3_2(self.x3)

            self.x9 = self.up4(self.x8, self.x1, weights_up)
            if self.skip_index[0][3]:
                self.x9 = self.x9 + self.alphas_skipdown[3] * self.skip_down_0_1(x)
            if self.skip_index[1][2]:
                self.x9 = self.x9 + self.alphas_skipup[2] * self.skip_up_2_1(self.x2)

            self.y = self.outc(self.x9)
            if self.skip_index[1][3]:
                self.y = self.y + self.alphas_skipup[3] * self.skip_up_1_0(self.x1)

            x = self.y + input


        # return input - x
        return x

    def _initialize_alphas(self):
        up_num_ops = len(UPSAMPLE_PRIMITIVE)
        skipup_num_ops = len(SKIPUP_PRIMITIVE)
        skipdown_num_ops = len(SKIPDOWN_PRIMITIVE)
        self.alphas_upsample = Variable(1e-3 * torch.randn(1, up_num_ops), requires_grad=True)
        self.alphas_skipup = torch.tensor(1e-3 * torch.randn(skipup_num_ops), requires_grad=True)
        self.alphas_skipdown = torch.tensor(1e-3 * torch.randn(skipdown_num_ops), requires_grad=True)
        self._arch_parameters = [
            self.alphas_upsample,
            self.alphas_skipup,
            self.alphas_skipdown
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        def parse_up(weights):  # up结点
            gene = []
            start = 0
            W = weights[start:].copy()
            i = 0
            while i < 2:
                k_best = None
                for k in range(2):
                    if k_best is None or W[0][k] > W[0][k_best]:
                        k_best = k
                W[0][k_best] = 0
                gene.append(UPSAMPLE_PRIMITIVE[k_best])
                i = i + 1
            return gene

        gene_upsample = parse_up(F.softmax(self.alphas_upsample, dim=-1).data.cpu().numpy())
        concat = range(2, 3)
        genotype = Genotype(
            upsample=gene_upsample, upsample_concat=concat
        )
        return genotype

    def select_skip(self):
        def parse_up(weights):
            gene = []
            start = 0
            W = weights[start:].copy()
            i = 0
            while i< 4:
                k_best = None
                for k in range(len(W)):
                    if k_best is None or W[k] > W[k_best]:
                        k_best = k
                W[k_best]=0
                gene.append(SKIPUP_PRIMITIVE[k_best])
                i = i+1
            return gene

        def parse_down(weights):
            gene = []
            start = 0
            W = weights[start:].copy()
            i = 0
            while i< 4:
                k_best = None
                for k in range(len(W)):
                    if k_best is None or W[k] > W[k_best]:
                        k_best = k
                W[k_best]=0
                gene.append(SKIPDOWN_PRIMITIVE[k_best])
                i = i+1
            return gene

        gene_skipup = parse_up(F.softmax(self.alphas_skipup, dim=-1).data.cpu().numpy())
        gene_skipdown = parse_down(F.softmax(self.alphas_skipdown, dim=-1).data.cpu().numpy())
        concat = range(2,3)
        genotype = Select(
            Skip_up=gene_skipup, skipup_concat=concat,
            Skip_down = gene_skipdown, skipdown_concat = concat,
        )
        return genotype

if __name__ == '__main__':
    a = torch.randn(4, 3, 256, 256)
    # print('-------------------------------------------------------------------------------------------------------------')
    # criterion = nn.MSELoss()
    mode = UNet()
    b = mode(a)
    # print('-------------------------------------------------------------------------------------------------------------')
    # print(b.size())
    b = a[:, 0, :, :]
    print(b.size())