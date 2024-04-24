import torch
import torch.nn as nn
# from tensor2tensor.layers.common_layers import sepconv_relu_sepconv

UPSAMPLE_OPS = {
  # 'nearest':       lambda C_in, C_out, stride: BilinearOp(C_in, C_out, stride, upsample_mode='nearest'),
  # 'bilinear':      lambda C_in, C_out, stride: BilinearOp(C_in, C_out, stride, upsample_mode='bilinear'),
  # 'bicubic':       lambda C_in, C_out, stride: BilinearOp(C_in, C_out, stride, upsample_mode='bicubic'),
  # 'skip_connect':  lambda C_in, C_out, stride: Identity() if stride == 1 else ConvUpBN(C_in, C_out),
  'bilinear' : lambda C_in, C_out, stride: nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
  'bicubic'  : lambda C_in, C_out, stride: nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
}

CONV_OPS = {
  'skip_connect' : lambda C_in, C_out: Identity(),
  'avg_pool_3x3' : lambda C_in, C_out: nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C_in, C_out: nn.MaxPool2d(3, stride=1, padding=1),
  'conv_3x3'     : lambda C_in, C_out: conv(C_in, C_out, 3, 1, padding=1),
  'conv_5x5'     : lambda C_in, C_out: conv(C_in, C_out, 5, 1, padding=2),
  'conv_7x7'     : lambda C_in, C_out: conv(C_in, C_out, 7, 1, padding=3),
  'double_conv'  : lambda C_in, C_out: double_conv(C_in, C_out),
  'sep_conv_3x3' : lambda C_in, C_out: SepConv(C_in, C_out, 3, 1, 1, affine=True),
  'sep_conv_5x5' : lambda C_in, C_out: SepConv(C_in, C_out, 5, 1, 2, affine=True),
  'sep_conv_7x7' : lambda C_in, C_out: SepConv(C_in, C_out, 7, 1, 3, affine=True),
  'dil_conv_3x3' : lambda C_in, C_out: DilConv(C_in, C_out, 3, 1, 2, 2, affine=True),
  'dil_conv_5x5' : lambda C_in, C_out: DilConv(C_in, C_out, 5, 1, 4, 2, affine=True),
  'dil_conv_7x7' : lambda C_in, C_out: DilConv(C_in, C_out, 7, 1, 6, 2, affine=True),
}

class conv(nn.Module):
  def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
    super(conv, self).__init__()
    self.op = nn.Sequential(
      nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, padding=padding),
      nn.BatchNorm2d(out_ch),
      nn.ReLU(inplace=True),
    )

  def forward(self, x):
    return self.op(x)

class double_conv(nn.Module):
  def __init__(self, in_ch, out_ch):
    super(double_conv, self).__init__()
    self.op = nn.Sequential(
          nn.Conv2d(in_ch, out_ch, 3, padding=1),
          nn.BatchNorm2d(out_ch),
          nn.ReLU(inplace=True),
          nn.Conv2d(out_ch, out_ch, 3, padding=1),
          nn.BatchNorm2d(out_ch),
          nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.op(x)

class SepConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size, padding=padding),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True)
        )

  def forward(self, x):
    return self.op(x)


class DilConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      # nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in,
                bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out),
      nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.op(x)

class BilinearOp(nn.Module):

  def __init__(self, C_in, C_out, stride, upsample_mode):
    super(BilinearOp, self).__init__()

    activation = nn.ReLU()
    if stride == 2:
      self.op = nn.Sequential(
        activation,  # 激活
        nn.Upsample(scale_factor=stride, mode=upsample_mode),#上采样
        nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),#卷积
        nn.BatchNorm2d(C_out),#归一
      )
    else:
      self.op = nn.Sequential(
        activation,  # 激活
        nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),  # 卷积
        nn.BatchNorm2d(C_out),  # 归一
      )

  def forward(self, x):
    return self.op(x)

class Identity(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return x


# class Zero(nn.Module):
#   def __init__(self):
#     super(Zero, self).__init__()
#
#   def forward(self, x):
#     return x.mul(0.)



#前一个cell是normal和reduce时的转变--------------------------------------------------------------------------------------
class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)


class ConvUpBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size=4, stride=2, padding=1, affine=True):
    super(ConvUpBN, self).__init__()

    activation = nn.ReLU()
    self.op = nn.Sequential(
      activation,
      nn.ConvTranspose2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out),
    )

  def forward(self, x):
    return self.op(x)

# class FactorizedUpsample(nn.Module):
#
#   def __init__(self, C_in, C_out, affine=True):
#     super(FactorizedUpsample, self).__init__()
#     assert C_out % 2 == 0
#     self.relu = nn.ReLU(inplace=False)
#     self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
#     self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
#     self.bn = nn.BatchNorm2d(C_out, affine=affine)
#
#   def forward(self, x):
#     x = self.relu(x)
#     out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
#     out = self.bn(out)
#     return out
