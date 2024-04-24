from collections import namedtuple

# Genotype = namedtuple('Genotype', 'conv conv_concat upsample upsample_concat')
Genotype = namedtuple('Genotype', 'upsample upsample_concat')
Select = namedtuple('Select', 'Skip_up skipup_concat Skip_down skipdown_concat')

UPSAMPLE_PRIMITIVE = [
  'bilinear',
  'bicubic',
]
# UPSAMPLE_PRIMITIVE = [
#   'bilinear',
# ]

# CONV_PRIMITIVE = [
#   # 'skip_connect',
#   # 'avg_pool_3x3',
#   # 'max_pool_3x3'
#   # 'conv_3x3',
#   # 'conv_5x5',
#   # 'conv_7x7',
#   # 'double_conv',
#   # 'sep_conv_3x3',
#   # 'sep_conv_5x5',
#   # 'sep_conv_7x7',
#   # 'dil_conv_3x3',
#   # 'dil_conv_5x5',
#   # 'dil_conv_7x7'
# ]

SKIPUP_PRIMITIVE = [
  'enc4_dec3',
  'enc3_dec2',
  'enc2_dec1',
  'enc1_dec0',
]

SKIPDOWN_PRIMITIVE = [
  'enc3_dec4',
  'enc2_dec3',
  'enc1_dec2',
  'enc0_dec1',
]

