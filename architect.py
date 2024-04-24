import torch
import numpy as np
import torch.nn as nn
from SSIM import SSIM
from percep_loss import networks
from percep_loss import vgg


mse = nn.MSELoss().cuda()
ssim = SSIM().cuda()

self_device = torch.device('cuda:{}'.format('0'))
self_vgg = vgg.Vgg19(requires_grad=False).to(self_device)
criterionVgg = networks.VGGLoss1(self_device, vgg=self_vgg, normalize=False)

class Architect () :
    def __init__(self, model, args):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
            lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

    def step (self, output_valid, target_valid, blended_valid) :
        self.optimizer.zero_grad ()
        self._backward_step(output_valid, target_valid, blended_valid)
        self.optimizer.step()

    def _backward_step (self, output_valid, target_valid, blended_valid) :
        loss = 0.1 * mse(output_valid, target_valid) + (1-ssim(output_valid, target_valid))
        vgg_loss = criterionVgg(target_valid, output_valid) / criterionVgg(blended_valid, output_valid)
        loss += 0.1 * vgg_loss
        loss.backward ()



