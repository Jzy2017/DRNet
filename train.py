import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import utils
import torch.nn.functional as F
from DerainDataset import *
from architect import Architect
from utils import *
from torch.optim.lr_scheduler import MultiStepLR
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from SSIM import SSIM
from utils.image_io import save_graph
# from networks import *
from P_3UNet_Darts import UNet
from percep_loss import networks
from percep_loss import vgg


parser = argparse.ArgumentParser(description="PReNet_train")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[30,50,80], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument("--save_path", type=str, default="logs/", help='path to save models and log files')
parser.add_argument("--save_freq",type=int,default=3,help='save intermediate model')
parser.add_argument("--data_path",type=str, default="./train/Rain100H",help='path to training data')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
opt = parser.parse_args()

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

self_device = torch.device('cuda:{}'.format('0'))
self_vgg = vgg.Vgg19(requires_grad=False).to(self_device)
criterionVgg = networks.VGGLoss1(self_device, vgg=self_vgg, normalize=False)

def main():
    losses = []

    print('Loading dataset ...\n')
    dataset_train = Dataset(data_path=opt.data_path)
    num_train = dataset_train.__len__()
    indices = list(range(num_train))
    split = int(np.floor(opt.train_portion * num_train))
    loader_train = DataLoader(dataset=dataset_train, num_workers=0, batch_size=opt.batch_size,
                              sampler=torch.utils.data.sampler.SequentialSampler(indices[:split]),pin_memory=True)
    loader_valid = DataLoader(dataset=dataset_train, num_workers=0, batch_size=opt.batch_size,
                              sampler=torch.utils.data.sampler.SequentialSampler(indices[split:num_train]),pin_memory=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build model
    model = UNet()

    # loss function
    # criterion = nn.MSELoss(size_average=False)
    criterion = SSIM()
    mse = torch.nn.MSELoss()

    # Move to GPU
    if opt.use_gpu:
        model = model.cuda()
        criterion.cuda()
        mse.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)  # learning rates

    architect = Architect(model, opt)


    # start training
    for epoch in range(opt.epochs):
        scheduler.step(epoch)
        print('epoch:', epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        # 选出权重值大的2个节点并保留操作
        genotype = model.genotype()
        print('genotype = %s', genotype)
        # print(F.softmax(model.alphas_conv, dim=-1))
        print(F.softmax(model.alphas_upsample, dim=-1))
        select = model.select_skip()
        print('select_skip = %s', select)
        print(F.softmax(model.alphas_skipup, dim=-1))
        print(F.softmax(model.alphas_skipdown, dim=-1))

        ## epoch training start
        objs = utils.AvgrageMeter()
        for i, (input_train, target_train) in enumerate(loader_train, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            input_train, target_train = Variable(input_train), Variable(target_train)
            n = input_train.size(0)
            if opt.use_gpu:
                input_train, target_train = input_train.cuda(), target_train.cuda()

            #处理α
            (input_search, target_search) = next(iter(loader_valid))
            blended_search = Variable(input_search, requires_grad=False).cuda()
            target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)
            output_search = model(blended_search)
            architect.step(output_search, target_search, blended_search)
            aloss = 0.1 * mse(output_search, target_search) + (1-criterion(output_search, target_search))
            vgg_loss = criterionVgg(target_search, output_search) / criterionVgg(blended_search, output_search)
            # a = vgg_loss / (vgg_loss+aloss)
            # v = aloss / (vgg_loss+aloss)
            print('aloss: %.4f' % (aloss.item()) + ' || vgg_loss: %.4f' % ( vgg_loss.item()))
            # aloss = a * aloss + v * vgg_loss
            aloss += 0.1 * vgg_loss


            out_train = model(input_train)
            pixel_metric = criterion(target_train, out_train)
            loss = 0.1 * mse(target_train, out_train) + (1-pixel_metric)
            vgg_loss = criterionVgg(target_train, out_train) / criterionVgg(input_train, out_train)
            # a = vgg_loss / (vgg_loss + loss)
            # v = loss / (vgg_loss + loss)
            # loss = a * loss + v * vgg_loss
            loss += 0.1 * vgg_loss

            loss.backward()
            optimizer.step()

            objs.update(aloss.data.item(), n)

        print(objs.avg)
        ## epoch training end


        # # save model
        # torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))
        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch+1)))
            # save model
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))

        losses.append(objs.avg)
        save_graph(str(epoch) + "_aloss", losses,
                   output_path=opt.save_path + '/')


if __name__ == "__main__":
    if opt.preprocess:
        if opt.data_path.find('RainTrainH') != -1:
            prepare_data_RainTrainH(data_path=opt.data_path, patch_size=128, stride=80)
        elif opt.data_path.find('RainTrainL') != -1:
            prepare_data_RainTrainL(data_path=opt.data_path, patch_size=128, stride=80)
        elif opt.data_path.find('Rain12600') != -1:
            prepare_data_Rain12600(data_path=opt.data_path, patch_size=128, stride=100)
        else:
            print('unkown datasets: please define prepare data function in DerainDataset.py')

    main()
