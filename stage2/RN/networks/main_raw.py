from __future__ import print_function
import argparse
from math import log10
import numpy as np

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from module_util import *
from dataset import build_dataloader
import pdb
import socket
import time
from skimage import io
# from skimage.measure import compare_psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from models import InpaintingModel
from tensorboardX import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description='Region Normalization for Image Inpainting')
parser.add_argument('--bs', type=int, default=38, help='training batch size')
parser.add_argument('--input_size', type=int, default=256, help='input image size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=2, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=67454, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=2, type=int, help='number of gpu')

parser.add_argument('--img_flist', type=str,
                    default='/data/exdata_zhr/wm/paris/train')
parser.add_argument('--mask_flist', type=str,
                    default='/data/exdata_zhr/wm/paris/1000_train_mask')

parser.add_argument('--model_type', type=str, default='RN')
parser.add_argument('--threshold', type=float, default=0.8)
parser.add_argument('--pretrained_sr', default='./ckpt/places2/raw/weights/x_gpu14RN_0p1GAN0p8thre_bs_38_epoch_2.pth', help='pretrained base modules')
parser.add_argument('--pretrained', type=bool, default=False)

parser.add_argument('--save_folder', default='./ckpt/paris/raw/weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='0p1GAN0p8thre', help='Location to save checkpoint models')
parser.add_argument('--print_interval', type=int, default=100, help='how many steps to print the results out')
parser.add_argument('--render_interval', type=int, default=1000, help='how many steps to save a checkpoint')
parser.add_argument('--l1_weight', type=float, default=1.0)
parser.add_argument('--gan_weight', type=float, default=0.1)
parser.add_argument('--update_weight_interval', type=int, default=5000, help='how many steps to update losses weighing')

parser.add_argument('--with_test', default=False, action='store_true', help='Train with testing?')
parser.add_argument('--test', default=False, action='store_true', help='Test modules')
parser.add_argument('--test_img_flist', type=str,
                    default='/home/Sxd13/Sxd14StuA/wangmeng/data/inpainting/celebA/celeba_val')
parser.add_argument('--test_mask_flist', type=str,
                    default='/home/Sxd13/Sxd14StuA/wangmeng/data/inpainting/celebA/500_val_mask')

parser.add_argument('--tb', default=True, action='store_true', help='Use tensorboardX?')

opt = parser.parse_args()
gpus_list = [0, 1]
hostname = str(socket.gethostname())
opt.save_folder += opt.prefix
cudnn.benchmark = True
if not os.path.exists(opt.save_folder):
    os.makedirs(opt.save_folder)
print(opt)
torch.autograd.set_detect_anomaly(True)


def train(epoch):
    iteration, avg_g_loss, avg_d_loss, avg_l1_loss, avg_gan_loss = 0, 0, 0, 0, 0
    last_l1_loss, last_gan_loss, cur_l1_loss, cur_gan_loss = 0, 0, 0, 0
    t0 = time.time()
    t_io1 = time.time()
    for batch in training_data_loader:
        gt, mask, index = batch
        t_io2 = time.time()
        if cuda:
            gt = gt.cuda()
            mask = mask.cuda()

        # render(epoch, iteration, mask, prediction.detach(), gt)
        # os._exit()

        # Compute Loss

        # 下面是训练判别器，要阻断prediction的梯度
        # 判别器想区分开真假-->样本:（gt, real）, (pred, fake)
        model.discriminator.train(True)
        model.generator.train(False)
        prediction = model.generator(gt, mask)
        merged_result = prediction * mask + gt * (1 - mask)

        d_real = model.discriminator(gt)
        d_fake = model.discriminator(merged_result.detach())
        d_real_loss = model.adversarial_loss(d_real, is_real=True, is_disc=False)
        d_fake_loss = model.adversarial_loss(d_fake, False, True)
        d_loss = (d_real_loss + d_fake_loss) / 2

        avg_d_loss += d_loss.data.item()

        d_loss.backward()
        model.dis_optimizer.step()
        model.dis_optimizer.zero_grad()

        # 下面是利用训练一次之后的判别器来得到对抗损失，prediction需要反向梯度
        # 生成器想让判别器认为pred是真的，-->（pred, real）
        model.discriminator.train(False)
        model.generator.train(True)
        # prediction = modules.generator(gt, mask)
        # merged_result = prediction * mask + gt * (1 - mask)

        g_fake = model.discriminator.forward(merged_result)
        g_gan_loss = model.adversarial_loss(g_fake, is_real=True, is_disc=False)

        g_l1_loss = model.l1_loss(gt, merged_result) / torch.mean(mask)
        # g_l1_loss = modules.l1_loss(gt, prediction) / torch.mean(mask)
        # TODO
        g_loss = model.l1_weight * g_l1_loss + model.gan_weight * g_gan_loss

        # Record
        cur_l1_loss += g_l1_loss.data.item()
        cur_gan_loss += g_gan_loss.data.item()
        avg_l1_loss += g_l1_loss.data.item()
        avg_gan_loss += g_gan_loss.data.item()
        avg_g_loss += g_loss.data.item()

        # Backward
        g_loss.backward()
        model.gen_optimizer.step()
        model.gen_optimizer.zero_grad()

        model.global_iter += 1
        iteration += 1
        t1 = time.time()
        td, t0 = t1 - t0, t1

        if iteration % opt.print_interval == 0:
            print("=> Epoch[{}]({}/{}): Avg L1 loss: {:.6f} | G loss: {:.6f} | Avg D loss: {:.6f} || Timer: {:.4f} sec. | IO: {:.4f}".format(
                epoch, iteration, len(training_data_loader), avg_l1_loss/opt.print_interval, avg_g_loss/opt.print_interval, avg_d_loss/opt.print_interval, td, t_io2-t_io1), flush=True)
            # print("=> Epoch[{}]({}/{}): Avg G loss: {:.6f} || Timer: {:.4f} sec. || IO: {:.4f}".format(
            # epoch, iteration, len(training_data_loader), avg_g_loss/opt.print_interval, td, t_io2-t_io1), flush=True)

            if opt.tb:
                writer.add_scalar('scalar/G_loss', avg_g_loss/opt.print_interval, model.global_iter)
                writer.add_scalar('scalar/D_loss', avg_d_loss/opt.print_interval, model.global_iter)
                writer.add_scalar('scalar/G_l1_loss', avg_l1_loss/opt.print_interval, model.global_iter)
                writer.add_scalar('scalar/G_gan_loss', avg_gan_loss/opt.print_interval, model.global_iter)

            avg_g_loss, avg_d_loss, avg_l1_loss, avg_gan_loss = 0, 0, 0, 0

        t_io1 = time.time()

        if iteration % opt.render_interval == 0:
            render(epoch, iteration, mask, merged_result.detach(), gt)
            if opt.with_test:
                print("Testing 1000 images...")
                test_psnr = test(model, test_data_loader)
                if opt.tb:
                    writer.add_scalar('scalar/test_PSNR', test_psnr, model.global_iter)
                    print("PSNR: ", test_psnr)

        # if iteration % opt.update_weight_interval == 0:
        #     if last_l1_loss == 0:
        #         last_l1_loss, last_gan_loss = cur_l1_loss, cur_gan_loss
        #     weights = dynamic_weigh([last_l1_loss, last_gan_loss], [cur_l1_loss, cur_gan_loss], T=1)
        #     modules.l1_weight, modules.gan_weight = weights[0], weights[1]
        #     print("===> losses weights changing: [l1, gan] = {:.4f}, {:.4f}".format(modules.l1_weight, modules.gan_weight))
        #     last_l1_loss, last_gan_loss = cur_l1_loss, cur_gan_loss


def dynamic_weigh(last_losses, cur_losses, T=20):
    # input lists
    last_losses, cur_losses = torch.Tensor(last_losses), torch.Tensor(cur_losses)
    w = torch.exp((cur_losses / last_losses) / T)
    return (last_losses.size(0) * w / torch.sum(w)).cuda()


def render(epoch, iter, mask, output, gt):

    name_pre = 'render/'+str(epoch)+'_'+str(iter)+'_'

    # input: (bs,3,256,256)
    input = gt * (1 - mask) + mask
    input = input[0].permute(1,2,0).cpu().numpy()
    io.imsave(name_pre+'input.png', (input*255).astype(np.uint8))

    # mask: (bs,1,256,256)
    mask = mask[0,0].cpu().numpy()
    io.imsave(name_pre+'mask.png', (mask*255).astype(np.uint8))

    # output: (bs,3,256,256)
    output = output[0].permute(1,2,0).cpu().numpy()
    io.imsave(name_pre+'output.png', (output*255).astype(np.uint8))

    # gt: (bs,3,256,256)
    gt = gt[0].permute(1,2,0).cpu().numpy()
    io.imsave(name_pre+'gt.png', (gt*255).astype(np.uint8))


def test(gen, dataloader):
    model = gen.eval()
    psnr = 0
    count = 0
    for batch in dataloader:
        gt_batch, mask_batch, index = batch
        if cuda:
            gt_batch = gt_batch.cuda()
            mask_batch = mask_batch.cuda()
        with torch.no_grad():
            pred_batch = model.generator(gt_batch, mask_batch)
        for i in range(gt_batch.size(0)):
            gt, pred = gt_batch[i], pred_batch[i]
            psnr += compare_psnr(pred.permute(1, 2, 0).cpu().numpy(), gt.permute(1, 2, 0).cpu().numpy(), data_range=1)
            count += 1
    return psnr / count


def checkpoint(epoch):
    model_out_path = opt.save_folder+'/'+'x_'+hostname + \
        opt.model_type+"_"+opt.prefix + "_bs_{}_epoch_{}.pth".format(opt.bs, epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':
    if opt.tb:
        writer = SummaryWriter("./ckpt/places2/raw/runs")

    # Set the GPU mode
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    # Set the random seed
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed_all(opt.seed)

    # Model
    model = InpaintingModel(g_lr=opt.lr, d_lr=(0.1 * opt.lr),
                            l1_weight=opt.l1_weight, gan_weight=opt.gan_weight, iter=0, threshold=opt.threshold)
    print('---------- Networks architecture -------------')
    print("Generator:")
    print_network(model.generator)
    print("Discriminator:")
    print_network(model.discriminator)
    print('----------------------------------------------')
    initialize_weights(model, scale=0.1)

    if cuda:
        model = model.cuda()
        if opt.gpus > 1:
            model.generator = torch.nn.DataParallel(model.generator, device_ids=gpus_list)
            model.discriminator = torch.nn.DataParallel(model.discriminator, device_ids=gpus_list)

    # Load the pretrain modules.
    if opt.pretrained:
        model_name = os.path.join(opt.pretrained_sr)
        print('pretrained modules: %s' % model_name)
        if os.path.exists(model_name):
            pretained_model = torch.load(model_name, map_location=lambda storage, loc: storage)
            model.load_state_dict(pretained_model)  # torch.save(modules.state_dict(), model_out_path)
            print('Pre-trained modules is loaded.')
            print(' Current: G learning rate:', model.g_lr,
                  ' | L1 loss weight:', model.l1_weight,
                  ' | GAN loss weight:', model.gan_weight)

    # Datasets
    print('===> Loading datasets')
    training_data_loader = build_dataloader(
        flist=opt.img_flist,
        mask_flist=opt.mask_flist,
        augment=True,
        training=True,
        input_size=opt.input_size,
        batch_size=opt.bs,
        num_workers=opt.threads,
        shuffle=True
    )

    print('===> Loaded datasets')

    if opt.test or opt.with_test:
        test_data_loader = build_dataloader(
            flist=opt.test_img_flist,
            mask_flist=opt.test_mask_flist,
            augment=False,
            training=False,
            input_size=opt.input_size,
            batch_size=64,
            num_workers=opt.threads,
            shuffle=False
        )
        print('===> Loaded test datasets')

    if opt.test:
        test_psnr = test(model, test_data_loader)
        os._exit(0)

    # Start training
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):

        train(epoch)

        count = (epoch-1)
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        for param_group in model.gen_optimizer.param_groups:
            param_group['lr'] = model.g_lr * (0.8 ** count)
            print('===> Current G learning rate: ', param_group['lr'])
        for param_group in model.dis_optimizer.param_groups:
            param_group['lr'] = model.d_lr * (0.8 ** count)
            print('===> Current D learning rate: ', param_group['lr'])

        if (epoch+1) % (opt.snapshots) == 0:
            checkpoint(epoch)
if opt.tb:
    writer.close()
