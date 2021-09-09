from MEDFE.training.loss import TotalLoss
from MEDFE.network.medfe import MEDFE
from MEDFE.training.wgan import Discriminator, train_discriminator

import torch
from torch import autograd
import torch.optim as optim
import torch.nn as nn
from torch.nn.functional import interpolate
from Model_TP.template import BASE


class LBAM(BASE):
    def __init__(self, in_channels, out_channels, args, path, ckpt):
        super().__init__(args.fine_tune, ckpt)

        self.save_hyperparameters()
        self.automatic_optimization = False

        self.generator = MEDFE(in_channels=in_channels,
                               batch_norm=True, use_bpa=False, use_branch=True, use_res=True,
                               branch_channels=512 // 4, channels=64 // 4)
        self.discriminator_wgan_global = Discriminator((args.batch_size, 3, args.size, args.size), name='global')

        self.criterion = TotalLoss(self.discriminator_wgan_global, None)

    def forward(self, x):
        mask_img, mask, = x
        self.generator.set_mask(mask)
        pred = self.generator(mask_img)
        return pred

    def training_step(self, batch, batch_index):
        img, mask_img, gradient, mask_gradient, img_gray, mask_gray, mask, index = batch
        inputs = None
        y = img
        mask_y = mask_img

        y_32 = interpolate(y, 32)

        if self.hparams.args.stage == 1:  # for raw
            inputs = torch.cat([mask_img, mask], dim=1)
        elif self.hparams.args.stage == 2:  # for train add_channel
            inputs = torch.cat([mask_img, mask, gradient], dim=1)
        elif self.hparams.args.stage == 3:  # for fine_tune add_channel
            inputs = torch.cat([mask_img, mask_gradient, mask], dim=1)
            pred_gradient = self.eg_model(inputs)
            com_gradient = pred_gradient * mask + mask_gradient
            inputs = torch.cat([mask_img, mask, com_gradient], dim=1)

        x = (inputs, mask)
        pred = self.forward(x)
        merged = pred * mask + mask_y

        # train discriminator
        wgan_global_real_hist = y
        if wgan_global_real_hist.shape[0] > 4 * self.hparams.args.batch_size:
            wgan_global_real_hist = wgan_global_real_hist[:-4 * self.hparams.args.batch_size]
        wgan_global_real_hist = torch.cat((wgan_global_real_hist, y), dim=0).detach()
        train_discriminator(self.discriminator_wgan_globa, self.dis_optimizer, gt=wgan_global_real_hist, out=merged)

        # generator
        # determine the loss
        single_loss = self.criterion(
            i_gt_small=y_32,
            i_st=batch["gt_smooth"],    # structural_image: raw: matlab or edge or gradient
            i_ost=model.struct_branch_img,
            i_ote=model.tex_branch_img,
            i_gt_large=y,
            i_out_large=merged,
            i_gt_sliced=None,
            i_out_sliced=None,
            mask_size=256 * 256 - torch.sum(model.mask[0]),
        )

        self.log('train/loss', single_loss)

        img = torch.cat([mask_y[:6], y[:6], pred[:6], merged[:6]], dim=0)
        if self.global_step % self.hparams.args.log_every_n_steps == 0:
            self.log_img(img, "train")

        self.train_val_step_mid(y, pred, merged, "train")
        self.log_mae_mse(y, pred, merged, "train")

        single_loss.backward()
        self.gen_optimizer.step()

    def validation_step(self, batch, batch_idx, optimizer_idx=None):
        img, mask_img, gradient, mask_gradient, img_gray, mask_gray, mask, index = batch
        inputs = None
        y = img
        mask_y = mask_img

        y_32 = interpolate(y, 32)

        if self.hparams.args.stage == 1:  # for raw
            inputs = torch.cat([mask_img, mask], dim=1)
        elif self.hparams.args.stage == 2:  # for train add_channel
            inputs = torch.cat([mask_img, mask, gradient], dim=1)
        elif self.hparams.args.stage == 3:  # for fine_tune add_channel
            inputs = torch.cat([mask_img, mask_gradient, mask], dim=1)
            pred_gradient = self.eg_model(inputs)
            com_gradient = pred_gradient * mask + mask_gradient
            inputs = torch.cat([mask_img, mask, com_gradient], dim=1)

        x = (inputs, mask)
        pred = self.forward(x)
        merged = pred * mask + mask_y

        # generator
        # determine the loss
        single_loss = self.criterion(
            i_gt_small=y_32,
            i_st=batch["gt_smooth"],
            i_ost=model.struct_branch_img,
            i_ote=model.tex_branch_img,
            i_gt_large=y,
            i_out_large=merged,
            i_gt_sliced=None,
            i_out_sliced=None,
            mask_size=256 * 256 - torch.sum(model.mask[0]),
        )

        self.log('val/gen_loss', single_loss)

        img = torch.cat([mask_y[:6], y[:6], pred[:6], merged[:6]], dim=0)
        self.log_img(img, "val")

        self.train_val_step_mid(y, pred, merged, "val")
        self.log_mae_mse(y, pred, merged, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        img, mask_img, gradient, mask_gradient, img_gray, mask_gray, mask, index = batch
        inputs = None
        y = img
        mask_y = mask_img

        y_32 = interpolate(y, 32)

        if self.hparams.args.stage == 1:  # for raw
            inputs = torch.cat([mask_img, mask], dim=1)
        elif self.hparams.args.stage == 2:  # for train add_channel
            inputs = torch.cat([mask_img, mask, gradient], dim=1)
        elif self.hparams.args.stage == 3:  # for fine_tune add_channel
            inputs = torch.cat([mask_img, mask_gradient, mask], dim=1)
            pred_gradient = self.eg_model(inputs)
            com_gradient = pred_gradient * mask + mask_gradient
            inputs = torch.cat([mask_img, mask, com_gradient], dim=1)

        x = (inputs, mask)
        pred = self.forward(x)
        merged = pred * mask + mask_y

        return self.test_step_mid(y, pred, merged, mask_y, dataloader_idx)

    def configure_optimizers(self):
        self.gen_optimizer = optim.Adam(
            params=self.generator.parameters(),
            lr=self.hparams.args.g_lr,  # 0.0001
            betas=(self.hparams.args.g_beta1, self.hparams.args.g_beta2)
        )

        self.dis_optimizer = optim.Adam(
            params=self.discriminator_wgan_global.parameters(),
            lr=self.hparams.args.d_lr,  # 0.00001
            betas=(self.hparams.args.d_beta1, self.hparams.args.d_beta2)
        )
        return ({"optimizer": self.dis_optimizer},
                {"optimizer": self.gen_optimizer})


if __name__ == '__main__':
    # self, lr, in_channels, out_channels, args):
    # input_channels=3, residual_blocks=8, threshold=threshold
    model = InpaintingModel(in_channels=4, out_channels=3, args=None)
    print(model.generator)
    print(model.discriminator)
