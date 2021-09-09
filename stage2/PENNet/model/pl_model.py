"""
Author: CraneLone
email: wang-zhizhen@outlook.com

file: pl_model
date: 2021/8/19 0019 上午 03:29
desc: 
"""

from PENNet.core.loss import AdversarialLoss, PerceptualLoss, StyleLoss, VGG19
from PENNet.modules.pennet import InpaintGenerator, Discriminator
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from Model_TP.template import BASE


class PENNet(BASE):
    def __init__(self, in_channels, out_channels, args, path, ckpt):
        super().__init__(args.fine_tune, ckpt)
        self.save_hyperparameters()

        self.gan_type = "hinge"
        self.hole_weight = 6
        self.valid_weight = 1
        self.pyramid_weight = 0.5
        self.adversarial_weight = 0.1

        self.adversarial_loss = AdversarialLoss(self.gan_type)
        self.l1_loss = nn.L1Loss()
        self.generator = InpaintGenerator(in_channels, out_channels)
        self.discriminator = Discriminator(in_channels=out_channels, use_sigmoid=self.gan_type != 'hinge')

    def forward(self, x):
        inputs, mask = x
        feats, pred = self.generator(inputs, mask)  # pyramid_imgs, output
        return pred, feats

    def training_step(self, batch, batch_index, optimizer_idx=None):
        img, mask_img, gradient, mask_gradient, img_gray, mask_gray, mask, index = batch
        inputs = None
        y = img
        mask_y = mask_img

        if self.hparams.args.stage == 1:    # for raw
            inputs = mask_img
        elif self.hparams.args.stage == 2:  # for train add_channel
            inputs = torch.cat([mask_img, gradient], dim=1)
        elif self.hparams.args.stage == 3:  # for fine_tune add_channel
            inputs = torch.cat([mask_img, mask_gradient, mask], dim=1)
            pred_gradient = self.eg_model(inputs)
            com_gradient = pred_gradient * mask + mask_gradient
            inputs = torch.cat([mask_img, com_gradient], dim=1)

        x = (inputs, mask)
        pred, feats = self.forward(x)
        merged = pred * mask + mask_y

        # train discriminator
        if optimizer_idx == 0:
            dis_real_feat = self.discriminator(y)
            dis_fake_feat = self.discriminator(merged.detach())
            dis_real_loss = self.adversarial_loss(dis_real_feat, True, True)
            dis_fake_loss = self.adversarial_loss(dis_fake_feat, False, True)
            dis_loss = (dis_real_loss + dis_fake_loss) / 2

            self.log('train/loss_dis', dis_loss)
            return dis_loss

        # train generator
        if optimizer_idx == 1:
            hole_loss = self.l1_loss(pred * mask, y * mask) / torch.mean(mask)
            gen_loss = hole_loss * self.hole_weight
            valid_loss = self.l1_loss(pred * (1 - mask), y * (1 - mask)) / torch.mean(1 - mask)
            gen_loss += valid_loss * self.valid_weight

            pyramid_loss = 0
            if feats is not None:
                for _, f in enumerate(feats):
                    pyramid_loss += self.l1_loss(f, F.interpolate(y,
                                                                  size=f.size()[2:4],
                                                                  mode='bilinear',
                                                                  align_corners=True))
                gen_loss += pyramid_loss * self.pyramid_weight

            self.log_dict({'train/pyramid_loss': pyramid_loss * self.pyramid_weight,
                           "train/hole_loss": hole_loss * self.hole_weight,
                           "train/valid_loss": valid_loss * self.valid_weight,
                           "train/gen_loss": gen_loss})

            img = torch.cat([mask_y[:6], y[:6], pred[:6], merged[:6]], dim=0)
            if self.global_step % self.hparams.args.log_every_n_steps == 0:
                self.log_img(img, "train")

            self.train_val_step_mid(y, pred, merged, "train")
            self.log_mae_mse(y, pred, merged, "train")
            return gen_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=None):
        img, mask_img, gradient, mask_gradient, img_gray, mask_gray, mask, index = batch
        inputs = None
        y = img
        mask_y = mask_img

        if self.hparams.args.stage == 1:    # for raw
            inputs = mask_img
        elif self.hparams.args.stage == 2:  # for train add_channel
            inputs = torch.cat([mask_img, gradient], dim=1)
        elif self.hparams.args.stage == 3:  # for fine_tune add_channel
            inputs = torch.cat([mask_img, mask_gradient, mask], dim=1)
            pred_gradient = self.eg_model(inputs)
            com_gradient = pred_gradient * mask + mask_gradient
            inputs = torch.cat([mask_img, com_gradient], dim=1)

        x = (inputs, mask)
        pred, feats = self.forward(x)
        merged = pred * mask + mask_y

        hole_loss = self.l1_loss(pred * mask, y * mask) / torch.mean(mask)
        gen_loss = hole_loss * self.hole_weight
        valid_loss = self.l1_loss(pred * (1 - mask), y * (1 - mask)) / torch.mean(1 - mask)
        gen_loss += valid_loss * self.valid_weight

        pyramid_loss = 0
        if feats is not None:
            for _, f in enumerate(feats):
                pyramid_loss += self.l1_loss(f, F.interpolate(y,
                                                              size=f.size()[2:4],
                                                              mode='bilinear',
                                                              align_corners=True))
            gen_loss += pyramid_loss * self.pyramid_weight

        self.log_dict({'val/pyramid_loss': pyramid_loss * self.pyramid_weight,
                       "val/hole_loss": hole_loss * self.hole_weight,
                       "val/valid_loss": valid_loss * self.valid_weight,
                       "val/gen_loss": gen_loss})

        img = torch.cat([mask_y[:8], y[:8], pred[:8], merged[:8]], dim=0)
        self.log_img(img, "val")

        self.train_val_step_mid(y, pred, merged, "val")
        self.log_mae_mse(y, pred, merged, "val")
        return gen_loss

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        img, mask_img, gradient, mask_gradient, img_gray, mask_gray, mask, index = batch
        inputs = None
        y = img
        mask_y = mask_img

        if self.hparams.args.stage == 1:    # for raw
            inputs = mask_img
        elif self.hparams.args.stage == 2:  # for train add_channel
            inputs = torch.cat([mask_img, gradient], dim=1)
        elif self.hparams.args.stage == 3:  # for fine_tune add_channel
            inputs = torch.cat([mask_img, mask_gradient, mask], dim=1)
            pred_gradient = self.eg_model(inputs)
            com_gradient = pred_gradient * mask + mask_gradient
            inputs = torch.cat([mask_img, com_gradient], dim=1)

        x = (inputs, mask)
        pred, feats = self.forward(x)
        merged = pred * mask + mask_y

        return self.test_step_mid(y, pred, merged, mask_y, dataloader_idx)

    def configure_optimizers(self):
        gen_optimizer = optim.Adam(
            params=self.generator.parameters(),
            lr=self.hparams.args.g_lr,  # 1e-4
            betas=(self.hparams.args.g_beta1, self.hparams.args.g_beta2)
        )
        g_stepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer=gen_optimizer, milestones=[50, 150], gamma=0.1)

        dis_optimizer = optim.Adam(
            params=self.discriminator.parameters(),
            lr=self.hparams.args.d_lr,  # 1e-4 * 1
            betas=(self.hparams.args.d_beta1, self.hparams.args.d_beta2)
        )
        d_stepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer=dis_optimizer, milestones=[50, ], gamma=0.1)

        return ({"optimizer": dis_optimizer, 'lr_scheduler': d_stepLR},
                {"optimizer": gen_optimizer, 'lr_scheduler': g_stepLR})


if __name__ == '__main__':
    # self, lr, in_channels, out_channels, args):
    # input_channels=3, residual_blocks=8, threshold=threshold
    model = PENNet(in_channels=4, out_channels=3, args=None)
    print(model.generator)
    print(model.discriminator)