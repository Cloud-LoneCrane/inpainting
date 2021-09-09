from GMCNN.modules.net import GMCNN as gmcnn
from GMCNN.modules.net import GlobalLocalDiscriminator
from GMCNN.modules.layer import init_weights, PureUpsampling, ConfidenceDrivenMaskLayer, SpectralNorm
from GMCNN.modules.loss import WGANLoss, IDMRFLoss

import torch.nn.functional as F
import torch
import torch.optim as optim
import torch.nn as nn
from Model_TP.template import BASE


class GMCNN(BASE):
    def __init__(self, in_channels, out_channels, args, path, ckpt):
        super().__init__(args.fine_tune, ckpt)
        self.save_hyperparameters()
        # self.confidence_mask_layer = ConfidenceDrivenMaskLayer()
        self.generator = gmcnn(in_channels, out_channels=out_channels, cnum=32, act=F.elu, norm=None)
        init_weights(self.generator)
        self.recloss = nn.L1Loss()
        self.aeloss = nn.L1Loss()

        self.discriminator = GlobalLocalDiscriminator(
            3, cnum=64, act=F.elu, spectral_norm=1,
            g_fc_channels=args.size // 16 * args.size // 16 * 64 * 4,
            l_fc_channels=args.size // 16 * args.size // 16 * 64 * 4)
        init_weights(self.discriminator)

        self.wganloss = WGANLoss()
        self.mrfloss = IDMRFLoss()

        self.lambda_adv = 1e-3
        self.lambda_rec = 1.4
        self.lambda_ae = 1.2
        self.lambda_gp = 10
        self.lambda_mrf = 0.05

    def forward(self, x):
        pred = self.generator(x)    # x = torch.cat((mask_img, mask), 1)
        return pred

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

        x = torch.cat([inputs, mask], dim=1)
        pred = self.forward(x)
        merged = pred * mask + mask_y

        # train discriminator
        if optimizer_idx == 0:
            completed_logit, completed_local_logit = self.discriminator(merged.detach(), merged.detach())
            gt_logit, gt_local_logit = self.discriminator(y, y)
            # hinge loss
            D_loss_local = nn.ReLU()(1.0 - gt_local_logit).mean() + nn.ReLU()(1.0 + completed_local_logit).mean()
            D_loss = nn.ReLU()(1.0 - gt_logit).mean() + nn.ReLU()(1.0 + completed_logit).mean()
            D_loss = D_loss + D_loss_local

            self.log('train/loss_dis', D_loss)
            return D_loss

        # train generator
        if optimizer_idx == 1:
            G_loss_reconstruction = self.recloss(merged * mask, y * mask) / torch.mean(mask)
            G_loss_ae = self.aeloss(pred * (1 - self.mask), y * (1 - self.mask)) / torch.mean(1 - self.mask)

            G_loss = self.lambda_rec * G_loss_reconstruction + self.lambda_ae * G_loss_ae

            # discriminator
            completed_logit, completed_local_logit = self.discriminator(merged, merged)
            G_loss_mrf = self.mrfloss((merged + 1) / 2.0, (y + 1) / 2.0)
            G_loss = G_loss + self.lambda_mrf * G_loss_mrf

            G_loss_adv = -completed_logit.mean()
            G_loss_adv_local = -completed_local_logit.mean()
            gan_loss = self.lambda_adv * (G_loss_adv + G_loss_adv_local)
            G_loss = G_loss + gan_loss

            self.log_dict({'train/loss_gan': gan_loss,
                           "train/merged_l1": G_loss_reconstruction,
                           "train/pred_l1": G_loss_ae,
                           "train/gen_loss": G_loss})

            img = torch.cat([mask_y[:6], y[:6], pred[:6], merged[:6]], dim=0)
            if self.global_step % self.hparams.args.log_every_n_steps == 0:
                self.log_img(img, "train")

            self.train_val_step_mid(y, pred, merged, "train")
            self.log_mae_mse(y, pred, merged, "train")
            return G_loss

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

        x = torch.cat([inputs, mask], dim=1)
        pred = self.forward(x)
        merged = pred * mask + mask_y

        G_loss_reconstruction = self.recloss(merged * mask, y * mask) / torch.mean(mask)
        G_loss_ae = self.aeloss(pred * (1 - mask), y * (1 - mask)) / torch.mean(1 - mask)

        G_loss = self.lambda_rec * G_loss_reconstruction + self.lambda_ae * G_loss_ae

        # discriminator
        completed_logit, completed_local_logit = self.discriminator(merged, merged)
        G_loss_mrf = self.mrfloss((merged + 1) / 2.0, (y + 1) / 2.0)
        G_loss = G_loss + self.lambda_mrf * G_loss_mrf

        G_loss_adv = -completed_logit.mean()
        G_loss_adv_local = -completed_local_logit.mean()
        gan_loss = self.lambda_adv * (G_loss_adv + G_loss_adv_local)
        G_loss = G_loss + gan_loss

        self.log_dict({'val/loss_gan': gan_loss,
                       "val/merged_l1": G_loss_reconstruction,
                       "val/pred_l1": G_loss_ae,
                       "val/gen_loss": G_loss})

        img = torch.cat([mask_y[:6], y[:6], pred[:6], merged[:6]], dim=0)
        self.log_img(img, "val")

        self.train_val_step_mid(y, pred, merged, "val")
        self.log_mae_mse(y, pred, merged, "val")

        return G_loss

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

        x = torch.cat([inputs, mask], dim=1)
        pred = self.forward(x)
        merged = pred * mask + mask_y

        return self.test_step_mid(y, pred, merged, mask_y, dataloader_idx)

    def configure_optimizers(self):
        gen_optimizer = optim.Adam(
            params=self.generator.parameters(),
            lr=self.hparams.args.g_lr,  # 1e-5
            betas=(self.hparams.args.g_beta1, self.hparams.args.g_beta2)
        )

        dis_optimizer = optim.Adam(
            filter(lambda x: x.requires_grad, self.discriminator.parameters()),
            lr=self.hparams.args.d_lr,  # 1e-5
            betas=(self.hparams.args.d_beta1, self.hparams.args.d_beta2)
        )
        return ({"optimizer": dis_optimizer},
                {"optimizer": gen_optimizer})
