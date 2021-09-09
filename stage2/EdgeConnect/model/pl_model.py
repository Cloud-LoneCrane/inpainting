from EdgeConnect.modules.loss import AdversarialLoss, PerceptualLoss, StyleLoss
from EdgeConnect.modules.networks import InpaintGenerator, EdgeGenerator, Discriminator

import torch
import torch.optim as optim
import torch.nn as nn
from Model_TP.template import BASE


class EdgeConnect(BASE):
    def __init__(self, in_channels, out_channels, args, path, ckpt):
        super().__init__(args.fine_tune, ckpt)
        self.save_hyperparameters()
        self.generator = InpaintGenerator(in_channels, out_channels)
        self.GAN_LOSS = 'hinge'   # nsgan | lsgan | hinge
        self.discriminator = Discriminator(in_channels=3, use_sigmoid=self.GAN_LOSS != 'hinge')

        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()
        self.adversarial_loss = AdversarialLoss(loss_type=self.GAN_LOSS)

        self.INPAINT_ADV_LOSS_WEIGHT = 0.1
        self.L1_LOSS_WEIGHT = 1
        self.CONTENT_LOSS_WEIGHT = 0.1
        self.STYLE_LOSS_WEIGHT = 250

    def forward(self, x):
        pred = self.generator(x)    # inputs = torch.cat((images_masked, edges), dim=1)
        return pred

    def training_step(self, batch, batch_index, optimizer_idx=None):
        img, mask_img, gradient, mask_gradient, img_gray, mask_gray, mask, index = batch
        inputs = None
        y = img
        mask_y = mask_img

        if self.hparams.args.stage == 1:    # for raw
            inputs = torch.cat([mask_img, gradient], dim=1)
        else:  # for fine tune raw
            inputs = torch.cat([mask_img, mask_gradient, mask], dim=1)
            pred_gradient = self.eg_model(inputs)
            com_gradient = pred_gradient * mask + mask_gradient
            inputs = torch.cat([mask_img, com_gradient], dim=1)

        pred = self.forward(inputs)
        merged = pred * mask + mask_y

        # train discriminator
        if optimizer_idx == 0:
            dis_real, _ = self.discriminator(img)  # in: [rgb(3)]
            dis_fake, _ = self.discriminator(pred.detach())  # in: [rgb(3)]
            dis_real_loss = self.adversarial_loss(dis_real, True, True)
            dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
            dis_loss = (dis_real_loss + dis_fake_loss) / 2

            self.log('train/loss_dis', dis_loss)
            return dis_loss

        # train generator
        if optimizer_idx == 1:
            gen_fake, _ = self.discriminator(pred)  # in: [rgb(3)]
            gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.INPAINT_ADV_LOSS_WEIGHT
            gen_loss = gen_gan_loss

            # generator l1 loss
            gen_l1_loss = self.l1_loss(pred, img) * self.L1_LOSS_WEIGHT / torch.mean(mask)
            gen_loss += gen_l1_loss

            # generator perceptual loss
            gen_content_loss = self.perceptual_loss(pred, img)
            gen_content_loss = gen_content_loss * self.CONTENT_LOSS_WEIGHT
            gen_loss += gen_content_loss

            # generator style loss
            gen_style_loss = self.style_loss(pred * mask, img * mask)
            gen_style_loss = gen_style_loss * self.STYLE_LOSS_WEIGHT
            gen_loss += gen_style_loss

            self.log_dict({'train/g_l1_loss': gen_l1_loss,
                           "train/g_perceptual_loss": gen_content_loss,
                           "train/g_style_loss": gen_style_loss,
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
            inputs = torch.cat([mask_img, gradient], dim=1)
        else:  # for fine tune raw
            inputs = torch.cat([mask_img, mask_gradient, mask], dim=1)
            pred_gradient = self.eg_model(inputs)
            com_gradient = pred_gradient * mask + mask_gradient
            inputs = torch.cat([mask_img, com_gradient], dim=1)

        pred = self.forward(inputs)
        merged = pred * mask + mask_y

        gen_fake, _ = self.discriminator(pred)  # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.INPAINT_ADV_LOSS_WEIGHT
        gen_loss = gen_gan_loss

        # generator l1 loss
        gen_l1_loss = self.l1_loss(pred, img) * self.L1_LOSS_WEIGHT / torch.mean(mask)
        gen_loss += gen_l1_loss

        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(pred, img)
        gen_content_loss = gen_content_loss * self.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss

        # generator style loss
        gen_style_loss = self.style_loss(pred * mask, img * mask)
        gen_style_loss = gen_style_loss * self.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        self.log_dict({'val/g_l1_loss': gen_l1_loss,
                       "val/g_perceptual_loss": gen_content_loss,
                       "val/g_style_loss": gen_style_loss,
                       "val/gen_loss": gen_loss})

        img = torch.cat([mask_y[:6], y[:6], pred[:6], merged[:6]], dim=0)
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
            inputs = torch.cat([mask_img, gradient], dim=1)
        else:  # for fine tune raw
            inputs = torch.cat([mask_img, mask_gradient, mask], dim=1)
            pred_gradient = self.eg_model(inputs)
            com_gradient = pred_gradient * mask + mask_gradient
            inputs = torch.cat([mask_img, com_gradient], dim=1)

        pred = self.forward(inputs)
        merged = pred * mask + mask_y

        return self.test_step_mid(y, pred, merged, mask_y, dataloader_idx)

    def configure_optimizers(self):
        gen_optimizer = optim.Adam(
            params=self.generator.parameters(),
            lr=self.hparams.args.g_lr,  # 0.0001
            betas=(self.hparams.args.g_beta1, self.hparams.args.g_beta2),
            eps=1e-6
        )

        dis_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.args.d_lr,  # g_lr * 0.1
            betas=(self.hparams.args.d_beta1, self.hparams.args.d_beta2),
            eps=1e-6
        )
        return ({"optimizer": dis_optimizer},
                {"optimizer": gen_optimizer})
