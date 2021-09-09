from RN.networks.module import G_Net, D_Net, AdversarialLoss
import torch
import torch.optim as optim
import torch.nn as nn
from Model_TP.template import BASE


class RN(BASE):
    def __init__(self, in_channels, out_channels, args, path, ckpt):
        super().__init__(args.fine_tune, ckpt)
        self.l1_weight = 1.0
        self.gan_weight = 0.1
        self.save_hyperparameters()
        self.generator = G_Net(in_channels, out_channels, residual_blocks=8, threshold=0.8)
        self.discriminator = D_Net(out_channels)
        self.l1_loss = nn.L1Loss()  # mae
        self.l2_loss = nn.MSELoss()  # mse
        self.adversarial_loss = AdversarialLoss('nsgan')  # BCE loss

    def forward(self, x):
        inputs, mask = x
        pred = self.generator(inputs, mask)
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

        x = (inputs, mask)
        pred = self.forward(x)
        merged = pred * mask + mask_y

        # train discriminator
        if optimizer_idx == 0:
            d_real = self.discriminator(y)
            loss_disc_real = self.adversarial_loss(d_real, is_real=True, is_disc=False)

            d_fake = self.discriminator(merged.detach())
            d_fake_loss = self.adversarial_loss(d_fake, False, True)
            loss_disc = (loss_disc_real + d_fake_loss) / 2
            self.log('train/loss_dis', loss_disc)
            return loss_disc

        # train generator
        if optimizer_idx == 1:
            g_fake = self.discriminator(merged)
            loss_gan = self.adversarial_loss(g_fake, is_real=True, is_disc=False)

            l1_loss1 = self.l1_loss(y, merged) / torch.mean(mask)
            l1_loss2 = self.l1_loss(y, pred) / torch.mean(mask)
            gen_loss = self.l1_weight * l1_loss1 + \
                       0.1 * self.l1_weight * l1_loss2 + \
                       self.gan_weight * loss_gan

            self.log_dict({'train/loss_gan': self.gan_weight * loss_gan,
                           "train/merged_l1": self.l1_weight * l1_loss1,
                           "train/pred_l1": 0.1 * self.l1_weight * l1_loss2,
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

        if self.hparams.args.stage == 1:  # for raw
            inputs = mask_img
        elif self.hparams.args.stage == 2:  # for train add_channel
            inputs = torch.cat([mask_img, gradient], dim=1)
        elif self.hparams.args.stage == 3:  # for fine_tune add_channel
            inputs = torch.cat([mask_img, mask_gradient, mask], dim=1)
            pred_gradient = self.eg_model(inputs)
            com_gradient = pred_gradient * mask + mask_gradient
            inputs = torch.cat([mask_img, com_gradient], dim=1)

        x = (inputs, mask)
        pred = self.forward(x)
        merged = pred * mask + mask_y

        g_fake = self.discriminator(merged)
        loss_gan = self.adversarial_loss(g_fake, is_real=True, is_disc=False)

        l1_loss1 = self.l1_loss(y, merged) / torch.mean(mask)
        l1_loss2 = self.l1_loss(y, pred) / torch.mean(mask)
        gen_loss = self.l1_weight * l1_loss1 + \
                   0.1 * self.l1_weight * l1_loss2 + \
                   self.gan_weight * loss_gan

        self.log_dict({'val/loss_gan': self.gan_weight * loss_gan,
                       "val/merged_l1": self.l1_weight * l1_loss1,
                       "val/pred_l1": 0.1 * self.l1_weight * l1_loss2,
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

        if self.hparams.args.stage == 1:  # for raw
            inputs = mask_img
        elif self.hparams.args.stage == 2:  # for train add_channel
            inputs = torch.cat([mask_img, gradient], dim=1)
        elif self.hparams.args.stage == 3:  # for fine_tune add_channel
            inputs = torch.cat([mask_img, mask_gradient, mask], dim=1)
            pred_gradient = self.eg_model(inputs)
            com_gradient = pred_gradient * mask + mask_gradient
            inputs = torch.cat([mask_img, com_gradient], dim=1)

        x = (inputs, mask)
        pred = self.forward(x)
        merged = pred * mask + mask_y

        return self.test_step_mid(y, pred, merged, mask_y, dataloader_idx)

    def configure_optimizers(self):
        gen_optimizer = optim.Adam(
            params=self.generator.parameters(),
            lr=self.hparams.args.g_lr,
            betas=(self.hparams.args.g_beta1, self.hparams.args.g_beta2)
        )

        dis_optimizer = optim.Adam(
            params=self.discriminator.parameters(),
            lr=self.hparams.args.d_lr,
            betas=(self.hparams.args.d_beta1, self.hparams.args.d_beta2)
        )
        return ({"optimizer": dis_optimizer},
                {"optimizer": gen_optimizer})


if __name__ == '__main__':
    # self, lr, in_channels, out_channels, args):
    # input_channels=3, residual_blocks=8, threshold=threshold
    model = InpaintingModel(in_channels=4, out_channels=3, args=None)
    print(model.generator)
    print(model.discriminator)