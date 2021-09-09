# from LBAM.loss.InpaintingLoss import InpaintingLossWithGAN
from LBAM.modules.LBAMModel import LBAMModel, VGG16FeatureExtractor
from LBAM.modules.discriminator import DiscriminatorDoubleColumn

import torch
from torch import autograd
import torch.optim as optim
import torch.nn as nn
from Model_TP.template import BASE


class LBAM(BASE):
    def __init__(self, in_channels, out_channels, args, path, ckpt):
        super().__init__(args.fine_tune, ckpt)

        self.save_hyperparameters()
        self.Lambda = 10.0

        self.generator = LBAMModel(in_channels, out_channels)

        # ---------------for generator
        self.l1 = nn.L1Loss()
        self.extractor = VGG16FeatureExtractor()
        # self.criterion = InpaintingLossWithGAN()
        self.discriminator = DiscriminatorDoubleColumn(3)

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
        if optimizer_idx == 0:
            D_real = self.discriminator(y, mask)
            D_real = D_real.mean().sum() * -1
            D_fake = self.discriminator(pred, mask)
            D_fake = D_fake.mean().sum() * 1
            gp = self.calc_gradient_penalty(y, pred, mask)
            D_loss = D_fake - D_real + gp

            self.log('train/loss_dis', D_loss)
            return D_loss

        # train generator
        if optimizer_idx == 1:
            holeLoss = 6 * self.l1((1 - mask) * pred, (1 - mask) * y)
            validAreaLoss = self.l1(mask * pred, mask * y)

            feat_output_comp = self.extractor(merged)
            feat_output = self.extractor(pred)
            feat_gt = self.extractor(y)

            prcLoss = 0.0
            for i in range(3):
                prcLoss += 0.01 * self.l1(feat_output[i], feat_gt[i])
                prcLoss += 0.01 * self.l1(feat_output_comp[i], feat_gt[i])

            styleLoss = 0.0
            for i in range(3):
                styleLoss += 120 * self.l1(self.gram_matrix(feat_output[i]),
                                           self.gram_matrix(feat_gt[i]))
                styleLoss += 120 * self.l1(self.gram_matrix(feat_output_comp[i]),
                                           self.gram_matrix(feat_gt[i]))

            D_fake = self.discriminator(pred, mask)
            D_fake = D_fake.mean().sum() * 1

            GLoss = holeLoss + validAreaLoss + prcLoss + styleLoss + 0.1 * D_fake

            self.log_dict({'train/hole_Loss': holeLoss,
                           "train/validAreaLoss": validAreaLoss,
                           "train/prcLoss": prcLoss,
                           "train/styleLoss": styleLoss,
                           "train/gen_loss": GLoss})

            img = torch.cat([mask_y[:6], y[:6], pred[:6], merged[:6]], dim=0)
            if self.global_step % self.hparams.args.log_every_n_steps == 0:
                self.log_img(img, "train")

            self.train_val_step_mid(y, pred, merged, "train")
            self.log_mae_mse(y, pred, merged, "train")
            return GLoss.sum()

    def validation_step(self, batch, batch_idx, optimizer_idx=None):
        img, mask_img, gradient, mask_gradient, img_gray, mask_gray, mask, index = batch
        inputs = None
        y = img
        mask_y = mask_img

        if self.hparams.args.stage == 1:    # for raw
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

        holeLoss = 6 * self.l1((1 - mask) * pred, (1 - mask) * y)
        validAreaLoss = self.l1(mask * pred, mask * y)

        feat_output_comp = self.extractor(merged)
        feat_output = self.extractor(pred)
        feat_gt = self.extractor(y)

        prcLoss = 0.0
        for i in range(3):
            prcLoss += 0.01 * self.l1(feat_output[i], feat_gt[i])
            prcLoss += 0.01 * self.l1(feat_output_comp[i], feat_gt[i])

        styleLoss = 0.0
        for i in range(3):
            styleLoss += 120 * self.l1(self.gram_matrix(feat_output[i]),
                                       self.gram_matrix(feat_gt[i]))
            styleLoss += 120 * self.l1(self.gram_matrix(feat_output_comp[i]),
                                       self.gram_matrix(feat_gt[i]))

        D_fake = self.discriminator(pred, mask)
        D_fake = D_fake.mean().sum() * 1

        GLoss = holeLoss + validAreaLoss + prcLoss + styleLoss + 0.1 * D_fake

        self.log_dict({'val/hole_Loss': holeLoss,
                       "val/validAreaLoss": validAreaLoss,
                       "val/prcLoss": prcLoss,
                       "val/styleLoss": styleLoss,
                       "val/gen_loss": GLoss})

        img = torch.cat([mask_y[:6], y[:6], pred[:6], merged[:6]], dim=0)
        self.log_img(img, "val")

        self.train_val_step_mid(y, pred, merged, "val")
        self.log_mae_mse(y, pred, merged, "val")

        return GLoss.sum()

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        img, mask_img, gradient, mask_gradient, img_gray, mask_gray, mask, index = batch
        inputs = None
        y = img
        mask_y = mask_img

        if self.hparams.args.stage == 1:    # for raw
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
        gen_optimizer = optim.Adam(
            params=self.generator.parameters(),
            lr=self.hparams.args.g_lr,  # 0.0001
            betas=(self.hparams.args.g_beta1, self.hparams.args.g_beta2)
        )

        dis_optimizer = optim.Adam(
            params=self.discriminator.parameters(),
            lr=self.hparams.args.d_lr,  # 0.00001
            betas=(self.hparams.args.d_beta1, self.hparams.args.d_beta2)
        )
        return ({"optimizer": dis_optimizer},
                {"optimizer": gen_optimizer})

    # modified from WGAN-GP
    def calc_gradient_penalty(self, real_data, fake_data, masks):
        BATCH_SIZE = real_data.size()[0]
        DIM = real_data.size()[2]
        alpha = torch.rand(BATCH_SIZE, 1)
        alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement() / BATCH_SIZE)).contiguous()
        alpha = alpha.view(BATCH_SIZE, 3, DIM, DIM)
        alpha = alpha.type_as(real_data)

        fake_data = fake_data.view(BATCH_SIZE, 3, DIM, DIM)
        interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
        interpolates = interpolates.type_as(real_data)

        interpolates.requires_grad_(True)

        disc_interpolates = self.discriminator(interpolates, masks)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).type_as(real_data),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.Lambda
        return gradient_penalty.sum().mean()

    def gram_matrix(self, feat):
        # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
        (b, ch, h, w) = feat.size()
        feat = feat.view(b, ch, h * w)
        feat_t = feat.transpose(1, 2)
        gram = torch.bmm(feat, feat_t) / (ch * h * w)
        return gram

    # tv loss
    def total_variation_loss(self, image):
        # shift one pixel and get difference (for both x and y direction)
        loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
               torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
        return loss


if __name__ == '__main__':
    # self, lr, in_channels, out_channels, args):
    # input_channels=3, residual_blocks=8, threshold=threshold
    model = InpaintingModel(in_channels=4, out_channels=3, args=None)
    print(model.generator)
    print(model.discriminator)