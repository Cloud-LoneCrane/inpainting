import torch
import torch.optim as optim
from EG_Model.pl_module.ViT_SEG import vit_seg
from Model_TP.template import BASE

from PRVS.modules.PRVSNet import PRVSNet, VGG16FeatureExtractor
from PRVS.modules.Losses import AdversarialLoss
import torch.nn.functional as F
from PRVS.modules.Discriminator import Discriminator


class PRVS(BASE):
    def __init__(self, input_channels, out_channels, args, path, ckpt):
        super().__init__(args.fine_tune, ckpt)
        self.save_hyperparameters()

        self.G = PRVSNet(input_channels=input_channels, output_channels=out_channels)
        self.lossNet = VGG16FeatureExtractor()
        self.D = Discriminator(out_channels)
        self.adversarial_loss = AdversarialLoss()

    def forward(self, x):
        masked_image, mask, masked_edge, gt_image, gt_edge = x
        fake_img, _, edge_small, edge_big = self.G(masked_image, mask, masked_edge)
        edge_fake = [edge_small, edge_big]

        return fake_img, edge_fake

    def on_train_epoch_start(self) -> None:
        self.G.train(finetune=False)

    def training_step(self, batch, batch_index, optimizer_idx=None):
        img, mask_img, gradient, mask_gradient, img_gray, mask_gray, mask, index = batch
        y = img
        mask_y = mask_img

        x = None
        if self.hparams.args.stage == 1:    # for raw
            x = (mask_y, mask, mask_gradient)
        elif self.hparams.args.stage == 2:  # for train use gt_edge or gt_gradient
            x = (mask_y, mask, gradient)
        elif self.hparams.args.stage == 3:  # for fine_tune use gt_edge or gt_gradient
            inputs = torch.cat([mask_img, mask_gradient, mask], dim=1)
            pred_gradient = self.eg_model(inputs)
            com_gradient = pred_gradient * mask + mask_gradient
            x = (mask_y, mask, com_gradient)

        pred, pred_gradient = self.forward(x)
        merged = pred * (1 - mask) + mask_y

        # train discriminator
        if optimizer_idx == 0:
            loss_disc = self.get_d_loss(gradient, pred_gradient, img_gray)
            self.log('train/loss_dis', loss_disc)
            return loss_disc

        # train generator
        if optimizer_idx == 1:
            gen_loss = self.get_g_loss(img, pred, merged, gradient, pred_gradient, mask)
            self.log('train/loss_gen', gen_loss)

            img = torch.cat([mask_y[:6], y[:6], pred[:6], merged[:6]], dim=0)
            if self.global_step % self.hparams.args.log_every_n_steps == 0:
                self.log_img(img, "train")
            self.train_val_step_mid(y, pred, merged, "train")
            self.log_mae_mse(y, pred, merged, stage="train")
            return gen_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=None):
        img, mask_img, gradient, mask_gradient, img_gray, mask_gray, mask, index = batch
        y = img
        mask_y = mask_img

        x = None
        if self.hparams.args.stage == 1:  # for raw
            x = (mask_y, mask, mask_gradient)
        elif self.hparams.args.stage == 2:  # for train use gt_edge or gt_gradient
            x = (mask_y, mask, gradient)
        elif self.hparams.args.stage == 3:  # for fine_tune use gt_edge or gt_gradient
            inputs = torch.cat([mask_img, mask_gradient, mask], dim=1)
            pred_gradient = self.eg_model(inputs)
            com_gradient = pred_gradient * mask + mask_gradient
            x = (mask_y, mask, com_gradient)

        pred, pred_gradient = self.forward(x)
        merged = pred * (1 - mask) + mask_y

        gen_loss = self.get_g_loss(img, pred, merged, gradient, pred_gradient, mask)
        self.log('val/loss_gen', gen_loss)

        img = torch.cat([mask_y[:8], y[:8], pred[:8], merged[:8]], dim=0)
        self.log_img(img, "val")

        self.train_val_step_mid(y, pred, merged, "val")
        self.log_mae_mse(y, pred, merged, stage="val")

        return gen_loss

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        img, mask_img, gradient, mask_gradient, img_gray, mask_gray, mask, index = batch
        y = img
        mask_y = mask_img

        x = None
        if self.hparams.args.stage == 1:  # for raw
            x = (mask_y, mask, mask_gradient)
        elif self.hparams.args.stage == 2:  # for train use gt_edge or gt_gradient
            x = (mask_y, mask, gradient)
        elif self.hparams.args.stage == 3:  # for fine_tune use gt_edge or gt_gradient
            inputs = torch.cat([mask_img, mask_gradient, mask], dim=1)
            pred_gradient = self.eg_model(inputs)
            com_gradient = pred_gradient * mask + mask_gradient
            x = (mask_y, mask, com_gradient)

        pred, pred_gradient = self.forward(x)
        merged = pred * (1 - mask) + mask_y

        return self.test_step_mid(y, pred, merged, mask_y, dataloader_idx)

    def configure_optimizers(self):
        gen_optimizer = optim.Adam(
            params=self.generator.parameters(),
            lr=self.hparams.args.g_lr,  # 2e-4
            betas=(self.hparams.args.g_beta1, self.hparams.args.g_beta2)
        )

        dis_optimizer = optim.Adam(
            params=self.discriminator.parameters(),
            lr=self.hparams.args.d_lr,  # 2e-5
            betas=(self.hparams.args.d_beta1, self.hparams.args.d_beta2)
        )
        return ({"optimizer": dis_optimizer},
                {"optimizer": gen_optimizer})

    def get_d_loss(self, real_edges, fake_edges, grey):
        loss_D = 0
        for i in range(2):
            fake_edge = fake_edges[i]
            real_edge = F.interpolate(real_edges, size=fake_edge.size()[2:])
            real_edge = torch.clamp(real_edge * 4, 0, 1)
            real_image = F.interpolate(grey, size=fake_edge.size()[2:])
            real_edge = real_edge.detach()
            fake_edge = fake_edge.detach()
            real_edge = torch.cat([real_edge, real_image], dim=1)
            fake_edge = torch.cat([fake_edge, real_image], dim=1)
            pred_real, _ = self.D(real_edge)
            pred_fake, _ = self.D(fake_edge)
            loss_D += (self.adversarial_loss(pred_real, True, True) + self.adversarial_loss(pred_fake, False, True)) / 2
        return loss_D.sum()

    def get_g_loss(self, real_B, fake_B, comp_B, real_edge, fake_edge, mask):
        real_B_feats = self.lossNet(real_B)
        fake_B_feats = self.lossNet(fake_B)
        comp_B_feats = self.lossNet(comp_B)

        tv_loss = self.TV_loss(comp_B * (1 - mask))
        style_loss = self.style_loss(real_B_feats, fake_B_feats) + self.style_loss(real_B_feats, comp_B_feats)
        preceptual_loss = self.preceptual_loss(real_B_feats, fake_B_feats) + self.preceptual_loss(real_B_feats,
                                                                                                  comp_B_feats)
        valid_loss = self.l1_loss(real_B, fake_B, mask)
        hole_loss = self.l1_loss(real_B, fake_B, (1 - mask))
        adv_loss_0 = self.edge_loss(fake_edge[1], real_edge)
        adv_loss_1 = self.edge_loss(fake_edge[0], F.interpolate(real_edge, scale_factor=0.5))

        adv_loss = adv_loss_0 + adv_loss_1

        loss_G = (tv_loss * 0.1
                  + style_loss * 150
                  + preceptual_loss * 0.05
                  + valid_loss * 50
                  + hole_loss * 50) + 0.1 * adv_loss
        self.l1_loss_val += valid_loss.detach() + hole_loss.detach()
        return loss_G

    def l1_loss(self, f1, f2, mask=1):
        return torch.mean(torch.abs(f1 - f2) * mask)

    def edge_loss(self, fake_edge, real_edge):
        grey_image = self.grey
        grey_image = F.interpolate(grey_image, size=fake_edge.size()[2:])
        fake_edge = torch.cat([fake_edge, grey_image], dim=1)
        pred_fake, features_edge1 = self.D(fake_edge)
        return self.adversarial_loss(pred_fake, True, False)

    def style_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            _, c, w, h = A_feat.size()
            A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
            B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
            A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
            B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
            loss_value += torch.mean(torch.abs(A_style - B_style) / (c * w * h))
        return loss_value

    def TV_loss(self, x):
        h_x = x.size(2)
        w_x = x.size(3)
        h_tv = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]))
        w_tv = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]))
        return h_tv + w_tv

    def preceptual_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            loss_value += torch.mean(torch.abs(A_feat - B_feat))
        return loss_value


if __name__ == '__main__':
    # self, lr, in_channels, out_channels, args):
    # input_channels=3, residual_blocks=8, threshold=threshold
    # modules = InpaintingModel(in_channels=4, out_channels=3, args=None)
    # print(modules.generator)
    # print(modules.discriminator)
    pass