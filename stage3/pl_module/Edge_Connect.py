import pytorch_lightning as pl
from pl_module.template import BASE
from networks.edge_connect import EdgeGenerator, Discriminator
import os
import torch
import torch.nn as nn
from torch.optim import *
from networks.edge_connect_loss import AdversarialLoss
import torchvision
from utils.metrics import ssim, psnr
import numpy as np
import torch.nn.functional as F


class edge_connect(pl.LightningModule):
    def __init__(self, in_channel=3+1+1, out_channel=1, lr=0.0001, args=None):
        super().__init__()
        self.save_hyperparameters()
        self.val_step = 0

        self.FM_LOSS_WEIGHT = 0.1

        self.generator = EdgeGenerator(in_channels=in_channel, out_channels=out_channel)
        # GAN_LOSS: nsgan | lsgan | hinge
        self.GAN_LOSS = "hinge"
        self.discriminator = Discriminator(in_channels=out_channel, use_sigmoid=self.GAN_LOSS != 'hinge')

        self.l1_loss = nn.L1Loss()
        self.adversarial_loss = AdversarialLoss(type=self.GAN_LOSS)

    def forward(self, x):
        """
        :param x: [mask_img, mask_gradient, mask]
        :return: gradient_hat
        """
        outputs = self.generator(x)
        return outputs

    def training_step(self, batch, batch_idx, optimizer_idx):
        (x, y), (mask_y, mask) = batch
        y_hat = self.forward(x)
        y_com = y_hat * mask + mask_y
        dis_real, real_feat = self.discriminator(y)
        dis_fake, fake_feat = self.discriminator(y_hat)

        if optimizer_idx == 0:
            # gen
            l1_loss_com = self.l1_loss(y, y_com)
            l1_loss_hat = self.l1_loss(y, y_hat)
            gan_loss = self.adversarial_loss(dis_fake, is_real=True, is_disc=False)

            gen_loss = l1_loss_com + l1_loss_hat + gan_loss

            # generator feature matching loss
            gen_fm_loss = 0
            for i in range(len(fake_feat)):
                gen_fm_loss += self.l1_loss(fake_feat[i], real_feat[i])
            gen_fm_loss = gen_fm_loss * self.FM_LOSS_WEIGHT

            gen_loss = gen_loss + gen_fm_loss

            self.log("train/gen_l1_loss_com", l1_loss_com)
            self.log("train/gen_l1_loss_hat", l1_loss_hat)
            self.log("train/gan_loss", gan_loss)
            self.log("train/gen_fm_loss", gen_fm_loss)
            self.log("train/gen_loss", gen_loss)
            if self.global_step % 100 == 0:
                img = torch.cat([mask_y[:8], y[:8], y_hat[:8], y_com[:8]], dim=0)
                self.log_img(img, "train")

            return gen_loss

        elif optimizer_idx == 1:
            # dis
            real_loss = self.adversarial_loss(dis_real, is_real=True, is_disc=False)
            fake_loss = self.adversarial_loss(dis_fake, is_real=False, is_disc=True)

            dis_loss = (real_loss + fake_loss)/2

            self.log("train/real_loss", real_loss)
            self.log("train/fake_loss", fake_loss)
            self.log("train/dis_loss", dis_loss)
            return dis_loss

    def training_step_end(self, losses):
        loss = torch.mean(self.all_gather(losses, sync_grads=True))
        return loss

    def validation_step(self, batch, batch_idx):
        (x, y), (mask_y, mask) = batch
        y_hat = self.forward(x)
        y_com = y_hat * mask + mask_y

        dis_real, real_feat = self.discriminator(y)
        dis_fake, fake_feat = self.discriminator(y_hat)

        l1_loss_com = self.l1_loss(y, y_com)
        l1_loss_hat = self.l1_loss(y, y_hat)
        gan_loss = self.adversarial_loss(dis_fake, is_real=True, is_disc=False)

        gen_loss = l1_loss_com + l1_loss_hat + gan_loss

        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(fake_feat)):
            gen_fm_loss += self.l1_loss(fake_feat[i], real_feat[i])
        gen_fm_loss = gen_fm_loss * self.FM_LOSS_WEIGHT

        gen_loss = gen_loss + gen_fm_loss

        self.log("val_loss", l1_loss_com)
        self.log("val/gen_l1_loss_hat", l1_loss_hat)
        self.log("val/gan_loss", gan_loss)
        self.log("val/gen_fm_loss", gen_fm_loss)
        self.log("val/gen_loss", gen_loss)

        img = torch.cat([mask_y[:8], y[:8], y_hat[:8], y_com[:8]], dim=0)
        self.log_img(img, "val")

        ssim_y_hat = [ssim(m1, m2) for m1, m2 in zip(y, y_hat)]
        ssim_y_compose = [ssim(m1, m2) for m1, m2 in zip(y, y_com)]
        psnr_y_hat = [psnr(m1, m2) for m1, m2 in zip(y, y_hat)]
        psnr_y_compose = [psnr(m1, m2) for m1, m2 in zip(y, y_com)]

        ssim_y_hat_mean = np.mean(ssim_y_hat)
        ssim_y_compose_mean = np.mean(ssim_y_compose)
        psnr_y_hat_mean = np.mean(psnr_y_hat)
        psnr_y_compose_mean = np.mean(psnr_y_compose)

        self.log("val/ssim_hat", ssim_y_hat_mean)
        self.log("val/ssim_com", ssim_y_compose_mean)
        self.log("val/psnr_hat", psnr_y_hat_mean)
        self.log("val/psnr_com", psnr_y_compose_mean)

        self.val_step += 1
        return gen_loss

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        (x, y), (mask_y, mask) = batch
        y_hat = self.forward(x)
        y_com = y_hat * mask + mask_y

        l1_loss_hat = self.l1_loss(y, y_hat)

        mae = F.l1_loss(y_com, y)
        mse = F.mse_loss(y_com, y)

        if self.hparams.args.save_result:
            img = torch.cat([mask_y, y, y_hat, y_com], dim=0)
            size = y.shape[0]
            torchvision.utils.save_image(img, "test/{}/{}/{}/{}.jpg".format(self.hparams.args.model,
                                                                            self.hparams.args.dataset_name,
                                                                            self.hparams.args.stage,
                                                                            self.test_counter), nrow=size)

        ssim_y_hat = [ssim(m1, m2) for m1, m2 in zip(y, y_hat)]
        ssim_y_compose = [ssim(m1, m2) for m1, m2 in zip(y, y_com)]
        psnr_y_hat = [psnr(m1, m2) for m1, m2 in zip(y, y_hat)]
        psnr_y_compose = [psnr(m1, m2) for m1, m2 in zip(y, y_com)]

        ssim_y_hat_mean = np.mean(ssim_y_hat)
        ssim_y_compose_mean = np.mean(ssim_y_compose)
        psnr_y_hat_mean = np.mean(psnr_y_hat)
        psnr_y_compose_mean = np.mean(psnr_y_compose)

        return {"ssim_y_hat": ssim_y_hat_mean,
                "ssim_y_com": ssim_y_compose_mean,
                "psnr_y_hat": psnr_y_hat_mean,
                "psnr_y_com": psnr_y_compose_mean,
                "l1_loss_hat": l1_loss_hat,
                "mae": mae,
                "mse": mse,
                "idx": dataloader_idx}

    def test_step_end(self, output_results):
        all_gpus_result = self.all_gather(output_results)
        self.test_counter += 1

        ssim_y_hat = torch.mean(all_gpus_result["ssim_y_hat"]).cpu()
        ssim_y_com = torch.mean(all_gpus_result["ssim_y_com"]).cpu()
        psnr_y_hat = torch.mean(all_gpus_result["psnr_y_hat"]).cpu()
        psnr_y_com = torch.mean(all_gpus_result["psnr_y_com"]).cpu()
        mae = torch.mean(all_gpus_result["mae"]).cpu()
        mse = torch.mean(all_gpus_result["mse"]).cpu()
        loss = torch.mean(all_gpus_result["l1_loss_hat"]).cpu()

        return {"ssim_y_hat": ssim_y_hat,
                "ssim_y_com": ssim_y_com,
                "psnr_y_hat": psnr_y_hat,
                "psnr_y_com": psnr_y_com,
                "l1_loss_hat": loss,
                "mae": mae,
                "mse": mse
                }

    def test_epoch_end(self, outputs):
        if not self.hparams.args.if_split:
            ssim_y_hat = []
            ssim_y_com = []
            psnr_y_hat = []
            psnr_y_com = []
            loss_l = []
            mae_l = []
            mse_l = []

            for test_result in outputs:
                ssim_y_hat.append(test_result["ssim_y_hat"])
                ssim_y_com.append(test_result["ssim_y_com"])
                psnr_y_hat.append(test_result['psnr_y_hat'])
                psnr_y_com.append(test_result['psnr_y_com'])
                loss_l.append(test_result["l1_loss_hat"])
                mae_l.append(test_result["mae"])
                mse_l.append(test_result["mse"])

            ssim_y_hat = np.mean(ssim_y_hat)
            ssim_y_com = np.mean(ssim_y_com)
            psnr_y_hat = np.mean(psnr_y_hat)
            psnr_y_com = np.mean(psnr_y_com)
            mae_l = torch.mean(torch.tensor(mae_l))
            mse_l = torch.mean(torch.tensor(mse_l))
            loss_l = torch.mean(torch.tensor(loss_l))

            print("test:\n"
                  "ssim_y_hat:{}\n".format(ssim_y_hat),
                  "ssim_y_com:{}\n".format(ssim_y_com),
                  "psnr_y_hat:{}\n".format(psnr_y_hat),
                  "psnr_y_com:{}\n".format(psnr_y_com),
                  "mae:{}\n".format(mae_l),
                  "mse:{}\n".format(mse_l),
                  "l1_loss_hat:{}".format(loss_l))

            return {"ssim_y_hat": ssim_y_hat,
                    "ssim_y_com": ssim_y_com,
                    "psnr_y_hat": psnr_y_hat,
                    "psnr_y_com": psnr_y_com,
                    "l1_loss_hat": loss_l,
                    "mae": mae_l,
                    "mse": mse_l}
        else:
            results = []
            rates = [0.5, 0.4, 0.07, 0.01, 0.01, 0.01]
            for i, outputs_i in enumerate(outputs):
                ssim_y_hat = []
                ssim_y_com = []
                psnr_y_hat = []
                psnr_y_com = []
                loss_l = []
                mae_l = []
                mse_l = []

                for test_result in outputs_i:   # list
                    ssim_y_hat.append(test_result["ssim_y_hat"])
                    ssim_y_com.append(test_result["ssim_y_com"])
                    psnr_y_hat.append(test_result['psnr_y_hat'])
                    psnr_y_com.append(test_result['psnr_y_com'])
                    loss_l.append(test_result["l1_loss_hat"])
                    mae_l.append(test_result["mae"])
                    mse_l.append(test_result["mse"])

                result = [np.mean(ssim_y_hat), np.mean(ssim_y_com), np.mean(psnr_y_hat), np.mean(psnr_y_com),
                          torch.mean(torch.tensor(mae_l)), torch.mean(torch.tensor(mse_l)),
                          torch.mean(torch.tensor(loss_l)), i]
                results.append(result)

            ssim_y_hat = np.sum([rate*result[0] for rate, result in zip(rates, results)])
            ssim_y_com = np.sum([rate*result[1] for rate, result in zip(rates, results)])
            psnr_y_hat = np.sum([rate*result[2] for rate, result in zip(rates, results)])
            psnr_y_com = np.sum([rate*result[3] for rate, result in zip(rates, results)])
            loss = np.sum([rate*result[4] for rate, result in zip(rates, results)])
            mae = np.sum([rate*result[5] for rate, result in zip(rates, results)])
            mse = np.sum([rate*result[6] for rate, result in zip(rates, results)])

            print("test_zero-1~6:\n"
                  "ssim_y_hat:{}\n".format(ssim_y_hat),
                  "ssim_y_com:{}\n".format(ssim_y_com),
                  "psnr_y_hat:{}\n".format(psnr_y_hat),
                  "psnr_y_com:{}\n".format(psnr_y_com),
                  "mae:{}\n".format(mae),
                  "mse:{}\n".format(mse),
                  "l1_loss_hat:{}".format(loss),
                  )

            return {"ssim_y_hat": ssim_y_hat,
                    "ssim_y_com": ssim_y_com,
                    "psnr_y_hat": psnr_y_hat,
                    "psnr_y_com": psnr_y_com,
                    "l1_loss_hat": loss,
                    "mae": mae,
                    "mse": mse}

    def configure_optimizers(self):
        weight_decay = 5e-7  # l2正则化系数
        beta1 = 0.0
        beta2 = 0.9
        d2g_lr = 0.1

        gen_optimizer = Adam(
            params=self.generator.parameters(),
            lr=self.hparams.lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay
        )
        # 这里设置20个epoch后学习率变为原来的0.5，之后不再改变
        gen_opt_stepLR = torch.optim.lr_scheduler.MultiStepLR(gen_optimizer, milestones=[20], gamma=0.5)

        dis_optimizer = Adam(
            params=self.discriminator.parameters(),
            lr=self.hparams.lr * d2g_lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay
        )
        # 这里设置20个epoch后学习率变为原来的0.5，之后不再改变
        dis_opt_stepLR = torch.optim.lr_scheduler.MultiStepLR(dis_optimizer, milestones=[20], gamma=0.5)

        optim_dict = ({'optimizer': gen_optimizer, 'lr_scheduler': gen_opt_stepLR},
                      {'optimizer': dis_optimizer, 'lr_scheduler': dis_opt_stepLR})

        # 只要在training_step()函数中返回了loss，就会自动反向传播，
        # 并自动调用loss.backward()和optimizer.step()和stepLR.step()了
        return optim_dict

    def log_img(self, batch_img, name):
        if name == "train":
            step = self.global_step
        else:
            step = self.val_step
        sample_pred = batch_img
        grid = torchvision.utils.make_grid(sample_pred)
        self.logger.experiment.add_image(name, grid, step)


class edge_connect_no_gan(pl.LightningModule):
    def __init__(self, in_channel=3+1+1, out_channel=1, lr=0.0001, args=None):
        super().__init__()
        self.save_hyperparameters()
        self.val_step = 0

        self.FM_LOSS_WEIGHT = 0.1

        self.generator = EdgeGenerator(in_channels=in_channel, out_channels=out_channel)
        # GAN_LOSS: nsgan | lsgan | hinge
        self.GAN_LOSS = "hinge"
        self.discriminator = Discriminator(in_channels=out_channel, use_sigmoid=self.GAN_LOSS != 'hinge')

        self.l1_loss = nn.L1Loss()
        self.adversarial_loss = AdversarialLoss(type=self.GAN_LOSS)

    def forward(self, x):
        """
        :param x: [mask_img, mask_gradient, mask]
        :return: gradient_hat
        """
        outputs = self.generator(x)
        return outputs

    def training_step(self, batch, batch_idx, optimizer_idx):
        (x, y), (mask_y, mask) = batch
        y_hat = self.forward(x)
        y_com = y_hat * mask + mask_y
        dis_real, real_feat = self.discriminator(y)
        dis_fake, fake_feat = self.discriminator(y_hat)

        if optimizer_idx == 0:
            # gen
            l1_loss_com = self.l1_loss(y, y_com)
            l1_loss_hat = self.l1_loss(y, y_hat)
            gan_loss = self.adversarial_loss(dis_fake, is_real=True, is_disc=False)

            gen_loss = l1_loss_com + l1_loss_hat + gan_loss

            # generator feature matching loss
            gen_fm_loss = 0
            for i in range(len(fake_feat)):
                gen_fm_loss += self.l1_loss(fake_feat[i], real_feat[i])
            gen_fm_loss = gen_fm_loss * self.FM_LOSS_WEIGHT

            gen_loss = gen_loss + gen_fm_loss

            self.log("train/gen_l1_loss_com", l1_loss_com)
            self.log("train/gen_l1_loss_hat", l1_loss_hat)
            self.log("train/gan_loss", gan_loss)
            self.log("train/gen_fm_loss", gen_fm_loss)
            self.log("train/gen_loss", gen_loss)
            if self.global_step % 100 == 0:
                img = torch.cat([mask_y[:8], y[:8], y_hat[:8], y_com[:8]], dim=0)
                self.log_img(img, "train")

            return gen_loss

        elif optimizer_idx == 1:
            # dis
            real_loss = self.adversarial_loss(dis_real, is_real=True, is_disc=False)
            fake_loss = self.adversarial_loss(dis_fake, is_real=False, is_disc=True)

            dis_loss = (real_loss + fake_loss)/2

            self.log("train/real_loss", real_loss)
            self.log("train/fake_loss", fake_loss)
            self.log("train/dis_loss", dis_loss)
            return dis_loss

    def training_step_end(self, losses):
        loss = torch.mean(self.all_gather(losses, sync_grads=True))
        return loss

    def validation_step(self, batch, batch_idx):
        (x, y), (mask_y, mask) = batch
        y_hat = self.forward(x)
        y_com = y_hat * mask + mask_y

        dis_real, real_feat = self.discriminator(y)
        dis_fake, fake_feat = self.discriminator(y_hat)

        l1_loss_com = self.l1_loss(y, y_com)
        l1_loss_hat = self.l1_loss(y, y_hat)
        gan_loss = self.adversarial_loss(dis_fake, is_real=True, is_disc=False)

        gen_loss = l1_loss_com + l1_loss_hat + gan_loss

        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(fake_feat)):
            gen_fm_loss += self.l1_loss(fake_feat[i], real_feat[i])
        gen_fm_loss = gen_fm_loss * self.FM_LOSS_WEIGHT

        gen_loss = gen_loss + gen_fm_loss

        self.log("val_loss", l1_loss_com)
        self.log("val/gen_l1_loss_hat", l1_loss_hat)
        self.log("val/gan_loss", gan_loss)
        self.log("val/gen_fm_loss", gen_fm_loss)
        self.log("val/gen_loss", gen_loss)

        img = torch.cat([mask_y[:8], y[:8], y_hat[:8], y_com[:8]], dim=0)
        self.log_img(img, "val")

        ssim_y_hat = [ssim(m1, m2) for m1, m2 in zip(y, y_hat)]
        ssim_y_compose = [ssim(m1, m2) for m1, m2 in zip(y, y_com)]
        psnr_y_hat = [psnr(m1, m2) for m1, m2 in zip(y, y_hat)]
        psnr_y_compose = [psnr(m1, m2) for m1, m2 in zip(y, y_com)]

        ssim_y_hat_mean = np.mean(ssim_y_hat)
        ssim_y_compose_mean = np.mean(ssim_y_compose)
        psnr_y_hat_mean = np.mean(psnr_y_hat)
        psnr_y_compose_mean = np.mean(psnr_y_compose)

        self.log("val/ssim_hat", ssim_y_hat_mean)
        self.log("val/ssim_com", ssim_y_compose_mean)
        self.log("val/psnr_hat", psnr_y_hat_mean)
        self.log("val/psnr_com", psnr_y_compose_mean)

        self.val_step += 1
        return gen_loss

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        (x, y), (mask_y, mask) = batch
        y_hat = self.forward(x)
        y_com = y_hat * mask + mask_y

        l1_loss_hat = self.l1_loss(y, y_hat)

        mae = F.l1_loss(y_com, y)
        mse = F.mse_loss(y_com, y)

        if self.hparams.args.save_result:
            img = torch.cat([mask_y, y, y_hat, y_com], dim=0)
            size = y.shape[0]
            torchvision.utils.save_image(img, "test/{}/{}/{}/{}.jpg".format(self.hparams.args.model,
                                                                            self.hparams.args.dataset_name,
                                                                            self.hparams.args.stage,
                                                                            self.test_counter), nrow=size)

        ssim_y_hat = [ssim(m1, m2) for m1, m2 in zip(y, y_hat)]
        ssim_y_compose = [ssim(m1, m2) for m1, m2 in zip(y, y_com)]
        psnr_y_hat = [psnr(m1, m2) for m1, m2 in zip(y, y_hat)]
        psnr_y_compose = [psnr(m1, m2) for m1, m2 in zip(y, y_com)]

        ssim_y_hat_mean = np.mean(ssim_y_hat)
        ssim_y_compose_mean = np.mean(ssim_y_compose)
        psnr_y_hat_mean = np.mean(psnr_y_hat)
        psnr_y_compose_mean = np.mean(psnr_y_compose)

        return {"ssim_y_hat": ssim_y_hat_mean,
                "ssim_y_com": ssim_y_compose_mean,
                "psnr_y_hat": psnr_y_hat_mean,
                "psnr_y_com": psnr_y_compose_mean,
                "l1_loss_hat": l1_loss_hat,
                "mae": mae,
                "mse": mse,
                "idx": dataloader_idx}

    def test_step_end(self, output_results):
        all_gpus_result = self.all_gather(output_results)
        self.test_counter += 1

        ssim_y_hat = torch.mean(all_gpus_result["ssim_y_hat"]).cpu()
        ssim_y_com = torch.mean(all_gpus_result["ssim_y_com"]).cpu()
        psnr_y_hat = torch.mean(all_gpus_result["psnr_y_hat"]).cpu()
        psnr_y_com = torch.mean(all_gpus_result["psnr_y_com"]).cpu()
        mae = torch.mean(all_gpus_result["mae"]).cpu()
        mse = torch.mean(all_gpus_result["mse"]).cpu()
        loss = torch.mean(all_gpus_result["l1_loss_hat"]).cpu()

        return {"ssim_y_hat": ssim_y_hat,
                "ssim_y_com": ssim_y_com,
                "psnr_y_hat": psnr_y_hat,
                "psnr_y_com": psnr_y_com,
                "l1_loss_hat": loss,
                "mae": mae,
                "mse": mse
                }

    def test_epoch_end(self, outputs):
        if not self.hparams.args.if_split:
            ssim_y_hat = []
            ssim_y_com = []
            psnr_y_hat = []
            psnr_y_com = []
            loss_l = []
            mae_l = []
            mse_l = []

            for test_result in outputs:
                ssim_y_hat.append(test_result["ssim_y_hat"])
                ssim_y_com.append(test_result["ssim_y_com"])
                psnr_y_hat.append(test_result['psnr_y_hat'])
                psnr_y_com.append(test_result['psnr_y_com'])
                loss_l.append(test_result["l1_loss_hat"])
                mae_l.append(test_result["mae"])
                mse_l.append(test_result["mse"])

            ssim_y_hat = np.mean(ssim_y_hat)
            ssim_y_com = np.mean(ssim_y_com)
            psnr_y_hat = np.mean(psnr_y_hat)
            psnr_y_com = np.mean(psnr_y_com)
            mae_l = torch.mean(torch.tensor(mae_l))
            mse_l = torch.mean(torch.tensor(mse_l))
            loss_l = torch.mean(torch.tensor(loss_l))

            print("test:\n"
                  "ssim_y_hat:{}\n".format(ssim_y_hat),
                  "ssim_y_com:{}\n".format(ssim_y_com),
                  "psnr_y_hat:{}\n".format(psnr_y_hat),
                  "psnr_y_com:{}\n".format(psnr_y_com),
                  "mae:{}\n".format(mae_l),
                  "mse:{}\n".format(mse_l),
                  "l1_loss_hat:{}".format(loss_l))

            return {"ssim_y_hat": ssim_y_hat,
                    "ssim_y_com": ssim_y_com,
                    "psnr_y_hat": psnr_y_hat,
                    "psnr_y_com": psnr_y_com,
                    "l1_loss_hat": loss_l,
                    "mae": mae_l,
                    "mse": mse_l}
        else:
            results = []
            rates = [0.5, 0.4, 0.07, 0.01, 0.01, 0.01]
            for i, outputs_i in enumerate(outputs):
                ssim_y_hat = []
                ssim_y_com = []
                psnr_y_hat = []
                psnr_y_com = []
                loss_l = []
                mae_l = []
                mse_l = []

                for test_result in outputs_i:   # list
                    ssim_y_hat.append(test_result["ssim_y_hat"])
                    ssim_y_com.append(test_result["ssim_y_com"])
                    psnr_y_hat.append(test_result['psnr_y_hat'])
                    psnr_y_com.append(test_result['psnr_y_com'])
                    loss_l.append(test_result["l1_loss_hat"])
                    mae_l.append(test_result["mae"])
                    mse_l.append(test_result["mse"])

                result = [np.mean(ssim_y_hat), np.mean(ssim_y_com), np.mean(psnr_y_hat), np.mean(psnr_y_com),
                          torch.mean(torch.tensor(mae_l)), torch.mean(torch.tensor(mse_l)),
                          torch.mean(torch.tensor(loss_l)), i]
                results.append(result)

            ssim_y_hat = np.sum([rate*result[0] for rate, result in zip(rates, results)])
            ssim_y_com = np.sum([rate*result[1] for rate, result in zip(rates, results)])
            psnr_y_hat = np.sum([rate*result[2] for rate, result in zip(rates, results)])
            psnr_y_com = np.sum([rate*result[3] for rate, result in zip(rates, results)])
            loss = np.sum([rate*result[4] for rate, result in zip(rates, results)])
            mae = np.sum([rate*result[5] for rate, result in zip(rates, results)])
            mse = np.sum([rate*result[6] for rate, result in zip(rates, results)])

            print("test_zero-1~6:\n"
                  "ssim_y_hat:{}\n".format(ssim_y_hat),
                  "ssim_y_com:{}\n".format(ssim_y_com),
                  "psnr_y_hat:{}\n".format(psnr_y_hat),
                  "psnr_y_com:{}\n".format(psnr_y_com),
                  "mae:{}\n".format(mae),
                  "mse:{}\n".format(mse),
                  "l1_loss_hat:{}".format(loss),
                  )

            return {"ssim_y_hat": ssim_y_hat,
                    "ssim_y_com": ssim_y_com,
                    "psnr_y_hat": psnr_y_hat,
                    "psnr_y_com": psnr_y_com,
                    "l1_loss_hat": loss,
                    "mae": mae,
                    "mse": mse}

    def configure_optimizers(self):
        weight_decay = 5e-7  # l2正则化系数
        beta1 = 0.0
        beta2 = 0.9
        d2g_lr = 0.1

        gen_optimizer = Adam(
            params=self.generator.parameters(),
            lr=self.hparams.lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay
        )
        # 这里设置20个epoch后学习率变为原来的0.5，之后不再改变
        gen_opt_stepLR = torch.optim.lr_scheduler.MultiStepLR(gen_optimizer, milestones=[20], gamma=0.5)

        dis_optimizer = Adam(
            params=self.discriminator.parameters(),
            lr=self.hparams.lr * d2g_lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay
        )
        # 这里设置20个epoch后学习率变为原来的0.5，之后不再改变
        dis_opt_stepLR = torch.optim.lr_scheduler.MultiStepLR(dis_optimizer, milestones=[20], gamma=0.5)

        optim_dict = ({'optimizer': gen_optimizer, 'lr_scheduler': gen_opt_stepLR},
                      {'optimizer': dis_optimizer, 'lr_scheduler': dis_opt_stepLR})

        # 只要在training_step()函数中返回了loss，就会自动反向传播，
        # 并自动调用loss.backward()和optimizer.step()和stepLR.step()了
        return optim_dict

    def log_img(self, batch_img, name):
        if name == "train":
            step = self.global_step
        else:
            step = self.val_step
        sample_pred = batch_img
        grid = torchvision.utils.make_grid(sample_pred)
        self.logger.experiment.add_image(name, grid, step)