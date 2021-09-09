import pytorch_lightning as pl
from pytorch_lightning.callbacks import *
import torch.nn.functional as F
import torch
import torchvision
from torch.optim import *
from utils.metrics import ssim, psnr
import numpy as np
import pprint
import os


class BASE(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.test_counter = 0
        self.parse_batch = None     # is a function_pointer, must return (x, y), (mask_y, mask)

    def forward(self, x):
        # for inference
        y_hat = ...
        return y_hat

    def set_parse_batch(self, parse_batch_function):
        self.parse_batch = parse_batch_function

    def training_step(self, batch, batch_idx):
        (x, y), (mask_y, mask) = self.parse_batch(batch)
        y_hat = self.forward(x)
        y_hat_area = y_hat * mask
        y_area = y * mask
        y_compose = y_hat * mask + mask_y

        hat_mse, hat_mae = self.cal_loss(y_hat, y)
        area_mae = torch.mean(torch.sum(torch.abs(y_hat_area-y_area), dim=(2, 3))/torch.sum(mask, dim=(2, 3)))
        area_mse = torch.mean(torch.sum(torch.square(y_hat_area-y_area), dim=(2, 3))/torch.sum(mask, dim=(2, 3)))

        loss = (hat_mse + hat_mae) + (area_mae + area_mse)

        com_mse, com_mae = self.cal_loss(y_compose, y)
        self.log_dict({"train/area_mae": area_mae,
                       "train/area_mse": area_mse,
                       "train/hat_mse": hat_mse,
                       "train/hat_mae": hat_mae,

                       "train/com_mse": com_mse,
                       "train/com_mae": com_mae,
                       "train_loss": loss})

        img = torch.cat([mask_y[:8], y[:8], y_hat[:8], y_compose[:8]], dim=0)
        if self.global_step % self.hparams.args.log_every_n_steps == 0:
            self.log_img(img, "train")

        #
        ssim_y_hat = [ssim(m1, m2) for m1, m2 in zip(y, y_hat)]
        ssim_y_compose = [ssim(m1, m2) for m1, m2 in zip(y, y_compose)]
        psnr_y_hat = [psnr(m1, m2) for m1, m2 in zip(y, y_hat)]
        psnr_y_compose = [psnr(m1, m2) for m1, m2 in zip(y, y_compose)]

        ssim_y_hat_mean = np.mean(ssim_y_hat)
        ssim_y_compose_mean = np.mean(ssim_y_compose)
        psnr_y_hat_mean = np.mean(psnr_y_hat)
        psnr_y_compose_mean = np.mean(psnr_y_compose)

        self.log("train/ssim_hat", ssim_y_hat_mean)
        self.log("train/ssim_com", ssim_y_compose_mean)
        self.log("train/psnr_hat", psnr_y_hat_mean)
        self.log("train/psnr_com", psnr_y_compose_mean)

        return loss

    # def training_step_end(self, results):
    #     all_gpus_results = self.all_gather(results, sync_grads=True)
    #     loss = torch.mean(all_gpus_results["train_loss"])
    #     self.log("train_loss", loss)
    #     return loss

    def validation_step(self, batch, batch_idx):
        (x, y), (mask_y, mask) = self.parse_batch(batch)

        y_hat = self.forward(x)
        y_hat_area = y_hat * mask
        y_area = y * mask
        y_compose = y_hat * mask + mask_y

        hat_mse, hat_mae = self.cal_loss(y_hat, y)
        area_mae = torch.mean(torch.sum(torch.abs(y_hat_area-y_area), dim=(2, 3))/torch.sum(mask, dim=(2, 3)))
        area_mse = torch.mean(torch.sum(torch.square(y_hat_area-y_area), dim=(2, 3))/torch.sum(mask, dim=(2, 3)))

        loss = (hat_mse + hat_mae) + (area_mae + area_mse)

        com_mse, com_mae = self.cal_loss(y_compose, y)
        self.log_dict({"val/area_mae": area_mae,
                       "val/area_mse": area_mse,
                       "val/hat_mse": hat_mse,
                       "val/hat_mae": hat_mae,

                       "val/com_mse": com_mse,
                       "val/com_mae": com_mae,
                       "val_loss": loss})

        img = torch.cat([mask_y, y, y_hat, y_compose], dim=0)
        self.log_img(img, "val")

        ssim_y_hat = [ssim(m1, m2) for m1, m2 in zip(y, y_hat)]
        ssim_y_compose = [ssim(m1, m2) for m1, m2 in zip(y, y_compose)]
        psnr_y_hat = [psnr(m1, m2) for m1, m2 in zip(y, y_hat)]
        psnr_y_compose = [psnr(m1, m2) for m1, m2 in zip(y, y_compose)]

        ssim_y_hat_mean = np.mean(ssim_y_hat)
        ssim_y_compose_mean = np.mean(ssim_y_compose)
        psnr_y_hat_mean = np.mean(psnr_y_hat)
        psnr_y_compose_mean = np.mean(psnr_y_compose)

        self.log("val/ssim_hat", ssim_y_hat_mean)
        self.log("val/ssim_com", ssim_y_compose_mean)
        self.log("val/psnr_hat", psnr_y_hat_mean)
        self.log("val/psnr_com", psnr_y_compose_mean)

        return loss

    # def validation_step_end(self, results):
    #     all_gpus_results = self.all_gather(results)
    #     loss = torch.mean(all_gpus_results["val_loss"])
    #     self.log("val_loss", loss)
    #     return loss

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        (x, y), (mask_y, mask) = self.parse_batch(batch)

        y_hat = self.forward(x)
        y_hat_area = y_hat * mask
        y_area = y * mask
        y_compose = y_hat * mask + mask_y

        hat_mse, hat_mae = self.cal_loss(y_hat, y)
        area_mae = torch.mean(torch.sum(torch.abs(y_hat_area-y_area), dim=(2, 3))/torch.sum(mask, dim=(2, 3)))
        area_mse = torch.mean(torch.sum(torch.square(y_hat_area-y_area), dim=(2, 3))/torch.sum(mask, dim=(2, 3)))

        loss = (hat_mse + hat_mae) + (area_mae + area_mse)
        com_mse, com_mae = self.cal_loss(y_compose, y)

        if self.hparams.args.save_result:
            img = torch.cat([mask_y, y, y_hat, y_compose], dim=0)
            nrow = img.shape[0]//4
            torchvision.utils.save_image(img, "test/{}/{}/{}/{}.jpg".format(self.hparams.args.model,
                                                                            self.hparams.args.dataset_name,
                                                                            self.hparams.args.stage,
                                                                            self.test_counter),
                                         nrow=nrow)

        ssim_y_hat = [ssim(m1, m2) for m1, m2 in zip(y, y_hat)]
        ssim_y_compose = [ssim(m1, m2) for m1, m2 in zip(y, y_compose)]
        psnr_y_hat = [psnr(m1, m2) for m1, m2 in zip(y, y_hat)]
        psnr_y_compose = [psnr(m1, m2) for m1, m2 in zip(y, y_compose)]

        ssim_y_hat_mean = np.mean(ssim_y_hat)
        ssim_y_compose_mean = np.mean(ssim_y_compose)
        psnr_y_hat_mean = np.mean(psnr_y_hat)
        psnr_y_compose_mean = np.mean(psnr_y_compose)

        return {"ssim_hat": ssim_y_hat_mean,
                "ssim_com": ssim_y_compose_mean,
                "psnr_hat": psnr_y_hat_mean,
                "psnr_com": psnr_y_compose_mean,
                "loss": loss,
                "mse_hat": hat_mse,
                "mae_hat": hat_mae,
                "mse_com": com_mse,
                "mae_com": com_mae,
                "idx": dataloader_idx}

    def test_step_end(self, output_results):
        all_gpus_result = self.all_gather(output_results)
        self.test_counter += 1

        ssim_y_hat = torch.mean(all_gpus_result["ssim_hat"]).cpu()
        ssim_y_com = torch.mean(all_gpus_result["ssim_com"]).cpu()
        psnr_y_hat = torch.mean(all_gpus_result["psnr_hat"]).cpu()
        psnr_y_com = torch.mean(all_gpus_result["psnr_com"]).cpu()
        mae_hat = torch.mean(all_gpus_result["mae_hat"]).cpu()
        mse_hat = torch.mean(all_gpus_result["mse_hat"]).cpu()
        mae_com = torch.mean(all_gpus_result["mae_com"]).cpu()
        mse_com = torch.mean(all_gpus_result["mse_com"]).cpu()
        loss = torch.mean(all_gpus_result["loss"]).cpu()

        return {"ssim_hat": ssim_y_hat,
                "ssim_com": ssim_y_com,
                "psnr_hat": psnr_y_hat,
                "psnr_com": psnr_y_com,
                "loss": loss,
                "mae_hat": mae_hat,
                "mse_hay": mse_hat,
                "mae_com": mae_com,
                "mse_com": mse_com
                }

    def test_epoch_end(self, outputs):
        self.test_counter = 0
        if not self.hparams.args.if_split:
            ssim_y_hat = []
            ssim_y_com = []
            psnr_y_hat = []
            psnr_y_com = []
            loss_l = []
            mae_hat = []
            mse_hat = []
            mae_com = []
            mse_com = []

            for test_result in outputs:
                ssim_y_hat.append(test_result["ssim_hat"])
                ssim_y_com.append(test_result["ssim_com"])
                psnr_y_hat.append(test_result['psnr_hat'])
                psnr_y_com.append(test_result['psnr_com'])
                loss_l.append(test_result["loss"])
                mae_hat.append(test_result["mae_hat"])
                mse_hat.append(test_result["mse_hat"])
                mae_com.append(test_result["mae_com"])
                mse_com.append(test_result["mse_com"])

            ssim_y_hat = np.mean(ssim_y_hat)
            ssim_y_com = np.mean(ssim_y_com)
            psnr_y_hat = np.mean(psnr_y_hat)
            psnr_y_com = np.mean(psnr_y_com)
            mae_hat = torch.mean(torch.tensor(mae_hat))
            mse_hat = torch.mean(torch.tensor(mse_hat))
            mae_com = torch.mean(torch.tensor(mae_com))
            mse_com = torch.mean(torch.tensor(mse_com))
            loss_l = torch.mean(torch.tensor(loss_l))

            self.print("test:\n"
                       "ssim_hat:{}\n".format(ssim_y_hat),
                       "ssim_com:{}\n".format(ssim_y_com),
                       "psnr_hat:{}\n".format(psnr_y_hat),
                       "psnr_com:{}\n".format(psnr_y_com),
                       "mae_hat:{}\n".format(mae_hat),
                       "mse_hat:{}\n".format(mse_hat),
                       "mae_com:{}\n".format(mae_com),
                       "mse_com:{}\n".format(mse_com),
                       "loss:{}".format(loss_l))

            return {"ssim_hat": ssim_y_hat,
                    "ssim_com": ssim_y_com,
                    "psnr_hat": psnr_y_hat,
                    "psnr_com": psnr_y_com,
                    "loss": loss_l,
                    "mae_hat": mae_hat,
                    "mse_hat": mse_hat,
                    "mae_com": mae_com,
                    "mse_com": mse_com}
        else:
            results = []
            rates = [0.5, 0.4, 0.07, 0.01, 0.01, 0.01]
            for i, outputs_i in enumerate(outputs):
                ssim_y_hat = []
                ssim_y_com = []
                psnr_y_hat = []
                psnr_y_com = []
                loss_l = []
                mae_hat = []
                mse_hat = []
                mae_com = []
                mse_com = []

                for test_result in outputs_i:   # list
                    ssim_y_hat.append(test_result["ssim_hat"])
                    ssim_y_com.append(test_result["ssim_com"])
                    psnr_y_hat.append(test_result['psnr_hat'])
                    psnr_y_com.append(test_result['psnr_com'])
                    mae_hat.append(test_result["mae_hat"])
                    mse_hat.append(test_result["mse_hat"])
                    mae_com.append(test_result["mae_com"])
                    mse_com.append(test_result["mse_com"])
                    loss_l.append(test_result["loss"])

                result = [np.mean(ssim_y_hat), np.mean(ssim_y_com), np.mean(psnr_y_hat), np.mean(psnr_y_com),
                          torch.mean(torch.tensor(mae_hat)), torch.mean(torch.tensor(mse_hat)),
                          torch.mean(torch.tensor(mae_com)), torch.mean(torch.tensor(mse_com)),
                          torch.mean(torch.tensor(loss_l)), i]
                results.append(result)

            ssim_y_hat = np.sum([rate*result[0] for rate, result in zip(rates, results)])
            ssim_y_com = np.sum([rate*result[1] for rate, result in zip(rates, results)])
            psnr_y_hat = np.sum([rate*result[2] for rate, result in zip(rates, results)])
            psnr_y_com = np.sum([rate*result[3] for rate, result in zip(rates, results)])
            mae_hat = np.sum([rate*result[4] for rate, result in zip(rates, results)])
            mse_hat = np.sum([rate*result[5] for rate, result in zip(rates, results)])
            mae_com = np.sum([rate*result[6] for rate, result in zip(rates, results)])
            mse_com = np.sum([rate*result[7] for rate, result in zip(rates, results)])
            loss = np.sum([rate*result[8] for rate, result in zip(rates, results)])

            self.print("test_result:\n"
                       "ssim_hat:{}\n".format(ssim_y_hat),
                       "ssim_com:{}\n".format(ssim_y_com),
                       "psnr_hat:{}\n".format(psnr_y_hat),
                       "psnr_com:{}\n".format(psnr_y_com),
                       "mae_hat:{}\n".format(mae_hat),
                       "mse_hat:{}\n".format(mse_hat),
                       "mae_com:{}\n".format(mae_com),
                       "mse_com:{}\n".format(mse_com),
                       "loss:{}".format(loss),
                       )

            self.print("zero-1~6 result:")
            self.print("ssim_hat:{}\n".format([result[0] for result in results]),
                       "ssim_com:{}\n".format([result[1] for result in results]),
                       "psnr_hat:{}\n".format([result[2] for result in results]),
                       "psnr_com:{}\n".format([result[3] for result in results]),
                       "mae_hat:{}\n".format([result[4] for result in results]),
                       "mse_hat:{}\n".format([result[5] for result in results]),
                       "mae_com:{}\n".format([result[6] for result in results]),
                       "mse_com:{}\n".format([result[7] for result in results]),
                       "loss:{}".format([result[8] for result in results]))

            return {"ssim_hat": ssim_y_hat,
                    "ssim_com": ssim_y_com,
                    "psnr_hat": psnr_y_hat,
                    "psnr_com": psnr_y_com,
                    "loss": loss,
                    "mae_hat": mae_hat,
                    "mse_hay": mse_hat,
                    "mae_com": mae_com,
                    "mse_com": mse_com
                    }

    def configure_optimizers(self):
        weight_decay = 1e-6  # l2正则化系数
        optimizer = Adam(self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)
        # 这里设置20个epoch后学习率变为原来的0.5，之后不再改变
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.5)

        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}

        # 只要在training_step()函数中返回了loss，就会自动反向传播，
        # 并自动调用loss.backward()和optimizer.step()和stepLR.step()了
        return optim_dict

    def cal_loss(self, pred, target):
        mse_loss = F.mse_loss(pred, target)
        mae_loss = F.l1_loss(pred, target)
        return mse_loss, mae_loss

    def log_img(self, batch_img, name):
        step = self.global_step
        sample_pred = batch_img
        nrow = sample_pred.shape[0]//4
        grid = torchvision.utils.make_grid(sample_pred, nrow=nrow)
        self.logger.experiment.add_image(name, grid, step)

