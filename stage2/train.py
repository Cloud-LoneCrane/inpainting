"""
Author: CraneLone
email: wang-zhizhen@outlook.com

file: train
date: 2021/8/16 0016 下午 08:36
desc: 
"""

from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
from RN.model.pl_model import InpaintingModel as RN
from data.dataset import MyDataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import *
import os


def main(args):
    seed_everything(100, workers=True)
    data_module = MyDataModule(opt=args)

    # import matplotlib.pyplot as plt
    # import numpy as np
    # import torch
    # data_module.setup("fit")
    # train_loader = data_module.train_dataloader()
    # for data in train_loader:
    #     img, mask_img, gradient, mask_gradient, img_gray, mask_gray, mask, index = data
    #     # mask_img, mask_gradient, mask_gray, img, gradient, gray, mask, index = data
    #     img = torch.cat([mask_img[:8], img[:8], gradient[:8]], dim=0)
    #
    #     return

    prefix = "/data/exdata_zhr/wm/model"

    if args.E_G == "edge":
        structure_model_ckpt = "/data/exdata_zhr/wm/modules/ckpt/edge/vit_seg/paris_1/last.ckpt"
    else:
        structure_model_ckpt = "/data/exdata_zhr/wm/modules/ckpt/gradient/vit_seg/paris_1/last.ckpt"

    save_path = None
    if args.save_result:
        save_path = prefix + "/test/{}/{}/{}/{}".format(args.model, args.dataset_name, args.E_G, args.stage)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    # /data/wm_data/models/
    ckpt_path = prefix + '/ckpt/{}/{}/{}/{}'.format(args.model, args.dataset_name, args.E_G, args.stage)

    if train:
        data_module.setup("fit")

        early_stop = EarlyStopping(monitor="val/gen_loss", mode="min", patience=10)
        checkpoint = ModelCheckpoint(monitor="val/gen_loss",
                                     dirpath=ckpt_path,
                                     filename='{epoch:03d}-{val_loss:.5f}',
                                     save_top_k=20,
                                     save_last=True,
                                     save_weights_only=False,
                                     every_n_train_steps=args.every_n_train_steps  # how many train_step save weights ckpt
                                     )
        callbacks = [checkpoint]

        logger = TensorBoardLogger(prefix+'/logs/{}/{}/{}/{}'.format(args.model, args.dataset_name, args.E_G, args.stage))

        model = None
        if not resume:
            if args.stage == 1:
                model = eval(args.model)(3, 3, args, save_path, structure_model_ckpt)
            else:
                model = eval(args.model)(3+1, 3, args, save_path, structure_model_ckpt)

            trainer = Trainer(callbacks=callbacks, logger=logger, max_epochs=args.max_epoch,  gpus=args.gpus,
                              val_check_interval=int(args.val_check_interval),
                              limit_val_batches=args.limit_val_batches,
                              log_every_n_steps=args.log_every_n_steps,
                              accelerator="ddp")
        else:
            resume_checkpoint_path = os.path.join(ckpt_path, "last.ckpt")
            model = eval(args.model).load_from_checkpoint(resume_checkpoint_path)

            trainer = Trainer(callbacks=callbacks, logger=logger, max_epochs=args.max_epoch, gpus=args.gpus,
                              val_check_interval=int(args.val_check_interval),
                              resume_from_checkpoint=resume_checkpoint_path,
                              limit_val_batches=args.limit_val_batches,
                              log_every_n_steps=args.log_every_n_steps,
                              accelerator="ddp")

        trainer.fit(model, datamodule=data_module)

    if test:
        data_module.setup("test")
        ckpt_name = "last.ckpt"
        resume_checkpoint_path = os.path.join(ckpt_path, ckpt_name)
        model = eval(args.model).load_from_checkpoint(resume_checkpoint_path)
        model.hparams["args"] = args
        trainer = Trainer(gpus=args.gpus, accelerator="ddp")
        result = trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    flag = 2    # 1 for train, 2 for resume train, 3 for test

    if flag == 1:
        train = True
        resume = False
        test = False
    elif flag == 2:
        train = True
        resume = True
        test = False
    else:
        train = False
        resume = False
        test = True

    parser = ArgumentParser()
    parser.add_argument('--gpus', default=[0, 1, 2, 3], help="gpu use list or str")
    parser.add_argument("--g_lr", default=0.0001, type=float, help="generator learning rate")
    parser.add_argument("--d_lr", default=0.00001, type=float, help="discriminator learning rate")
    # parser.add_argument("--batch_size", default=50, type=int)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--size", default=256, type=int, help="input image size")
    parser.add_argument("--num_workers", default=4, type=int, help="parallel load example num")
    parser.add_argument("--shuffle", default=True, type=bool, help="shuffle the dataloader")
    parser.add_argument("--max_epoch", default=500, type=int)
    parser.add_argument("--save_result", default=True, type=bool, help="whether save test result")
    parser.add_argument("--every_n_train_steps", default=294, type=int, help="how many train_setp to save a modules")
    parser.add_argument("--val_check_interval", default=50, type=int, help="how many train_step val_step")
    parser.add_argument("--limit_val_batches", default=1, type=int, help="how many val batch to validate")
    parser.add_argument("--log_every_n_steps", default=50, type=int, help="how many train_step to log once")
    parser.add_argument("--model", default="RN",
                        choices=["RN", "LBAM", "MEDFE", "PRVS", "ShiftNet", "GMCNN", "PENNet", "EdgeConnect"])
    parser.add_argument("--dataset_name", default="paris", choices=["celebA", "places2", "paris"],
                        help="which dataset use save ckpt and logs")
    parser.add_argument("--stage", default=1, choices=[1, 2, 3])
    parser.add_argument("--E_G", default="edge", choices=["edge", "gradient"])
    parser.add_argument("--fine_tune", default=False, type=bool)
    parser.add_argument("--if_split", default=False, type=bool, help="whether seperate test zero-1 - zero-6 mask")

    # below the args not need to set
    parser.add_argument("--train_image_path", default=None, type=str, help="train image path")
    parser.add_argument("--train_mask_path", default=None, type=str, help="train mask path")
    parser.add_argument("--val_image_path", default=None, type=str, help="validation image path")
    parser.add_argument("--val_mask_path", default=None, type=str, help="validation mask path")
    parser.add_argument("--test_image_path", default=None, type=str, help="test image path")
    parser.add_argument("--test_mask_path", default=None, type=str, help="test mask path")
    args = parser.parse_args()

    if args.if_split:
        test_mask_path = "/data/exdata_zhr/wm/celebA/split_mask_yeah"
    else:
        test_mask_path = "/data/exdata_zhr/wm/celebA/500_test_mask"

    train_mask_path = "/data/exdata_zhr/wm/celebA/1000_train_mask"
    val_mask_path = "/data/exdata_zhr/wm/celebA/500_val_mask"

    if args.dataset_name == "celebA":
        train_image_path = "/data/exdata_zhr/wm/celebA/celeba_train"
        val_image_path = "/data/exdata_zhr/wm/celebA/celeba_val"
        test_image_path = "/data/exdata_zhr/wm/celebA/celeba_test"

    elif args.dataset_name == "places2":
        train_image_path = "/data/exdata_zhr/wm/places2/train"
        val_image_path = "/data/exdata_zhr/wm/places2/test"
        test_image_path = "/data/exdata_zhr/wm/places2/test"

    elif args.dataset_name == "paris":
        # train_image_path = "/data/exdata_zhr/wm/celebA/celeba_train"
        train_image_path = "/data/exdata_zhr/wm/paris/train"
        val_image_path = "/data/exdata_zhr/wm/paris/test"
        test_image_path = "/data/exdata_zhr/wm/paris/eval/paris_eval_gt"

    else:
        train_image_path = "/data/exdata_zhr/wm/paris/train"
        val_image_path = ""
        test_image_path = ""

    args.train_image_path = train_image_path
    args.val_image_path = val_image_path
    args.test_image_path = test_image_path
    args.train_mask_path = train_mask_path
    args.val_mask_path = val_mask_path
    args.test_mask_path = test_mask_path

    main(args)
