"""
Author: CraneLone
email: wang-zhizhen@outlook.com

file: train
date: 2021/8/16 0016 下午 08:36
desc: 
"""

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import *
import os
import importlib

from Options.Args import MyArg

from data.dataset import MyDataModule
from EdgeConnect.model.pl_model import EdgeConnect
from GMCNN.model.pl_model import GMCNN
from LBAM.model.pl_model import LBAM
# from MEDFE.model.pl_model import MEDFE
from PENNet.model.pl_model import PENNet
from PRVS.model.pl_model import PRVS
from RN.model.pl_model import RN


def main(args):
    seed_everything(100, workers=True)
    data_module = MyDataModule(opt=args)

    # import matplotlib.pyplot as plt
    # import numpy as np
    # import torch
    # data_module.setup("fit")
    # train_loader = data_module.train_dataloader()
    # print(train_loader.__len__())
    # for data in train_loader:
    #     img, mask_img, gradient, mask_gradient, img_gray, mask_gray, mask, index = data
    #     # mask_img, mask_gradient, mask_gray, img, gradient, gray, mask, index = data
    #     img = torch.cat([mask_img[:8], img[:8], gradient[:8]], dim=0)
    #
    #     return

    prefix = os.getcwd()

    if args.E_G == "edge":
        structure_model_ckpt = "EG_Model/ckpt/vit_seg/paris/edge/last.ckpt"
    else:
        structure_model_ckpt = "EG_Model/ckpt/vit_seg/paris/gradient/last.ckpt"
    structure_model_ckpt = os.path.join(prefix, structure_model_ckpt)

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
                                     save_top_k=10,
                                     save_last=True,
                                     save_weights_only=False,
                                     every_n_train_steps=args.every_n_train_steps  # how many train_step save weights ckpt
                                     )
        callbacks = [checkpoint]

        logger = TensorBoardLogger(prefix+'/logs/{}/{}/{}/{}'.format(args.model, args.dataset_name, args.E_G, args.stage))

        model = None
        if not resume:
            if args.stage == 1:
                if args.model == "RN":
                    model = eval(args.model)(3, 3, args, save_path, structure_model_ckpt)
                elif args.model == "PRVS":
                    model = eval(args.model)(3, args, save_path, structure_model_ckpt)
                elif args.model == "GMCNN":
                    model = eval(args.model)(4, 3, args, save_path, structure_model_ckpt)
                elif args.model == "LBAM":
                    model = eval(args.model)(4, 3, args, save_path, structure_model_ckpt)
                elif args.model == "MEDFE":
                    model = eval(args.model)(4, 3, args, save_path, structure_model_ckpt)
                elif args.model == "PENNet":
                    model = eval(args.model)(3, 3, args, save_path, structure_model_ckpt)
                elif args.model == "ShiftNet":
                    pass
                elif args.model == "EdgeConnect":
                    model = eval(args.model)(4, 3, args, save_path, structure_model_ckpt)
            else:
                if args.model == "RN":
                    model = eval(args.model)(3+1, 3, args, save_path, structure_model_ckpt)
                elif args.model == "PRVS":
                    model = eval(args.model)(3, 3, args, save_path, structure_model_ckpt)
                elif args.model == "GMCNN":
                    model = eval(args.model)(4+1, 3, args, save_path, structure_model_ckpt)
                elif args.model == "LBAM":
                    model = eval(args.model)(4+1, 3, args, save_path, structure_model_ckpt)
                elif args.model == "MEDFE":
                    model = eval(args.model)(4+1, 3, args, save_path, structure_model_ckpt)
                elif args.model == "PENNet":
                    model = eval(args.model)(3+1, 3, args, save_path, structure_model_ckpt)
                elif args.model == "ShiftNet":
                    pass
                elif args.model == "EdgeConnect":
                    model = eval(args.model)(4, 3, args, save_path, structure_model_ckpt)

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
        logger = TensorBoardLogger('test_logs/{}/{}/{}/{}'.format(args.model, args.dataset_name, args.E_G, args.stage))
        data_module.setup("test")
        ckpt_name = "last.ckpt"
        resume_checkpoint_path = os.path.join(ckpt_path, ckpt_name)
        model = eval(args.model).load_from_checkpoint(resume_checkpoint_path)
        model.hparams["args"] = args
        trainer = Trainer(gpus=args.gpus, accelerator="ddp", logger=logger)
        result = trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    data_root = "/home/yangdehe/Wangmeng/data"
    flag = 1    # 1 for train, 2 for resume train, 3 for test

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

    args = MyArg().args
    ARGS = None
    module = "{}.model.Parser".format(args.model)
    ARGS = importlib.import_module(module)
    args = ARGS.MyArgs().args

    if args.dataset_name == "celebA":
        train_image_path = data_root + "/celebA/celeba_train"
        val_image_path = data_root + "/celebA/celeba_val"
        test_image_path = data_root + "/celebA/celeba_test"

    elif args.dataset_name == "places2":
        train_image_path = data_root + "/places2/train"
        val_image_path = data_root + "/places2/test"
        test_image_path = data_root + "/places2/test"

    elif args.dataset_name == "paris":
        train_image_path = data_root + "/paris/train"
        val_image_path = data_root + "/paris/test"
        test_image_path = data_root + "/paris/eval/paris_eval_gt"

    else:
        train_image_path = ""
        val_image_path = ""
        test_image_path = ""

    if args.if_split:
        test_mask_path = data_root + "/celebA/split_mask_yeah"
    else:
        test_mask_path = data_root + "/celebA/500_test_mask"

    train_mask_path = data_root + "/celebA/1000_train_mask"
    val_mask_path = data_root + "/celebA/500_val_mask"

    args.train_image_path = train_image_path
    args.val_image_path = val_image_path
    args.test_image_path = test_image_path
    args.train_mask_path = train_mask_path
    args.val_mask_path = val_mask_path
    args.test_mask_path = test_mask_path

    main(args)


