import logger as logger
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
from pl_module.ViT_SEG import vit_seg
from pl_module.SWIN import SwinUnet
from pl_module.Edge_Connect import edge_connect
from data.dataset import MyDataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import *
import os
import torch


def parse_batch_1(batch):
    x = torch.cat([batch["mask_image"], batch["mask"]], dim=1)
    return (x, batch["image"]), (batch["mask_image"], batch["mask"])


def parse_batch_2(batch):
    x = torch.cat([batch["mask_image"], batch["gradient"], batch["mask"]], dim=1)
    return (x, batch["image"]), (batch["mask_image"], batch["mask"])


def parse_batch_3(batch):
    x = torch.cat([batch["mask_image"], batch["gray"], batch["mask"]], dim=1)
    return (x, batch["image"]), (batch["mask_image"], batch["mask"])


def parse_batch_4(batch):
    x = torch.cat([batch["mask_image"], batch["gradient"], batch["gray"], batch["mask"]], dim=1)
    return (x, batch["image"]), (batch["mask_image"], batch["mask"])


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
    seed_everything(100, workers=True)
    data_module = MyDataModule(opt=args)
    parser_batch_func = eval("parse_batch_{}".format(args.stage))

    prefix = "/data/wm_data/ablation"
    if args.save_result:
        path = "test/{}/{}/{}".format(args.model, args.dataset_name, args.stage)
        if not os.path.exists(path):
            os.makedirs(path)

    ckpt_path = os.path.join(prefix, 'ckpt/{}/{}_{}'.format(args.model, args.dataset_name, args.stage))

    if train:
        data_module.setup("fit")

        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=10)
        checkpoint = ModelCheckpoint(monitor="val_loss",
                                     dirpath=ckpt_path,
                                     filename='{epoch:03d}-{val_loss:.8f}',
                                     save_top_k=10,
                                     save_last=True,
                                     save_weights_only=False,
                                     every_n_train_steps=args.every_n_train_steps    # how many train_step save weights ckpt
                                     )
        callbacks = [checkpoint]
        logger_path = 'logs/{}/{}/{}'.format(args.model, args.dataset_name, args.stage)
        logger = TensorBoardLogger(os.path.join(prefix, logger_path))

        trainer = None
        model = None
        if not resume:
            if args.stage == 1:
                model = eval(args.model)(3+1, 3, args.lr, args)
            elif args.stage == 2 or args.stage == 3:
                model = eval(args.model)(3+1+1, 3, args.lr, args)
            else:
                model = eval(args.model)(3+1+1+1, 3, args.lr, args)

            trainer = Trainer(callbacks=callbacks, logger=logger, max_epochs=args.max_epoch,  gpus=args.gpus,
                              accelerator="ddp",    # if model is edge_connect set "ddp2" will distribute error
                              val_check_interval=int(args.val_check_interval),
                              limit_val_batches=args.limit_val_batches,
                              log_every_n_steps=args.log_every_n_steps)

        else:
            resume_checkpoint_path = os.path.join(ckpt_path, "last.ckpt")
            model = eval(args.model).load_from_checkpoint(resume_checkpoint_path)

            trainer = Trainer(callbacks=callbacks, logger=logger, max_epochs=args.max_epoch, gpus=args.gpus,
                              accelerator="ddp",
                              val_check_interval=int(args.val_check_interval),
                              resume_from_checkpoint=resume_checkpoint_path,
                              limit_val_batches=args.limit_val_batches,
                              log_every_n_steps=args.log_every_n_steps)
        model.set_parse_batch(parser_batch_func)
        trainer.fit(model, datamodule=data_module)

    if test:
        data_module.setup("test")
        ckpt_name = "last.ckpt"
        resume_checkpoint_path = os.path.join(ckpt_path, ckpt_name)
        model = eval(args.model).load_from_checkpoint(resume_checkpoint_path)
        model.set_parse_batch(parser_batch_func)
        model.hparams["args"] = args
        trainer = Trainer(gpus=args.gpus, accelerator="ddp")

        result = trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    data_prefix = "/data/wm_data"

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

    parser = ArgumentParser()
    parser.add_argument('--gpus', default=[0, 1], help="gpu use list or str")
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
    parser.add_argument("--batch_size", default=35, type=int)
    parser.add_argument("--size", default=256, type=int, help="input image size")
    parser.add_argument("--num_workers", default=2, type=int, help="parallel load example num")
    parser.add_argument("--shuffle", default=True, type=bool, help="shuffle the dataloader")
    parser.add_argument("--max_epoch", default=500, type=int)
    parser.add_argument("--save_result", default=True, type=bool, help="whether save test result")
    parser.add_argument("--every_n_train_steps", default=500, type=int, help="how many train_setp to save a model")
    parser.add_argument("--val_check_interval", default=100, type=int, help="how many train_step val_step")
    parser.add_argument("--limit_val_batches", default=1, type=int, help="how many val batch to validate")
    parser.add_argument("--log_every_n_steps", default=100, type=int, help="how many train_step to log once")

    parser.add_argument("--model", default="SwinUnet", choices=["vit_seg", "SwinUnet", "edge_connect"])
    parser.add_argument("--dataset_name", default="celebA", choices=["celebA", "places2", "paris"], help="which dataset use save ckpt and logs")
    parser.add_argument("--stage", default=4, choices=[1, 2, 3, 4], type=int)     # 消融实验：验证的是第几组消融实验
    parser.add_argument("--if_split", default=False, type=bool, help="whether seperate test zero-1 - zero-6 mask")

    # below the args not need to set
    parser.add_argument("--train_image_path", default=None, type=str, help="train image path")
    parser.add_argument("--train_mask_path", default=None, type=str, help="train mask path")
    parser.add_argument("--val_image_path", default=None, type=str, help="validation image path")
    parser.add_argument("--val_mask_path", default=None, type=str, help="validation mask path")
    parser.add_argument("--test_image_path", default=None, type=str, help="test image path")
    parser.add_argument("--test_mask_path", default=None, type=str, help="test mask path")
    args = parser.parse_args()

    # change your data and mask below
    train_image_path = None
    val_image_path = None
    test_image_path = None
    if args.dataset_name == "celebA":
        train_image_path = "celebA/celeba_train"
        val_image_path = "celebA/celeba_val"
        test_image_path = "celebA/celeba_test"
    elif args.dataset_name == "places2":
        train_image_path = "places2/train"
        val_image_path = "places2/test"
        test_image_path = val_image_path
    elif args.dataset_name == "paris":
        train_image_path = "paris/train"
        val_image_path = "paris/eval/paris_eval_gt"
        test_image_path = "paris/test"

    if args.if_split:
        test_mask_path = "celebA/split_mask_yeah"
    else:
        test_mask_path = "celebA/500_test_mask"

    train_mask_path = "celebA/1000_train_mask"
    val_mask_path = "celebA/500_val_mask"

    args.train_image_path = os.path.join(data_prefix, train_image_path)
    args.val_image_path = os.path.join(data_prefix, val_image_path)
    args.test_image_path = os.path.join(data_prefix, test_image_path)
    args.train_mask_path = os.path.join(data_prefix, train_mask_path)
    args.val_mask_path = os.path.join(data_prefix, val_mask_path)
    args.test_mask_path = os.path.join(data_prefix, test_mask_path)

    main(args)