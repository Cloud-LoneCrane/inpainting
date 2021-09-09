from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
from pl_finetune.FT import FineTuneSec, FineTune3TH
from pl_module.ViT_SEG import vit_seg
from data.dataset import MyDataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import *
import os


def main(args):
    seed_everything(100, workers=True)
    data_module = MyDataModule(opt=args)
    prefix = "/data/wm_data/models/"

    if args.stage == 2:
        if args.save_result:
            path = "test/{}/{}/{}_ft2/".format(args.stage2_model, args.dataset_name, args.stage)
            if not os.path.exists(path):
                os.makedirs(path)
        ckpt1_path = prefix + 'ckpt/{}/{}_{}'.format(args.stage1_model, args.dataset_name, 1)   # ckpt/vit_seg/places2_1
        ckpt2_path = prefix + 'ckpt/{}/{}_{}'.format(args.stage2_model, args.dataset_name, 2)   # ckpt/vit_seg/places2_2

        ckpt1_name = "last.ckpt"
        ckpt2_name = "last.ckpt"

        resume_checkpoint_path1 = os.path.join(ckpt1_path, ckpt1_name)
        resume_checkpoint_path2 = os.path.join(ckpt2_path, ckpt2_name)

        ckpt_path = ckpt2_path
        model_name = args.stage2_model

        if fine_tune:
            data_module.setup("fit")
            early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)
            checkpoint = ModelCheckpoint(monitor="val_loss",
                                         dirpath=ckpt_path + "_finetune",
                                         filename='{epoch:03d}-{val_loss:.8f}',
                                         save_top_k=15,  # -1 for saving all
                                         save_last=True,
                                         save_weights_only=False
                                         )
            callbacks = [checkpoint]

            logger = TensorBoardLogger('/data/wm_data/models/logs/{}/{}/{}_finetune'.format(model_name, args.dataset_name, args.stage))
            # logs/vit_seg/places2/2_finetune

            # stage1_model = eval(args.stage1_model).load_from_checkpoint(resume_checkpoint_path1)
            # stage1_model.eval()
            # stage1_predict = stage1_model.forward
            model = FineTuneSec(args.lr, resume_checkpoint_path1, resume_checkpoint_path2, args=args)

            if resume:
                resume_ckpt = ckpt_path + "_finetune"
                ckpt_name = "last.ckpt"
                resume_path = os.path.join(resume_ckpt, ckpt_name)
                trainer = Trainer(callbacks=callbacks, logger=logger, max_epochs=args.max_epoch, gpus=args.gpus,
                                  resume_from_checkpoint=resume_path,
                                  val_check_interval=int(args.val_check_interval),
                                  limit_val_batches=args.limit_val_batches,
                                  log_every_n_steps=args.log_every_n_steps,
                                  accelerator="ddp")
            else:
                trainer = Trainer(callbacks=callbacks, logger=logger, max_epochs=args.max_epoch, gpus=args.gpus,
                                  val_check_interval=int(args.val_check_interval),
                                  limit_val_batches=args.limit_val_batches,
                                  log_every_n_steps=args.log_every_n_steps,
                                  accelerator="ddp")

            trainer.fit(model, datamodule=data_module)

        if test:
            MODEL = FineTuneSec

            data_module.setup("test")
            ckpt_name = "last.ckpt"
            resume_checkpoint_path = os.path.join(ckpt_path + "_finetune", ckpt_name)
            model = MODEL.load_from_checkpoint(resume_checkpoint_path)

            model.hparams["args"] = args
            trainer = Trainer(gpus=args.gpus, accelerator="ddp")
            result = trainer.test(model, datamodule=data_module)
            print(result)

    elif args.stage == 3:
        if args.save_result:
            path = "test/{}/{}/{}_ft3/".format(args.stage2_model, args.dataset_name, args.stage)
            if not os.path.exists(path):
                os.makedirs(path)

        ckpt1_path = prefix + 'ckpt/{}/{}_{}_finetune'.format(args.stage1_model, args.dataset_name, 2)  # ckpt/vit_seg/places2_2_finetune
        ckpt2_path = prefix + 'ckpt/{}/{}_{}'.format(args.stage2_model, args.dataset_name, 3)

        ckpt1_name = "last.ckpt"
        ckpt2_name = "last.ckpt"

        resume_checkpoint_path1 = os.path.join(ckpt1_path, ckpt1_name)
        resume_checkpoint_path2 = os.path.join(ckpt2_path, ckpt2_name)

        ckpt_path = ckpt2_path
        model_name = args.stage3_model

        if fine_tune:
            data_module.setup("fit")
            early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)
            checkpoint = ModelCheckpoint(monitor="val_loss",
                                         dirpath=ckpt_path + "_finetune",
                                         filename='{epoch:03d}-{val_loss:.8f}',
                                         save_top_k=10,  # -1 for saving all
                                         save_last=True,
                                         save_weights_only=False
                                         )
            callbacks = [checkpoint]

            logger = TensorBoardLogger('/data/wm_data/models/logs/{}/{}/{}_finetune'.format(model_name, args.dataset_name, args.stage))
            # logs/vit_seg/places2/2_finetune

            model = FineTune3TH(args.lr, resume_checkpoint_path1, resume_checkpoint_path2, args=args)

            if resume:
                resume_ckpt = ckpt_path + "_finetune"
                ckpt_name = "last.ckpt"
                resume_path = os.path.join(resume_ckpt, ckpt_name)
                trainer = Trainer(callbacks=callbacks, logger=logger, max_epochs=args.max_epoch,  gpus=args.gpus,
                                  resume_from_checkpoint=resume_path,
                                  val_check_interval=int(args.val_check_interval),
                                  limit_val_batches=args.limit_val_batches,
                                  log_every_n_steps=args.log_every_n_steps,
                                  accelerator="ddp")
            else:
                trainer = Trainer(callbacks=callbacks, logger=logger, max_epochs=args.max_epoch,  gpus=args.gpus,
                                  val_check_interval=int(args.val_check_interval),
                                  limit_val_batches=args.limit_val_batches,
                                  log_every_n_steps=args.log_every_n_steps,
                                  accelerator="ddp")
            trainer.fit(model, datamodule=data_module)

        if test:
            MODEL = FineTune3TH
            data_module.setup("test")
            ckpt_name = "last.ckpt"
            resume_checkpoint_path = os.path.join(ckpt_path + "_finetune", ckpt_name)
            model = MODEL.load_from_checkpoint(resume_checkpoint_path)

            model.hparams["args"] = args
            trainer = Trainer(gpus=args.gpus, accelerator="ddp")
            result = trainer.test(model, datamodule=data_module)
            print(result)


if __name__ == '__main__':
    fine_tune = True
    resume = True
    test = False

    parser = ArgumentParser()
    parser.add_argument('--gpus', default=[0, 1, 2, 3], help="gpu use list or str")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--size", default=256, type=int, help="input image size")
    parser.add_argument("--num_workers", default=4, type=int, help="parallel load example num")
    parser.add_argument("--shuffle", default=True, type=bool, help="shuffle the dataloader")
    parser.add_argument("--max_epoch", default=50+10+20, type=int)
    parser.add_argument("--save_result", default=True, type=bool, help="whether save test result")

    parser.add_argument("--every_n_train_steps", default=1000, type=int, help="how many train_setp to save a model")
    parser.add_argument("--val_check_interval", default=100, type=int, help="how many train_step val_step")
    parser.add_argument("--limit_val_batches", default=1, type=int, help="how many val batch to validate")
    parser.add_argument("--log_every_n_steps", default=100, type=int, help="how many train_step to log once")

    parser.add_argument("--fine_tune", default=True, type=bool, help="use to denote which dataset result to return")
    parser.add_argument("--stage1_model", default="vit_seg", choices=["vit_seg", ])
    parser.add_argument("--stage2_model", default="vit_seg", choices=["vit_seg", ])
    parser.add_argument("--stage3_model", default="vit_seg", choices=["vit_seg", ])

    parser.add_argument("--stage", default=2, choices=[2, 3], help="which stage model to fine tune")
    parser.add_argument("--dataset_name", default="paris", choices=["celebA", "places2", "paris"],
                        help="which dataset use save ckpt and logs")
    parser.add_argument("--if_split", default=False, type=bool, help="whether seperate test zero-1 - zero-6 mask")

    # below the args not set here
    parser.add_argument("--train_image_path", default="", type=str, help="train image path")
    parser.add_argument("--train_mask_path", default="", type=str, help="train mask path")
    parser.add_argument("--val_image_path", default="", type=str, help="validation image path")
    parser.add_argument("--val_mask_path", default="", type=str, help="validation mask path")
    parser.add_argument("--test_image_path", default="", type=str, help="test image path")
    parser.add_argument("--test_mask_path", default="", type=str, help="test mask path")
    args = parser.parse_args()

    # change your data and mask below
    train_image_path = None
    val_image_path = None
    test_image_path = None
    if args.dataset_name == "celebA":
        train_image_path = "/data/wm_data/celebA/celeba_train"
        val_image_path = "/data/wm_data/celebA/celeba_val"
        test_image_path = "/data/wm_data/celebA/celeba_test"
    elif args.dataset_name == "places2":
        train_image_path = "/data/wm_data/places2/train"
        val_image_path = "/data/wm_data/places2/test"
        test_image_path = val_image_path
    elif args.dataset_name == "paris":
        train_image_path = "/data/wm_data/paris/train"
        val_image_path = "/data/wm_data/paris/eval/paris_eval_gt"
        test_image_path = "/data/wm_data/paris/test"

    if args.if_split:
        test_mask_path = "/data/wm_data/celebA/split_mask_yeah"
    else:
        test_mask_path = "/data/wm_data/celebA/500_test_mask"

    train_mask_path = "/data/wm_data/celebA/1000_train_mask"
    val_mask_path = "/data/wm_data/celebA/500_val_mask"

    args.train_image_path = train_image_path
    args.train_mask_path = train_mask_path
    args.val_image_path = val_image_path
    args.val_mask_path = val_mask_path
    args.test_image_path = test_image_path
    args.test_mask_path = test_mask_path
    main(args)