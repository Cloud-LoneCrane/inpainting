import pytorch_lightning as pl
from argparse import ArgumentParser
from pl_module.ViT_SEG import vit_seg
from data.dataset_ import MyDataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import *
import os


def main(args):
    seed_everything(100, workers=True)
    data_module = MyDataModule(opt=args)
    if args.stage == 1:
        model = vit_seg(4, 1, args.lr, args.size)
    elif args.stage == 2:
        model = vit_seg(5, 1, args.lr, args.size)
    else:
        model = vit_seg(6, 3, args.lr, args.size)

    ckpt_path = 'ckpt\{}_{}'.format(args.dataset_name, args.stage)

    early_stop = EarlyStopping(monitor="val_loss", mode="min")
    checkpoint = ModelCheckpoint(monitor="val_loss",
                                 dirpath=ckpt_path,
                                 filename='{epoch:03d}-{val_loss:.2f}')
    callbacks = [early_stop, checkpoint]

    if resume:
        ckpt = os.listdir(ckpt_path)[-1]
        resume_checkpoint_path = os.path.join(ckpt_path, ckpt)
        trainer = Trainer(max_epochs=args.max_epoch, gpus=args.gpus,
                          resume_from_checkpoint=resume_checkpoint_path)
    else:
        trainer = Trainer(max_epochs=args.max_epoch, gpus=args.gpus)

    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    resume = False

    parser = ArgumentParser()
    parser.add_argument('--gpus', default=[0, 1], help="gpu use list or str")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--batch_size", default=24, type=int)
    parser.add_argument("--size", default=256, type=int, help="input image size")
    parser.add_argument("--num_workers", default=4, type=int, help="parallel load example num")
    parser.add_argument("--shuffle", default=True, type=bool, help="shuffle the dataloader")
    parser.add_argument("--max_epoch", default=50, type=int)

    parser.add_argument("--dataset_name", default=["celebA", "places2"], help="which dataset use save ckpt and logs")
    parser.add_argument("--stage", default=3, choices=[1, 2, 3])

    parser.add_argument("--train_image_path", default="/data/wm_data/celebA/celeba_train", type=str, help="train image path")
    parser.add_argument("--train_mask_path", default="/data/wm_data/celebA/1000_train_mask", type=str, help="train mask path")
    parser.add_argument("--val_image_path", default="/data/wm_data/celebA/celeba_val", type=str, help="validation image path")
    parser.add_argument("--val_mask_path", default="/data/wm_data/celebA/500_val_mask", type=str, help="validation mask path")

    parser.add_argument("--with_test", default=False, type=bool, help="whether with test")
    parser.add_argument("--test_image_path", default="", type=str, help="test image path")
    parser.add_argument("--test_mask_path", default="", type=str, help="test mask path")
    args = parser.parse_args()
    main(args)