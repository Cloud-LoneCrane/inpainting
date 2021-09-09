"""
Author: CraneLone
email: wang-zhizhen@outlook.com

file: Args
date: 2021/8/25 0025 下午 08:22
desc: 
"""
from argparse import ArgumentParser


class MyArg:
    def __init__(self):
        self.parser = ArgumentParser()
        self.parser.add_argument('--gpus', default=[0, ], help="gpu use list or str")
        self.parser.add_argument("--model", default="PENNet",
                                 choices=["RN", "LBAM", "MEDFE", "PRVS", "ShiftNet", "GMCNN", "PENNet", "EdgeConnect"])
        self.parser.add_argument("--dataset_name", default="paris", choices=["celebA", "places2", "paris"],
                                 help="which dataset use save ckpt and logs")
        self.parser.add_argument("--max_epoch", default=500, type=int)
        self.parser.add_argument("--stage", default=1, choices=[1, 2, 3])
        self.parser.add_argument("--E_G", default="raw", choices=["raw", "edge", "gradient"])
        self.parser.add_argument("--fine_tune", default=False, type=bool)
        self.parser.add_argument("--if_split", default=False, type=bool,
                                 help="whether seperate test zero-1 - zero-6 mask")
        self.parser.add_argument("--save_result", default=False, type=bool, help="whether save test result")

        self.parser.add_argument("--g_lr", default=0.0001, type=float, help="generator learning rate")
        self.parser.add_argument("--d_lr", default=0.00001, type=float, help="discriminator learning rate")
        self.parser.add_argument("--g_beta1", default=0.0, type=float)
        self.parser.add_argument("--g_beta2", default=0.9, type=float)
        self.parser.add_argument("--d_beta1", default=0.0, type=float)
        self.parser.add_argument("--d_beta2", default=0.9, type=float)

        self.parser.add_argument("--batch_size", default=1, type=int)
        self.parser.add_argument("--size", default=256, type=int, help="input image size")
        self.parser.add_argument("--num_workers", default=0, type=int, help="parallel load example num")
        self.parser.add_argument("--shuffle", default=True, type=bool, help="shuffle the dataloader")

        self.parser.add_argument("--every_n_train_steps", default=500, type=int,
                                 help="how many train_setp to save a modules")
        self.parser.add_argument("--val_check_interval", default=100, type=int, help="how many train_step val_step")
        self.parser.add_argument("--limit_val_batches", default=1, type=int, help="how many val batch to validate")
        self.parser.add_argument("--log_every_n_steps", default=100, type=int, help="how many train_step to log once")

        # below the args not need to set
        self.parser.add_argument("--train_image_path", default=None, type=str, help="train image path")
        self.parser.add_argument("--train_mask_path", default=None, type=str, help="train mask path")
        self.parser.add_argument("--val_image_path", default=None, type=str, help="validation image path")
        self.parser.add_argument("--val_mask_path", default=None, type=str, help="validation mask path")
        self.parser.add_argument("--test_image_path", default=None, type=str, help="test image path")
        self.parser.add_argument("--test_mask_path", default=None, type=str, help="test mask path")
        self.my_args = self.parser.parse_args()

    @property
    def args(self):
        return self.my_args