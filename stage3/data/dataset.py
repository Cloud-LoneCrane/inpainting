import os
import glob
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms
from skimage.feature import canny
import cv2
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from utils.tools import dict_to_object


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_path, mask_path, target_size=256, training=True, augment=True, mask_reverse=False):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.image_names = self.load_list(image_path)
        self.mask_names = self.load_list(mask_path)
        self.target_size = target_size
        self.mask_reverse = mask_reverse

        self.sigma = 2

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.image_names[index])
            item = self.load_item(0)

        return item

    def load_item(self, index):
        # load image
        img_bgr = cv2.imread(self.image_names[index], 1)  # BGR
        if self.training:
            img_bgr = self.resize(img_bgr)  # img, aspect_ratio_kept=True, fixed_size=False, centerCrop=False
        else:
            img_bgr = self.resize(img_bgr, True, True, True)

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)    # img_gray: uint8
        gradient = self.harf_canny(img_gray) / 255.
        edge = self.load_edge(img_gray)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255.
        mask = self.load_mask(index)

        img_gray = img_gray / 255.

        img = img.transpose(2, 0, 1)

        revers_mask = 1.0 - mask

        mask_img = img * revers_mask
        mask_gray = img_gray * revers_mask
        mask_gradient = gradient * revers_mask
        mask_edge = edge * revers_mask

        img = img.transpose(1, 2, 0)
        mask_img = mask_img.transpose(1, 2, 0)

        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            mask_img = mask_img[:, ::-1, ...]
            gradient = gradient[:, ::-1, ...]
            mask_gradient = mask_gradient[:, ::-1, ...]
            img_gray = img_gray[:, ::-1, ...]
            mask_gray = mask_gray[:, ::-1, ...]
            mask = mask[:, ::-1, ...]
            edge = edge[:, ::-1, ...]
            mask_edge = mask_edge[:, ::-1, ...]

        img, mask_img = self.to_tensor(img.copy()), self.to_tensor(mask_img.copy())
        gradient, mask_gradient = self.to_tensor(gradient.copy()), self.to_tensor(mask_gradient.copy())
        edge, mask_edge = self.to_tensor(edge.copy()), self.to_tensor(mask_edge.copy())
        img_gray, mask_gray = self.to_tensor(img_gray.copy()), self.to_tensor(mask_gray.copy())
        mask = self.to_tensor(mask.copy())

        return {"image": img, "mask_image": mask_img,
                "gradient": gradient, "mask_gradient": mask_gradient,
                "edge": edge, "mask_edge": mask_edge,
                "gray": img_gray, "mask_gray": mask_gray,
                "mask": mask,
                "index": index}

    def load_edge(self, img):
        return canny(img, sigma=self.sigma).astype(np.float)

    def harf_canny(self, img):
        new_gray = cv2.GaussianBlur(img, (5, 5), 1.4)
        x = cv2.Sobel(new_gray, cv2.CV_16S, 1, 0, ksize=3)
        y = cv2.Sobel(new_gray, cv2.CV_16S, 0, 1, ksize=3)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        return dst

    def load_mask(self, index):
        if self.augment:
            mask_index = random.randint(0, len(self.mask_names) - 1)
        else:
            _, mask_index = np.divmod(index, len(self.mask_names))

        mask_BRG = cv2.imread(self.mask_names[mask_index])
        mask = cv2.cvtColor(mask_BRG, cv2.COLOR_BGR2GRAY)
        mask = self.resize(mask, False)
        mask = (mask > 127.5).astype(np.uint8).astype(np.float)  # threshold due to interpolation

        if self.mask_reverse:
            return 1.0 - mask
        else:
            return mask * 1.0

    def resize(self, img, aspect_ratio_kept=True, fixed_size=False, centerCrop=False):
        if aspect_ratio_kept:
            imgh, imgw = img.shape[0:2]
            side = np.minimum(imgh, imgw)
            if fixed_size:
                if centerCrop:
                    # center crop
                    j = (imgh - side) // 2
                    i = (imgw - side) // 2
                    img = img[j:j + side, i:i + side, ...]
                else:
                    j = (imgh - side)
                    i = (imgw - side)
                    h_start = 0
                    w_start = 0
                    if j != 0:
                        h_start = random.randrange(0, j)
                    if i != 0:
                        w_start = random.randrange(0, i)
                    img = img[h_start:h_start + side, w_start:w_start + side, ...]
            else:
                if side <= self.target_size:
                    j = (imgh - side)
                    i = (imgw - side)
                    h_start = 0
                    w_start = 0
                    if j != 0:
                        h_start = random.randrange(0, j)
                    if i != 0:
                        w_start = random.randrange(0, i)
                    img = img[h_start:h_start + side, w_start:w_start + side, ...]
                else:
                    side = random.randrange(self.target_size, side)
                    j = (imgh - side)
                    i = (imgw - side)
                    h_start = random.randrange(0, j)
                    w_start = random.randrange(0, i)
                    img = img[h_start:h_start + side, w_start:w_start + side, ...]
        img = cv2.resize(img, (self.target_size, self.target_size))
        return img

    def to_tensor(self, img):
        img_t = F.to_tensor(img).float()
        return img_t

    def load_list(self, flist):
        flist = list(glob.glob(flist + '/*.jpg')) + \
                list(glob.glob(flist + '/*.JPG')) + \
                list(glob.glob(flist + '/*.png')) + \
                list(glob.glob(flist + '/*.PNG'))
        flist.sort()
        return flist


class MyDataModule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.test_datasets = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            # image_path, mask_path, target_size=256, training=True, augment=True, mask_reverse=False
            self.train_dataset = Dataset(self.opt.train_image_path, self.opt.train_mask_path, self.opt.size)
            self.val_dataset = Dataset(self.opt.val_image_path, self.opt.val_mask_path, self.opt.size, False, False)
        if stage == "test":
            if self.opt.if_split:
                folders = [os.path.join(self.opt.test_mask_path, folder) for folder in
                           sorted(os.listdir(self.opt.test_mask_path))]
                self.test_datasets = [
                    Dataset(self.opt.test_image_path, mask_path, self.opt.size, False, False)
                    for mask_path in folders]
            else:
                self.test_dataset = Dataset(self.opt.test_image_path, self.opt.test_mask_path, self.opt.size, False,
                                            False)

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset, batch_size=self.opt.batch_size, shuffle=self.opt.shuffle,
                                num_workers=self.opt.num_workers)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.val_dataset, batch_size=self.opt.batch_size, shuffle=False)
        return dataloader

    def test_dataloader(self):
        if self.opt.if_split:
            dataloaders = [DataLoader(test_dataset, batch_size=self.opt.batch_size, shuffle=False)
                           for test_dataset in self.test_datasets]
            return dataloaders
        else:
            dataloader = DataLoader(self.test_dataset, batch_size=self.opt.batch_size, shuffle=False)
            return dataloader


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torchvision
    import copy

    img_flist = '/home/GPU24User/data/celebA/celeba_train'
    mask_flist = '/home/GPU24User/data/celebA/1000_train_mask'

    val_image_path = "/home/GPU24User/data/wm_data/celebA/celeba_val"
    val_mask_path = "/home/GPU24User/data/wm_data/celebA/500_val_mask"

    args = {"train_image_path": img_flist,
            "train_mask_path": mask_flist,
            "val_image_path": val_mask_path,
            "val_mask_path": val_mask_path,
            "size": 256,
            "batch_size": 3,
            "shuffle": True,
            "num_workers": 1}
    args = dict_to_object(args)
    data_mudule = MyDataModule(args)
    data_mudule.setup("fit")

    train_dataloader = data_mudule.train_dataloader()
    for batch in train_dataloader:
        image = batch["image"]
        mask_image = batch["mask_image"]
        gradient = batch["gradient"]
        mask_gradient = batch["mask_gradient"]
        edge = batch["edge"]
        mask_edge = batch["mask_edge"]
        gray = batch["gray"]
        mask_gray = batch["mask_gray"]
        mask = batch["mask"]
        index = batch["index"]
        # rgb = torch.cat([image, mask_image], dim=0)
        # gray = torch.cat([gradient, mask_gradient,
        #                   edge, mask_edge,
        #                   gray, mask_gray,
        #                   mask], dim=0)
        # rgb_grid = torchvision.utils.make_grid(rgb, nrow=args.batch_size)
        # gray_grid = torchvision.utils.make_grid(gray, nrow=args.batch_size)
        # torchvision.utils.save_image(rgb_grid, "rgb.jpg")
        # torchvision.utils.save_image(gray_grid, "gray.jpg")
        # print(rgb_grid.min(), rgb_grid.max())
        # print(gray_grid.min(), gray_grid.max())
        image = image[:, :, :100, :100]
        img2 = copy.deepcopy(image)
        mask = mask[:, :, :100, :100]
        img2[0, 0, 0, 0] = 1

        sum = torch.sum(torch.abs(image - img2), dim=(2, 3))    # (10, 3)
        mean = torch.sum(mask, dim=(2, 3))     # (10, 1)

        area_mean = sum/mean
        loss_mean = torch.mean(area_mean)
        print(loss_mean)
        print(sum)
        print(mean)
        print(area_mean)
        print(torch.mean(torch.abs(image - img2)))
        break