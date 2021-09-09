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


def my_transforms():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return transform


class Dataset(torch.utils.data.Dataset):
    def __init__(self, stage, image_path, mask_path, target_size=256,  training=True, augment=True, mask_reverse=False):
        super(Dataset, self).__init__()
        self.stage = stage
        self.augment = augment
        self.training = training
        self.data = self.load_list(image_path)
        self.mask_data = self.load_list(mask_path)

        self.target_size = target_size
        self.mask_reverse = mask_reverse

        self.sigma = 2
        self.mask_reverse = mask_reverse

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        # load image
        img_BRG = cv2.imread(self.data[index], 1)   # BGR
        if self.training:
            img_BRG = self.resize(img_BRG)
        else:
            img_BRG = self.resize(img_BRG, True, True, True)

        img_gray = cv2.cvtColor(img_BRG, cv2.COLOR_BGR2GRAY)
        # gradient = self.harf_canny(img_gray)/255.
        gradient = self.load_edge(img_gray)
        img = cv2.cvtColor(img_BRG, cv2.COLOR_BGR2RGB)/255.
        mask = self.load_mask(index)/255.
        # 0 for hole, 1 for valid

        img = img.transpose(2, 0, 1)

        mask_img = img * (1-mask)
        mask_gray = img_gray * (1-mask)/255.
        mask_gradient = gradient * (1-mask)

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
            # print(index)

        img, mask_img = self.to_tensor(img.copy()), self.to_tensor(mask_img.copy())
        gradient, mask_gradient = self.to_tensor(gradient.copy()), self.to_tensor(mask_gradient.copy())
        img_gray, mask_gray = self.to_tensor(img_gray.copy()), self.to_tensor(mask_gray.copy())
        mask = self.to_tensor(mask.copy())

        if self.stage == 1:     # mask_img, mask_gradient, mask, gradient
            x = torch.cat([mask_img, mask_gradient, mask], dim=0)
            return (x, gradient), (mask_gradient, mask)
        elif self.stage == 2:   # [mask_img, gradient, mask], img_gray
            x = torch.cat([mask_img, gradient, mask], dim=0)
            return (x, img_gray), (mask_gray, mask)
        elif self.stage == 3:   # [mask_img, gradient, img_gray], img
            x = torch.cat([mask_img, gradient, img_gray, mask], dim=0)
            return (x, img), (mask_img, mask)
        else:
            return img, mask_img, gradient, mask_gradient, img_gray, mask_gray, mask, index

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
            mask_index = random.randint(0, len(self.mask_data) - 1)
        else:
            _, mask_index = np.divmod(index, len(self.mask_data))

        mask_BRG = cv2.imread(self.mask_data[mask_index])
        mask = cv2.cvtColor(mask_BRG, cv2.COLOR_BGR2GRAY)
        mask = self.resize(mask, False)
        mask = (mask > 0).astype(np.uint8)  # threshold due to interpolation
        if self.mask_reverse:
            return (1 - mask) * 255
        else:
            return mask * 255

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
        flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.JPG')) + list(glob.glob(flist + '/*.png'))
        flist.sort()
        return flist


def generate_stroke_mask(im_size, parts=15, maxVertex=25, maxLength=100, maxBrushWidth=24, maxAngle=360):
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)
    mask = np.concatenate([mask, mask, mask], axis=2)
    return mask


def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)
        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)
        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask


class MyDataModule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            # image_path, mask_path, target_size=256,  training=True, augment=True, mask_reverse=False
            self.train_dataset = Dataset(4, self.opt.train_image_path, self.opt.train_mask_path, self.opt.size)
            self.val_dataset = Dataset(4, self.opt.val_image_path, self.opt.val_mask_path, self.opt.size, False, False)
        if stage == "test":
            if self.opt.if_split:
                folders = [os.path.join(self.opt.test_mask_path, folder) for folder in sorted(os.listdir(self.opt.test_mask_path))]
                self.test_datasets = [Dataset(4, self.opt.test_image_path, mask_path, self.opt.size, False, False)
                                      for mask_path in folders]
            else:
                self.test_dataset = Dataset(4, self.opt.test_image_path, self.opt.test_mask_path, self.opt.size, False, False)

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset, batch_size=self.opt.batch_size, shuffle=self.opt.shuffle,
                                num_workers=self.opt.num_workers)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.val_dataset, batch_size=self.opt.batch_size, shuffle=False,
                                num_workers=self.opt.num_workers)
        return dataloader

    def test_dataloader(self):
        if self.opt.if_split:
            dataloaders = [DataLoader(test_dataset, batch_size=self.opt.batch_size, shuffle=False)
                           for test_dataset in self.test_datasets]
            return dataloaders
        else:
            dataloader = DataLoader(self.test_dataset, batch_size=self.opt.batch_size, shuffle=False)
            return dataloader


def build_dataset(flist, mask_flist, augment, mask_mode, mask_reverse, training, input_size):
    # tage, image_path, mask_path, target_size=256,  training=True, augment=True, mask_reverse=False
    dataset = Dataset(
        3,
        image_path=flist,
        mask_path=mask_flist,
        target_size=input_size,
        training=training,
        augment=augment,
        )

    print('Total instance number:', dataset.__len__())

    return dataset


def build_dataloader(dataset, batch_size, num_workers, shuffle):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=shuffle
    )
    return dataloader


if __name__ == '__main__':
    img_flist = '/data/wm_data/paris/train'
    mask_flist = '/home/GPU24User/data/celebA/1000_train_mask'

    dataset = Dataset(3, img_flist, mask_flist, 256, training=False, augment=True)
    dataloader = build_dataloader(dataset, 1, 1, True)
    import matplotlib.pyplot as plt

    for batch in dataloader:
        (x, y), (mask_x, mask) = batch
        plt.imshow(np.transpose(y[0], (1, 2, 0)))
        plt.show()

        plt.imshow(np.transpose(mask_x[0], (1, 2, 0)))
        plt.show()

        compose = y * mask
        plt.imshow(np.transpose(compose[0], (1, 2, 0)))
        plt.show()
        plt.imshow(np.transpose(mask[0], (1, 2, 0)))
        plt.show()

        break