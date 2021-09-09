import os
import numpy as np
import skimage
import skimage.io
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def save_img(path, name, img):
    # img (H,W,C) or (H,W) np.uint8
    skimage.io.imsave(path+'/'+name+'.png', img)


def PSNR(pred, gt, shave_border=0):
    return compare_psnr(pred, gt, data_range=255)
    # imdff = pred - gt
    # rmse = math.sqrt(np.mean(imdff ** 2))
    # if rmse == 0:
    #     return 100
    # return 20 * math.log10(255.0 / rmse)


def L1(pred, gt):
    return np.mean(np.abs((np.mean(pred, 2) - np.mean(gt, 2))/255))


def L2(pred, gt):
    return np.mean(np.square((np.mean(pred, 2) - np.mean(gt, 2))/255))


def SSIM(pred, gt, data_range=255, win_size=11, multichannel=True):
    return compare_ssim(pred, gt, data_range=data_range, multichannel=multichannel, win_size=win_size)


def reslut_compare(img_path, predict_path):
    names = os.listdir(img_path)
    names.sort()
    num = len(names)

    total_psnr = 0.0
    total_ssim = 0.0
    total_l1 = 0.0
    total_l2 = 0.0
    for name in names:
        img_name = os.path.join(img_path, name)
        pred_name = os.path.join(predict_path, name)
        img = skimage.io.imread(img_name)
        pred = skimage.io.imread(pred_name)

        psnr = PSNR(img, pred)
        ssim = SSIM(img, pred)
        l1 = L1(img, pred)
        l2 = L2(img, pred)

        total_psnr += psnr
        total_ssim += ssim
        total_l1 += l1
        total_l2 += l2

    return total_psnr/num, total_ssim/num, total_l1/num, total_l2/num


if __name__ == '__main__':
    print("111")