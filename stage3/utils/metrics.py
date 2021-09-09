from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
import torch


def ssim(m1, m2):
    n1 = np.transpose(np.array(m1.cpu()), [1, 2, 0])
    n2 = m2.detach().permute(1, 2, 0).cpu().numpy()
    # n2 = np.transpose(np.array(m2.detach().numpy()), [1, 2, 0])
    channel = n1.shape[-1]
    multichannel = False
    if channel > 1:
        multichannel = True
    return structural_similarity(n1, n2, data_range=1.0, multichannel=True)


def psnr(m1, m2):
    n1 = np.transpose(np.array(m1.cpu()), [1, 2, 0])
    n2 = m2.detach().permute(1, 2, 0).cpu().numpy()
    # n2 = np.transpose(np.array(m2.detach().numpy()), [1, 2, 0])
    return peak_signal_noise_ratio(n1, n2, data_range=1.0)
