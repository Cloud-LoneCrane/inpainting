"""
Author: CraneLone
email: wang-zhizhen@outlook.com

file: t
date: 2021/8/23 0023 下午 11:48
desc: 
"""

import torch

gpus = torch.cuda.current_device()

print(gpus)