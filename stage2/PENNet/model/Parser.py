"""
Author: CraneLone
email: wang-zhizhen@outlook.com

file: Parser
date: 2021/8/25 0025 下午 08:26
desc: 
"""

from Options.Args import MyArg


class MyArgs(MyArg):
    def __init__(self):
        super().__init__()

    @property
    def args(self):
        self.my_args.model = "PENNet"
        self.my_args.g_lr = 1e-4
        self.my_args.g_beta1 = 0.0
        self.my_args.g_beta2 = 0.999

        self.my_args.d_lr = 1e-4
        self.my_args.d_beta1 = 0.0
        self.my_args.d_beta2 = 0.999

        return self.my_args