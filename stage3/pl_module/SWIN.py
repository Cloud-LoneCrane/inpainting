"""
Author: CraneLone
email: wang-zhizhen@outlook.com

file: SWIN
date: 2021/8/27 0027 上午 01:43
desc: 
"""

from pl_module.template import BASE
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
import os


class SwinUnet(BASE):
    def __init__(self, in_channel, out_channel, lr, args):
        super().__init__()
        self.save_hyperparameters()

        self.zero_head = False
        window_size = 8
        assert (256/4) % window_size == 0, "size must divide by 256 integral"
        self.swin_unet = SwinTransformerSys(img_size=256,
                                            patch_size=4,
                                            in_chans=self.hparams.in_channel,
                                            num_classes=self.hparams.out_channel,
                                            embed_dim=96,
                                            depths=[2, 2, 6, 2],
                                            num_heads=[3, 6, 12, 24],
                                            window_size=8,
                                            mlp_ratio=4.,
                                            qkv_bias=True,
                                            qk_scale=None,
                                            drop_rate=0.0,
                                            drop_path_rate=0.1,
                                            ape=False,
                                            patch_norm=True,
                                            use_checkpoint=False)

    def forward(self, x):
        return self.swin_unet(x)


if __name__ == '__main__':
    model = SwinUnet(3, 3, 0.001, None)
    print(model)