from EG_Model.pl_module.template import BASE
from EG_Model.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from EG_Model.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import os


class vit_seg(BASE):
    def __init__(self, in_channel, out_channel, lr, args):
        super().__init__()
        self.save_hyperparameters()

        vit_patches_size = 16
        vit_name = "R50-ViT-B_16"
        config_vit = CONFIGS_ViT_seg[vit_name]
        config_vit.n_classes = out_channel
        config_vit.n_skip = 3

        config_vit.patches.grid = (int(args.size / vit_patches_size), int(args.size / vit_patches_size))
        self.model = ViT_seg(config_vit, args.size, num_classes=out_channel, in_channels=in_channel)

    def forward(self, x):
        return self.model(x)