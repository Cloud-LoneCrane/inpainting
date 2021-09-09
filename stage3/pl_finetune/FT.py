from pl_finetune.template_merged import BASE, BASESec, BASESec2, BASE3TH
from pl_module.ViT_SEG import vit_seg
from pl_module.Edge_Connect import edge_connect


class FineTune(BASE):
    def __init__(self, lr, resume_checkpoint_path1, resume_checkpoint_path2, resume_checkpoint_path3, args):
        super().__init__()
        self.save_hyperparameters()
        self.stage1_model = eval(self.hparams.args.stage1_model).load_from_checkpoint(resume_checkpoint_path1)
        self.stage2_model = eval(self.hparams.args.stage2_model).load_from_checkpoint(resume_checkpoint_path2)

        self.stage1_model.eval()
        if self.hparams.args.state == 2:
            self.stage2_model.train()
        else:
            self.stage2_model.eavl()
            self.stage3_model = eval(self.hparams.args.stage3_model).load_from_checkpoint(resume_checkpoint_path3)
            self.stage3_model.train()


class FineTuneSec(BASESec):
    def __init__(self, lr, resume_checkpoint_path1, resume_checkpoint_path2, args):
        super().__init__()
        self.save_hyperparameters()
        self.stage1_model = eval(self.hparams.args.stage1_model).load_from_checkpoint(resume_checkpoint_path1)
        self.stage2_model = eval(self.hparams.args.stage2_model).load_from_checkpoint(resume_checkpoint_path2)

        self.stage1_model.eval()    # fixed model1
        self.stage2_model.train()   # fine tune model2


class FineTune3TH(BASE3TH):
    def __init__(self, lr, resume_checkpoint_path1, resume_checkpoint_path2, args):
        super().__init__()
        self.save_hyperparameters()
        self.stage1_model = FineTuneSec.load_from_checkpoint(resume_checkpoint_path1)
        self.stage2_model = eval(self.hparams.args.stage3_model).load_from_checkpoint(resume_checkpoint_path2)

        self.stage1_model.eval()    # fixed model1
        self.stage2_model.train()   # fine tune model2