from incr_modules.fenced_module_masked import E2VIDRecurrentIncr
from ev_projs.rpg_e2depth.model.model import E2VIDRecurrent

import torch.nn as nn 
import spconv.pytorch as spconv

class ModelGetter:

    @staticmethod
    def get_e2vid_incr_model(pth=None) -> E2VIDRecurrentIncr:
        from ev_projs.rpg_e2depth.utils.loading_utils import load_model_incr
        if pth is None:
            pth='/home/sankeerth/ev/experiments/pretrained_models/E2DEPTH_si_grad_loss_mixed.pth.tar'
        return load_model_incr(pth)

    @staticmethod
    def get_e2vid_model(pth=None) -> E2VIDRecurrent:
        from ev_projs.rpg_e2depth.utils.loading_utils import load_model
        if pth is None:
            pth = '/home/sankeerth/ev/experiments/pretrained_models/E2DEPTH_si_grad_loss_mixed.pth.tar'
        return load_model(pth)

    @staticmethod
    def get_dense_conv() -> nn.Conv2d:
        conv2d = nn.Conv2d(32, 64, (5, 5), (1,1), bias=None)
        return conv2d

    @staticmethod
    def get_spconv_incr() -> spconv.SparseConv2d:
        return None

    @staticmethod
    def get_sbnet_conv():
        return None



