import torch
import torch.nn as nn
from incr_modules.evflownet_incr import ConvLayerIncr, ResidualBlockIncr
from .mask_incr_modules import nnConvIncr

class FireFlowNetIncr(nn.Module):
    """
    FireFlowNet architecture for (dense/sparse) optical flow estimation from event-data.
    "Back to Event Basics: Self Supervised Learning of Image Reconstruction from Event Data via Photometric Constancy", Paredes-Valles et al., 2020
    """

    def __init__(self, unet_kwargs, num_bins):
        super().__init__()
        base_num_channels = unet_kwargs["base_num_channels"]
        kernel_size = unet_kwargs["kernel_size"]
        self.mask = unet_kwargs["mask_output"]

        padding = kernel_size // 2
        self.E1 = ConvLayerIncr(num_bins, base_num_channels, kernel_size, padding=padding)
        self.E2 = ConvLayerIncr(base_num_channels, base_num_channels, kernel_size, padding=padding)
        self.R1 = ResidualBlockIncr(base_num_channels, base_num_channels)
        self.E3 = ConvLayerIncr(base_num_channels, base_num_channels, kernel_size, padding=padding)
        self.R2 = ResidualBlockIncr(base_num_channels, base_num_channels)
        self.pred = ConvLayerIncr(base_num_channels, out_channels=2, kernel_size=1, activation="tanh")



    def forward_refresh_reservoirs(self, x):
        """
        :param inp_voxel: N x num_bins x H x W
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        """
        inp_voxel = x
        # forward pass
        x = self.E1.forward_refresh_reservoirs(inp_voxel)
        x = self.E2.forward_refresh_reservoirs(x)
        x, _ = self.R1.forward_refresh_reservoirs(x)
        x = self.E3.forward_refresh_reservoirs(x)
        x, _ = self.R2.forward_refresh_reservoirs(x)
        flow = self.pred.forward_refresh_reservoirs(x)

        # # mask flow
        # if self.mask:
        #     mask = torch.sum(inp_cnt, dim=1, keepdim=True)
        #     mask[mask > 0] = 1
        #     flow = flow * mask

        return {"flow": [flow]}

    def forward(self, x_incr):
        """
        :param inp_voxel: N x num_bins x H x W
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        """
        inp_voxel = x_incr

        # forward pass
        x_incr = inp_voxel
        x_incr = self.E1(x_incr)
        x_incr = self.E2(x_incr)
        x_incr,_ = self.R1(x_incr)
        x_incr = self.E3(x_incr)
        x_incr,_ = self.R2(x_incr)
        flow = self.pred(x_incr)

        # # mask flow
        # if self.mask:
        #     mask = torch.sum(inp_cnt, dim=1, keepdim=True)
        #     mask[mask > 0] = 1
        #     flow = flow * mask

        return {"flow": [flow]}


