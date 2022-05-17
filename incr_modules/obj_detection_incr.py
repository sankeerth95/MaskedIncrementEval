import torch
import torch.nn as nn
from .mask_incr_modules import KFencedMaskModule, nnReluIncr, nnSequentialIncr, nnConvIncr, nnBatchNorm2dIncr, nnLinearIncr, nnMaxPool2dIncr

class DenseObjectDetIncr(nn.Module):
    def __init__(self, nr_classes, in_c=2, nr_box=2, small_out_map=True):
        super().__init__()
        self.nr_box = nr_box
        self.nr_classes = nr_classes

        self.kernel_size = 3
        self.conv_layers = nnSequentialIncr(
            self.conv_block(in_c=in_c, out_c=16),
            self.conv_block(in_c=16, out_c=32),
            self.conv_block(in_c=32, out_c=64),
            self.conv_block(in_c=64, out_c=128),
            self.conv_block(in_c=128, out_c=256, max_pool=False),
            nnConvIncr(in_channels=256, out_channels=512, kernel_size=self.kernel_size, stride=2, bias=False),
            nnBatchNorm2dIncr(512),
            nnReluIncr(),
        )

        self.relu1 = nnReluIncr()

        if small_out_map:
            self.cnn_spatial_output_size = [5, 7]
        else:
            self.cnn_spatial_output_size = [6, 8]

        spatial_size_product = self.cnn_spatial_output_size[0] * self.cnn_spatial_output_size[1]

        self.linear_input_features = spatial_size_product * 512
        self.linear_1 = nnLinearIncr(self.linear_input_features, 1024)
        self.linear_2 = nnLinearIncr(1024, spatial_size_product*(nr_classes + 5*self.nr_box))

    def conv_block(self, in_c, out_c, max_pool=True):
        if max_pool:
            return nnSequentialIncr(
                nnConvIncr(in_c, out_c, kernel_size=self.kernel_size, padding=(1, 1), bias=False),
                nnBatchNorm2dIncr(out_c),
                nnReluIncr(),
                nnConvIncr(out_c, out_c, kernel_size=self.kernel_size, padding=(1, 1), bias=False),
                nnBatchNorm2dIncr(out_c),
                nnReluIncr(),
                nnMaxPool2dIncr(kernel_size=self.kernel_size, stride=2),
                KFencedMaskModule()
            )
        else:
            return nnSequentialIncr(
                nnConvIncr(in_c, out_c, kernel_size=self.kernel_size, padding=(1, 1), bias=False),
                nnBatchNorm2dIncr(out_c),
                nnReluIncr(),
                nnConvIncr(out_c, out_c, kernel_size=self.kernel_size, padding=(1, 1), bias=False),
                nnBatchNorm2dIncr(out_c),
                nnReluIncr(),
            )


    def forward(self, x_incr):
        x = self.conv_layers(x_incr)
        x[0] = torch.flatten(x[0], 1)
        # x = x.view(-1, self.linear_input_features)
        x = self.linear_1(x)
        x = self.relu1(x)
        x = self.linear_2(x)
        return x[0].view([-1] + self.cnn_spatial_output_size + [(self.nr_classes + 5*self.nr_box)]), x[1]

    def forward_refresh_reservoirs(self, x_incr):
        x_incr = self.conv_layers.forward_refresh_reservoirs(x_incr)
        x_incr = torch.flatten(x_incr, 1)
        x_incr = self.linear_1.forward_refresh_reservoirs(x_incr)
        x_incr = self.relu1.forward_refresh_reservoirs(x_incr)
        x_incr = self.linear_2.forward_refresh_reservoirs(x_incr)
        x_incr = x_incr.view([-1] + self.cnn_spatial_output_size + [(self.nr_classes + 5*self.nr_box)])
        return x_incr


