import torch
import torch.nn as nn
from incr_modules.evflownet_incr import ConvLayerIncr, ResidualBlockIncr, UpsampleConvLayerIncr
from .mask_incr_modules import nnConvIncr
import torch.nn.functional as F

from torch.nn import ZeroPad2d


def skip_concat(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    if diffY != 0:
        dummy = 1
    if diffX != 0:
        dummy = 1
    # assert diffY == 0
    # assert diffX == 0
    if diffX < 0:
        first_X, second_X = diffX - diffX // 2, diffX // 2
    else:
        first_X, second_X = diffX // 2, diffX - diffX // 2
    if diffY < 0:
        first_Y, second_Y = diffY - diffY // 2, diffY // 2
    else:
        first_Y, second_Y = diffY // 2, diffY - diffY // 2
    # padding = ZeroPad2d((diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
    padding = ZeroPad2d((first_X, second_X, first_Y, second_Y))
    # if diffY != 0 or diffX != 0:
    #     print(f"{padding=}")
    x1 = padding(x1)
    return torch.cat([x1, x2], dim=1)



def skip_concat_incr(x1, x2):
    diffY = x2[0].size()[2] - x1[0].size()[2]
    diffX = x2[0].size()[3] - x1[0].size()[3]
    if diffY != 0 or diffX != 0:
        dummy = 1
    # assert diffY == 0
    # assert diffX == 0
    if diffX < 0:
        first_X, second_X = diffX - diffX // 2, diffX // 2
    else:
        first_X, second_X = diffX // 2, diffX - diffX // 2
    if diffY < 0:
        first_Y, second_Y = diffY - diffY // 2, diffY // 2
    else:
        first_Y, second_Y = diffY // 2, diffY - diffY // 2
    # padding = ZeroPad2d((diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
    padding = ZeroPad2d((first_X, second_X, first_Y, second_Y))
    # if diffY != 0 or diffX != 0:
    #     print(f"{padding=}")
    x1[0] = padding(x1[0])
    return torch.cat([x1[0], x2[0]], dim=1), None


class BaseUNet(nn.Module):
    """
    Base class for conventional UNet architecture.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(
        self,
        base_num_channels,
        num_encoders,
        num_residual_blocks,
        num_output_channels,
        skip_type,
        norm,
        use_upsample_conv,
        num_bins,
        recurrent_block_type=None,
        kernel_size=5,
        channel_multiplier=2,
    ):
        super(BaseUNet, self).__init__()
        self.base_num_channels = base_num_channels
        self.num_encoders = num_encoders
        self.num_residual_blocks = num_residual_blocks
        self.num_output_channels = num_output_channels
        self.kernel_size = kernel_size
        self.skip_type = skip_type
        self.norm = norm
        self.num_bins = num_bins
        self.recurrent_block_type = recurrent_block_type
        self.channel_multiplier = channel_multiplier

        self.skip_ftn = eval("skip_" + skip_type)
        if use_upsample_conv:
            self.UpsampleLayer = UpsampleConvLayerIncr
        else:
            self.UpsampleLayer = TransposedConvLayer
        assert self.num_output_channels > 0

        self.encoder_input_sizes = [
            int(self.base_num_channels * pow(self.channel_multiplier, i)) for i in range(self.num_encoders)
        ]
        self.encoder_output_sizes = [
            int(self.base_num_channels * pow(self.channel_multiplier, i + 1)) for i in range(self.num_encoders)
        ]
        self.max_num_channels = self.encoder_output_sizes[-1]

    def build_encoders(self):
        encoders = nn.ModuleList()
        for (input_size, output_size) in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            encoders.append(
                ConvLayer(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=2,
                    padding=self.kernel_size // 2,
                    activation=self.activation,
                    norm=self.norm,
                )
            )
        return encoders

    def build_resblocks(self):
        resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            resblocks.append(ResidualBlockIncr(self.max_num_channels, self.max_num_channels, norm=self.norm))
        return resblocks

    def build_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for input_size, output_size in zip(decoder_input_sizes, decoder_output_sizes):
            decoders.append(
                self.UpsampleLayer(
                    input_size if self.skip_type == "sum" else 2 * input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    norm=self.norm,
                )
            )
        return decoders

    def build_prediction_layer(self, num_output_channels, norm=None):
        return ConvLayer(
            self.base_num_channels if self.skip_type == "sum" else 2 * self.base_num_channels,
            num_output_channels,
            1,
            activation=None,
            norm=norm,
        )



class DelayedMultiResUNetIncr(BaseUNet):
    """
    Conventional UNet architecture.
    Symmetric, skip connections on every encoding layer.
    Predictions at each decoding layer.
    Predictions are added as skip connection (concat) to the input of the subsequent layer.
    """

    def __init__(self, unet_kwargs):
        self.final_activation = unet_kwargs.pop("final_activation", "none")
        self.skip_type = "concat"
        super().__init__(**unet_kwargs)

        self.encoders = self.build_encoders()
        self.resblocks = self.build_resblocks()
        self.decoders = self.build_multires_prediction_decoders()
        self.preds = self.build_multires_prediction_layer()
        self.delayed_pred_layer_list = self.build_delayed_prediction_layer()

    def build_encoders(self):
        encoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(self.encoder_input_sizes, self.encoder_output_sizes)):
            if i == 0:
                input_size = self.num_bins
            encoders.append(
                ConvLayerIncr(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=2,
                    padding=self.kernel_size // 2,
                    norm=self.norm,
                )
            )
        return encoders

    def build_delayed_prediction_layer(self):
        # 2 is the number of channel for prediction at each level
        delayed_prediction_input_channel = sum(self.encoder_input_sizes) + 2 * self.num_encoders
        delayed_pred_layer = nn.ModuleList()
        delayed_pred_layer.append(ConvLayerIncr(delayed_prediction_input_channel, self.base_num_channels, 1, activation="relu", norm=self.norm))
        delayed_pred_layer.append(ConvLayerIncr(self.base_num_channels + 2 * self.num_encoders, 2, 1, activation=self.final_activation, norm=self.norm))
        return delayed_pred_layer

    def build_multires_prediction_layer(self):
        preds = nn.ModuleList()
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        for output_size in decoder_output_sizes:
            preds.append(
                ConvLayerIncr(output_size, self.num_output_channels, 1, activation=self.final_activation, norm=self.norm)
            )
        return preds

    def build_multires_prediction_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(decoder_input_sizes, decoder_output_sizes)):
            prediction_channels = 0 if i == 0 else self.num_output_channels
            multiplier = 2 if i == 0 else 1
            decoders.append(
                self.UpsampleLayer(
                    multiplier * input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    norm=self.norm,
                )
            )
        return decoders

    def forward(self, x):

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            blocks.append(x)

        # residual blocks
        last_resblock_output = None
        for resblock in self.resblocks:
            x, _ = resblock(x)
            last_resblock_output = x

        # decoder with delayed concatenation
        predictions = []
        decoder_out_list = []
        for i, (decoder, pred) in enumerate(zip(self.decoders, self.preds)):
            if i == 0:
                decoder_in = skip_concat_incr(last_resblock_output, blocks[self.num_encoders - i - 1])
            else:
                decoder_in = blocks[self.num_encoders - i - 1]
            decoder_out = decoder(decoder_in)
            decoder_out_list.append(decoder_out)
            predictions.append(pred(decoder_out))

        delayed_all_combined = decoder_out_list[-1][0]
        for i, decoder_out in enumerate(reversed(decoder_out_list[:-1]), start=1):
            delayed_all_combined = torch.cat((delayed_all_combined,
                                              F.interpolate(decoder_out[0], size=(delayed_all_combined.shape[-2], delayed_all_combined.shape[-1]), mode="bilinear", align_corners=False)), dim=1)

        delayed_all_combined = torch.cat((delayed_all_combined, predictions[-1][0]), dim=1)
        for i, prediction in enumerate(reversed(predictions[:-1]), start=1):
            delayed_all_combined = torch.cat((delayed_all_combined,
                                              F.interpolate(prediction[0], size=(delayed_all_combined.shape[-2], delayed_all_combined.shape[-1]), mode="bilinear", align_corners=False)), dim=1)
        
        
        delayed_pred = self.delayed_pred_layer_list[0]([delayed_all_combined, None])
        delayed_pred = [torch.cat((delayed_pred[0], delayed_all_combined[:,-2*self.num_encoders:,:,:]), dim=1), None]
        delayed_pred = self.delayed_pred_layer_list[1]([delayed_pred[0], None])

        # # decoder and multires predictions
        # predictions = []
        # for i, (decoder, pred) in enumerate(zip(self.decoders, self.preds)):
        #     x = self.skip_ftn(x, blocks[self.num_encoders - i - 1])
        #     if i > 0:
        #         x = self.skip_ftn(predictions[-1], x)
        #     x = decoder(x)
        #     predictions.append(pred(x))

        return predictions, delayed_pred

    def forward_refresh_reservoirs(self, x):

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x = encoder.forward_refresh_reservoirs(x)
            blocks.append(x)

        # residual blocks
        last_resblock_output = None
        for resblock in self.resblocks:
            x, inp_ = resblock.forward_refresh_reservoirs(x)
            last_resblock_output = x

        # decoder with delayed concatenation
        predictions = []
        decoder_out_list = []
        for i, (decoder, pred) in enumerate(zip(self.decoders, self.preds)):
            if i == 0:
                decoder_in = self.skip_ftn(last_resblock_output, blocks[self.num_encoders - i - 1])
            else:
                decoder_in = blocks[self.num_encoders - i - 1]
            decoder_out = decoder.forward_refresh_reservoirs(decoder_in)
            decoder_out_list.append(decoder_out)
            predictions.append(pred.forward_refresh_reservoirs(decoder_out))

        delayed_all_combined = decoder_out_list[-1]
        for i, decoder_out in enumerate(reversed(decoder_out_list[:-1]), start=1):
            scale_factor = 2 ** i
            delayed_all_combined = torch.cat((delayed_all_combined,
                                              F.interpolate(decoder_out, size=(delayed_all_combined.shape[-2], delayed_all_combined.shape[-1]), mode="bilinear", align_corners=False)), dim=1)

        delayed_all_combined = torch.cat((delayed_all_combined, predictions[-1]), dim=1)
        for i, prediction in enumerate(reversed(predictions[:-1]), start=1):
            scale_factor = 2 ** i
            delayed_all_combined = torch.cat((delayed_all_combined,
                                              F.interpolate(prediction, size=(delayed_all_combined.shape[-2], delayed_all_combined.shape[-1]), mode="bilinear", align_corners=False)), dim=1)

        delayed_pred = self.delayed_pred_layer_list[1].forward_refresh_reservoirs(torch.cat((self.delayed_pred_layer_list[0].forward_refresh_reservoirs(delayed_all_combined), delayed_all_combined[:,-2*self.num_encoders:,:,:]), dim=1))

        return predictions, delayed_pred


class DelayedEVFlowNetIncr(nn.Module):
    """
    FireFlowNet architecture for (dense/sparse) optical flow estimation from event-data.
    "Back to Event Basics: Self Supervised Learning of Image Reconstruction from Event Data via Photometric Constancy", Paredes-Valles et al., 2020
    """

    def __init__(self, unet_kwargs, num_bins):
        super().__init__()
        self.crop = None
        self.mask = unet_kwargs["mask_output"]
        EVFlowNet_kwargs = {
            "base_num_channels": unet_kwargs["base_num_channels"],
            "num_encoders": 4,
            "num_residual_blocks": 2,
            "num_output_channels": 2,
            "skip_type": "concat",
            "norm": None,
            "num_bins": num_bins,
            "use_upsample_conv": True,
            "kernel_size": unet_kwargs["kernel_size"],
            "channel_multiplier": 2,
            "final_activation": "tanh",
        }
        self.num_encoders = EVFlowNet_kwargs["num_encoders"]
        unet_kwargs.update(EVFlowNet_kwargs)
        unet_kwargs.pop("name", None)
        unet_kwargs.pop("eval", None)
        unet_kwargs.pop("encoding", None)  # TODO: remove
        unet_kwargs.pop("mask_output", None)
        unet_kwargs.pop("mask_smoothing", None)  # TODO: remove
        if "flow_scaling" in unet_kwargs.keys():
            unet_kwargs.pop("flow_scaling", None)

        self.multires_unet = DelayedMultiResUNetIncr(unet_kwargs)

    def reset_states(self):
        pass

    def init_cropping(self, width, height, safety_margin=0):
        self.crop = CropParameters(width, height, self.num_encoders, safety_margin)

    def forward(self, x_incr):

        inp_voxel = x_incr

        # forward pass
        x_incr = inp_voxel

        # pad input
        if self.crop is not None:
            x_incr = self.crop.pad(x_incr)

        # forward pass
        multires_flow = self.multires_unet.forward(x_incr)
        if isinstance(multires_flow, tuple):
            assert len(multires_flow) == 2
            delayed_flow = multires_flow[1]
            multires_flow = multires_flow[0]
        else:
            delayed_flow = None
        
        # upsample flow estimates to the original input resolution
        flow_list = []
        for flow in multires_flow:
            flow_list.append(
                torch.nn.functional.interpolate(
                    flow[0],
                    scale_factor=(
                        multires_flow[-1][0].shape[2] / flow[0].shape[2],
                        multires_flow[-1][0].shape[3] / flow[0].shape[3],
                    ),
                )
            )
        if delayed_flow is not None:
            flow_list.append(delayed_flow[0])

        # crop output
        if self.crop is not None:
            for i, flow in enumerate(flow_list):
                flow_list[i] = flow[:, :, self.crop.iy0 : self.crop.iy1, self.crop.ix0 : self.crop.ix1]
                flow_list[i] = flow_list[i].contiguous()

        # mask flow
        if self.mask and False:
            mask = torch.sum(inp_cnt, dim=1, keepdim=True)
            mask[mask > 0] = 1
            for i, flow in enumerate(flow_list):
                flow_list[i] = flow * mask

        return {"flow": flow_list}


    def forward_refresh_reservoirs(self, x):
        """
        :param inp_voxel: N x num_bins x H x W
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        """

        # pad input
        inp_voxel = x
        if self.crop is not None:
            x = self.crop.pad(x)

        # forward pass
        multires_flow = self.multires_unet.forward_refresh_reservoirs(x)
        if isinstance(multires_flow, tuple):
            assert len(multires_flow) == 2
            delayed_flow = multires_flow[1]
            multires_flow = multires_flow[0]
        else:
            delayed_flow = None

        # upsample flow estimates to the original input resolution
        flow_list = []
        for flow in multires_flow:
            flow_list.append(
                torch.nn.functional.interpolate(
                    flow,
                    scale_factor=(
                        multires_flow[-1].shape[2] / flow.shape[2],
                        multires_flow[-1].shape[3] / flow.shape[3],
                    ),
                )
            )
        if delayed_flow is not None:
            flow_list.append(delayed_flow)

        # crop output
        if self.crop is not None:
            for i, flow in enumerate(flow_list):
                flow_list[i] = flow[:, :, self.crop.iy0 : self.crop.iy1, self.crop.ix0 : self.crop.ix1]
                flow_list[i] = flow_list[i].contiguous()

        # mask flow
        if self.mask and False:
            mask = torch.sum(inp_cnt, dim=1, keepdim=True)
            mask[mask > 0] = 1
            for i, flow in enumerate(flow_list):
                flow_list[i] = flow * mask

        return {"flow": flow_list}

