import torch
import torch.nn as nn
import torch.nn.functional as F
from incr_modules.mask_incr_functional import IncrPointwiseMultiply

from incr_modules.mask_incr_modules import nnBatchNorm2dIncr, nnConvIncr, nnReservedActivation, nnReservedMultiplication, nnSigmoidIncr, nnTanhIncr



def skip_concat(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    padding = nn.ZeroPad2d((diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
    x1 = padding(x1)
    return torch.cat([x1, x2], dim=1)



class ConvGRUIncr(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size, activation=None):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nnConvIncr(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nnConvIncr(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nnConvIncr(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        assert activation is None, "ConvGRU activation cannot be set (just for compatibility)"

        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)
        nn.init.constant_(self.reset_gate.bias, 0.0)
        nn.init.constant_(self.update_gate.bias, 0.0)
        nn.init.constant_(self.out_gate.bias, 0.0)

        self.sigmoid1 = nnSigmoidIncr()
        self.sigmoid2 = nnSigmoidIncr()
        self.tanh     = nnTanhIncr()

        self.mult = nnReservedMultiplication()

    def forward(self, input_incr, prev_state):

        # get batch and spatial sizes
        batch_size = input_incr[0].data.size()[0]
        spatial_size = input_incr[0].data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = [torch.zeros(state_size, dtype=input_incr[0].dtype, device=input_incr[0].device).to(memory_format=torch.channels_last), None]

        # data size is [batch, channel, height, width]
        stacked_inputs = [torch.cat([input_incr[0], prev_state[0]], dim=1), None]


        update = self.sigmoid1(self.update_gate(stacked_inputs))
        # reset = self.sigmoid2(self.reset_gate(stacked_inputs))

        out_inputs = self.tanh(self.out_gate(stacked_inputs))

        # new_state = prev_state * (1 - update) + out_inputs * update
        # new_state = prev_state + (out_inputs-prev_state) * update

        new_state = [prev_state[0] + self.mult([out_inputs[0]-prev_state[0], None], update)[0], None]


        return new_state, new_state


    def forward_refresh_reservoirs(self, x, prev_state):
        # get batch and spatial sizes
        batch_size = x.data.size()[0]
        spatial_size = x.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size, dtype=x.dtype, device=x.device).to(memory_format=torch.channels_last)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([x, prev_state], dim=1)

        update = self.sigmoid1.forward_refresh_reservoirs(self.update_gate.forward_refresh_reservoirs(stacked_inputs))
        reset = self.sigmoid2.forward_refresh_reservoirs(self.reset_gate.forward_refresh_reservoirs(stacked_inputs))
        out_inputs = self.tanh.forward_refresh_reservoirs(self.out_gate.forward_refresh_reservoirs(torch.cat([x, prev_state * reset], dim=1)))
        

        
        # new_state = prev_state * (1 - update) + out_inputs * update

        new_state = prev_state + self.mult.forward_refresh_reservoirs(out_inputs-prev_state, update)


        return new_state, new_state        


class ConvLayerIncr(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        activation="relu",
        norm=None,
        BN_momentum=0.1,
        w_scale=None,
    ):
        super(ConvLayerIncr, self).__init__()

        bias = False if norm == "BN" else True
        padding = kernel_size // 2
        self.conv2d = nnConvIncr(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if w_scale is not None:
            nn.init.uniform_(self.conv2d.weight, -w_scale, w_scale)
            nn.init.zeros_(self.conv2d.bias)

        if activation is not None:
            if hasattr(torch, activation):
                activation = getattr(torch, activation)
                self.activation = nnReservedActivation(activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nnBatchNorm2dIncr(out_channels, momentum=BN_momentum)
        elif norm == "IN":
            raise NotImplementedError

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out

    def forward_refresh_reservoirs(self, x):
        out = self.conv2d.forward_refresh_reservoirs(x)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer.forward_refresh_reservoirs(out)

        if self.activation is not None:
            out = self.activation.forward_refresh_reservoirs(out)

        return out



class RecurrentConvLayerIncr(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        activation_ff="relu",
        activation_rec=None,
        norm=None,
        BN_momentum=0.1,
    ):
        super().__init__()

        self.conv = ConvLayerIncr(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            activation_ff,
            norm,
            BN_momentum=BN_momentum,
        )
        self.recurrent_block = ConvGRUIncr(
            input_size=out_channels, hidden_size=out_channels, kernel_size=3, activation=activation_rec
        )

    def forward(self, x, prev_state):
        x = self.conv(x)
        x, state = self.recurrent_block(x, prev_state)
        return x, state

    def forward_refresh_reservoirs(self, x, prev_state):
        x = self.conv.forward_refresh_reservoirs(x)
        x, state = self.recurrent_block.forward_refresh_reservoirs(x, prev_state)
        return x, state


class ResidualBlockIncr(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        activation="relu",
        downsample=None,
        norm=None,
        BN_momentum=0.1
    ):
        super(ResidualBlockIncr, self).__init__()
        bias = False if norm == "BN" else True
        self.conv1 = nnConvIncr(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
        )

        if activation is not None:
            if hasattr(torch, activation):
                activation = getattr(torch, activation)
                self.activation = nnReservedActivation(activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.bn1 = nnBatchNorm2dIncr(out_channels, momentum=BN_momentum)
            self.bn2 = nnBatchNorm2dIncr(out_channels, momentum=BN_momentum)
        elif norm == "IN":
            raise NotImplementedError

        self.conv2 = nnConvIncr(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out1 = self.conv1(x)
        if self.norm in ["BN", "IN"]:
            out1 = self.bn1(out1)

        if self.activation is not None:
            out1 = self.activation(out1)

        out2 = self.conv2(out1)
        if self.norm in ["BN", "IN"]:
            out2 = self.bn2(out2)

        if self.downsample:
            residual = self.downsample(x)

        out2 += residual
        if self.activation is not None:
            out2 = self.activation(out2)

        return out2, out1



    def forward_refresh_reservoirs(self, x):
        residual = x
        out1 = self.conv1.forward_refresh_reservoirs(x)
        if self.norm in ["BN", "IN"]:
            out1 = self.bn1.forward_refresh_reservoirs(out1)

        if self.activation is not None:
            out1 = self.activation.forward_refresh_reservoirs(out1)

        out2 = self.conv2.forward_refresh_reservoirs(out1)
        if self.norm in ["BN", "IN"]:
            out2 = self.bn2.forward_refresh_reservoirs(out2)

        if self.downsample:
            residual = self.downsample.forward_refresh_reservoirs(x)

        out2 += residual
        if self.activation is not None:
            out2 = self.activation.forward_refresh_reservoirs(out2)

        return out2, out1




class UpsampleConvLayerIncr(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        activation="relu",
        norm=None,
    ):
        super(UpsampleConvLayerIncr, self).__init__()

        bias = False if norm == "BN" else True
        padding = kernel_size // 2
        self.conv2d = nnConvIncr(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if activation is not None:
            activation = getattr(torch, activation)
            self.activation = nnReservedActivation(activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nnBatchNorm2dIncr(out_channels)
        elif norm == "IN":
            raise NotImplementedError


    def forward(self, x):
        x_upsampled = [F.interpolate(x[0], scale_factor=2, mode="bilinear", align_corners=False), None]
        out = self.conv2d(x_upsampled)
        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)
        if self.activation is not None:
            out = self.activation(out)
        return out


    def forward_refresh_reservoirs(self, x):
        x_upsampled = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.conv2d.forward_refresh_reservoirs(x_upsampled)
        if self.norm in ["BN", "IN"]:
            out = self.norm_layer.forward_refresh_reservoirs(out)
        if self.activation is not None:
            out = self.activation.forward_refresh_reservoirs(out)
        return out


class MultiResUNetRecurrentIncr(nn.Module):

    def __init__(self):
        
        super().__init__()
        self.final_activation = 'tanh'
        self.w_scale_pred = None


        self.num_encoders = 4
        self.num_residual_blocks = 2
        self.ff_act, self.rec_act = ['relu', None]
        self.skip_ftn = skip_concat
        self.kernel_size = 3
        self.num_bins = 2
        self.norm = None

        self.base_num_channels = 32
        self.channel_multiplier = 2
        self.num_output_channels = 2

        self.encoder_input_sizes = [
            int(self.base_num_channels * pow(self.channel_multiplier, i)) for i in range(self.num_encoders)
        ]
        self.encoder_output_sizes = [
            int(self.base_num_channels * pow(self.channel_multiplier, i + 1)) for i in range(self.num_encoders)
        ]

        self.max_num_channels = self.encoder_output_sizes[-1]


        self.encoders = self.build_recurrent_encoders()
        self.resblocks = self.build_resblocks()
        self.decoders = self.build_multires_prediction_decoders()
        self.preds = self.build_multires_prediction_layer()

        self.num_states = self.num_encoders
        self.states = [None] * self.num_states

    def build_recurrent_encoders(self):
        encoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(self.encoder_input_sizes, self.encoder_output_sizes)):
            if i == 0:
                input_size = self.num_bins
            encoders.append(
                RecurrentConvLayerIncr(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=2,
                    activation_ff=self.ff_act,
                    activation_rec=self.rec_act,
                    norm=self.norm,
                )
            )
        return encoders


    def build_resblocks(self):
        resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            resblocks.append(
                ResidualBlockIncr(
                    self.max_num_channels,
                    self.max_num_channels,
                    activation=self.ff_act,
                    norm=self.norm,
                )
            )
        return resblocks

    def build_multires_prediction_layer(self):
        preds = nn.ModuleList()
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        for output_size in decoder_output_sizes:
            preds.append(
                ConvLayerIncr(
                    output_size,
                    self.num_output_channels,
                    1,
                    activation=self.final_activation,
                    norm=self.norm
                )
            )
        return preds

    def build_multires_prediction_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(decoder_input_sizes, decoder_output_sizes)):
            prediction_channels = 0 if i == 0 else self.num_output_channels
            decoders.append(
                UpsampleConvLayerIncr(
                    2 * input_size + prediction_channels,
                    output_size,
                    kernel_size=self.kernel_size,
                    activation=self.ff_act,
                    norm=self.norm,
                )
            )
        return decoders

    def forward(self, x_incr):


        states_incr = [None] * self.num_states
        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x_incr, state = encoder(x_incr, states_incr[i])
            blocks.append(x_incr)
            states_incr[i] = state

        # residual blocks
        for resblock in self.resblocks:
            x_incr, _ = resblock(x_incr)

        # decoder and multires predictions
        predictions = []
        for i, (decoder, pred) in enumerate(zip(self.decoders, self.preds)):
            x_incr = [self.skip_ftn(x_incr[0], blocks[self.num_encoders - i - 1][0]), None]
            if i > 0:
                x_incr = [self.skip_ftn(predictions[-1][0], x_incr[0]), None]
            x_incr = decoder(x_incr)
            predictions.append(pred(x_incr))

        return predictions


    def forward_refresh_reservoirs(self, x):
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder.forward_refresh_reservoirs(x, self.states[i])
            blocks.append(x)
            self.states[i] = state

        # residual blocks
        for resblock in self.resblocks:
            x, _ = resblock.forward_refresh_reservoirs(x)

        # decoder and multires predictions
        predictions = []
        for i, (decoder, pred) in enumerate(zip(self.decoders, self.preds)):
            x = self.skip_ftn(x, blocks[self.num_encoders - i - 1])
            if i > 0:
                x = self.skip_ftn(predictions[-1], x)
            x = decoder.forward_refresh_reservoirs(x)
            predictions.append(pred.forward_refresh_reservoirs(x))

        return predictions


