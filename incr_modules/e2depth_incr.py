import torch
import torch.nn as nn
import torch.nn.functional as F
from ev_projs.rpg_e2depth.model.model import BaseE2VID
from .masked_types import Masked, DenseT, SpOrDense
import ext.pointops.pointops_functional as pf

from typing import Tuple, overload
from .mask_incr_modules import IncrementMaskModule, print_sparsity, NonlinearPointOpIncr, IncrementReserve, conv2d_from_module,bn2d_from_module,interpolate_from_module, KFencedMaskModule, PointwiseMultiplyIncr



# does not include a sparse version
class ConvLayerIncr(IncrementMaskModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super().__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv2d_weights = pf.convert_filter_out_channels_last(self.conv2d.weight).cuda()

        # self.sparseconv2d = SparseNet(self.conv2d)
        self.kf = KFencedMaskModule()
        if activation is not None:
            op = getattr(torch, activation, 'relu')
            self.activation_in_res = IncrementReserve()
            self.activation = NonlinearPointOpIncr(self.activation_in_res, op=op)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)


    # fully connected implementation
    def forward(self, x_incr: Masked) -> Masked:
        out_incr = conv2d_from_module(x_incr, self.conv2d_weights, conv_bias=None, stride=self.conv2d.stride, padding=self.conv2d.padding, forward_refresh=False)
        out_incr = self.kf(out_incr)
        if self.norm == 'BN':
            out_incr = bn2d_from_module(out_incr, self.norm_layer, bias=False)
        elif self.norm == 'IN':
            raise NotImplementedError # TODO: handle instance norm

        if self.activation is not None:
            out_incr_act = self.activation(out_incr)
            self.activation_in_res.accumulate(out_incr)
            out_incr = out_incr_act

        return out_incr

    def forward_refresh_reservoirs(self, x: DenseT):
        out = conv2d_from_module(x, self.conv2d_weights, conv_bias=self.conv2d.bias, stride=self.conv2d.stride, padding=self.conv2d.padding, forward_refresh=True)
        out = self.kf.forward_refresh_reservoirs(out)
        if self.norm == 'BN':
            out = self.norm_layer(out)
        elif self.norm == 'IN':
            raise NotImplementedError # TODO: instance Norm

        if self.activation is not None:
            out_act = self.activation.forward_refresh_reservoirs(out)
            self.activation_in_res.update_reservoir(out)
            out = out_act

        return out

# multiple inheritance model!
class ConvLSTMIncr(IncrementMaskModule):

    def __init__(self, input_size: int, hidden_size: int, kernel_size: int, prev_c_res: IncrementReserve = None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        pad = kernel_size // 2
        
        self.zero_tensors = {}
        self.zero_tensors_incr = {}

        # convfilter:
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)
        self.Gates_weights = pf.convert_filter_out_channels_last(self.Gates.weight).cuda()

        self.kf = KFencedMaskModule()
        #nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)


        # define reserves:
        self.prev_c_res = IncrementReserve() if prev_c_res is None else prev_c_res
        self.rm_res = IncrementReserve()
        self.rm_s_res = IncrementReserve()
        self.in_res = IncrementReserve()
        self.in_s_res = IncrementReserve()
        self.cg_res = IncrementReserve()
        self.cg_th_res = IncrementReserve()
        self.o_s_res = IncrementReserve()
        self.o_res = IncrementReserve()
        self.c_res = IncrementReserve()
        self.c_th_res = IncrementReserve()

        self.prev_c_rm_s_mult = PointwiseMultiplyIncr(self.prev_c_res, self.rm_s_res)
        self.in_s_cg_th_mult = PointwiseMultiplyIncr(self.in_s_res, self.cg_th_res)
        self.o_s_c_th_rm_mult = PointwiseMultiplyIncr(self.o_s_res, self.c_th_res)

        self.rm_sigmoid = NonlinearPointOpIncr(self.rm_res, torch.sigmoid)
        self.in_sigmoid = NonlinearPointOpIncr(self.in_res, torch.sigmoid)
        self.o_sigmoid = NonlinearPointOpIncr(self.o_res, torch.sigmoid)

        self.cg_tanh = NonlinearPointOpIncr(self.cg_res, torch.tanh)
        self.c_tanh = NonlinearPointOpIncr(self.c_res, torch.tanh)


    def forward(self, input_incr: Masked, prev_state_incr: Masked):

        # get batch and spatial sizes
        batch_size = input_incr.data.size()[0]
        spatial_size = input_incr.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state_incr is None:
            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors_incr:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors_incr[state_size] = (
                    torch.zeros(state_size, device=input_incr.device).to(memory_format=torch.channels_last),
                    torch.zeros(state_size, device=input_incr.device).to(memory_format=torch.channels_last)
                )

            prev_state_incr = self.zero_tensors_incr[state_size]

        prev_h_incr, prev_c_incr = prev_state_incr

        stacked_inputs_incr = torch.cat((input_incr, prev_h_incr), 1)
        gates_incr = conv2d_from_module(stacked_inputs_incr, self.Gates_weights, conv_bias=None, stride=self.Gates.stride, padding=self.Gates.padding)

        gates_incr = self.kf(gates_incr)

        in_incr, rm_incr, o_incr, cg_incr = gates_incr.chunk(4, 1)

        # set of pointwise nonlinear operations: output is guaranteed to be sparse
        rm_s_incr = self.rm_sigmoid(rm_incr)
        in_s_incr = self.in_sigmoid(in_incr)
        cg_th_incr = self.cg_tanh(cg_incr)
        o_s_incr = self.o_sigmoid(o_incr)  

        # this output is not guaranteed to be sparse :(
        c_incr = self.in_s_cg_th_mult(in_s_incr, cg_th_incr) + self.prev_c_rm_s_mult(rm_s_incr, prev_c_incr)
#        cell_incr = (remember_gate_incr * prev_cell_incr) + (in_gate_incr * cell_gate_incr)

        # nonlinear operation: incrmenet as well as a 
        c_th_incr = self.c_tanh(c_incr)
        h_incr = self.o_s_c_th_rm_mult(o_s_incr, c_th_incr)    
#        hidden_incr = out_gate_incr * self.cell_incr_tanh(cell_incr)

        # accumulate
        self.prev_c_res.accumulate(prev_c_incr)
        self.rm_res.accumulate(rm_incr)
        self.rm_s_res.accumulate(rm_s_incr)
        self.in_res.accumulate(in_incr)
        self.in_s_res.accumulate(in_s_incr)
        self.cg_res.accumulate(cg_incr)
        self.cg_th_res.accumulate(cg_th_incr)
        self.o_s_res.accumulate(o_s_incr)
        self.o_res.accumulate(o_incr)
        self.c_res.accumulate(c_incr)
        self.c_th_res.accumulate(c_th_incr)

        return h_incr, c_incr


    # update all reservoirs in the meanwhile
    # call forward_refresh_reservoirs for each sub-increment module
    # same as before but now it's forward_refresh instead, and update_accumulate instead
    def forward_refresh_reservoirs(self, input_, prev_state):
        # get batch and spatial sizes

        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        if prev_state is None:

            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors: # TODO: Need to deal with sparsifying this later.
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors[state_size] = (
                    torch.zeros(state_size, device=input_.device).to(memory_format=torch.channels_last),
                    torch.zeros(state_size, device=input_.device).to(memory_format=torch.channels_last)
                )

            prev_state = self.zero_tensors[tuple(state_size)]


        prev_c, prev_h = prev_state
        
        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_h), 1)
        gates = conv2d_from_module(stacked_inputs, self.Gates_weights, conv_bias=self.Gates.bias, stride=self.Gates.stride, padding=self.Gates.padding, forward_refresh=True)

        gates = self.kf.forward_refresh_reservoirs(gates)
        # chunk across channel dimension
        ing, rm, o, cg = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_s = self.in_sigmoid.forward_refresh_reservoirs(ing)
        rm_s = self.rm_sigmoid.forward_refresh_reservoirs(rm)
        o_s = self.o_sigmoid.forward_refresh_reservoirs(o)

        # apply tanh non linearity
        cg_th = self.cg_tanh.forward_refresh_reservoirs(cg)

        c = self.in_s_cg_th_mult.forward_refresh_reservoirs(in_s, cg_th) +\
             self.prev_c_rm_s_mult.forward_refresh_reservoirs(rm_s, prev_c)

        # compute current cell and hidden state
        c_th = self.c_tanh.forward_refresh_reservoirs(c)
        h = self.o_s_c_th_rm_mult.forward_refresh_reservoirs(o_s, c_th)

        self.prev_c_res.update_reservoir(prev_c)
        self.rm_res.update_reservoir(rm)
        self.rm_s_res.update_reservoir(rm_s)
        self.in_res.update_reservoir(ing)
        self.in_s_res.update_reservoir(in_s)
        self.cg_res.update_reservoir(cg)
        self.cg_th_res.update_reservoir(cg_th)
        self.o_s_res.update_reservoir(o_s)
        self.o_res.update_reservoir(o)
        self.c_res.update_reservoir(c)
        self.c_th_res.update_reservoir(c_th)
        return h,c


class RecurrentConvLayerIncr(IncrementMaskModule):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
    recurrent_block_type='convlstm', activation='relu', norm=None):
        super().__init__()

        self.conv = ConvLayerIncr(in_channels, out_channels, kernel_size, stride, padding, activation, norm)

        # TODO: only suppoerts ConvLSTM, not ConvGRU for now
        self.recurrent_block = ConvLSTMIncr(input_size=out_channels, hidden_size=out_channels, kernel_size=3)

    def forward(self, x_incr, prev_state_incr):
        x_incr = self.conv(x_incr)
        state_incr = self.recurrent_block(x_incr, prev_state_incr) 
        x_incr = state_incr[0]
        return x_incr, state_incr

    def forward_refresh_reservoirs(self, x, prev_state):
        x = self.conv.forward_refresh_reservoirs(x)
        state = self.recurrent_block.forward_refresh_reservoirs(x, prev_state)
        x = state[0]
        return x, state

# took out the instance norm terms: for default clases don't change init. that's how you preserve load_state_dict
class ResidualBlockIncr(IncrementMaskModule):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm=None):
        super().__init__()
        bias = False if norm == 'BN' else True
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.conv1_weights = pf.convert_filter_out_channels_last(self.conv1.weight).cuda()

        self.kf1 = KFencedMaskModule()

        self.norm = norm
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.bn2 = nn.InstanceNorm2d(out_channels)

#        self.relu = nn.ReLU(inplace=True)

        self.c1_in_res = IncrementReserve()
        self.relu1 = NonlinearPointOpIncr(res_in=self.c1_in_res, op=torch.relu)

        self.c2_in_res = IncrementReserve()
        self.relu2 = NonlinearPointOpIncr(res_in=self.c2_in_res, op=torch.relu)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv2_weights = pf.convert_filter_out_channels_last(self.conv2.weight).cuda()

        self.kf2 = KFencedMaskModule()

        self.downsample = downsample

    def forward(self, x_incr: Masked) -> Masked:
        residual_incr = x_incr
        out_incr = conv2d_from_module(x_incr, self.conv1_weights, conv_bias=None, stride=self.conv1.stride, padding=self.conv1.padding)
        out_incr = self.kf1(out_incr)

        if self.norm  == 'BN':
            out_incr = bn2d_from_module(out_incr, self.bn1, bias=False)
        elif self.norm == 'IN':
            raise NotImplementedError # TODO: InstanceNorm

        out_incr_relu = self.relu1(out_incr)
        self.c1_in_res.accumulate(out_incr)

        out_incr = conv2d_from_module(out_incr_relu, self.conv2_weights, conv_bias=None, stride=self.conv2.stride, padding=self.conv2.padding)
        out_incr = self.kf2(out_incr)
        if self.norm =='BN':
            out_incr = bn2d_from_module(out_incr, self.bn2, bias=False)
        elif self.norm == 'IN':
            raise NotImplementedError # TODO: InstanceNorm


        if self.downsample:
            residual_incr = self.downsample(x_incr) # TODO: check if this is linear

        out_incr += residual_incr

        out_incr_relu = self.relu2(out_incr)
        self.c2_in_res.accumulate(out_incr)

        return out_incr_relu

    def forward_refresh_reservoirs(self, x):
        residual = x
        out = conv2d_from_module(x, self.conv1_weights, conv_bias=self.conv1.bias, stride=self.conv1.stride, padding=self.conv1.padding, forward_refresh=True)
        out = self.kf1.forward_refresh_reservoirs(out)
        if self.norm == 'BN':
            out = F.batch_norm(out, running_mean=self.bn1.running_mean, running_var=self.bn1.running_var, weight=self.bn1.weight, bias=self.bn2.bias, training=self.bn1.training, momentum=self.bn1.momentum, eps=self.bn1.eps)
        elif self.norm == 'IN':
            raise NotImplementedError # TODO: InstanceNorm

        out_relu = self.relu1.forward_refresh_reservoirs(out)
        self.c1_in_res.update_reservoir(out)

        out = conv2d_from_module(x, self.conv2_weights, conv_bias=self.conv2.bias, stride=self.conv2.stride, padding=self.conv2.padding, forward_refresh=True)
        out = self.kf2.forward_refresh_reservoirs(out)
        if self.norm == 'BN':
            out = F.batch_norm(out, running_mean=self.bn2.running_mean, running_var=self.bn2.running_var, weight=self.bn2.weight, bias=self.bn2.bias, training=self.bn2.training, momentum=self.bn2.momentum, eps=self.bn2.eps)
        elif self.norm == 'IN':
            raise NotImplementedError # TODO: InstanceNorm

        if self.downsample:
            residual = self.downsample(x)

        out += residual

        out_relu = self.relu2.forward_refresh_reservoirs(out)
        self.c2_in_res.update_reservoir(out)

        return out_relu



class TransposedConvLayerIncr(IncrementMaskModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super().__init__()

        bias = False if norm == 'BN' else True
        self.transposed_conv2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)

        if activation is not None:
            op = getattr(torch, activation, 'relu')
            self.activation_in_res = IncrementReserve()
            self.activation = NonlinearPointOpIncr(self.activation_in_res, op=op)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)


    def forward(self, x_incr: Masked) -> Masked:
        out_incr = transposed_conv2d_from_module(x_incr, self.transposed_conv2d, bias=False) 

        if self.norm =='BN':
            out_incr = bn2d_from_module(out_incr, self.norm_layer, bias=False)
        elif self.norm == 'IN':
            raise NotImplementedError # TODO: Instance norm

        if self.activation is not None:
            out_act_incr = self.activation(out_incr)
            self.activation_in_res.accumulate(out_incr)
            out_incr = out_act_incr
        return out_incr

    def forward_refresh_reservoirs(self, x):
        out = self.transposed_conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out_act = self.activation.forward_refresh_reservoirs(out)
            self.activation_in_res.update_reservoir(out)
            out = out_act
            
        return out

class UpsampleConvLayerIncr(IncrementMaskModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super().__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv2d_weights = pf.convert_filter_out_channels_last(self.conv2d.weight).cuda()
        self.kf = KFencedMaskModule()
        if activation is not None:
            op = getattr(torch, activation, 'relu')
            self.act_in_res = IncrementReserve()
            self.activation = NonlinearPointOpIncr(res_in=self.act_in_res, op=op)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x_incr: SpOrDense) -> SpOrDense:

        x_upsampled_incr = interpolate_from_module(x_incr)
        out_incr = conv2d_from_module(x_upsampled_incr, self.conv2d_weights, conv_bias=None, stride=self.conv2d.stride, padding=self.conv2d.padding)

        out_incr = self.kf(out_incr)

        if self.norm =='BN':
            out_incr = bn2d_from_module(out_incr, self.norm_layer, bias=False)
        elif self.norm == 'IN':
            raise NotImplementedError # TODO: Instance norm

        if self.activation is not None:
            out_incr_act = self.activation(out_incr)
            self.act_in_res.accumulate(out_incr)
            out_incr = out_incr_act
        
        return out_incr

    def forward_refresh_reservoirs(self, x):

        x_upsampled = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = conv2d_from_module(x_upsampled, self.conv2d_weights, conv_bias=self.conv2d.bias, stride=self.conv2d.stride, padding=self.conv2d.padding, forward_refresh=True)
        out = self.kf.forward_refresh_reservoirs(out)
        if self.norm =='BN':
            out = bn2d_from_module(out, self.norm_layer, bias=True)
        elif self.norm == 'IN':
            raise NotImplementedError # TODO: Implement instancenorm

        if self.activation is not None:
            out_act = self.activation.forward_refresh_reservoirs(out)
            self.act_in_res.update_reservoir(out)
            out = out_act
        return out


class BaseUNetIncr(IncrementMaskModule):
    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum', activation='sigmoid',
                 num_encoders=4, base_num_channels=32, num_residual_blocks=2, norm=None, use_upsample_conv=True):
        super().__init__()

        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.skip_type = skip_type
        self.apply_skip_connection = lambda x1,x2: x1+x2
        self.norm = norm

        if use_upsample_conv:
            print('Using UpsampleConvLayer (slow, but no checkerboard artefacts)')
            self.UpsampleLayer = UpsampleConvLayerIncr
        else:
            print('Using TransposedConvLayer (fast, with checkerboard artefacts)')
            self.UpsampleLayer = TransposedConvLayerIncr

        self.num_encoders = num_encoders
        self.base_num_channels = base_num_channels
        self.num_residual_blocks = num_residual_blocks
        self.max_num_channels = self.base_num_channels * pow(2, self.num_encoders)

        assert(self.num_input_channels > 0)
        assert(self.num_output_channels > 0)

        self.encoder_input_sizes = []
        for i in range(self.num_encoders):
            self.encoder_input_sizes.append(self.base_num_channels * pow(2, i))

        self.encoder_output_sizes = [self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]

        self.act_baseunet_in_res = IncrementReserve()
        op = getattr(torch, activation, 'sigmoid')
        self.activation = NonlinearPointOpIncr(self.act_baseunet_in_res, op)

    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlockIncr(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_decoders(self):
        decoder_input_sizes = list(reversed([self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]))

        self.decoders = nn.ModuleList()
        for input_size in decoder_input_sizes:
            self.decoders.append(self.UpsampleLayer(input_size if self.skip_type == 'sum' else 2 * input_size,
                                                    input_size // 2,
                                                    kernel_size=5, padding=2, norm=self.norm))

    def build_prediction_layer(self):
        self.pred = ConvLayerIncr(self.base_num_channels if self.skip_type == 'sum' else 2 * self.base_num_channels,
                              self.num_output_channels, 1, activation=None, norm=self.norm)


class UNetRecurrentIncr(BaseUNetIncr):
    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum',
                 recurrent_block_type='convlstm', activation='sigmoid', num_encoders=4, base_num_channels=32,
                 num_residual_blocks=2, norm=None, use_upsample_conv=True):
        super().__init__(num_input_channels, num_output_channels, skip_type, activation,
                                            num_encoders, base_num_channels, num_residual_blocks, norm,
                                            use_upsample_conv)

        self.head = ConvLayerIncr(self.num_input_channels, self.base_num_channels,
                              kernel_size=5, stride=1, padding=2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(RecurrentConvLayerIncr(input_size, output_size,
                                                    kernel_size=5, stride=2, padding=2,
                                                    recurrent_block_type=recurrent_block_type,
                                                    norm=self.norm))

        self.build_resblocks()
        self.build_decoders()
        self.build_prediction_layer()


    def forward(self, x_incr: Masked, prev_states_incr: Masked):

        print_sparsity(x_incr, "before convhead")

        x_incr = self.head(x_incr)

        head = x_incr
        
        print_sparsity(x_incr, "after convhead")

        if prev_states_incr is None:
            prev_states_incr = [None] * self.num_encoders

        return x_incr
        # encoder
        blocks = []
        states_incr = []
        for i, encoder in enumerate(self.encoders):
            x_incr, state_incr = encoder(x_incr, prev_states_incr[i])
            blocks.append(x_incr)
            states_incr.append(state_incr)
            print_sparsity(x_incr, "after encoder{}".format(i) )


        # residual blocks
        for i,resblock in enumerate(self.resblocks):
            x_incr = resblock(x_incr)
            print_sparsity(x_incr, "after resblock{}".format(i) )

        # decoder
        for i, decoder in enumerate(self.decoders):
            x_incr = decoder(self.apply_skip_connection(x_incr, blocks[self.num_encoders - i - 1]))
            print_sparsity(x_incr, "after decoder{}".format(i) )

        # tail
        pred_incr = self.pred(self.apply_skip_connection(x_incr, head))
        img_incr = self.activation(pred_incr)
        self.act_baseunet_in_res.accumulate(pred_incr)

        print_sparsity(x_incr, "final")

        return img_incr, states_incr


    def forward_refresh_reservoirs(self, x, prev_states):
        x = self.head.forward_refresh_reservoirs(x)
        head = x
        if prev_states is None:
            prev_states = [None] * self.num_encoders

        # encoder
        blocks = []
        states = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder.forward_refresh_reservoirs(x, prev_states[i])
            blocks.append(x)
            states.append(state)

        # residual blocks
        for resblock in self.resblocks:
            x = resblock.forward_refresh_reservoirs(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder.forward_refresh_reservoirs(self.apply_skip_connection(x, blocks[self.num_encoders - i - 1]))

        # tail
        pred = self.pred.forward_refresh_reservoirs(self.apply_skip_connection(x, head))
        img = self.activation.forward_refresh_reservoirs(pred)
        self.act_baseunet_in_res.update_reservoir(pred)

        return img, states


class E2VIDRecurrentIncr(BaseE2VID, IncrementMaskModule):

    def __init__(self, config):
        super().__init__(config)

        try:
            self.recurrent_block_type = str(config['recurrent_block_type'])
        except KeyError:
            self.recurrent_block_type = 'convlstm'  # or 'convgru'

        self.unetrecurrent = UNetRecurrentIncr(num_input_channels=self.num_bins,
                                           num_output_channels=1,
                                           skip_type=self.skip_type,
                                           recurrent_block_type=self.recurrent_block_type,
                                           activation='sigmoid',
                                           num_encoders=self.num_encoders,
                                           base_num_channels=self.base_num_channels,
                                           num_residual_blocks=self.num_residual_blocks,
                                           norm=self.norm,
                                           use_upsample_conv=self.use_upsample_conv)

    def forward(self, input_sample: Tuple[Masked]):
        event_tensor, prev_states = input_sample
        img_pred, states = self.unetrecurrent.forward(event_tensor, prev_states)
        return img_pred, states

    def forward_refresh_reservoirs(self, input_sample):
        event_tensor, prev_states = input_sample
        img_pred, states = self.unetrecurrent.forward_refresh_reservoirs(event_tensor, prev_states)
        return img_pred, states


