import torch
import torch.nn as nn
from ev_projs.rpg_event_representation_learning.utils.models import Classifier
from incr_modules.mask_incr_functional import IncrementReserve
from incr_modules.mask_incr_modules import NonlinearPointOpIncr, nnAdaptiveAvgPool2dIncr, nnBatchNorm2dIncr, nnConvIncr, nnLinearIncr, nnMaxPool2dIncr, nnSequentialIncr


class BasicBlockIncr(nn.Module):

    def __init__(self, sz, downsample=False):
        nn.Module.__init__(self)

        sz_in = sz/2 if downsample else sz

        self.conv1 = nnConvIncr(sz_in, sz, 3, stride=1, padding=1, bias=False)
        self.bn1 = nnBatchNorm2dIncr(sz, sz, 3, stride=1, padding=1, bias=False)

        self.res_in = IncrementReserve()
        self.relu = NonlinearPointOpIncr(self.res_in, op=torch.relu)

        self.conv2 = nnConvIncr(sz, sz, 3, stride=1, padding=1, bias=False)
        self.bn2 = nnBatchNorm2dIncr(sz, sz, 3, stride=1, padding=1, bias=False)

        
        self.downsample = nnSequentialIncr([
            nnConvIncr(sz_in, sz, 1, stride=2, bias=False),
            nnBatchNorm2dIncr(sz, eps=1e-5, momentum=0.1, affine=True)
        ]) if downsample else None


    def forward(self, x_incr):
        out = self.conv1(x_incr)
        in_activation = self.bn1(out)
        out = self.relu(in_activation)
        out = self.bn2(self.conv2(out))
        
        if self.downsample:
            out = self.downsample(out)

        self.res_in.accumulate(in_activation)
        return out

    def forward_refresh_reservoir(self, x_incr):
        out = self.conv1.forward_refresh_reservoirs(x_incr)
        in_activation = self.bn1.forward_refresh_reservoirs(out)
        out = self.relu.forward_refresh_reservoirs(in_activation)
        out = self.bn2.forward_refresh_reservoirs(self.conv2(out))
        
        if self.downsample:
            out = self.downsample.forward_refresh_reservoirs(out)

        self.res_in.update_reservoir(in_activation)
        return out



class ResnetIncr(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv1 = nnConvIncr(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nnBatchNorm2dIncr(64, eps=1e-5, momentum=0.1, affine=True)
        
        self.in_res = IncrementReserve()
        self.relu = NonlinearPointOpIncr(self.in_res, torch.relu)

        self.maxpool = nnMaxPool2dIncr(kernel_size=3, stride=2, padding=1)

        self.layers1 = nnSequentialIncr([BasicBlockIncr(64)]*3)
        self.layers2 = nnSequentialIncr([BasicBlockIncr(128, downsample=True)] + [BasicBlockIncr(128)]*3)
        self.layers3 = nnSequentialIncr([BasicBlockIncr(256, downsample=True)] + [BasicBlockIncr(256)]*5)
        self.layers4 = nnSequentialIncr([BasicBlockIncr(512, downsample=True)] + [BasicBlockIncr(512)]*2)

        self.avgpool = nnAdaptiveAvgPool2dIncr(output_size=(1, 1))
        self.fc = nnLinearIncr(512, 1000, bias=True)


    def forward(self, x_incr):

        out = self.conv1(x_incr)
        in_activation = self.bn1(out)
        out = self.relu(in_activation)

        out = self.maxpool(out)
        
        out = self.layers1(out)
        out = self.layers2(out)
        out = self.layers3(out)
        out = self.layers4(out)

        out = self.avgpool(out)
        out = self.fc(out)

        self.in_res.accumulate(in_activation)
        return out

    def forward_refresh_reservoirs(self, x):
        out = self.conv1.forward_refresh_reservoirs(x)
        in_activation = self.bn1.forward_refresh_reservoirs(out)
        out = self.relu.forward_refresh_reservoirs(in_activation)
        out = self.maxpool.forward_refresh_reservoirs(out)
        out = self.layers1.forward_refresh_reservoirs(out)
        out = self.layers2.forward_refresh_reservoirs(out)
        out = self.layers3.forward_refresh_reservoirs(out)
        out = self.layers4.forward_refresh_reservoirs(out)
        out = self.avgpool.forward_refresh_reservoirs(out)
        out = self.fc.forward_refresh_reservoirs(out)

        self.in_res.update_reservoir(in_activation)
        return out


class ClassifierIncrEval(Classifier):

    def __init__(self,
                 voxel_dimension=(9,180,240),  # dimension of voxel will be C x 2 x H x W
                 crop_dimension=(224, 224),  # dimension of crop before it goes into classifier
                 num_classes=101,
                 mlp_layers=[1, 30, 30, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1),
                 pretrained=True):
        Classifier.__init__(self, self,
                 voxel_dimension,  # dimension of voxel will be C x 2 x H x W
                 crop_dimension,  # dimension of crop before it goes into classifier
                 num_classes,
                 mlp_layers,
                 activation,
                 pretrained)


        self.classifier_incr = ResnetIncr()
        self.classifier_incr.load_state_dict(self.classifier.state_dict)

        input_channels = 2*voxel_dimension[0]
        self.classifier_incr.conv1 = nnConvIncr(input_channels, )
        self.classifier_incr.fc = nnLinearIncr(self.classifier.fc.in_features, num_classes)

 
    def forward(self, x):

        vox = self.quantization_layer.forward(x)
        vox_cropped = self.crop_and_resize_to_resolution(vox, self.crop_dimension)

        pred = self.classifier_incr(vox_cropped - prev)


        return pred, vox




    def forward_refresh_reservoir(self, x):
        vox = self.quantization_layer.forward(x)
        vox_cropped = self.crop_and_resize_to_resolution(vox, self.crop_dimension)
        pred = self.classifier_incr.forward_refresh_reservoirs(vox_cropped)
        return pred, vox




