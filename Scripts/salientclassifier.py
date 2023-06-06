import torch
import torch.nn as nn
from .sc_utils import MinimalBlock, BasicBlock, Bottleneck
from .sc_utils import conv1x1, get_model_params, salinet_params
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.functional import conv2d
import torch.nn.functional as F
import numpy as np
import clip

class LinearKernel(torch.nn.Module):
    def __init__(self):
        super(LinearKernel, self).__init__()
    
    def forward(self, x_unf, w, b):
        t = x_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
        if b is not None:
            return t + b
        return t
        
class PolynomialKernel(LinearKernel):
    def __init__(self, cp=0.5, dp=2, train_cp=True):
        super(PolynomialKernel, self).__init__()
        self.cp = torch.nn.parameter.Parameter(torch.tensor(cp, requires_grad=train_cp))
        self.dp = dp

    def forward(self, x_unf, w, b):
        return (self.cp + super(PolynomialKernel, self).forward(x_unf, w, b))**self.dp

class GaussianKernel(torch.nn.Module):
    def __init__(self, gamma=0.1):
        super(GaussianKernel, self).__init__()
        self.gamma = torch.nn.parameter.Parameter(
                            torch.tensor(gamma, requires_grad=True))
    
    def forward(self, x_unf, w, b):
        l = x_unf.transpose(1, 2)[:, :, :, None] - w.view(1, 1, -1, w.size(0))
        l = torch.sum(l**2, 2)
        t = torch.exp(-self.gamma * l)
        if b:
            return t + b
        return t

class KConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_fn=PolynomialKernel,
                 stride=1, padding=0, dilation=1, groups=1, bias=None,
                 padding_mode='zeros'):
        '''
        Follows the same API as torch Conv2d except kernel_fn.
        kernel_fn should be an instance of the above kernels.
        '''
        super(KConv2d, self).__init__(in_channels, out_channels, 
                                           kernel_size, stride, padding,
                                           dilation, groups, bias, padding_mode)
        self.kernel_fn = kernel_fn()
   
    def compute_shape(self, x):
        h = (x.shape[2] + 2 * self.padding[0] - 1 * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        w = (x.shape[3] + 2 * self.padding[1] - 1 * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        return h, w
    
    def forward(self, x):
        x_unf = torch.nn.functional.unfold(x, self.kernel_size, self.dilation,self.padding, self.stride)
        h, w = self.compute_shape(x)
        return self.kernel_fn(x_unf, self.weight, self.bias).view(x.shape[0], -1, h, w)

nn.KConv2d = KConv2d

class SaliNet(nn.Module):

    def __init__(self, layers=None, global_params=None):
        super(SaliNet, self).__init__()
        assert isinstance(layers, tuple), "blocks_args should be a tuple"
        assert len(layers) > 0, "layers must be greater than 0"

        if global_params.norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if global_params.replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = global_params.groups
        self.base_width = global_params.width_per_group
        self.conv1 = nn.KConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(global_params.block, 64, layers[0])
        self.layer2 = self._make_layer(global_params.block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(global_params.block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(global_params.block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * global_params.block.expansion, global_params.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if global_params.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        return x

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        layers, global_params = get_model_params(model_name, override_params)
        return cls(layers, global_params)

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, res = resnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name"""
        num_models = ["2m", "4m", "8m", "16m", "2b", "4b", "8b", "16b", "2n", "4n", "8n", "16n"]
        valid_models = ["salinet" + i for i in num_models]
        if model_name not in valid_models:
            raise ValueError("model_name should be one of: " + ", ".join(valid_models))

class SalientClassifier(nn.Module):
    def __init__(self, salinet_arch="salinet2m", num_classes=3):
        super(SalientClassifier, self).__init__()
        self.salinet_arch = salinet_arch
        self.num_classes = num_classes

        self.model = SaliNet.from_name(self.salinet_arch)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)