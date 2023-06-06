import collections

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

########################################################################
############### HELPERS FUNCTIONS FOR MODEL ARCHITECTURE ###############
########################################################################


# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple("GlobalParams", [
    "block", "zero_init_residual",
    "groups", "width_per_group", "replace_stride_with_dilation",
    "norm_layer", "num_classes", "image_size"])

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class MinimalBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(MinimalBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


########################################################################
############## HELPERS FUNCTIONS FOR LOADING MODEL PARAMS ##############
########################################################################

def salinet_params(model_name):
    """ Map resnet_pytorch model name to parameter coefficients. """

    params_dict = {
        # Coefficients:   block, res
        "salinet2m": (MinimalBlock, 224),
        "salinet4m": (MinimalBlock, 224),
        "salinet8m": (MinimalBlock, 224),
        "salinet16m": (MinimalBlock, 224),

        "salinet2b": (BasicBlock, 224),
        "salinet4b": (BasicBlock, 224),
        "salinet8b": (BasicBlock, 224),
        "salinet16b": (BasicBlock, 224),

        "salinet2n": (Bottleneck, 224),
        "salinet4n": (Bottleneck, 224),
        "salinet8n": (Bottleneck, 224),
        "salinet16n": (Bottleneck, 224),
    }
    return params_dict[model_name]


def salinet(arch, block, num_classes=3, zero_init_residual=False,
           groups=1, width_per_group=64, replace_stride_with_dilation=None,
           norm_layer=None, image_size=224):
    """ Creates a salinet model. """

    global_params = GlobalParams(
        block=block,
        num_classes=num_classes,
        zero_init_residual=zero_init_residual,
        groups=groups,
        width_per_group=width_per_group,
        replace_stride_with_dilation=replace_stride_with_dilation,
        norm_layer=norm_layer,
        image_size=image_size,
    )

    layers_dict = {
        "salinet2m": (1, 1, 0, 0),
        "salinet4m": (1, 1, 1, 1),
        "salinet8m": (2, 2, 2, 2),
        "salinet16m": (3, 4, 6, 3),

        "salinet2b": (1, 1, 0, 0),
        "salinet4b":  (1, 1, 1, 1),
        "salinet8b": (2, 2, 2, 2),
        "salinet16b": (3, 4, 6, 3),

        "salinet2n": (1, 1, 0, 0),
        "salinet4n": (1, 1, 1, 1),
        "salinet8n": (2, 2, 2, 2),
        "salinet16n": (3, 4, 6, 3),
    }
    layers = layers_dict[arch]

    return layers, global_params


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith("salinet"):
        b, s = salinet_params(model_name)
        layers, global_params = salinet(arch=model_name, block=b, image_size=s)
    else:
        raise NotImplementedError(f"model name is not pre-defined: {model_name}")
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return layers, global_params