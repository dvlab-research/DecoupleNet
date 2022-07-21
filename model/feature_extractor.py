import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from . import resnet

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d_v2, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        output = F.batch_norm(x, self.running_mean, self.running_var, weight=self.weight, bias=self.bias, training=False)
        return output

class resnet_feature_extractor(nn.Module):
    def __init__(self, backbone_name, pretrained_weights=None, aux=False, pretrained_backbone=True, freeze_bn=False):
        super(resnet_feature_extractor, self).__init__()
        bn_layer = nn.BatchNorm2d
        if freeze_bn:
            bn_layer = FrozenBatchNorm2d
        backbone = resnet.__dict__[backbone_name](
                pretrained=pretrained_backbone,
                replace_stride_with_dilation=[False, True, True], pretrained_weights=pretrained_weights, norm_layer=bn_layer)
        return_layers = {'layer4': 'out'}
        if aux:
            return_layers['layer3'] = 'aux'
        self.aux = aux
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    
    def forward(self, x):
        if self.aux == True:
            output = self.backbone(x)
            aux, out = output['aux'], output['out']
            return aux, out
        else:
            out = self.backbone(x)['out']
        return out
