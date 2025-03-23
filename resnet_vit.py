import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchvision.models as models


import torch
import torch.nn as nn
# from utils import load_state_dict_from_url



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
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
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
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

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


# def _resnet(arch, block, layers, pretrained, progress, **kwargs):
#     model = ResNet(block, layers, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls[arch],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model


# def resnet18(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-18 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
#                    **kwargs)
#
#
# def resnet34(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-34 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
#                    **kwargs)
#
#
# def resnet50(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-50 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
#                    **kwargs)
#
#
# def resnet101(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-101 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
#                    **kwargs)
#
#
# def resnet152(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-152 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
#                    **kwargs)
#
#
# def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
#     r"""ResNeXt-50 32x4d model from
#     `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 4
#     return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
#                    pretrained, progress, **kwargs)
#
#
# def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
#     r"""ResNeXt-101 32x8d model from
#     `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 8
#     return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
#                    pretrained, progress, **kwargs)
#
#
# def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
#     r"""Wide ResNet-50-2 model from
#     `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
#
#     The model is the same as ResNet except for the bottleneck number of channels
#     which is twice larger in every block. The number of channels in outer 1x1
#     convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
#     channels, and in Wide ResNet-50-2 has 2048-1024-2048.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['width_per_group'] = 64 * 2
#     return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
#                    pretrained, progress, **kwargs)
#
#
# def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
#     r"""Wide ResNet-101-2 model from
#     `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
#
#     The model is the same as ResNet except for the bottleneck number of channels
#     which is twice larger in every block. The number of channels in outer 1x1
#     convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
#     channels, and in Wide ResNet-50-2 has 2048-1024-2048.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['width_per_group'] = 64 * 2
#     return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
#                    pretrained, progress, **kwargs)





class Embedding(nn.Module):  # Patch Embedding + Position Embedding + Class Embedding
    def __init__(self, image_channels=3, image_size=224, patch_size=32, dim=768, drop_ratio=0.):
        super(Embedding, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2  # Patch数量

        self.patch_conv = nn.Conv2d(2048, dim, 1, 1)  # 使用卷积将图像划分成Patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))  # class embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))  # position embedding
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x):
        x = self.patch_conv(x)
        x = rearrange(x, "B C H W -> B (H W) C")
        cls_token = torch.repeat_interleave(self.cls_token, x.shape[0], dim=0)  # (1,1,dim) -> (B,1,dim)
        x = torch.cat([cls_token, x], dim=1)  # (B,1,dim) cat (B,num_patches,dim) --> (B,num_patches+1,dim)
        x = x + self.pos_emb
        return self.dropout(x)  # token


class MultiHeadAttention(nn.Module):  # Multi-Head Attention
    def __init__(self, dim, num_heads=8, drop_ratio=0.):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)  # 使用一个Linear，计算得到qkv
        self.dropout = nn.Dropout(drop_ratio)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # B: Batch Size / P: Num of Patches / D: Dim of Patch / H: Num of Heads / d: Dim of Head
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "B P (C H d) -> C B H P d", C=3, H=self.num_heads, d=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分离qkv
        k = rearrange(k, "B H P d -> B H d P")
        # Attention(Q, K, V ) = softmax(QKT/dk)V （T表示转置)
        attn = torch.matmul(q, k) * self.head_dim ** -0.5  # QKT/dk
        attn = F.softmax(attn, dim=-1)  # softmax(QKT/dk)
        attn = self.dropout(attn)
        x = torch.matmul(attn, v)  # softmax(QKT/dk)V
        x = rearrange(x, "B H P d -> B P (H d)")
        x = self.proj(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):  # MLP
    def __init__(self, in_dims, hidden_dims=None, drop_ratio=0.):
        super(MLP, self).__init__()
        if hidden_dims is None:
            hidden_dims = in_dims * 4  # linear的hidden_dims默认为in_dims的4倍

        self.fc1 = nn.Linear(in_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, in_dims)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x):
        # Linear + GELU + Dropout + Linear + Dropout
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):  # Transformer Encoder Block
    def __init__(self, dim, num_heads=8, drop_ratio=0.):
        super(EncoderBlock, self).__init__()

        self.layernorm1 = nn.LayerNorm(dim)
        self.multiheadattn = MultiHeadAttention(dim, num_heads)
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x):
        # 两次残差连接，分别在Multi-Head Attention和MLP之后
        x0 = x
        x = self.layernorm1(x)
        x = self.multiheadattn(x)
        x = self.dropout(x)
        x1 = x + x0  # 第一次残差连接
        x = self.layernorm2(x1)
        x = self.mlp(x)
        x = self.dropout(x)
        return x + x1  # 第二次残差连接


class MLPHead(nn.Module):  # MLP Head
    def __init__(self, dim, num_classes=1000):
        super(MLPHead, self).__init__()
        self.layernorm = nn.LayerNorm(dim)
        # 对于一般数据集，此处为1层Linear; 对于ImageNet-21k数据集，此处为Linear+Tanh+Linear
        self.mlphead = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.layernorm(x)
        cls = x[:, 0, :]  # 去除class token
        return self.mlphead(cls)




class ViT(nn.Module):  # Vision Transformer
    def __init__(self, image_channels=3, image_size=224, num_classes=1000, patch_size=32, dim=768, num_heads=12,
                 layer=2, block=Bottleneck, zero_init_residual=False,layers=[3, 4, 6, 3],
                     groups=1, width_per_group=64, replace_stride_with_dilation=None,
                     norm_layer=None):
        super(ViT, self).__init__()
        self.embedding = Embedding(image_channels, image_size, patch_size, dim)
        self.encoder = nn.Sequential(
            *[EncoderBlock(dim, num_heads) for i in range(layer)]  # encoder结构为layers(L)个Transformer Encoder Block
        )
        self.head = MLPHead(dim, num_classes)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv2=nn.Conv2d(2048,768, kernel_size=1,stride=1,padding=0)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x_emb = self.embedding(x)
        feature = self.encoder(x_emb)
        return self.head(feature)


def vit_base(num_classes=1000,layer=3):  # ViT-Base
    return ViT(image_channels=3, image_size=224, num_classes=num_classes, patch_size=32, dim=768, num_heads=12,
               layer=layer)


def vit_large(num_classes=1000):  # ViT-Large
    return ViT(image_channels=3, image_size=224, num_classes=num_classes, patch_size=16, dim=1024, num_heads=16,
               layers=24)


def vit_huge(num_classes=1000):  # ViT-Huge
    return ViT(image_channels=3, image_size=224, num_classes=num_classes, patch_size=16, dim=1280, num_heads=16,
               layers=32)


# if __name__ == "__main__":
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     images = torch.randn(8, 3, 224, 224).cuda()
#     vb = vit_base()
#     vb.to(device)
#     # vl = vit_large()
#     # vh = vit_huge()
#     print(vb(images).shape)
#     # print(vl(images).shape)
#     # print(vh(images).shape)

