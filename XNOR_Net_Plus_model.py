"""
This Network is Strong Baseline in 'Training Binary Neural Networks with Real-To-Binary Convolution'.
It use Bi-Real Network & scaling factor of XNOR-Net++'s learnable parameter(per output channel)

"""
from telnetlib import X3PAD
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

import sys

#global variable

__all__ = ['xnornetplus18_case1', 'xnornetplus18_case4', 'xnornetplus18_BAN']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BinActive(torch.autograd.Function):

    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        input = input.sign()
        return input

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class BConv_case1(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(BConv_case1, self).__init__()
        self.stride = stride
        self.padding = padding
        self.out_chn = out_chn
        self.in_chn = in_chn
        self.kernel_size = kernel_size
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.scale_shape = (1, out_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights,1), device='cuda') * 0.001, requires_grad=True)
        self.scaling_factor = torch.rand(self.scale_shape, device='cuda') * 0.001
        self.real_scaling_factor = nn.Parameter(torch.mean(torch.mean(torch.mean(abs(self.scaling_factor), dim=3, keepdim=True),
                                                dim=2, keepdim=True), dim=0, keepdim=True), requires_grad=True)

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        
        binary_weights_no_grad = torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)
        y = y * self.real_scaling_factor.expand(y.size())

        return y
    
class BConv_case4(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(BConv_case4, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_chn = out_chn
        self.in_chn = in_chn
        self.kernel_size = kernel_size
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.scale_shape = (1, out_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.randn((self.number_of_weights,1), device='cuda') * 0.001, requires_grad=True)
        self.init = True

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        if self.init == True :
            out_size = int((x.size()[2] + 2*self.padding - self.kernel_size) / self.stride) + 1
            self.a = nn.Parameter(torch.rand((self.out_chn, 1), device='cuda') * 0.1, requires_grad=True)
            self.b = nn.Parameter(torch.rand((out_size, 1), device='cuda') * 0.1, requires_grad=True)
            self.c = nn.Parameter(torch.rand((out_size, 1), device='cuda') * 0.1, requires_grad=True)
            self.init = False
        
        binary_weights_no_grad = torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)
        scale = torch.matmul(self.a, torch.matmul(self.b, self.c.T).flatten().unsqueeze(dim=0)).reshape(y.size()[1], y.size()[2], y.size()[3])
        scale = scale.unsqueeze(dim=0)
        y = y * scale.expand(y.size())

        return y

class BAN_Conv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(BAN_Conv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.out_chn = out_chn
        self.in_chn = in_chn
        self.kernel_size = kernel_size
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.scale_shape = (1, out_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights,1), device='cuda') * 0.001, requires_grad=True)
        self.scaling_factor = torch.rand(self.scale_shape, device='cuda') * 0.001
        self.real_scaling_factor = nn.Parameter(torch.mean(torch.mean(torch.mean(abs(self.scaling_factor), dim=3, keepdim=True),
                                                dim=2, keepdim=True), dim=0, keepdim=True), requires_grad=True)

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        y = F.conv2d(x, real_weights, stride=self.stride, padding=self.padding)
        return y

class BasicBlock_case1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size, stride=1, downsample=None):
        super(BasicBlock_case1, self).__init__()

        self.conv = BConv_case1(inplanes, planes, kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = BinActive.apply(x)
        out = self.conv(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out
    
class BasicBlock_case4(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size, stride=1, downsample=None):
        super(BasicBlock_case4, self).__init__()

        self.conv = BConv_case4(inplanes, planes, kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = BinActive.apply(x)
        out = self.conv(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out


class BasicBlock_BAN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size, stride=1, downsample=None):
        super(BasicBlock_BAN, self).__init__()

        self.conv = BAN_Conv(inplanes, planes, kernel_size, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = BinActive.apply(x)
        out = self.conv(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual

        return out

class XNORNet_Plus(nn.Module):

    def __init__(self, block, layers, kernel_size=3, num_classes=1000, zero_init_residual=False):
        super(XNORNet_Plus, self).__init__()
        self.show_weight_scale = 0
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], kernel_size)
        self.layer2 = self._make_layer(block, 128, layers[1], kernel_size, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], kernel_size, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], kernel_size, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, kernel_size, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def xnornetplus18_case1(pretrained=False, **kwargs):
    model = XNORNet_Plus(BasicBlock_case1, [4, 4, 4, 4], **kwargs)
    return model

def xnornetplus18_case4(pretrained=False, **kwargs):
    model = XNORNet_Plus(BasicBlock_case4, [4, 4, 4, 4], **kwargs)
    return model

def xnornetplus18_BAN(pretrained=False, **kwargs):
    model = XNORNet_Plus(BasicBlock_BAN, [4, 4, 4, 4], **kwargs)
    return model



