import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size = 3, dilation=1, groups=1, mask_prob=0.005):
        super(Conv2d, self).__init__()
        self.in_channel = in_channels
        self.rge = kernel_size//2
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.mask_prob = mask_prob
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,kernel_size,kernel_size))
        self.bias = nn.Parameter(torch.Tensor(self.rge+1, out_channels))
        self.alpha = nn.Parameter(torch.Tensor(self.rge+1, out_channels,1,1))
        self.beta = nn.Parameter(torch.Tensor(self.rge+1, out_channels,1,1))
        self.tanh = nn.Tanh()
        nn.init.uniform_(self.weight, a=-0.2, b=0.2)
        nn.init.uniform_(self.bias, a=-2.0, b=2.0)
        nn.init.uniform_(self.alpha, a=-2.0, b=2.0)
        nn.init.uniform_(self.beta, a=-2.0, b=2.0)

    def charge_val(self, old, new, output, alpha, beta, lock):
        cmp_val = new > old
        if self.training:
            select = (torch.rand_like(old) + self.mask_prob).floor_() == 1.0
            cmp_val = ((cmp_val) | select) & ~lock
            lock = lock & select
        return torch.where(cmp_val , new, old), torch.where(cmp_val, new*(1+self.tanh(alpha))+beta, output), lock

    def forward(self, x):
        weight = self.weight
        bias = self.bias
        scc = F.conv2d(x, weight[:, :, self.rge:self.rge + 1, self.rge:self.rge + 1],
                               bias[0], self.stride, 0, 1, self.groups)
        output = scc*(1+self.tanh(self.alpha[0]))+self.beta[0]
        lock = (torch.rand_like(output) + self.mask_prob).floor_() == 1.0
        for i in range(1,self.rge+1):
            new_scc = F.conv2d(x, weight[:, :, self.rge - i:self.rge + i + 1, self.rge - i:self.rge + i + 1],
                               bias[i], self.stride, i*self.dilation, self.dilation, self.groups)
            scc, output, lock = self.charge_val(scc, new_scc, output, self.alpha[i], self.beta[i], lock)
        return output

class BasicBlock(nn.Module):
    expansion=1

    def __init__(self,in_channels,out_channels,stride=1,downsample=None, kernel_size=3,**kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(in_channels,out_channels,stride=stride,kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = Conv2d(out_channels,out_channels, stride=1,kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self,x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity

        return out


class SCCNet(nn.Module):
    def __init__(self,block,block_num,num_classes=1000,include_top=True,groups=1,width_per_group=64):
        super(SCCNet, self).__init__()
        self.groups=groups
        self.width_per_group=width_per_group
        self.include_top = include_top
        self.in_channels = 64

        self.conv = nn.Conv2d(3,self.in_channels,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,64,block_num[0],3)
        self.layer2 = self._make_layer(block,128,block_num[1],3,stride=2)
        self.layer3 = self._make_layer(block,256,block_num[2],3,stride=2)
        self.layer4 = self._make_layer(block,512,block_num[3],3,stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.head = nn.Linear(512*block.expansion,num_classes)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,Conv2d):
                nn.init.uniform_(m.weight, a=-1.0, b=1.0)
                nn.init.uniform_(m.bias, a=-1.0, b=1.0)
                nn.init.uniform_(m.alpha, a=-1.0, b=1.0)
                nn.init.uniform_(m.beta, a=-1.0, b=1.0)

    def _make_layer(self,block,channels,block_num,kernel_size,stride=1):
        downsample = None
        if stride!=1 or self.in_channels != channels*block.expansion:
            downsample = nn.Sequential(
                Conv2d(self.in_channels,channels*block.expansion,kernel_size=5,stride=stride),
                nn.BatchNorm2d(channels*block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels,channels,downsample=downsample,kernel_size=kernel_size,stride=stride,groups=self.groups,width_per_group=self.width_per_group))
        self.in_channels = channels*block.expansion

        for _ in range(1,block_num):
            layers.append(block(self.in_channels,channels,kernel_size=kernel_size,groups=self.groups,width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x=torch.flatten(x,1)
            x=self.head(x)

        return x
        
def sccnet18(num_classes=1000,include_top=True):
    return SCCNet(BasicBlock,[2,2,2,2],num_classes, include_top)

def sccnet34(num_classes=1000,include_top=True):
    return SCCNet(BasicBlock,[3,4,6,3],num_classes, include_top)

