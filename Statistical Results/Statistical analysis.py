import torch
import torch.nn as nn
import torch.nn.functional as F
from dlcode.ResNet.model import resnet34 as model3
from idea.SCCNet.model import resnet34 as model4

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size = 3, dilation=1, groups=1, lock_prob=0.0):
        super(Conv2d, self).__init__()
        self.in_channel = in_channels
        self.rge = kernel_size//2
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.lock_prob = lock_prob
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,kernel_size,kernel_size))
        self.bias = nn.Parameter(torch.Tensor(self.rge+1, out_channels))
        self.alpha = nn.Parameter(torch.Tensor(self.rge+1, out_channels,1,1))
        self.beta = nn.Parameter(torch.Tensor(self.rge+1, out_channels,1,1))
        self.tanh = nn.Tanh()
        p=1.
        nn.init.uniform_(self.weight, a=-p, b=p)
        nn.init.uniform_(self.bias, a=-p, b=p)
        nn.init.uniform_(self.alpha, a=-p, b=p)
        nn.init.uniform_(self.beta, a=-p, b=p)

    def charge_val(self, old, new, output, alpha, beta, lock):
        cmp_val = new > old
        if self.training:
            select = (torch.rand_like(old) + self.lock_prob).floor_() == 1.0
            cmp_val = ((cmp_val) | select) & ~lock
            lock = lock & select
        return torch.where(cmp_val , new, old), torch.where(cmp_val, new*(1+self.tanh(alpha))+beta, output), lock

    def forward(self, x):
        weight = self.weight
        bias = self.bias
        scc = F.conv2d(x, weight[:, :, self.rge:self.rge + 1, self.rge:self.rge + 1],
                               bias[0], self.stride, 0, 1, self.groups)
        output = scc*(1+self.tanh(self.alpha[0]))+self.beta[0]
        lock = (torch.rand_like(output) + self.lock_prob).floor_() == 1.0
        for i in range(1,self.rge+1):
            new_scc = F.conv2d(x, weight[:, :, self.rge - i:self.rge + i + 1, self.rge - i:self.rge + i + 1],
                               bias[i], self.stride, i*self.dilation, self.dilation, self.groups)
            scc, output, lock = self.charge_val(scc, new_scc, output, self.alpha[i], self.beta[i], lock)
        return output

class SkewnessTracker:
    def __init__(self, model):
        self.model = model
        self.skewness = {}  
        self.max_output = {}
        self.mean = {}
        self.var = {}
        self.handles = []  


        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Identity)):
                handle = layer.register_forward_hook(
                    self._hook_compute_skewness(name)
                )
                self.handles.append(handle)

    def _hook_compute_skewness(self, layer_name):
        def hook_func(module, input, output):
            self.skewness[layer_name] = output

            with torch.no_grad():

                flattened = output.view(output.size(0), -1)  # [B, Features]

                mu = flattened.mean(dim=1, keepdim=True)  # [B, 1]
                sigma = flattened.std(dim=1, keepdim=True)  # [B, 1]
                centered = flattened - mu
                moment3 = (centered ** 3).mean(dim=1)  # [B]
                k = moment3 / (sigma.squeeze() ** 3 + 1e-8)  # [B]

                self.skewness[layer_name] = k.mean().item()  
                self.max_output[layer_name] = torch.max(output.abs()).item()
                self.mean[layer_name] = mu.mean().item()
                self.var[layer_name] = sigma.mean().item()

        return hook_func

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()



n=998

layers1 = []
layers2 = []

layers1.extend([
    nn.Conv2d(3, 64, 3),
    nn.BatchNorm2d(64),
    nn.Mish(),
    nn.Identity(),
    nn.Conv2d(64, 128, 3),
    nn.BatchNorm2d(128),
    nn.Mish(),
    nn.Identity(),
])

layers2.extend([
    Conv2d(3, 64, 3),
    nn.BatchNorm2d(64),
    nn.Identity(),
    Conv2d(64, 128, 3),
    nn.BatchNorm2d(128),
    nn.Identity(),
])


for i in range(2,n):
    layers1.extend([
        nn.Conv2d(128, 128, 1,bias=False),
        nn.BatchNorm2d(128),
        nn.Mish(),
        nn.Identity(),
    ])
    layers2.extend([
        Conv2d(128, 128,),
        nn.BatchNorm2d(128),
        nn.Identity(),
    ])

model1 = nn.Sequential(*layers1)
model2 = nn.Sequential(*layers2)
path0 = 'res34in1k-best.pth'
path1 = './resmishe100-best.pth'
path2 = './flin1kp0.5ds5x-best.pth'

model = model4()
path = path2

weights_dict = torch.load(path, map_location=torch.device('cuda:0'))
print(model.load_state_dict(weights_dict, strict=False))
model.to(torch.device('cuda:0'))

tracker = SkewnessTracker(model)


x = torch.randn(32, 3, 224, 224).to(torch.device('cuda:0'))
with torch.no_grad():
    _ = model(x)


res1 ,res2, res3 = [],[],[]
for name, sk in tracker.skewness.items():
    res1.append(sk)
    res2.append(tracker.mean[name])
    res3.append(tracker.var[name])
    print(f"{name}: Skewness = {sk:.4f}, max_output = {tracker.max_output[name]:.4f}, mean = {tracker.mean[name]:.4f}, var = {tracker.var[name]:.4f}")
ave1 = sum(res1) / len(res1)
ave2 = sum(res2) / len(res2)
ave3 = sum(res3) / len(res3)
print(f"smv: {ave1:.4f}, {ave2:.4f}, {ave3:.4f}")

tracker.remove_hooks()