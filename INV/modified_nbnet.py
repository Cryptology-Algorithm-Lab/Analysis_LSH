# Implementation of modified NbNet

import torch
from torch import nn
import torch.nn.functional as F

# Equalized Learning Rate Modules
class EqConv2d(nn.Module):
    def __init__(self, ic, oc, k, s, p, bias = False):
        super(EqConv2d, self).__init__()
        
        self.scale = nn.init.calculate_gain("conv2d") * ((ic+oc)*k*k/2)**(-0.5)
        self.weight = nn.Parameter(torch.randn(oc, ic, k, k))        
        self.bias = None
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(oc))
            
        self.s = s
        self.p = p
        
    def forward(self, x):
        x = F.conv2d(x, self.weight, self.bias, self.s, self.p)
        return x*self.scale

    
class EqDeconv2d(nn.Module):
    def __init__(self,ic,oc,k,s,p, bias = False):
        super(EqDeconv2d, self).__init__()
        
        self.scale = nn.init.calculate_gain("conv2d") * ((ic+oc)*k*k/2)**(-0.5)
        self.weight = nn.Parameter(torch.randn(ic,oc,k,k))
        self.bias = None
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(oc))
            
        self.s = s
        self.p = p
        
    def forward(self, x):
        x = F.conv_transpose2d(x, self.weight, self.bias, self.s, self.p)
        return x * self.scale

class EqLinear(nn.Module):
    def __init__(self, ic,oc):
        super(EqLinear, self).__init__()
        self.scale = nn.init.calculate_gain("linear") * (((ic+oc)/2) **(-0.5))
        self.weight = nn.Parameter(torch.randn((oc,ic)))
        self.bias = nn.Parameter(torch.zeros(oc))
    
    def forward(self, x):
        x = F.linear(x, self.weight, self.bias)
        return x * self.scale
    
    
# Minibatch Standard Deviation module
class MinStddev(nn.Module):
    def __init__(self):
        super(MinStddev,  self).__init__()

    def forward(self, x):
        x_std = x-x.mean(dim = 0, keepdims = True)
        x_std = torch.sqrt(x_std.pow(2) +1e-8).mean(dim=[1,2,3], keepdims = True)
        x_std = x_std * torch.ones((x.size(0), 1, x.size(2), x.size(3)), device = x.device)
        return torch.cat([x, x_std], dim=1)
      
# Upscale by Factor 2
def Upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1:
        return x
    s = x.size()
    x = x.view(-1, s[1], s[2], 1, s[3], 1)
    x = x.expand(-1, s[1], s[2], factor, s[3], factor)
    x = x.contiguous().view(-1, s[1], s[2] * factor, s[3] * factor)
    return x

# Pixel Normalization Module
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, epsilon=1e-8):
        return x * (((x**2).mean(dim=1, keepdim=True) + epsilon).rsqrt())

    
# Adaptive Instance Normalization (AdaIN)
class AdaIN(nn.Module):
    def __init__(self, nc, emb_size=512):
        super().__init__()
        self.lin_op = nn.Linear(emb_size, 2*nc)
    
    def forward(self, x, z):
        batch, nc = x.size(0), x.size(1)
        coeff = self.lin_op(z).unsqueeze(-1).unsqueeze(-1)
        alpha, beta = coeff[:,0::2,:,:], coeff[:,1::2,:,:]
        x_std, x_mean = torch.std_mean(x, dim = [2,3], keepdims = True)
        return alpha* (x-x_mean) / (x_std + 1e-8) + beta


# Fully-Connected Layer
class FC(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.lin = EqLinear(ic, oc)
        self.norm = PixelNorm()
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.lin(x)
        x = self.norm(x)
        x = self.act(x)
        return x
    
# Convolution with Style Injection (AdaIN)
class StyleBlock(nn.Module):
    def __init__(self, ic, oc, k=3,s=1,p=1, emb_size=512):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc, k, s, p)
        self.adain = AdaIN(oc, emb_size)
        self.act = nn.ReLU()
        
    def forward(self, x, z):
        x = self.conv(x)
        x = self.adain(x,z)
        x = self.act(x)
        
        return x

# NbNet Block
class NbBlock(nn.Module):
    def __init__(self, ic, oc, nblk, gr=8):
        super().__init__()
        assert(oc>nblk*gr)
        oc_start = oc - nblk*gr
        self.init_layer = nn.Sequential(
            nn.ConvTranspose2d(ic, oc_start, 4,2,1),
            PixelNorm(),
            nn.ReLU(),
        )
        self.main_module = nn.ModuleList()
        
        for i in range(nblk):
            self.main_module.add_module("layer%d"%(i+1), StyleBlock(oc_start+i*gr, gr))
            
        self.final_layer = StyleBlock(oc,oc)
    
    def forward(self, x, z):
        x = self.init_layer(x)
        
        for layer in self.main_module:
            out = layer(x, z)
            x = torch.cat([x, out], dim=1)
    
        x = self.final_layer(x,z)
        
        return x

# Modified NbNet
class NbNet(nn.Module):
    def __init__(self, nblk=5, nc = 32, emb_size = 512, fp16 = True):
        super().__init__()
        self.fp16 = fp16
        self.synthesis = nn.Sequential(*[FC(emb_size, emb_size) for _ in range(8)])
        self.emb_size = emb_size
        self.init_layer = nn.Sequential(
            nn.Linear(emb_size, emb_size*4*4),
            nn.ReLU()
        )
        nc=nc
        
        ic = emb_size
        self.main_module = nn.ModuleList()
        self.main_module.add_module("layer1", NbBlock(ic, ic, nc))
        nc//=2
        
        for i in range(1,nblk):
            self.main_module.add_module("layer%d"%(i+1), NbBlock(ic,ic//2, nc))
            ic //= 2
            nc //= 2
        
        self.to_rgb = nn.Sequential(
            nn.Conv2d(ic,3,3,1,1),
            nn.Tanh()        
        )
    
    def forward(self, z):
        z_syn = self.synthesis(z)
        
        x = self.init_layer(z)
        x = x.view(-1,self.emb_size,4,4)
        
        with torch.cuda.amp.autocast(True):        
            for layer in self.main_module:
                x = layer(x, z_syn)
        x = self.to_rgb(x.float())
        return x