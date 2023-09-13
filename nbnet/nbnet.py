import torch
from torch import nn
import torch.nn.functional as F

def act():
    return nn.ReLU()

def fc(ic, oc):
    return nn.Sequential(
        nn.Linear(ic,oc),
        PixelNorm(),
        act())
    
class PixelNorm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, epsilon=1e-8):
        return x * (((x**2).mean(dim=1, keepdim=True) + epsilon).rsqrt())
        
    
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
    
class StyleBlock(nn.Module):
    def __init__(self, ic, oc, k=3,s=1,p=1, emb_size=512):
        super().__init__()
        self.conv1 = nn.Conv2d(ic, oc, k, s, p)
        self.adain1 = AdaIN(oc, emb_size)
        self.act1 = act()
        
    def forward(self, x, z):
        x = self.conv1(x)
        x = self.adain1(x,z)
        x = self.act1(x)
        
        return x
    
class NbBlock(nn.Module):
    def __init__(self, ic, oc, nblk, gr=8):
        super().__init__()
        assert(oc>nblk*gr)
        oc_start = oc - nblk*gr
        self.init_layer = nn.Sequential(
            nn.ConvTranspose2d(ic, oc_start, 4,2,1),
            nn.BatchNorm2d(oc_start),
            act()
            
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
    
class NbNet(nn.Module):
    def __init__(self, nblk=5, nc = 32, emb_size = 512):
        super().__init__()
        self.synthesis = nn.Sequential(*[fc(emb_size, emb_size) for _ in range(8)])
        self.emb_size = emb_size
        self.init_layer = nn.Sequential(
            nn.Linear(emb_size, emb_size*4*4),
            act()
        
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
        
        for layer in self.main_module:
            x = layer(x, z_syn)
            
        x = self.to_rgb(x)
        
        return x