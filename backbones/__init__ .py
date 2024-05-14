import torch
from torch import nn

from .iresnet import iresnet50, iresnet100
from .sfnet import sfnet20, sfnet64
from .vit import get_vit, ViTs_face

""" 
Forked from Insightface(https://github.com/deepinsight/insightface/), Opensphere(https://github.com/ydwen/opensphere), Face-Transformer(https://github.com/zhongyy/Face-Transformer)
"""


def get_backbone(arch, param_dir, device, **kwargs):
    if arch == "r50":
        net = iresnet50(**kwargs)
    elif arch == "r100":
        net = iresnet100(**kwargs)
    elif arch == "sf20":
        net = sfnet20(**kwargs)
    elif arch == "sf64":
        net = sfnet20(**kwargs)       
    else:
        raise ValueError(f"Architeecture {arch} does not supported.")
    net.load_state_dict(torch.load(param_dir, map_location ="cpu"))
    net = net.to(device).eval()
    return net

class Rescale(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        assert(high > low)
        self.low = low
        self.high = high
    
    def forward(self, x):
        x = (x + 1) / 2
        return x * (self.high - self.low) + self.low

class dummy(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.module = net
    def forward(self, x):
        return self.module(x) 

@torch.no_grad()
def get_backbone_v2(config, device, **kwargs):
    arch, param_dir, is_onnx = config
    
    if not is_onnx:
    
        if arch == "r50":
            net = iresnet50(**kwargs)
        elif arch == "r100":
            net = iresnet100(**kwargs)
        elif arch == "sf20":
            net = sfnet20(**kwargs)
            net = dummy(net)
        elif arch == "sf64":
            net = sfnet64(**kwargs)       
            net = dummy(net)
        elif arch == "vit":
            net = get_vit()
        elif arch == "facenet":
            from facenet_pytorch import InceptionResnetV1
            net = InceptionResnetV1(pretrained = "casia-webface").eval()
        else:
            raise ValueError(f"Architecture {arch} does not supported.")
            
        net.load_state_dict(torch.load(param_dir, map_location ="cpu"))
        
        if arch == "vit":
            net = nn.Sequential(Rescale(0, 255), net)
        
        net = net.to(device).eval()
        return net
    
    else:
        from onnx2torch import convert
        net = convert(param_dir)
        net = net.to(device).eval()
        return net 

def get_vit():
    ViT = ViTs_face(
                loss_type='CosFace',
                GPU_ID='0',
                num_class=93431,
                image_size=112,
                patch_size=8,
                ac_patch_size=12,
                pad=4,
                dim=512,
                depth=20,
                heads=8,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1
            )
    return ViT        

class Rescale(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        assert(high > low)
        self.low = low
        self.high = high
    
    def forward(self, x):
        x = (x + 1) / 2
        return x * (self.high - self.low) + self.low

class dummy(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.module = net
    def forward(self, x):
        return self.module(x)    