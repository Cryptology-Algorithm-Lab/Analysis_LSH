import torch
from .iresnet import iresnet50, iresnet100

def get_backbone(arch, param_dir, device, **kwargs):
    if arch == "r50":
        net = iresnet50(**kwargs)
        
    elif arch == "r100":
        net = iresnet100(**kwargs)

    else:
        raise ValueError(f"Architecture {arch} does NOT supported.")
        
    net.load_state_dict(torch.load(param_dir, map_location = "cpu"))
    net = net.to(device).eval()
    return net

