from .modified_nbnet import NbNet
import torch

def get_nbnet(param_dir, device):
    net = NbNet()
    net.load_state_dict(torch.load(param_dir, map_location = "cpu"))
    net = net.to(device).eval()
    return net

