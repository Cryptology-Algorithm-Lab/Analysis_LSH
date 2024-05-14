import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from INV.dataset import MXFaceDataset
from INV.modified_nbnet import NbNet
from backbones.iresnet import *

def L1Loss(x, y):
    return (x-y).abs().mean(dim=[1,2,3]).mean()

def PerceptLoss(feat1, feat2):
    return (1 - F.cosine_similarity(feat1, feat2)).mean()

import time

# Simple logger
class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.time = None
        
    def write(self, payload):
        f = open(self.log_dir, 'a')
        f.write(payload + "\n")
        f.close()
    
    def timer_start(self):
        print("Timer Starts...")
        self.time = time.time()
    
    def timer_elapsed(self):
        curr_time = time.time()
        elapsed = curr_time - self.time
        self.time = curr_time
        return elapsed

def train(cfg):
    device = cfg.device
    # logger
    logger = Logger(cfg.log_dir)
    
    dataset = MXFaceDataset(cfg.img_dir, 0)
    dataloader = DataLoader(dataset, cfg.batch_size, shuffle= True, num_workers = 8, pin_memory = True)
    
    curr_step = 0
    total_step = cfg.total_step
    verbose = cfg.verbose
    
    
    # Target
    target = iresnet50(fp16 = True)
    target.load_state_dict(torch.load(cfg.target_dir, map_location = "cpu"))
    target.eval().to(device)
    
    percepts = []
    for info in cfg.percept_list:
        name, arch = info[0], info[1]
        
        if arch == "r50":
            percept = iresnet50(fp16 = True)
        else:
            percept = iresnet100(fp16 = True)
            
        percept.load_state_dict(torch.load(name, map_location = "cpu"))
        percept.eval().to(device)
        percepts.append(percept)
    
    nbnet = NbNet(fp16 = True)
    nbnet.train().to(device)
    opt = torch.optim.Adam(nbnet.parameters(), cfg.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, [total_step // 2, total_step // 4 * 3], gamma = 0.3
    )
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval = 100)
    
    logger.timer_start()
    
    
    while curr_step < total_step:
        for x, _ in dataloader:
            x = x.to(device)
            with torch.no_grad():
                feat = F.normalize(target(x))
            
            img_recon = F.interpolate(nbnet(feat), (112,112))
            l1_loss = L1Loss(x, img_recon)
            
            percept_loss = 0
            
            for model in percepts:
                feat_perc = F.normalize(model(x))
                feat_perc_recon = F.normalize(model(img_recon))
                percept_loss += PerceptLoss(feat_perc, feat_perc_recon)
                                              
            percept_loss = percept_loss / len(percepts)
            loss = 10 * l1_loss + percept_loss
            
    
            
            amp.scale(loss).backward()
            amp.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(nbnet.parameters(), 5)
            amp.step(opt)
            amp.update()
            opt.zero_grad()
            curr_step += 1
            
            if curr_step % verbose == 0:
                elapsed = logger.timer_elapsed()
                required = (total_step - curr_step) * (elapsed / verbose) / 3600                
                payload = "[{}/{}] L1LOSS: {:.3f}, Percept: {:.3f}, Elapsed: {:.3f}, Required: {:.3f}hrs".format(
                   curr_step, total_step, l1_loss.item(), percept_loss.item(),
                   elapsed, required
                )
                print(payload)
                logger.write(payload)
            
            if curr_step > total_step:
                break
                
        
    torch.save(nbnet.state_dict(), "nbnet_" + cfg.save_suffix + ".pt")
    

if __name__ == "__main__":
    from INV.config import cfg
    torch.backends.cudnn.benchmark = True
    train(cfg)
    
    

    
    