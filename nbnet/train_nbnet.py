from nbnet.nbnet import NbNet
from fr_utils.iresnet import iresnet50
from fr_utils.dataset import MXFaceDataset
from nbnet.utils import Logger, TimeManager

import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, args):
        self.dataset = MXFaceDataset(args.img_dir)
        self.dataloader = DataLoader(
            self.dataset,
            args.batch_size,
            shuffle = True,
            num_workers = 2,
            drop_last = True,
            pin_memory = True
        )
        self.device = args.device
        
        self.backbone = iresnet50()
        self.backbone.load_state_dict(torch.load(args.bb_dir, map_location = "cpu"))
        self.backbone = self.backbone.to(self.device).eval()
        
        self.nbnet = NbNet().to(self.device)
        
        self.target = iresnet50()
        self.target.load_state_dict(torch.load(args.std_dir, map_location = "cpu"))
        self.target = self.target.to(self.device).eval()
        
        
        self.optim = torch.optim.Adam(self.nbnet.parameters(),
                                   lr = args.lr,
                                  betas = [0.5, 0.99])
        self.sched = torch.optim.lr_scheduler.ExponentialLR(self.optim,
                                                            gamma = 0.94)
        
        self.max_step = args.max_step
        self.infer_step = args.infer_step
        self.curr_step = 0
        
        self.logger = Logger(args.logging_path)
        self.timer = TimeManager()
        
        self.alpha = 1
        self.beta = 0
    
    
    def renew_alpha_beta(self):
        self.alpha *= 0.9
        self.beta = 1-(1-self.beta)*0.85
        
        
    def train(self):
        self.timer.start()
        while True:
            if self.curr_step>self.max_step:
                return
            
            for idx, (img, _) in enumerate(self.dataloader):
                img = img.to(self.device)
                with torch.no_grad():
                    feat = self.backbone(img)
                    feat_target = self.target(img)
                    feat = F.normalize(feat)
                    feat_target = F.normalize(feat)
                    
                    
                img_recon = self.nbnet(feat)
                img_recon = F.interpolate(img_recon, 112, mode = 'nearest')
                feat_recon = self.target(img_recon)
                
                
                mae = (img-img_recon).abs().mean()
                perceptual = (1 - F.cosine_similairty(feat_target, feat_recon)).mean()
                loss = 10 * self.alpha * mae + self.beta * perceptual
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                self.curr_step += 1
                
                if self.curr_step % 5000 == 0:
                    self.sched.step()
                    
                if self.curr_step % 5000 == 0:
                    self.renew_alpha_beta()
                    
                if self.curr_step % self.infer_step == 0:
                    self.logger.logging("[{}/{}] MAE: {}, Perceptual: {}".format(self.curr_step, self.max_step, mae.item(), perceptual.item()))
                    self.timer.renew(self.curr_step, self.infer_step)
                    self.infer(img)
        
        torch.save(nbnet.state_dict(), "nbnet.pt")
        
    @torch.no_grad()
    def infer(self, img):
        ten2img = lambda x: (x.detach().cpu().numpy().transpose(1,2,0)+1) * 0.5
        self.nbnet.eval()
        feat = self.backbone(img)
        feat = F.normalize(feat)
        img_recon = self.nbnet(feat)
        img_recon = F.interpolate(img_recon, 112, mode = 'nearest')
        feat_recon = self.backbone(img_recon)
        
        print("Cosine Similarity: ", F.cosine_similarity(feat, feat_recon).mean())
        plt.subplot(1,2,1)
        plt.imshow(ten2img(img[0]))
        plt.axis("off")
        
        plt.subplot(1,2,2)
        plt.imshow(ten2img(img_recon[0]))
        plt.axis("off")
        plt.show()
        self.nbnet.train() 