import torch
import torch.nn as nn
import torch.nn.functional as F

class DIoMNetwork(nn.Module):
    def __init__(self, m, q, emb_size, l):
        super().__init__()
        self.m = m
        self.q = q
        self.emb_size = emb_size
        self.l = l        
        self.lin1 = nn.Linear(emb_size, l)
        self.lin2 = nn.Linear(l, l)
        self.lin3 = nn.Linear(l, m * q, bias = False)
        self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.lin1(x)       
        x = self.act(x)
        x = self.lin2(x)
        x = self.act(x)
        x = self.lin3(x)
        return x
        
# Loss functions for DIoM
# Helper: Softmaxout
def softmaxout(x, m, q, beta=9):
    # Assume that x is a batch of feature vectors of size (B, m*q).
    B = x.size(0)
    x = x.reshape(-1, m, q)
    x = F.softmax(x * beta, dim = 2)
    indices = torch.arange(1, q+1).to(x.device).reshape(1, q, 1).expand(B, q, 1).float()
    # (B, m)
    x = torch.bmm(x, indices).reshape(B,-1)
    return x


def pairwiseCE_b(network, z, s, m, q, balance = True):
    x = network(z)
    h = softmaxout(x, m, q)    
    s = s.float()
    h_l, h_r = h.chunk(2, dim = 0)
    pairwise_cos = F.cosine_similarity(h_l, h_r)
    
    loss = (torch.log(1 + torch.exp(pairwise_cos)) - s * pairwise_cos).sum()

    if balance:        
        b_loss = (h.mean(dim=1) - (q+1)/2).abs()
        loss = loss + (0.01) * b_loss.sum()        
    return loss

### logger

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

from torch.optim.lr_scheduler import _LRScheduler


class PolyScheduler(_LRScheduler):
    def __init__(self, optimizer, base_lr, max_steps, warmup_steps, last_epoch=-1):
        self.base_lr = base_lr
        self.warmup_lr_init = 0.0001
        self.max_steps: int = max_steps
        self.warmup_steps: int = warmup_steps
        self.power = 2
        super(PolyScheduler, self).__init__(optimizer, -1, False)
        self.last_epoch = last_epoch

    def get_warmup_lr(self):
        alpha = float(self.last_epoch) / float(self.warmup_steps)
        return [self.base_lr * alpha for _ in self.optimizer.param_groups]

    def get_lr(self):
        if self.last_epoch == -1:
            return [self.warmup_lr_init for _ in self.optimizer.param_groups]
        if self.last_epoch < self.warmup_steps:
            return self.get_warmup_lr()
        else:
            alpha = pow(
                1
                - float(self.last_epoch - self.warmup_steps)
                / float(self.max_steps - self.warmup_steps),
                self.power,
            )
            return [self.base_lr * alpha for _ in self.optimizer.param_groups]

def train(cfg, device):
    logger = Logger(cfg.log_dir)
    
    net = DIoMNetwork(cfg.m, cfg.q, 512, cfg.l)
    net = net.train().to(device)
        
    total_step = cfg.epoch * 8000 // cfg.batch_size
    opt = torch.optim.SGD(net.parameters(), lr = 5e-4, momentum=0.8, weight_decay=0.005)
    curr_step = 0
    
    logger.timer_start()
    
    lr_scheduler = PolyScheduler(
    optimizer=opt,
    base_lr=cfg.lr,
    max_steps = total_step,
    warmup_steps=0,
    last_epoch=-1
    )
    
    print(cfg)
    
    batch_size = cfg.batch_size
    
    feat_lfw = cfg.feat_lfw.to(device)
    left = feat_lfw[::2][:4200]
    right = feat_lfw[1::2][:4200]
    len_pairs = 6000
    stepsize = len_pairs // 20
    issame = []
    for i in range(len_pairs):
        if (i//300) % 2 == 0:
            issame += [True]
        else:
            issame += [False]
    
    issame = torch.BoolTensor(issame).int().to(device)

    print("Train start!")
    for _ in range(cfg.epoch):
        ba = 0
        ind_set = torch.randperm(4200)
        while (ba < left.size(0)):
            bb = min(left.size(0), ba+batch_size//2)
            ind = ind_set[ba:bb]
            l = left[ind]
            r = right[ind]
            s = issame[ind]
            feat = torch.cat((l,r), dim = 0)
            loss = pairwiseCE_b(net, feat, s, cfg.m, cfg.q, balance = cfg.balance)
            opt.zero_grad()
            loss.backward()
            opt.step()  
#             lr_scheduler.step()
            ba = bb
            curr_step += 1
                        
            if curr_step % cfg.verbose == 0:
                elapsed = logger.timer_elapsed()
                required = (total_step - curr_step) * (elapsed / cfg.verbose) / 3600                
                payload = "[{}/{}] Loss: {:.4f}, Required: {:.3f}".format(curr_step, total_step, loss.item(), required)
                print(payload, "lr : {:.4f}".format(lr_scheduler.get_last_lr()[0]))
                logger.write(payload)
    torch.save(net, "./DIOM/diom.pt")
    
    return net