import torch
from torch import nn
import torch.nn.functional as F

# Abstract LSH-based BTP
# Will be instantiated.
class LSH:
    def __init__(self):
        pass
    
    def setup(self):
        NotImplemented
    
    def hashing(self, x, R):
        NotImplemented
    
    def score(self, y, template):
        NotImplemented
        
    def lsp(self, template):
        NotImplemented
        
# Implementation of BioHashing
# Reference: [ADA04] Biohashing: two factor authentication featuring fingerprint data and tokenised random number
class BH(LSH):
    def __init__(self, m, th, emb_size, device="cpu"):
        super().__init__()
        self.m = m
        self.th = th
        self.emb_size = emb_size
        self.device = device
        
    def setup(self):
        R_c = torch.randn(self.emb_size, self.emb_size)
        svd = torch.linalg.svd(R_c)
        R = (svd[0] @ svd[2])[:self.m]
        return R.to(self.device)
    
    def hashing(self, x, R = None):
        if R == None:
            R = self.setup()
        x = x.to(self.device)
        code = torch.mm(x, R.T)
        code = torch.where(code>self.th ,1, 0)
        template = (code, R)
        return template
    
    def score(self, y, template):
        code, R = template
        code = code.to(self.device)
        code_y, _ = self.hashing(y,R)
        return (code == code_y).sum(dim=1)
    
    def lsp(self, template):
        code, R = template
        def BH_lsp(x):
            return (-1)**(1-code) * torch.mm(x, R.T)
        return BH_lsp
    
# Implementation of GRP-IoM
# Reference: [JLH+17] Ranking Based Locality Sensitive Hashing Enabled Cancelable Biometrics: Index-of-Max Hashing
class GRP(LSH):
    def __init__(self, m, q, emb_size, device="cpu"):
        super().__init__()
        self.m = m
        self.q = q
        self.emb_size = emb_size
        self.device = device
    
    def setup(self):
        R = torch.randn((self.m * self.q, self.emb_size), device = self.device)
        return R.to(self.device)
           
    def hashing(self, x, R = None):
        if R == None:
            R = self.setup()
        x = x.to(self.device)
        R = R.to(self.device)
        code = torch.mm(x, R.T)
        code = code.reshape(-1, self.m, self.q)
        code = code.argmax(dim = -1)
        template = (code, R)
        return template
    
    def score(self, y, template):
        code, R = template
        code = code.to(self.device)
        R = R.to(self.device)
        code_y, _ = self.hashing(y, R)
        return (code == code_y).sum(dim=-1)
    
    def lsp(self, template):
        code, R = template
        code = code.to(self.device)
        def GRP_lsp(x):                
            temp = torch.mm(x.to(self.device), R.T).reshape(-1, self.m, self.q)
            value = temp.gather(-1, code.unsqueeze(-1)).reshape(-1, self.m,1).expand(-1, self.m, self.q)
            return value - temp         
        return GRP_lsp
    
# Implemenation of URP-IoM
# Reference: [JLH+17] Ranking Based Locality Sensitive Hashing Enabled Cancelable Biometrics: Index-of-Max Hashing
class URP(LSH):
    def __init__(self, m, w, p, emb_size, device="cpu"):
        super().__init__()
        self.m = m
        self.w = w
        self.p = p
        self.emb_size = emb_size
        self.device = device
        
    def setup(self):
        R = torch.randn(self.m * self.p, self.emb_size).argsort()
        return R.to(self.device)
    
    def hashing(self, x, R=None):
        if R == None:
            R = self.setup()
        R = R.to(self.device)    
        x = x.to(self.device)
        code = x[:, R].reshape(-1, self.m, self.p, self.emb_size)
        code = code.prod(dim = 2)[:,:,:self.w].argmax(2)
        template = (code, R)
        return template
    
    def score(self, y, template):
        code, R = template
        code = code.to(self.device)
        code_y, _ = self.hashing(y, R)
        return (code == code_y).sum(dim=-1)
    
    def lsp(self, template):
        code, R = template
        code = code.to(self.device)
        R = R.to(self.device)
        def URP_lsp(x):
            temp = x[:,R].reshape(-1, self.m, self.p, self.emb_size)
            temp = temp.prod(dim = 2)[:,:,:self.w]
            value = temp.gather(-1, code.unsqueeze(-1)).reshape(-1, self.m, 1).expand(-1, self.m, self.w)
            return (value - temp)
        return URP_lsp
    
# Implementation of DIoM
# Reference: [CT20] Deep index-of-maximum hashing for face template protection.
class DIoM(LSH):
    def __init__(self, m, q, emb_size, pre_NN, device="cpu"):
        super().__init__()
        self.m = m
        self.q = q
        self.emb_size = emb_size
        self.pre_NN = pre_NN.to(device)
        self.device= device
        
    def setup(self):
        R = torch.randperm(self.emb_size, device = self.device)
        return R.to(torch.long).to(self.device)
    
    def softmaxout(self, x, beta=1):
        # Assume that x is a batch of feature vectors of size (B, m*q).
        B = x.size(0)
        x = x.reshape(-1, self.m, self.q)
        x = F.softmax(x * beta, dim = 2)
        indices = torch.arange(1, self.q+1).to(x.device).reshape(1, self.q, 1).expand(B, self.q, 1).float()
        # (B, m)
        x = torch.bmm(x, indices).squeeze(-1)
        return x
    
    def hashing(self, x, R=None):
        if R == None:
            R = self.setup()  
        R = R.to(self.device)    
        x = x.to(self.device)    
        code = x[:, R]
        code = self.pre_NN(code)
        code = code.reshape(-1, self.m, self.q)
        code = code.argmax(2)+1
        template = (code, R)
        return template
    
    def score(self, y, template):
        code, R = template
        code = code.to(self.device)
        code_y, _ = self.hashing(y, R)
        return (code == code_y).sum(dim=-1)
    
    def lsp(self, template):
        code, R = template
        code = code.to(self.device)
        R = R.to(self.device)
        def DIoM_lsp(x):
            x = x.to(self.device)
            temp = self.pre_NN(x[:, R]).reshape(-1, self.m, self.q)
            value = temp.gather(-1, (code-1).unsqueeze(-1)).reshape(-1, self.m, 1).expand(-1, self.m, self.q)
            return value - temp
        return DIoM_lsp
    
# Hashing Network for DIoM
# according to paper, l = 1024, m = 512, q = 30
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
def softmaxout(x, m, q, beta):
    # Assume that x is a batch of feature vectors of size (B, m*q).
    x = x.reshape(-1, m, q)
    x = F.softmax(x * beta, dim = 2)
    indices = torch.arange(q).to(x.device).reshape(1, q, 1).expand(B, q, 1)
    # (B, m)
    x = torch.bmm(x, indices).squeeze(-1)
    return x
    
def pairwiseCE(network, z, m, q):
    x = network(z)
    h = softmaxout(x, m, q)
    h_hat = F.normalize(h, dim = 1)    
    s = 1 - F.normalize(z, dim = 1) @ F.normalize(z, dim = 1).T
    pairwise_cos = 1 - h_hat @ h_hat.T
    
# Implementation of ABH
# Referecnce: [LJWT21] Efficient known-sample attack for distance-preserving hashing biometric template protection schemes.
class ABH(LSH):
    def __init__(self, s, u, b, th, emb_size, device="cpu"):
        super().__init__()
        self.s = s
        self.u = u
        self.b = b
        self.n = s * u * b
        self.th = th
        self.emb_size = emb_size
        self.device = device
        
        
    def setup(self):
        R = torch.randn(self.n, self.emb_size)
        return R.to(self.device)
    
    def hashing(self, x, R=None):
        if R == None:
            R = self.setup()
        R = R.to(self.device)    
        x = x.to(self.device)
        code = torch.where((x @ R.T)>0, 1, 0).reshape(x.size(0), -1, self.b).float().to(self.device)
        dec = torch.Tensor([2**(self.b-1-i) for i in range(self.b)]).to(self.device)
        code = (code @ dec.T).reshape(-1, self.s, self.u)
        template = (code, R)
        return template
    
    def score(self, y, template):
        code, R = template
        code = code.to(self.device)
        code_y, _ = self.hashing(y, R)
        return ((code == code_y).sum(dim=-1) > self.th).sum(dim = -1)
    
    def lsp(self, template):
        code, R = template
        code = code.to(self.device)
        R = R.to(self.device)
        def ABH_lsp(x):
            x = x.to(self.device)
            temp = x @ R.T
            value = torch.zeros(x.size(0), self.b, self.s * self.u).to(self.device)
            dec = code.reshape(code.size(0), -1)
            for i in range(self.b):
                value[:, self.b-1-i] = dec % 2
                dec = dec // 2
            value = value.transpose(1,2).reshape(x.size(0), -1)
            return (-1)**(1-value) * temp
        return ABH_lsp

    
# Implementation of Jiang et al.   
# Reference: [JSZ+23] Cancelable biometric schemes for Euclidean metric and Cosine metric
class Jiang(LSH):
    def __init__(self, l, d, M, emb_size, device="cpu"):
        super().__init__()
        self.l = l
        self.d = d
        self.M = M
        self.emb_size = emb_size
        self.device = device
        
    def setup(self):
        w = torch.randn(self.l*self.d, self.emb_size).to(self.device)
#         u_max = 2**self.d-1
        u_max = self.M
        a = torch.randint(1, u_max, (1, self.l)).to(self.device)
        b = torch.randint(1, u_max, (1, self.l)).to(self.device) 
        R = (w, a, b)
        return R
    
    def hashing(self, x, R=None):
        if R == None:
            R = self.setup()
        w, a, b  = R
        w = w.to(self.device)
        a = a.to(self.device)
        b = b.to(self.device)
        x = x.to(self.device)
        code = torch.where((x @ w.T)>=0, 1, 0).reshape(-1, self.l, self.d).float().to(self.device)
        dec = torch.Tensor([2**i for i in range(self.d)]).to(self.device)
        code = code @ dec.T
        code = code * a + b
        code = code % self.M
        template = (code, R)
        return template
    
    def score(self, y, template):
        code, R = template
        code = code.to(self.device)
        code_y, _ = self.hashing(y, R)
        return (code == code_y).sum(dim=-1)
    
    def lsp(self, template):
        code, R = template
        code = code.to(self.device)
        w, a, b = R
        w = w.to(self.device)
        a = a.to(self.device)
        b = b.to(self.device)
        def inv(a,b):
            p=b
            s0, s1, t0, t1 = 1, 0, 0, 1
            while b >0:
                q, r = divmod(a, b)
                s0, s1 = s1, s0 - q * s1
                t0, t1 = t1, t0 - q * t1
                a, b = b, r

            return s0%p

        a_inv = torch.zeros_like(a)
        for i in range(a.size(1)):
            a_inv[0, i] = inv(a[0, i].item(), self.M)
       
        def Jiang_lsp(x):
            x = x.to(self.device)
            temp = (x @ w.T)
            code_inv = ((code - b) * a_inv) % self.M
            ret = torch.zeros(x.size(0), code_inv.size(1), self.d).to(self.device)
            for i in range(self.d):
                ret[:, :, i] = code_inv % 2
                code_inv //= 2
            ret = ret.reshape(x.size(0), -1)    
            return (-1)**(1-ret) * temp
        return Jiang_lsp    
    
# Implemenation of IMM
# Reference: [YLH+22] Indexing-Min–Max Hashing: Relaxing the Security–Performance Tradeoff for Cancelable Fingerprint Templates
from scipy.linalg import hadamard
class IMM(LSH):
    def __init__(self, m, r, emb_size, device="cpu"):
        super().__init__()
        self.m = m
        self.r = r
        self.emb_size = emb_size
        self.device = device
    
    def setup(self):
        H = torch.Tensor(hadamard(self.emb_size))
        R = torch.argsort(torch.rand(self.m, self.emb_size), dim=1)[:,:self.r]
        H_n = H[R]        
        return H_n.to(self.device)
        
    def hashing(self, x, H=None):
        if H == None:
            H = self.setup() 
        H = H.to(self.device)    
        mat = H.matmul(x.T).transpose(0,2).transpose(1,2)
        code = torch.cat((mat.argmax(dim =2), mat.argmin(dim =2)), dim = 1)
        template = (code, H)
        return template
    
    def score(self, y, template):
        code, R = template
        code = code.to(self.device)
        code_y, _ = self.hashing(y, R)
        return (code == code_y).sum(dim=-1)
    
    def lsp(self, template):
        code, R = template
        code = code.to(self.device)
        R = R.to(self.device)
        code_max = code[:,:code.size(1)//2]
        code_min = code[:,code.size(1)//2:]
        def IMM_lsp(x):
            x = x.to(self.device)
            temp = R.matmul(x.T).transpose(0,2).transpose(1,2)
            value_max = temp.gather(-1, code_max.unsqueeze(-1)).expand_as(temp)
            value_min = temp.gather(-1, code_min.unsqueeze(-1)).expand_as(temp)
            return torch.cat(((value_max - temp), (temp - value_min)), dim = 0)
        return IMM_lsp