import torch
from torch import nn
import torch.nn.functional as F


class GRP:
    def __init__(self, m, n, d, device):
        self.m = m
        self.n = n
        self.d = d
        self.device = device
        self.rand = self.generate_random()
        
    def generate_random(self):
        return torch.randn((self.m, self.n, self.d), device = self.device)
    
    def hashing(self, x, rand = None):
        if rand == None:
            rand = self.rand
        x = x.to(self.device).T.unsqueeze(0).expand(self.m, self.d, -1)
        hashed = torch.bmm(rand, x).argmax(dim=1).T
        return hashed, rand
    
    def verify(self, y, template):
        code, mat = template
        code_y, _ = self.hashing(y, mat)
        return (code == code_y).sum(dim=-1)
        
       
    
class URP:
    def __init__(self, n, w, p, d, device):
        self.n = n
        self.p = p
        self.w = w
        self.d = d
        self.device = device
        self.rand = self.generate_random()

    def generate_random(self):
        perms = torch.zeros((self.n, self.p, self.d), device = self.device)
        for i in range(self.n):
            for j in range(self.p):
                perms[i, j, :] = torch.randperm(self. d)
        return perms
    
    def hashing(self, x, perms = None):
        if perms == None:
            perms = self.rand
        hashed = torch.zeros((x.size(0), self.n), device = self.device)
        x = x.to(self.device)
        
        for i in range(self.n):
            prod = torch.ones(x.shape, device = self.device)
            for j in range(self.p):
                perm = perms[i, j, :]
                perm = perm.to(torch.long)
                prod = prod * x[:, perm]
            hashed[:, i] = torch.argmax(prod[:, :self.w])
        return hashed, perms
 
    def verify(self, y, template):
        code, mat = template
        code_y, _ = self.hashing(y, mat)
        return (code == code_y).sum(dim=-1)
    
class ABH:
    def __init__(self, p, q, r, th, d, device):
        self.p = p
        self.q = q
        self.r = r
        self.th = th
        self.n = p * q * r
        self.d = d
        self.device = device
        
        self.rand = self.generate_random()
        
    def generate_random(self):
        return torch.randn((self.n, self.d), device = self.device)
    
    def hashing(self, x, rand = None):
        if rand == None:
            rand = self.rand
        sgns = (torch.mm(x, rand.T) > 0).reshape(-1, self.p, self.q*self.r)
        hashed = torch.zeros((x.size(0), self.p, self.q), device = self.device)
        for i in range(self.r):
            hashed += sgns[:, :, i::self.r] * (2 ** i)
        
        return hashed, rand
            
    def verify(self, y, template):
        code, mat = template
        code_y, _ = self.hashing(y, mat)
        
        return ((code == code_y).sum(dim=-1) > self.th).sum(dim=-1)