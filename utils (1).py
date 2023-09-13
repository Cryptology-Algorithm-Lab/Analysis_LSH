import torch

### Utilities for ABH
def ten2bin(x):
    l = x.size(0)
    #print(torch.arange(l-1,-1,-1) * x)
    return (2**(torch.arange(l-1,-1, -1)) * x).sum().item()

def bin2ten(x, b):
    l = [0] * b
    for idx in range(b):
        l[idx] = x%2
        x//=2
        
    return torch.Tensor(l[::-1])
    
'''
Implementation of TBW

Reference: []
'''


def lin_minima(A,x,d,e):
    c = torch.matmul(A,d)
    b = torch.matmul(A,x) - e 
    mu = -b/c
    
    S1, S2 = [], []
    for i in range(c.shape[0]):
        if c[i].item()>0:
            S1.append(mu[i])
        else:
            S2.append(mu[i])
    
    S1 = torch.tensor(S1, device = x.device)
    S2 = torch.tensor(S2, device = x.device)
    
    mu_up = torch.max(S1)
    
    if len(S2) == 0:
        return mu_up+torch.rand(1).item()
    else:
        mu_dn = torch.min(S2)
        diff = mu_up - mu_dn
        
        if diff<0:
            return (mu_up + mu_dn)/2
        elif diff == 0: 
            return mu_up
        else:
            mu_hat = max(0, mu_dn)
            sum_mu, _ = torch.sort(torch.cat([S1,S2], dim=0), dim=0)
            low = (sum_mu >=mu_hat)
            high = (sum_mu <=mu_up)
            
            sum_id = (sum_mu*low*high != 0).nonzero().to(torch.long)
            sum_tg = sum_mu[sum_id]
            idx = -1
            
            while True:
                idx += 1
                mut = sum_tg[idx,:]
                phimu = torch.sum(c*(b+mut*c-torch.abs(b+mut*c))).item()
                if phimu<0:
                    mu_hat = mut
                    continue
                    
                elif phimu == 0:
                    return mut
                
                else:
                    mu_nohat = mut
                    I = []
                    for i in range(idx, sum_tg.shape[0]-1):
                        if (c[i].item()>0 and mu[i]>mu_hat) or \
                           (c[i].item()<0 and mu[i]<mu_nohat):
                            I.append(i)
                    return -(torch.sum(b[I]*c[I])/(torch.sum(c[I]*c[I]) + 1e-8)).item()

                
def solver(target_matrix, m):
    emb_size = target_matrix.shape[1]
    device = target_matrix.device
    r = 10
    # Step 1
    k =0

    x = torch.mean(target_matrix, dim = 0, keepdim = True).T
    ax = torch.mm(target_matrix,x)
    gamma = torch.sum(ax>0).item()
    start = m-gamma
    
    if 0<gamma<m/2:
        x = -x
        ax = -ax
        gamma = m-gamma
    
    
    while True:
        if k>100:
            return x
        e = torch.rand(1, device = device).clamp(0, 1)
        # Step 2
        if gamma == m:
            return x
        elif gamma == 0:
            return -x
        
        # Step 3
        g = torch.mm(target_matrix.T, torch.abs(ax-e) - (ax-e))
        
        if torch.sum(g!=0) == 0:
            break
        
        lambd = 1./(torch.linalg.norm(g).pow(2) + 1e-8)
        
        # Step 4
        if k%r == 0:
            theta = 1
        else:
            theta = 0
        
        v = -torch.mm(x.T,g)/(x.norm().pow(2) +1e-8)
        
        if k==0:
            d=0
        d = g+v*x        
        
        # Step 5
        mu = lin_minima(target_matrix, x, d, e)
    
        # Step 6
        x += mu * d
        ax += mu * torch.mm(target_matrix, d)
        gamma = torch.sum(ax>0).item()
    
        # Step 7
        k += 1
        #Goto step 2

        
                
        