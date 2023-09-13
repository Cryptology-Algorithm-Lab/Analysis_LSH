import torch
import torch.nn.functional as F

### Attack Method with Conjugate Gradient Method
### Only suitable for GRP or ABH
from utils import bin2ten, solver

def attack_grp_efficient(GRP, template):
    hashed_value, matrices = template
    m,q,emb_size = matrices.shape
    device = matrices.device
    
    target_matrix = torch.zeros((m*(q-1), emb_size), device = device)
    hashed_value = hashed_value.to(device)
    
    for i in range(m):
        idx = hashed_value[0,i].to(torch.long)
        widx = matrices[i, idx, :]
        int_idx = 0
        for j in range(q):
            if idx==j:
                continue
            else:
                target_matrix[i*(q-1)+int_idx,:] = widx - matrices[i,j,:]
                int_idx += 1
    
    recon_x = None
    while recon_x == None:
        recon_x = solver(target_matrix, target_matrix.shape[0])
    
    return recon_x.T


def attack_abh_efficient(ABH, template):
    hashed_value, matrices = template
    p = ABH.p
    q = ABH.q
    r = ABH.r
    d = ABH.d
    device = ABH.device
    n = ABH.n
    hashed_value = hashed_value.to(torch.long)[0]
    mask = 2**torch.arange(r)
    hashed_value = hashed_value.unsqueeze(-1).bitwise_and(mask).ne(0)
    sgn = hashed_value.reshape(p, q*r).view(1, -1) * 2 - 1
    target_matrix = matrices * sgn.T
    
    recon_x = None
    while recon_x == None:
        recon_x = solver(target_matrix, target_matrix.shape[0])

    return recon_x.T

### Attack Method with Newton's Method
def attack_LSH(template, loss_fn):
    hashed_value, helper_data = template
    emb_size = helper_data.size(-1)
    device = helper_data.device

    # Starting Point
    x = F.normalize(torch.randn((1, emb_size)))
    x = x.requires_grad_(True).to(device)

    for idx in range(50):
        # Construct the objective function using LSP reduction.
        loss = loss_fn(template, x)

        grad= torch.autograd.grad(outputs = loss,
                                  inputs = x,
                                  grad_outputs = torch.ones(loss.shape, device=device),
                                  create_graph = True,
                                  retain_graph = True,
                                )[0]

        if grad.norm() == 0:
            find = True
            break
        
        ## Simple Newton's Method
        with torch.no_grad():
            x = x - grad / (grad.norm()**2+1e-8)*loss
        x.requires_grad = True
    return x


### Attack Method with Newton's Method including Restarting (Useful for breaking URP-IoM)
def attack_LSH_anom(template, loss_fn, k,  base_grad = 0.001):
    hashed_value, helper_data = template
    emb_size = helper_data.size(-1)
    device = helper_data.device
    find = False
    
    while not find:
        x = F.normalize(torch.randn((1, emb_size)))
        x = x.requires_grad_(True).to(device)

        for idx in range(50):
            loss = loss_fn(template, x, k)

            grad= torch.autograd.grad(outputs = loss,
                                      inputs = x,
                                      grad_outputs = torch.ones(loss.shape, device=device),
                                      create_graph = True,
                                      retain_graph = True,
                                    )[0]

            if grad.norm() == 0:
                find = True
                break

            with torch.no_grad():
                x = x - grad / (grad.norm()**2+1e-8)*loss
            x.requires_grad = True

            
        if grad.norm() < base_grad:
            find = True
        else:
            print("Restarting...", grad.norm())
    return x

# LSP reduction for GRP
def loss_fn_grp(template, x, **kwargs):
    hashed_value, matrices = template
    m,q,emb_size = matrices.shape
    device = x.device
    target_matrix = torch.zeros((m*(q-1), emb_size), device = device)
    
    for i in range(m):
        idx = hashed_value[0,i].to(torch.long)
        widx = matrices[i, idx, :]
        int_idx = 0
        for j in range(q):
            if idx==j:
                continue
            else:
                target_matrix[i*(q-1)+int_idx,:] = widx - matrices[i,j,:]
                int_idx += 1

    mat = torch.mm(x, target_matrix.T)
    
    return (mat-mat.abs()).abs().mean()

# LSP reduction for URP
def loss_fn_urp(template, x, k):
    hashed_value, perms = template
    m,p,emb_size = perms.shape
    device = x.device
    loss = torch.zeros(1).to(device)
    mask = (torch.arange(0,emb_size)<k).to(device)
    
    for i in range(m):
        curr_idx = hashed_value[:,i].to(torch.long)
        out = torch.ones((1,emb_size)).to(device)
        
        for j in range(p):
            curr_perm = perms[i,j,:]            
            out *= x[:, curr_perm.to(torch.long)]
            
        sol = torch.gather(out, dim=1, index = curr_idx.unsqueeze(0)).expand_as(out)
        rel = (sol-out) * mask
        
        loss += (rel.abs() - rel).abs().mean()
    
    return loss

# LSP reduction for ABH
def loss_fn_abh(template, x, **kwargs):
    hashed_value, matrices = template
    s,u = hashed_value.shape
    n, emb_size = matrices.shape
    b = n//(s*u)
    device = hashed_value.device

    temp = torch.zeros((s, u*b))
    target = torch.zeros((s*u*b, emb_size))
    
    
    for idx_s in range(s):
        curr_row = hashed_value[idx_s, :]
        for idx_u in range(u):
            temp[idx_s, idx_u*b:idx_u*b+b] = bin2ten(hashed_value[idx_s, idx_u], b)
    
    temps = temp.view(1, n)
    sgn = temps * 2 - 1
    
    target_matrix = matrices * sgn.T
    mat = torch.mm(x, target_matrix.T)
    
    return (mat-mat.abs()).abs().mean()


### Implementation of Genetic Algorithm-based Attack
from tqdm import tqdm

def attack_GA(LSH, template, n_sample = 200):
    start = F.normalize(torch.randn((1, LSH.d)))
    best_score = LSH.verify(start, template)
    
    # No. of Generation: 1000
    for _ in tqdm(range(1000)):
        # Mutation / No. of siblings: 200
        z = torch.randn(n_sample, LSH.d) * 0.1
        start_new = z + start
        
        # Scoring
        scores = LSH.verify(start_new, template)
        
        # Selection
        max_score = scores.max()
        max_pos = scores.argmax()
        start = start_new[max_pos:max_pos + 1, :]
    return start



### Implementation of Ghammam et al.'s Attack
### Using CVXOPT Library

import numpy as np
import cvxopt
from cvxopt import matrix

def attack_urp_efficient(URP, template):
    hashed_value, matrices = template
    m, p, emb_size = matrices.shape
    k = URP.w
    device = matrices.device
    
    target_matrix = torch.zeros((m * (k-1), emb_size), device = device)
    hashed_value = hashed_value.to(device)
    
    for i in range(m):
        idx = hashed_value[0, i].to(torch.long)
        perms = matrices[i]
        widx = perms[:, idx].reshape(-1).to(torch.long)
        int_idx = 0
        
        for j in range(k):
            if j==idx:
                continue
            else:
                # Constructing the constraints system as Ghammam's method
                widx_low = perms[:, j].reshape(-1).to(torch.long)        
                target_matrix[i * (k-1) + int_idx, widx[0] ] -=1
                target_matrix[i * (k-1) + int_idx, widx[1] ] -= 1
                target_matrix[i * (k-1)+ int_idx, widx_low[0] ] += 1
                target_matrix[i * (k-1)+ int_idx, widx_low[1] ] += 1
                int_idx += 1
        
    t_size = target_matrix.size(0)
    target_matrix = target_matrix.numpy().astype(float)
    target_matrix = matrix(target_matrix)
        
    # We used Quadratic Programming algorithm
    P = matrix(np.zeros((512, 512)))
    q = matrix(np.zeros(512))
    G = target_matrix
    h = matrix(-np.random.rand(t_size) * 0.1)
    outr = cvxopt.solvers.qp(P, q, G,h)

    res = torch.Tensor(np.array(outr['x'])).T
    res = torch.exp(res)

    print("CODE: ", URP.verify(res, template))
    return res
