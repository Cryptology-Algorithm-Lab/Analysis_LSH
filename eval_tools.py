import torch
from torch import nn
from LSH import *
import torch.nn.functional as F
from feats.utils_feat import feat_ext

@torch.no_grad()
def enroll(LSHs, lfw_feat, device, out_dir = "./template_sh/"):
    print("Run Enrollment...")
    # Preprocessing
    lfw_feat = lfw_feat.reshape(20, 600, 512).to(device)
    
    for name, LSH, _ in LSHs:
        print("Current LSH: ", name, end = "\t")
        code_name = out_dir + name + "_code.pt"
        R_name = out_dir + name + "_R.pt"
        
        mem_code = []
        mem_R = []
        
        if name == "Jiang":
            a_name = out_dir + name + "_a.pt"
            b_name = out_dir + name + "_b.pt"
            mem_a = []
            mem_b = []
        
        for i in range(20):
            code, helper = LSH.hashing(lfw_feat[i])
            mem_code.append(code.unsqueeze(0))
            if name == "Jiang":
                R, a, b = helper
                mem_R.append(R.unsqueeze(0))
                mem_a.append(a.unsqueeze(0))
                mem_b.append(b.unsqueeze(0))
                
            else:
                mem_R.append(helper.unsqueeze(0))
                
        mem_code = torch.cat(mem_code, dim = 0)
        mem_R = torch.cat(mem_R, dim = 0)

        if name == "Jiang":
            mem_a = torch.cat(mem_a, dim = 0)
            mem_b = torch.cat(mem_b, dim = 0)            
            
            torch.save(mem_code, code_name)
            torch.save(mem_R, R_name)
            torch.save(mem_a, a_name)
            torch.save(mem_b, b_name)
        else:
            torch.save(mem_code, code_name)
            torch.save(mem_R, R_name)
            
        print("Done!") 
        
        
# Verification
@torch.no_grad()
def verify(LSHs, lfw_feat, device, tmp_dir = "./template_sh/", target_far = None):
    print("Run Verification...")
    # Preprocessing
    lfw_feat = lfw_feat.reshape(20, 600, 512).to(device)
    
    issame = []
    
    for i in range(12000):
        if (i // 300) % 2 == 0:
            issame += [True]
        else:
            issame += [False]
    
    issame = torch.BoolTensor(issame).to(device)
    
    # Data Table
    ret = []
    
    for name, LSH, thx_upper in LSHs:
        print("Current LSH: ", name, end = "\t")
        code = torch.load(tmp_dir + name + "_code.pt")
        
        if name == "Jiang":
            R = torch.load(tmp_dir + name + "_R.pt")
            a = torch.load(tmp_dir + name + "_a.pt")
            b = torch.load(tmp_dir + name + "_b.pt")
        else:
            helper = torch.load(tmp_dir + name + "_R.pt")

        # Calculating Score
        # lr: Left-Enroll, Right-Query
        # rl: Left-Query, Right-Enroll
        match_scores_lr = []
        match_scores_rl = []
        for i in range(20):
            curr_code = code[i]
            curr_feat = lfw_feat[i]
                        
            # Split Left & Right
            left_code, right_code = curr_code[::2], curr_code[1::2]
            left_feat, right_feat = curr_feat[::2], curr_feat[1::2]
                        
        
            if name == "Jiang":
                t_l = (left_code, (R[i], a[i], b[i]))
                t_r = (right_code, (R[i], a[i], b[i]))
            else:
                t_l = (left_code, helper[i])
                t_r = (right_code, helper[i])
            
            score_lr = LSH.score(right_feat, t_l)
            score_rl = LSH.score(left_feat, t_r)   
            
            match_scores_lr.append(score_lr.float())
            match_scores_rl.append(score_rl.float())
                    
        # Scoring
        match_scores_lr = torch.cat(match_scores_lr, dim=0)
        match_scores_rl = torch.cat(match_scores_rl, dim=0)
        match_scores = torch.cat([match_scores_lr, match_scores_rl], dim = 0)
        
        # Find 
        best_tar, best_far, best_acc, best_thx = find_bests(match_scores, issame, thx_upper, target_far)
        
        
        print("Done!")
        print("TAR: {:.4f}".format(best_tar.item()))
        print("FAR: {:.4f}".format(best_far.item()))
        print("ACC: {:.4f}".format(best_acc.item()))
        print("THX: {:.4f}".format(best_thx.item(), end = "\n\n"))
        
        ret.append((name, best_tar, best_far, best_acc, best_thx))
        
    return ret


@torch.no_grad()
def find_bests(scores, issame, thx_upper, target_far):
    num_pairs = issame.size(0)
    
    best_tar = 0
    best_far = 1
    best_acc = 0
    best_thx = 0
    
    mem_acc = []
    mem_thx = []
    mem_tar = []
    mem_far = []    
    
    thxs = torch.linspace(0, thx_upper, thx_upper + 1)
    
    # Find BEST ACC or ACC/TAR at Target FAR
    for thx in thxs:
        pred =  scores > thx

        TA = (pred & issame).sum()
        FA = (pred & ~issame).sum()
        TR = (~pred & ~issame).sum()
        FR = (~pred & issame).sum()
        ACC = (TA + TR) / num_pairs

        if ACC > best_acc:
            best_acc = ACC
            best_tar = TA / (num_pairs / 2)
            best_far = FA / (num_pairs / 2)
            best_thx = thx
  
        if (target_far != None):
            mem_acc.append(ACC)
            mem_tar.append(TA / (num_pairs / 2))
            mem_far.append(FA / (num_pairs / 2))
            mem_thx.append(thx)

    if (target_far != None):
        far = torch.Tensor(mem_far)
        diff = far - target_far
        ind = diff.abs().argmin()
        return mem_tar[ind], mem_far[ind], mem_acc[ind], mem_thx[ind]        
    else:
        return best_tar, best_far, best_acc, best_thx

# if LSH == NONE, then it just runs the plain recognition algorithm
@torch.no_grad()
def eval_perf(feats, tau_upper, LSH = None, target_far = None, batch_size = 512):
    device = feats.device
    if LSH != None :
        LSH.device = device
    
    # Prepricessing for evaluation
    left, right = feats[::2], feats[1::2]
    len_pairs = left.size(0)
    stepsize = len_pairs // 20
    issame = []
    
    for i in range(len_pairs):
        if (i//stepsize) % 2 == 0:
            issame += [True]
        else:
            issame += [False]
    
    issame = torch.BoolTensor(issame).to(device)
    
    # Score Calculation
    if LSH != None:
        score = []
        # Evalaute LSHs
        # Left: Enroll, Right: Verify
        # TODO: Batched Evaluation?
        ba = 0
        while (ba < left.size(0)):
            bb = min(left.size(0), ba+batch_size)
            blk_l = left[ba:bb]
            blk_r = right[ba:bb]
            temp_l = LSH.hashing(blk_l, None)
            s = LSH.score(blk_r, temp_l)
            score.append(s)
            ba = bb
        score = torch.cat(score)    
        
        taus = torch.linspace(0, tau_upper, 1000)
            
    else:
        # Just Cosine similarity
        score = F.cosine_similarity(left, right)
        taus = torch.linspace(-1, 1, 1000)
#         pred = score > tau
    
    best_acc = 0
    best_tau = 0
    best_tar = 0
    best_far = 0
    
    acc = []
    thx = []
    tar = []
    far = []
    
    for tau in taus:
        pred = score > tau
    
        # Evaluation
        TA = (pred & issame).sum()
        TR = (~pred & ~issame).sum()
        FA = (pred & ~issame).sum()
        FR = (~pred & issame).sum()

        # Eval Criteria
        TAR = TA / (len_pairs / 2)
        FAR = FA / (len_pairs / 2)
        ACC = (TA + TR) / len_pairs
        
        if ACC > best_acc:
            best_acc = ACC
            best_tar = TAR
            best_far = FAR
            best_tau = tau
        
        if (far != None):
            acc.append(ACC)
            tar.append(TAR)
            far.append(FAR)
            thx.append(tau)
            
    if (target_far != None):
        far = torch.Tensor(far)
        diff = far - target_far
        ind = diff.abs().argmin()
        return tar[ind], far[ind], acc[ind], thx[ind]    
    else:        
        return best_tar, best_far, best_acc, best_tau

@torch.no_grad()
def eval_bbs(backbones, dataset, device, target_far = 1e-3):
    result = []
    for bb, backbone_name in backbones :
        feat = feat_ext(dataset, bb, 512, device)
        tar, far, acc, tau = eval_perf(feat, 1, target_far = 1e-3)
        print("Backbone Name:", backbone_name)
        print("TAR:{:.4f}".format(tar.item()), end="\t")
        print("FAR:{:.4f}".format(far.item()))
        print("ACC:{:.4f}".format(acc.item()), end="\t")
        print("TAU:{:.4f}".format(tau.item()))
        print("==========================================")
        tmp = (backbone_name, bb, tau.item())
        result.append(tmp)
    return result 