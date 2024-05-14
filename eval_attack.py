import torch
from torch import nn
import torch.nn.functional as F

### ATTACKS
from tqdm import tqdm

# Proposed Attack with Newton's Alg

def Newton(LSPs, batch_size, device, init=None, l_type = "s", mode = "abs", tol = 1e-3, n_iter = 50, emb_size = 512):
    find = False
    restart = 10
    
    # 10 Restarts
    while not find:
        if (l_type == "s") and (restart < 0):
            l_type = "m"
        
        # Initialization Trick
        if init != None :
            x = init.expand(batch_size, emb_size)
        else:
            x = F.normalize(torch.randn(batch_size, emb_size))
        
        x = x.requires_grad_(True).to(device)
        
        # Newton Method Itereation
        for _ in range(n_iter):
            G_x = (LSPs(x) - LSPs(x).abs())
            if mode == "pow":
                G_x = G_x.pow(2)
            elif mode == "abs":
                G_x = G_x.abs()
            else:
                print("Invalid Mode: ", mode, " Quitting...")
                return None
            
            if l_type == "s":
                loss = G_x.mean(dim=0).sum()
            else:
                loss = G_x.mean(dim=0).mean()
                
            grad= torch.autograd.grad(
                outputs = loss, inputs = x, create_graph = True, retain_graph = True,
            )[0]
            
            if grad.norm() == 0:
                find = True
                break
            
            with torch.no_grad():
                # Numerical
                x = x - grad / (grad.norm() ** 2 + 1e-8) * loss

            x.requires_grad_(True).to(device)
            
        if grad.norm() < tol or init != None:
            find = True
        else:
            restart -= 1
            continue
    return x.detach().cpu()

def Recover_pre(LSHs, lfw_feat, atk_configs, device, tmp_dir = "./template/", out_dir = "./Pre-image/"):
    print("Pre-Image Attack: Proposed")
    # Preprocessing 
    lfw_feat = lfw_feat.reshape(20, 600, 512)
    
    ret = []
    
    for name, LSH, _ in LSHs:
        print("Current LSH: ", name, end = "\t")
        code = torch.load(tmp_dir + name + "_code.pt").to(device)
        
        if name == "Jiang":
            R = torch.load(tmp_dir + name + "_R.pt").to(device)
            a = torch.load(tmp_dir + name + "_a.pt").to(device)
            b = torch.load(tmp_dir + name + "_b.pt").to(device)
        else:
            helper = torch.load(tmp_dir + name + "_R.pt").to(device)
                    
        preimgs = []
        scores = []
        purities = []
        
        for i in tqdm(range(20)):
            if name == "Jiang":
                t = (code[i], (R[i], a[i], b[i]))
            else:
                t = (code[i], helper[i])
            
            batch_size = code[i].size(0)
            init = None
            
            is_init, l_type, mode = atk_configs[name]
            
            # Initialization
            if is_init:
                if name == "DIoM":
                    init = F.normalize(torch.randn((600, 512), device = device)) * 1e-3
                    
                elif name == "Jiang":
                    def inv(a,b):
                        p=b
                        s0, s1, t0, t1 = 1, 0, 0, 1
                        while b >0:
                            q, r = divmod(a, b)
                            s0, s1 = s1, s0 - q * s1
                            t0, t1 = t1, t0 - q * t1
                            a, b = b, r

                        return s0%p                    
                    a_inv = torch.zeros_like(a[i])
                                        
                    for j in range(a[i].size(1)):
                        a_inv[0, j] = inv(a[i][0][j].item(), LSH.M)
                    code_inv = (code[i] - b[i]) * a_inv 
                    code_inv = code_inv % LSH.M
                    
                    init = torch.zeros((600, 10000), device = device)
                    
                    init[:,  ::5] = (code[i] // 16) % 2
                    init[:, 1::5] = (code[i] // 8) % 2
                    init[:, 2::5] = (code[i] // 4) % 2
                    init[:, 3::5] = (code[i] // 2) % 2
                    init[:, 4::5] = code[i] % 2
                    
                    init = (init.float() * 2 - 1)
                    init = (init @ F.normalize(R[i]))                    
                    
                elif name == "BH":
                    init = ((2 * code[i].float() - 1) @ F.normalize(helper[i]))                    
                elif name == "GRP":
                    helper_r = helper[i].reshape(LSH.m, LSH.q, LSH.emb_size)
                    init_tmp = []
                    for j in range(600):
                        tmpp = []
                        for k in range(300):
                            tmpp.append(helper_r[k][code[i][j][k]].reshape(1, 1, 512))
                        tmp = torch.cat(tmpp, dim = 0)
                        init_tmp.append(F.normalize((tmp - helper_r), dim = 2).mean(dim=[0, 1]).reshape(1, LSH.emb_size))
                    
                    init = torch.cat(init_tmp, dim = 0)
                    
                    
                elif name == "IMM":
                    helper_r = helper[i].reshape(LSH.m, LSH.r, LSH.emb_size)
                    code_max, code_min = code[i].chunk(2, dim = -1)
                    init_tmp = []
                    for j in range(600):
                        tmpp1 = []
                        for k in range(300):
                            tmpp1.append(helper_r[k][code_max[j][k]].reshape(1, 1, 512))
                        
                        tmpp2 = []
                        for k in range(300):
                            tmpp2.append(helper_r[k][code_min[j][k]].reshape(1, 1, 512))
                            
                        tmp1 = torch.cat(tmpp1, dim = 0)
                        tmp2 = torch.cat(tmpp2, dim = 0)
                        rr = torch.cat([(tmp1 - helper_r), (helper_r - tmp2)], dim = 0).mean(dim=[0,1]).reshape(1, 512)
                        init_tmp.append(rr)
                        
                    init = torch.cat(init_tmp, dim = 0)
                    
                elif name == "ABH":
                    R = helper[i]
                    code[i] 
                    init = torch.zeros((600, 50, 120), device = device)
                    init[:, :, ::2] = (code[i] // 2) % 2
                    init[:, :, 1::2] = code[i] % 2
                    init = (init.reshape(600, -1).float() * 2 - 1)
                    init = (init @ F.normalize(R))
                
                else:
                    init = F.normalize(helper[i].mean(dim = 0, keepdim = True))                    
            
            if name == "URP":
                sol = []
                score = []
                purity = []
                for j in range(600):
                    t = (code[i][j:j+1], helper[i])
                    LSPs = LSH.lsp(t)
                    tmp_sol = Newton(LSPs, 1, device, init, l_type, mode).to(device)
                    
                    tmp_score = LSH.score(tmp_sol, t)
                    tmp_purity = F.cosine_similarity(tmp_sol, lfw_feat[i][j:j+1])
                    
                    if tmp_purity < 0:
                        tmp_sol = -tmp_sol
                        tmp_purity = -tmp_purity
                    
                    sol.append(tmp_sol)
                    score.append(tmp_score)
                    purity.append(tmp_purity)
                                                    
                sol = torch.cat(sol, dim = 0)
                score = torch.cat(score, dim = 0)
                purity = torch.cat(purity, dim = 0)
            
            else:
                LSPs = LSH.lsp(t)
                sol = Newton(LSPs, batch_size, device, init, l_type, mode).to(device)

                # matching score
                score = LSH.score(sol, t)
                purity = F.cosine_similarity(sol, lfw_feat[i])
            

            preimgs.append(sol)
            scores.append(score.float())
            purities.append(purity)
        
        
        
        preimgs = torch.cat(preimgs, dim=0)
        scores = torch.cat(scores, dim = 0)
        purities = torch.cat(purities, dim = 0)
        
        
        avg_score = scores.mean()
        purity = purities.mean()
        
        torch.save(preimgs, out_dir + name + ".pt")
        
        ret.append((name, avg_score, purity))
        
        print("Done!")
        print("Avg. Score: ", avg_score)
        print("Purity: ", purity, end = "\n\n")
        
        torch.cuda.empty_cache()
        
    return ret

# Genetic Algorithm-based Attacks
@torch.no_grad()
def GenAttack(LSH, template, device, n_group = 200, n_iter = 1000):
    start = F.normalize(torch.randn((1, LSH.emb_size), device = device))
    best_score = LSH.score(start, template)
    
    for _ in range(1000):
        z = torch.randn((n_group, LSH.emb_size), device = device) * 0.1
        start_new = z + start
        scores = LSH.score(start_new, template)
        
        max_score = scores.max()
        max_pos = scores.argmax()
        
        if max_score > best_score:
            start = start_new[max_pos:max_pos + 1, :]
            best_score = max_score
            
    return start
    
@torch.no_grad()
def Recover_pre_GEN(LSHs, lfw_feat, atk_configs, device, tmp_dir = "./template/", out_dir = "./Pre-image_GEN/", n_pairs = 100):
    print("Pre-Image Attack: GEN")
    # Preprocessing 
    # We will attack only 100 pairs from the positive pair
    rand_indices = torch.randperm(3000)[:100]    
    lfw_pos = lfw_feat.reshape(20, -1, 512)[::2].reshape(-1, 512)
    lfw_pos_l, lfw_pos_r = lfw_pos[::2], lfw_pos[1::2]
    lfw_tp = torch.cat([lfw_pos_l[rand_indices], lfw_pos_r[rand_indices]], dim = 0)

    # Save Rand Indices
    torch.save(rand_indices, out_dir + "indices.pt")    
    ret = []
    
    for name, LSH, _ in LSHs:
        print("Current LSH: ", name, end = "\t")
        code = torch.load(tmp_dir + name + "_code.pt")
        code_t = code[::2].reshape(-1, code.size(-1))
        code_l, code_r = code_t[::2], code_t[1::2]
        code_l_tp = code_l[rand_indices]
        code_r_tp = code_r[rand_indices]        
        code_tp = torch.cat([code_l_tp, code_r_tp], dim = 0)
        
        if name == "Jiang":
            R = torch.load(tmp_dir + name + "_R.pt")
            a = torch.load(tmp_dir + name + "_a.pt")
            b = torch.load(tmp_dir + name + "_b.pt")
            
            R_t = R[::2]
            a_t = a[::2]
            b_t = b[::2]            
            
            R_tp = R_t[rand_indices // 300]
            a_tp = a_t[rand_indices // 300]
            b_tp = b_t[rand_indices // 300]

            R_tp = torch.cat([R_tp, R_tp], dim = 0)
            a_tp = torch.cat([a_tp, a_tp], dim = 0)
            b_tp = torch.cat([b_tp, b_tp], dim = 0)            
            
        else:
            R = torch.load(tmp_dir + name + "_R.pt")
            R_t = R[::2]
            R_tp = R_t[rand_indices // 300]
            R_tp = torch.cat([R_tp, R_tp], dim = 0)
            
            
        preimgs = []
        scores = []
        purities = []
        
        for i in tqdm(range(200)):            
            if name == "Jiang":
                t = (code_tp[i], (R_tp[i], a_tp[i], b_tp[i]))
            else:
                t = (code_tp[i], R_tp[i])
            
            # Setting Up Genetic Algorithm
            sol = GenAttack(LSH, t, device)
            
            
            # matching score
            score = LSH.score(sol, t)
            purity = F.cosine_similarity(sol, lfw_tp[i])
            
            if name == "URP":
                if purity < 0:
                    sol = -sol
                    purity = -purity
            
            preimgs.append(sol)
            scores.append(score.float())
            purities.append(purity)
        
        
        
        preimgs = torch.cat(preimgs, dim=0)
        scores = torch.cat(scores, dim = 0)
        purities = torch.cat(purities, dim = 0)
        
        avg_score = scores.mean()
        purity = purities.mean()
        
        torch.save(preimgs, out_dir + name + ".pt")
        
        ret.append((name, avg_score, purity))
        
        print("Done!")
        
        print("Avg. Score: ", avg_score.item())
        print("Purity: ", purity.item(), end = "\n\n")
        
        torch.cuda.empty_cache()
    
    return ret

### Attack Evaluation
@torch.no_grad()
def get_att_feat(feats, backbone, nbnet, device, batch_size = 512):
    n_feats = feats.size(0)
    ba = 0
    ret = []
    while ba < n_feats:
        bb = min(ba + batch_size, n_feats)
        blk = F.normalize(feats[ba:bb])
        recon_img = F.interpolate(nbnet(blk), (112,112))
        recon_feat = F.normalize(backbone(recon_img) + backbone(recon_img.flip(3)))
        ret.append(recon_feat)
        ba = bb
    ret = torch.cat(ret, dim = 0)
    return ret
    

@torch.no_grad()
def eval_ASR(LSHs, thx_configs, lfw_feat, backbone, nbnet, device, tmp_dir = "./template/", preimg_dir = "./Pre-image/"):
    print("Evaluating ASRs...")
    
    ret = []
    
    for name, LSH, _ in LSHs:
        print("Current LSH: ", name, end = "\t")
        code = torch.load(tmp_dir + name + "_code.pt")
        a = None
        b = None
        if name == "Jiang":
            R = torch.load(tmp_dir + name + "_R.pt")
            a = torch.load(tmp_dir + name + "_a.pt")
            b = torch.load(tmp_dir + name + "_b.pt")
            
        else:
            R = torch.load(tmp_dir + name + "_R.pt")
     
        thx_lsh = thx_configs[name]
        thx_bb = thx_configs["backbone"]
    
        preimgs = torch.load(preimg_dir + name + ".pt")
        att_feat = get_att_feat(preimgs, backbone, nbnet, device)
        att_feat_1 = att_feat.reshape(20, -1 ,512)
        # Type-1 Attack
        asr1, scores = att_type1(LSH, thx_lsh, att_feat_1, name, code, R, a, b, is_gen = False)
        # Type-2 Attack
        asr2, angles = att_type2(lfw_feat, att_feat, thx_bb, is_gen = False)
        
        print("Done!")
        print("ASR1: {:.2f}%".format(asr1.item()*100))
        print("Avg. Score: {:.2f}".format(scores.item()))
        print("ASR2: {:.2f}%".format(asr2.item()*100))
        print("Avg. Angle: {:.2f}".format(angles.item()), end = "\n\n")
             
        ret.append((name, asr1, scores, asr2, angles))
    
    return ret

# Evaluation for Genetic Algorithm
@torch.no_grad()
def eval_ASR_GEN(LSHs, thx_configs, lfw_feat, backbone, nbnet, device, tmp_dir = "./template/", preimg_dir = "./Pre-image_GEN/"):
    print("Evaluating ASRs...")
    rand_indices = torch.load(preimg_dir + "indices.pt")
    left, right = lfw_feat[::2], lfw_feat[1::2]
    left_pos = left.reshape(20, -1, 512)[::2].reshape(-1, 512)
    right_pos = right.reshape(20, -1, 512)[::2].reshape(-1, 512)
    # Selected Pairs!
    left_sel, right_sel = left_pos[rand_indices], right_pos[rand_indices]
    ret = []
    ori_feat = [left_sel, right_sel]
    for name, LSH, _ in LSHs:
        print("Current LSH: ", name, end = "\t")
        code = torch.load(tmp_dir + name + "_code.pt")
        code_t = code[::2].reshape(-1, code.size(-1))
        code_l, code_r = code_t[::2], code_t[1::2]
        code_l_tp = code_l[rand_indices]
        code_r_tp = code_r[rand_indices]        
        code_tp = torch.cat([code_l_tp, code_r_tp], dim = 0)        
        thx_lsh = thx_configs[name]
        thx_bb = thx_configs["backbone"]
        a_tp = None
        b_tp = None
        if name == "Jiang":
            R = torch.load(tmp_dir + name + "_R.pt")
            a = torch.load(tmp_dir + name + "_a.pt")
            b = torch.load(tmp_dir + name + "_b.pt")
            
            R_t = R[::2]
            a_t = a[::2]
            b_t = b[::2]            
            
            R_tp = R_t[rand_indices // 300]
            a_tp = a_t[rand_indices // 300]
            b_tp = b_t[rand_indices // 300]

            R_tp = torch.cat([R_tp, R_tp], dim = 0)
            a_tp = torch.cat([a_tp, a_tp], dim = 0)
            b_tp = torch.cat([b_tp, b_tp], dim = 0)            
            
        else:
            R = torch.load(tmp_dir + name + "_R.pt")
            R_t = R[::2]
            R_tp = R_t[rand_indices // 300]
            R_tp = torch.cat([R_tp, R_tp], dim = 0)
    
        preimgs = torch.load(preimg_dir + name + ".pt")
        att_feat = get_att_feat(preimgs, backbone, nbnet, device)

        # Type-1 Attack
        asr1, scores = att_type1(LSH, thx_lsh, att_feat, name, code_tp, R_tp, a_tp, b_tp, is_gen = True)
        
        # Type-2 Attack
        asr2, angles = att_type2(ori_feat, att_feat, thx_bb, is_gen = True)

        print("Done!")
        print("ASR1: {:.2f}%".format(asr1.item()*100))
        print("Avg. Score: {:.2f}".format(scores.item()))
        print("ASR2: {:.2f}%".format(asr2.item()*100))
        print("Avg. Angle: {:.2f}".format(angles.item()), end = "\n\n")

        ret.append((name, asr1, scores, asr2, angles))
    
    return ret

### Transfer Attack
from feats.utils_feat import feat_ext

@torch.no_grad()
def tf_attack_ours(lsh_name, lfw, targets, nbnet, feat_configs, device):
    print("LSH :", lsh_name)
    feat_dir = feat_configs[lsh_name]
    mem = []    
    for name, target, thx in targets:
        ori_feat = feat_ext(lfw, target, 512, device)
        att_feat = F.normalize(torch.load(feat_dir, map_location = "cpu")).to(device)
        att_feat = get_att_feat(att_feat, target, nbnet, device)
        print("Attack Start :", name)

        asr2, angles = att_type2(ori_feat, att_feat, thx, is_gen = False)
        
        print("Model :", name, end="\t")
        print("ASR : {:.2f}%".format(asr2.item()*100), end = "\t")
        print("Angle : {:.2f}".format(angles.item()), end = "\n\n")
        mem.append([name, asr2.item(), angles.item()])
        
    return mem

@torch.no_grad()
def att_type1(LSH, thx_lsh, att_feat, name, code, R, a, b, is_gen = False):
    if is_gen:
        total_num = 200
        loop_num = 200
    else: 
        total_num = 12000
        loop_num = 20
    asr1 = 0
    scores = 0
    for i in range(loop_num):
        if name == "Jiang":
            t = (code[i], (R[i], a[i], b[i]))
        else:
            t = (code[i], R[i])
        if is_gen:    
            score = LSH.score(att_feat[i:i+1], t)
        else:
            score = LSH.score(att_feat[i], t)
        asr1 += (score > thx_lsh).sum()
        scores += score.sum()
        
    asr1 = asr1 / total_num 
    scores = scores / total_num
    return asr1, scores


@torch.no_grad()
def att_type2(ori_feat, att_feat, thx_bb, is_gen = False):
    if is_gen:
        ori_left, ori_right = ori_feat
        att_left, att_right = att_feat.chunk(2, dim=0)
        total_num = 200
    else:
        att_left, att_right = att_feat[::2], att_feat[1::2]
        ori_left, ori_right = ori_feat[::2], ori_feat[1::2]
        
        att_left= att_left.reshape(20, -1, 512)[::2].reshape(-1, 512)        
        att_right = att_right.reshape(20, -1, 512)[::2].reshape(-1, 512)
        ori_left = ori_left.reshape(20, -1, 512)[::2].reshape(-1, 512)
        ori_right = ori_right.reshape(20, -1, 512)[::2].reshape(-1, 512)
        total_num = 6000

    score_lr = F.cosine_similarity(att_left, ori_right)
    score_rl = F.cosine_similarity(att_right, ori_left)
    
    asr2 = (score_lr > thx_bb).sum() + (score_rl > thx_bb).sum()
    angles = (score_lr.acos() * 180/3.14).sum() + (score_rl.acos() * 180 / 3.14).sum()
    
    asr2 = asr2 / total_num
    angles = angles / total_num  
    
    return asr2, angles

# for figure
import random

def for_figure(dataset, LSHs, backbone, nbnet, device, index = None):
    dset = dataset[0]
    
    if index == None:
        r = random.randint(0, dset[0].size(0))
    else:
        r = index
        
    print("Index: ", r)
        
    source = (dset[0][r:r+1] - 127.5) / 127.5
    source = source.to(device)
    source_f = source.flip(3)
    orig_feat= F.normalize(backbone(source) + backbone(source_f))
    
    mem_gen = []
    mem_ours = []
    
    for name, LSH, _ in LSHs:
        # Enroll
        t = LSH.hashing(orig_feat)
        
        # Gen
        preimg = GenAttack(LSH, t, device)
        purity = F.cosine_similarity(orig_feat, preimg)
        
        match_score1 = LSH.score(preimg.to(device), t)
        
        recon_img = nbnet(F.normalize(preimg.to(device)))
        recon_img = F.interpolate(recon_img, (112,112))
        recon_feat = F.normalize(backbone(recon_img) + backbone(recon_img.flip(3)))
        
        match_score2 = LSH.score(recon_feat.to(device), t)
        
        cosine = F.cosine_similarity(recon_feat, orig_feat)
        mem_gen.append((name, recon_img, match_score1.item(), match_score2.item(), purity.item(), cosine.item()))
        
    
    for name, LSH, _ in LSHs:
        
        t = LSH.hashing(orig_feat)
   
        # Proposed   
        if name == "Jiang":
            code, helper = t
            R, a, b = helper
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
                                
            for j in range(a.size(1)):
                a_inv[0, j] = inv(a[0][j].item(), LSH.M)
            code_inv = (code - b) * a_inv 
            code_inv = code_inv % LSH.M
            
            init = torch.zeros((1, 10000), device = device)
            
            init[0, ::5] = (code // 16) % 2
            init[0,1::5] = (code // 8) % 2
            init[0,2::5] = (code // 4) % 2
            init[0,3::5] = (code // 2) % 2
            init[0,4::5] = code % 2
            
            init = (init.float() * 2 - 1)
            init = (init @ F.normalize(R)) 

            preimg = Newton(LSH.lsp(t), 1, device, init, "s", "pow")
        elif name == "ABH":
            code, R = t
            init = torch.zeros((1, 50, 120), device = device)
            init[:, :, ::2] = (code // 2) % 2
            init[:, :, 1::2] = code % 2
            init = (init.reshape(1, -1).float() * 2 - 1)
            init = (init @ F.normalize(R))
        elif name == "DIoM":
            init = F.normalize(torch.randn((1, 512), device = device)) * 1e-3
            preimg = Newton(LSH.lsp(t), 1, device, init, "s")
        elif name == "BH":
            code, R = t
            init = ((2 * code.float() - 1) @ F.normalize(R))
            preimg = Newton(LSH.lsp(t), 1, device, init)
        elif name == "URP":
            preimg = Newton(LSH.lsp(t), 1, device, None, "s")        
            
        elif name == "GRP":
            # 300 * 16 * 512
            code, R = t
            helper_r = R.reshape(LSH.m, LSH.q, LSH.emb_size)
            init_tmp = []
            # Init engine
            for j in range(1):
                # 300
                tmpp = []
                for k in range(300):
                    tmpp.append(helper_r[k][code[j][k]].reshape(1, 1, 512))
                tmp = torch.cat(tmpp, dim = 0)
                init_tmp.append(F.normalize((tmp - helper_r), dim = 2).mean(dim=[0, 1]).reshape(1, LSH.emb_size))

            init = torch.cat(init_tmp, dim = 0)
            preimg = Newton(LSH.lsp(t), 1, device, init)


        elif name == "IMM":
            # dd
            code, R = t
            helper_r = R.reshape(LSH.m, LSH.r, LSH.emb_size)
            code_max, code_min = code.chunk(2, dim = -1)
            init_tmp = []
            for j in range(1):
                tmpp1 = []
                for k in range(300):
                    tmpp1.append(helper_r[k][code_max[j][k]].reshape(1, 1, 512))

                tmpp2 = []
                for k in range(300):
                    tmpp2.append(helper_r[k][code_min[j][k]].reshape(1, 1, 512))

                tmp1 = torch.cat(tmpp1, dim = 0)
                tmp2 = torch.cat(tmpp2, dim = 0)
                rr = torch.cat([(tmp1 - helper_r), (helper_r - tmp2)], dim = 0).mean(dim=[0,1]).reshape(1, 512)
                init_tmp.append(rr)

            init = torch.cat(init_tmp, dim = 0)
            preimg = Newton(LSH.lsp(t), 1, device, init)

        else:
            preimg = Newton(LSH.lsp(t), 1, device)        
        purity = F.cosine_similarity(orig_feat.cpu(), preimg)
        
        if purity < 0:
            preimg = -preimg
            purity = -purity
        
        match_score1 = LSH.score(preimg.to(device), t)
        
        recon_img = nbnet(F.normalize(preimg.to(device)))
        recon_img = F.interpolate(recon_img, (112,112))
        recon_feat = F.normalize(backbone(recon_img) + backbone(recon_img.flip(3)))
        
        match_score2 = LSH.score(recon_feat.to(device), t)
        
        cosine = F.cosine_similarity(recon_feat, orig_feat)
        mem_ours.append((name, recon_img, match_score1.item(), match_score2.item(), purity.item(), cosine.item()))
    return mem_gen, mem_ours, source