import torch
from torch import nn
import torch.nn.functional as F

import random 
import matplotlib.pyplot as plt

# Visualize the reconstructed facial image, along with the cosine similarity between features
@torch.no_grad()
def attack_visl(dataset, backbone, nbnet, device):
    dat = dataset[0][0]
    N = random.randint(0, dat.size(0) - 1)
    img = (dat[N] - 127.5) / 127.5
    img_recon = F.interpolate(nbnet(F.normalize(backbone(img.unsqueeze(0).to(device).detach()))), (112,112))

    feat = backbone(img.unsqueeze(0).to(device).detach())
    feat_recon = backbone(img_recon)

    print("Cos_Sim: {:.4f}".format(F.cosine_similarity(feat, feat_recon).item()))
    plt.subplot(1,2,1)
    plt.imshow((img.numpy().transpose(1,2,0) + 1) / 2)
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow((img_recon[0].detach().cpu().numpy().transpose(1,2,0) + 1) / 2)
    plt.axis("off")
    plt.show()

# Evaluate the attack success rate and epsilon 
@torch.no_grad()
def attack_eval(feats, backbone, nbnet, batch_size, device, thx):
    left, right = feats[::2], feats[1::2]
    
    score_orig = F.cosine_similarity(left, right)
    
    recon_feats = []
    ba = 0
    while ba < left.size(0):
        bb = min(left.size(0), ba + batch_size)
        blk = feats[ba:bb]
        imgs = F.interpolate(nbnet(left[ba:bb]), (112,112))
        feats_recon = F.normalize(backbone(imgs))
        ba = bb
        recon_feats.append(feats_recon)
    recon_feats = torch.cat(recon_feats, dim = 0)
    ### TYPE-1 TEST
    score_1 = F.cosine_similarity(left, recon_feats)
    score_2 = F.cosine_similarity(right, recon_feats)
    score_2_true = score_2.reshape(20, -1)[::2].reshape(-1)
    
    ASR1 = score_1 > thx
    ASR2 = score_2_true > thx
    eps = (1 - score_1).mean()
    
    print("ASR1: {:.3f}%".format(ASR1.sum().item() / ASR1.size(0) * 100))
    print("ASR2: {:.3f}%".format(ASR2.sum().item() / ASR2.size(0) * 100))
    print("epsilon: {:.4f}".format(eps.item()))
    
    s0_true = score_orig.reshape(20, -1)[::2].reshape(-1).cpu()
    s2_true = score_2_true

    # Visualization of Cosine similarity between original-original vs. original-recovered
    plt.hist(s0_true.numpy(), bins = 40, histtype = "step", color = 'r', alpha = 0.5)
    plt.hist(s2_true.cpu().numpy(), bins = 40, histtype = "step", color = 'b', alpha = 0.5)
    plt.show()
    
    return score_orig, score_1, score_2