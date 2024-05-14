import torch
import torch.nn.functional as F
from feats.utils_feat import feat_ext, load_bin
from LSH import * # LSH ZOO

def get_LSH_preset(device="cuda:0"):
    print("load preset LSHs...")
    print()
    bh = BH(500, 0, 512, device=device)
    grp = GRP(300, 16, 512, device)
    urp = URP(600, 100, 2, 512, device)
    imm = IMM(300, 16, 512, device)
    
    from LSH import DIoMNetwork
    NN = torch.load("./DIOM/diom.pt", map_location="cpu").to(device)
    NN = NN.eval()
    diom  = DIoM(512, 30, 512, NN, device)
    
    abh = ABH(50, 60, 2, 28, 512, device)
    j = Jiang(2000, 5, 17, 512, device)
    
    LSHs = [
        ("BH", bh, 500),
        ("GRP", grp, 300),
        ("URP", urp, 600),
        ("DIoM", diom, 512),
        ("IMM", imm, 600),
        ("ABH", abh, 50),
        ("Jiang", j, 2000),
    ]

    # print parameter setting
    print("LSHs")
    print("==========================")
    print("BH : (500, 0, 512)")
    print("GRP : (300, 16, 512)")
    print("URP : (600, 100, 2, 512)")
    print("IMM : (300, 16, 512)")
    print("DIOM : (512, 30, 512)")
    print("ABH : (50, 60, 2, 28, 512)")
    print("Jiang : (2000, 5, 17, 512)")
    print("==========================")
    print("Done.")
    return LSHs

##### 

from backbones import get_backbone_v2
from backbones.iresnet import iresnet50, iresnet100
from backbones.sfnet import sfnet20, sfnet64
from backbones.vit import ViTs_face
    
def get_transfer_preset(param_dir = "./params/", device = "cuda:0"):
    print("Load transfer backbones preset..")
    bb1 = ("r100", param_dir+"r100_glint_cosface.pth", False, "R100_Glint_Cosface") 
    bb2 = ("r50", param_dir+"r50_webface_arcface.onnx", True, "R50_Webface_Arcface")
    bb3 = ("sf20", param_dir+"sf20_vggface2_spface2.pth", False, "SF20_VGGface_Sphereface")
    bb4 = ("sf64", param_dir+"sf64_ms1m_spplus.pth", False, "SF64_MS1MV2_Sphereface")
    bb5 = ("vit", param_dir+"vit_ms1mv3_cosface.pth", False, "Vit_MS1MV3_Cosface")
    backbone_list = [bb1, bb2, bb3, bb4, bb5]
    backbones = []
    for bb in backbone_list:
        backbone = get_backbone_v2(bb[:3], device)
        backbones.append((backbone, bb[3]))
        print("Model :", bb[3], end="\t")
        print("Done..")
    print("Done!")
    return backbones

##### 
