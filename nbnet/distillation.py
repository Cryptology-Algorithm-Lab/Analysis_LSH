from fr_utils.iresnet import iresnet50
from fr_utils.dataset import MXFaceDataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def train(args):
    device = args.device
    
    dataset = MXFaceDataset(args.img_dir)
    dataloader = DataLoader(
        dataset,
        args.batch_size,
        shuffle = True,
        num_workers = 2,
        drop_last = True,
        pin_memory = True
    )
    
    teacher = iresnet50()
    teacher.load_state_dict(torch.load(args.bb_dir, map_location = "cpu"))
    teacher = teacher.eval().to(device)
    
    student = iresnet50()
    student = student.eval().to(device)
    
    optim = torch.optim.Adam(student.parameters(), lr = 1e-2)
    
    torch.backends.cudnn.benchmark = True
    
    print("Train Starts...")
    
    for i in range(3):
        for idx, (img, _) in enumerate(dataloader):
            img = img.to(device)
            with torch.no_grad():
                teacher_feat = teacher(img)
            student_feat = student(img)
            loss = (1 - F.cosine_similarity(teacher_feat, student_feat)).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            if idx%1000 == 0:
                print(idx, loss.item())
                
    
    torch.save(student.state_dict(), "./student.pt")
    
    