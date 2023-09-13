from easydict import EasyDict as edict

args = edict()
args.lr = 0.0002
args.max_step = 100000
args.img_dir = "your_dir"
args.bb_dir = "your_dir"
args.std_dir = "your_dir"
args.logging_path = './logs_nbnet.txt'
args.batch_size = 128
args.device = "cpu"                        # Modify in accordance with your environment
args.emb_size = 512
args.infer_step = 50

