from easydict import EasyDict as edict

cfg = edict()

cfg.device = "your device"

cfg.log_dir = "./INV/log_nbnet.txt"
cfg.img_dir = "your img dir"
cfg.batch_size = 128
cfg.total_step = 20000
cfg.verbose = 100
cfg.target_dir = "your target dir"
cfg.percept_list = [
    # List of Models for Perceptual Loss
    # (directory of your parameter, architecture)
    ("your percet dir", "your arch")
]
cfg.lr = 1e-3
cfg.sched_step = 5000
cfg.save_suffix = "default"