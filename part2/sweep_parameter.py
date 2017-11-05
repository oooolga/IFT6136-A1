from subprocess import call
import os

root_dir = "./mode0_lr_exps"
os.mkdir(root_dir)
lrs = [0.01, 0.05, 0.1]

for lr in lrs:
    out_dir = os.path.join(root_dir, "lr{}".format(lr))
    cmd = ["python", "main.py", "--lr", str(lr), "--out", out_dir]
    call(cmd)

