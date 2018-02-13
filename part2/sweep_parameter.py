from subprocess import call
import os

root_dir = "./exps"
lrs = [0.001, 0.01, 0.05]

for lr in lrs:
    out_dir = os.path.join(root_dir, "lr{}".format(lr))
    cmd = ["python", "main.py", "--lr", str(lr), "--out", out_dir]
    call(cmd)

