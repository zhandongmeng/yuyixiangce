#!/usr/bin/env python3
"""从局域网里的模型源机器拉人脸模型：yunet.onnx + arcface.onnx

密码一律从环境变量读，**不写在代码里**（这个仓库要公开）：
    set PHONE_PASS=你的密码        ← 必填
    set PHONE_HOST=192.168.31.50   ← 可选
    set PHONE_USER=pm              ← 可选
    set PHONE_DIR=/path/to/models/face  ← 可选

没有源机器也没关系：见 README「模型从哪来」，把两个 onnx 放进 models/face/ 就行。
"""
import os
from pathlib import Path

import paramiko

HOST = os.environ.get("PHONE_HOST", "192.168.31.50")
USER = os.environ.get("PHONE_USER", "pm")
PASS = os.environ.get("PHONE_PASS", "")
if not PASS:
    raise SystemExit("请先设置环境变量 PHONE_PASS（SSH 密码）；代码里不保存明文密码")
REMOTE = os.environ.get("PHONE_DIR", "/home/pm/ai/models/face")
LOCAL = str(Path(__file__).resolve().parent.parent / "models" / "face")
FILES = ["yunet.onnx", "arcface.onnx"]

os.makedirs(LOCAL, exist_ok=True)
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, port=22, username=USER, password=PASS, timeout=15,
            banner_timeout=15, auth_timeout=15,
            look_for_keys=False, allow_agent=False)
sftp = cli.open_sftp()
try:
    remote_files = set(sftp.listdir(REMOTE))
    print("手机上人脸目录：", sorted(remote_files))
    for f in FILES:
        lp = os.path.join(LOCAL, f)
        if f not in remote_files:
            print("!! 手机上没有", f)
            continue
        if os.path.exists(lp) and os.path.getsize(lp) > 1000:
            print("已存在，跳过", f)
            continue
        print("拉取", f, "...")
        sftp.get(os.path.join(REMOTE, f), lp)
        print("  ok %s (%.1f MB)" % (f, os.path.getsize(lp) / 1048576))
finally:
    sftp.close()
    cli.close()
print("FACE_PULL_DONE")
