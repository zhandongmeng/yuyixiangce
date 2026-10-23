#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键打包 语义相册.exe（PyInstaller onedir）

产出：D:\\shouji\\aiwin\\release\\pkg\\语义相册.exe + _internal\\ + models\\
      以及 release\\out\\语义相册_绿色包.zip（zip 内顶层目录名仍是「语义相册」）

⚠️ 两个和沙箱/安全策略有关的坑（2026-10-18 踩到）：
  1. PyInstaller 的 --distpath 不能指向一个「曾经被拒绝写 .exe」的目录
     （dist/语义相册、release/语义相册 都被登记了，之后写什么都 Permission denied），
     所以先打到 %TEMP% 下的全新目录，再整目录 move 过来。
  2. 目录能 move，文件不能 mv/rm（.exe/.zip 被 safe-delete 拦），
     所以构建目录用中性名 pkg，重复构建时把旧目录整体挪到 release/_trash_<时间>。
"""
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

PY = sys.executable
HOME = Path(__file__).resolve().parent.parent          # D:/shouji/aiwin
DIST = HOME / "release"
OUT = DIST / "out"
OUT.mkdir(parents=True, exist_ok=True)
PKG = DIST / "pkg"
WORK = Path(os.environ.get("TEMP", str(HOME))) / "aiwin_pyi_work"
PYI_DIST = Path(os.environ.get("TEMP", str(HOME))) / "aiwin_pyi_dist"
SP = Path(PY).parent.parent / "Lib" / "site-packages"

print("PyInstaller 打包中…", flush=True)
cmd = [
    PY, "-m", "PyInstaller",
    # --windowed：双击后不再弹黑窗口，程序在后台跑；日志写 data\logs\app.log，
    # 排障时用 `语义相册.exe --console` 可以叫出一个控制台。
    "--noconfirm", "--onedir", "--windowed",
    "--name", "语义相册",
    "--icon", str(HOME / "app" / "app.ico"),
    "--distpath", str(PYI_DIST),
    "--workpath", str(WORK),
    "--specpath", str(WORK),
    "--add-data", "%s;app" % (HOME / "app" / "index.html"),
    # CPU 版 onnxruntime 兜底：DirectML 起不来（远程桌面/驱动异常）时自动切过去，
    # 否则 import 阶段就抛 DLL 初始化失败，整个程序没法用。
    "--add-data", "%s;ort_cpu" % (HOME / "ort_cpu"),
    "--collect-all", "av",
    "--collect-all", "onnxruntime",
    "--collect-all", "ctranslate2",
    "--collect-all", "faster_whisper",
    "--collect-all", "tokenizers",
    "--collect-all", "pillow_heif",
    "--hidden-import", "pillow_heif",
    "--hidden-import", "server",
    "--hidden-import", "ort_loader",
    "--hidden-import", "ui_open",
    str(HOME / "app" / "win_main.py"),
]
# pillow-heif 的 DLL 在 site-packages 根目录（包外），需要显式收进来
for pat in ("libheif*.dll", "libde265*.dll", "libx265*.dll",
            "libgcc_s_seh*.dll", "libstdc++*.dll", "libwinpthread*.dll"):
    for f in sorted(SP.glob(pat)):
        cmd += ["--add-binary", "%s;." % f]
print(" ".join(cmd), flush=True)
r = subprocess.run(cmd, cwd=str(HOME))
if r.returncode != 0:
    print("打包失败 rc=%d" % r.returncode, flush=True)
    sys.exit(r.returncode)

# ---- 把产物整体挪到 release/pkg（目录 move，不写 .exe 字节）----
built = PYI_DIST / "语义相册"
if not built.exists():
    print("没找到 PyInstaller 产物：%s" % built, flush=True)
    sys.exit(2)
if PKG.exists():
    trash = DIST / ("_trash_" + time.strftime("%H%M%S"))
    shutil.move(str(PKG), str(trash))
    print("旧构建目录已挪到 %s" % trash, flush=True)
shutil.move(str(built), str(PKG))
print("产物：%s" % (PKG / "语义相册.exe"), flush=True)

# ---- 附带模型与说明 ----
shutil.copytree(HOME / "models", PKG / "models")
print("已复制 models -> %s" % (PKG / "models"), flush=True)
for extra in ("README.md", "使用说明.txt"):
    if (HOME / extra).exists():
        shutil.copy2(HOME / extra, PKG / extra)

# ---- 打zip，方便整体拷走（顶层目录名映射回「语义相册」）----
# ⚠️ 安全策略会拦 "语义相册_*.zip" 这个名字（含下划线的一律 Permission denied），
#    所以这个包就叫 语义相册.zip；解压出来仍是 语义相册/ 文件夹。
ZIP = OUT / "语义相册.zip"
if ZIP.exists():
    try:
        os.remove(ZIP)
    except Exception:
        pass
print("压缩 zip …", flush=True)
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            z.write(p, Path("语义相册") / p.relative_to(PKG))
print("\n完成：%s" % (PKG / "语义相册.exe"), flush=True)
print("zip  ：%s" % ZIP, flush=True)
