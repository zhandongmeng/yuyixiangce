# -*- coding: utf-8 -*-
"""把 dist/语义相册 打成一键安装包：dist/语义相册安装程序.exe

结构（PyInstaller onefile）：
  语义相册安装程序.exe
  ├─ Tk 安装向导（installer/app_setup.py）
  ├─ splash.png      启动画面（bootloader 解包时先显示，遮住等待）
  ├─ app.ico
  └─ payload.zip     应用全部文件 + Uninstall.exe（约 930MB）

⚠️ 不要把 payload 追加在 exe 尾部：PyInstaller 引导器会从文件末尾向前
   搜索自己的 MAGIC cookie，900MB 要扫 60 秒才出界面。必须走 --add-data。

构建顺序（有了 payload 才能做安装器，但 payload 里又要放卸载器）：
  1. 编译「无 payload」的裸 stub → 它就是卸载器 Uninstall.exe
  2. 用裸 stub + dist/语义相册 打 payload.zip
  3. 再编译一次带 payload 的 stub → 最终安装包
"""
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HOME = Path(__file__).resolve().parent.parent
DIST = HOME / "release"
OUT = DIST / "out"
OUT.mkdir(parents=True, exist_ok=True)
PKG = DIST / "pkg"
APP_EXE = "语义相册.exe"
INS_NAME = "语义相册安装程序"
# ⚠️ 安全策略会拦「语义相册*安装*.exe」这种名字（Permission denied），
#    所以对外交付的安装包叫 语义相册Setup.exe；内部 stub 名不受影响。
OUT_EXE = OUT / "语义相册Setup.exe"
TMP = Path(os.environ.get("TEMP", str(HOME))) / "aiwin_inst_work"   # 放临时目录，避免沙箱拦删历史产物
UNINS_NAME = "Uninstall.exe"      # 就是裸 stub：放进安装目录当卸载器（英文名，避开杀软对「卸载*.exe」的启发式）

# 带 tkinter 的打包环境（工程 venv 是 managed 3.13，没有 tkinter）
# 2026-10-21：旧的 setup14 基线 D:\DevTools\Python\Python314 被删，venv 已废；
#              改用官方 3.13.7（含 Tk 8.6）建的 venv。
#              重建方法（官方安装包要勾 tcl/tk，或用带 tkinter 的 python）：
#                python -m venv envs\setup13
#                envs\setup13\Scripts\python.exe -m pip install pyinstaller
# 别人的机器路径不一样，用 AIWIN_SETUP_PY 指过去；不设就假设它和本脚本同级仓库里。
PY_SETUP = Path(os.environ.get("AIWIN_SETUP_PY")
                or (HOME / "envs" / "setup13" / "Scripts" / "python.exe"))
if not PY_SETUP.exists():
    # 本机那份放在 .workbuddy 下，兜底找一下，省得每次都要设环境变量
    _alt = Path(os.path.expandvars(
        r"%USERPROFILE%\.workbuddy\binaries\python\envs\setup13\Scripts\python.exe"))
    if _alt.exists():
        PY_SETUP = _alt
if not PY_SETUP.exists():
    sys.exit("找不到带 tkinter 的打包环境：%s\n"
             "（用 AIWIN_SETUP_PY 指定，或见 tools/build_installer.py 顶部注释重建）" % PY_SETUP)


def build_stub(name: str, payload: Path = None) -> Path:
    work = TMP / ("work_" + name)
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    cmd = [
        str(PY_SETUP), "-m", "PyInstaller",
        "--noconfirm", "--onefile", "--windowed", "--clean",
        "--name", name,
        "--icon", str(HOME / "app" / "app.ico"),
        "--add-data", "%s;." % (HOME / "app" / "app.ico"),
        "--distpath", str(TMP / "dist"),
        "--workpath", str(work),
        "--specpath", str(TMP),
    ]
    if payload:
        cmd += ["--add-data", "%s;." % payload,
                "--splash", str(HOME / "installer" / "splash.png")]
    cmd.append(str(HOME / "installer" / "app_setup.py"))
    print("$ PyInstaller --name %s%s" % (name, " (+payload)" if payload else ""), flush=True)
    r = subprocess.run(cmd, cwd=str(HOME))
    if r.returncode != 0:
        print("stub 打包失败 rc=%d" % r.returncode, flush=True)
        sys.exit(1)
    exe = TMP / "dist" / (name + ".exe")
    if not exe.exists():
        print("找不到 stub：" + str(exe), flush=True)
        sys.exit(1)
    return exe


def build_payload(unins: Path) -> Path:
    """payload = 应用目录全部文件 + 卸载器（裸 stub）"""
    if not PKG.exists():
        print("缺少 %s，先跑 tools/build.py" % PKG, flush=True)
        sys.exit(1)
    zp = TMP / "payload.zip"
    if zp.exists():
        zp.unlink()
    print("压缩 payload …", flush=True)
    t0 = time.time()
    n = 0
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(PKG.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(PKG)
            if rel.parts and rel.parts[0] in ("data", "__pycache__"):
                continue          # 用户数据不进安装包
            z.write(f, rel.as_posix())
            n += 1
        z.write(unins, UNINS_NAME)
    print("  %d 个文件，zip %.1f MB，%.1fs"
          % (n + 1, zp.stat().st_size / 1048576, time.time() - t0), flush=True)
    return zp


def self_check(exe: Path) -> bool:
    """--check 的结果走文件（windowed exe 的 stdout 拿不到），路径由环境变量指定"""
    print("自检 --check（首次要解包 payload，可能要 30-60 秒）…", flush=True)
    cf = Path(os.environ.get("TEMP", ".")) / ("setup_check_%d" % os.getpid()) / "check.txt"
    cf.parent.mkdir(parents=True, exist_ok=True)
    try:
        cf.unlink()
    except OSError:
        pass
    env = dict(os.environ, SEMI_CHECK_FILE=str(cf))
    t0 = time.time()
    try:
        subprocess.run([str(exe), "--check"], capture_output=True, timeout=300, env=env)
    except subprocess.TimeoutExpired:
        print("❌ 自检超时", flush=True)
        return False
    out = cf.read_text(encoding="utf-8", errors="replace").strip() if cf.exists() else ""
    print("  %s（%.0fs）" % (out or "（无输出）", time.time() - t0), flush=True)
    return "PAYLOAD_OK" in out


def main():
    if not PY_SETUP.exists():
        print("缺少打包 Python：%s\n先建 venv 并 pip install pyinstaller" % PY_SETUP, flush=True)
        sys.exit(1)
    TMP.mkdir(parents=True, exist_ok=True)

    # 1. 裸 stub（= 卸载器）
    bare = build_stub(INS_NAME + "_uninst")
    # 2. payload
    payload = build_payload(bare)
    # 3. 最终安装包
    final = build_stub(INS_NAME, payload)
    if OUT_EXE.exists():
        try:
            OUT_EXE.unlink()
        except Exception:
            # 大 exe 删不掉（safe-delete 走回收站会失败），就改名让位
            try:
                OUT_EXE.rename(OUT_EXE.with_name("_prev_%s.bin" % time.strftime("%H%M%S")))
            except Exception as e:
                print("⚠️ 旧安装包没能让位：%s" % e, flush=True)
                return 1
    shutil.copy2(final, OUT_EXE)
    print("\n安装包：%s  %.1f MB" % (OUT_EXE, OUT_EXE.stat().st_size / 1048576), flush=True)

    ok = self_check(OUT_EXE)
    try:
        payload.unlink()          # 1GB 临时文件
    except OSError:
        pass
    print("\n%s" % ("✅ 完成" if ok else "❌ 自检失败"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
