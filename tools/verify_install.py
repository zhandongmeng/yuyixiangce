#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""安装版端到端验收：卸载旧版 → 静默安装新包 → 校验产物 → 跑 verify_pkg

为什么单独一个脚本：安装/卸载都是"跑外部 exe + 等它自己退出"，
手工在 shell 里做容易踩「卸载器自删除卡住」和「装完没校验就往下走」两个坑。

用法：
    python tools/verify_install.py                 # 默认装到 _verify_install
    python tools/verify_install.py --fresh         # 顺带跑 verify_pkg --fresh（从零索引）
    python tools/verify_install.py --dir D:\\tmp\\x  # 换目标目录
    python tools/verify_install.py --no-uninstall  # 不先卸载（目录已干净时）

返回码：0 全过，1 安装失败，2 verify_pkg 有失败项
"""
import os
import subprocess
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parent.parent
SETUP = HOME / "release" / "out" / "语义相册Setup.exe"
TARGET = HOME / "_verify_install"
EXE_NAME = "语义相册.exe"
NO_UNINSTALL = "--no-uninstall" in sys.argv
FRESH = "--fresh" in sys.argv

for i, a in enumerate(sys.argv):
    if a == "--dir" and i + 1 < len(sys.argv):
        TARGET = Path(sys.argv[i + 1])


def say(msg=""):
    print(msg, flush=True)


def run(cmd, timeout, desc):
    """跑外部 exe。超时/异常都只告警不中断 —— 卸载器自删除在受管环境必然失败。"""
    say("  $ " + " ".join(str(c) for c in cmd))
    try:
        p = subprocess.run([str(c) for c in cmd], timeout=timeout,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        out = (p.stdout or "") + (p.stderr or "")
        if out.strip():
            for line in out.strip().splitlines()[-6:]:
                say("    " + line)
        say("    rc=%s（%s）" % (p.returncode, desc))
        return p.returncode
    except subprocess.TimeoutExpired:
        say("    [警告] %s 超时 %ds，继续" % (desc, timeout))
        return None
    except Exception as e:
        say("    [警告] %s 起不来：%r，继续" % (desc, e))
        return None


def main():
    say("=" * 62)
    say("安装版验收")
    say("  安装包 : %s" % SETUP)
    say("  目标   : %s" % TARGET)
    say("=" * 62)

    if not SETUP.is_file():
        say("[失败] 找不到安装包，先跑 tools/build.py + tools/build_installer.py")
        return 1

    # 1. 卸载旧版
    if not NO_UNINSTALL:
        un = TARGET / "Uninstall.exe"
        if un.is_file():
            say("\n[1/3] 卸载旧版")
            run([un, "--silent"], 600, "卸载")
        else:
            say("\n[1/3] 没有旧版，跳过卸载")

    # 2. 静默安装
    say("\n[2/3] 静默安装")
    rc = run([SETUP, "--silent", "--dir=%s" % TARGET], 3600, "安装")
    exe = TARGET / EXE_NAME
    if not exe.is_file():
        say("\n[失败] 装完没有主程序：%s" % exe)
        say("       安装器返回 %s，看 %%TEMP%%\\语义相册_setup.log 或换 --dir" % rc)
        return 1
    data = TARGET / "data"
    say("  主程序 OK（%s）" % exe)
    say("  带 data 目录：%s（全新安装应为 否）" % ("是" if data.is_dir() else "否"))

    # 3. verify_pkg
    say("\n[3/3] verify_pkg（AIWIN_PKG=%s）" % TARGET)
    env = dict(os.environ, AIWIN_PKG=str(TARGET))
    cmd = [sys.executable, str(HOME / "tools" / "verify_pkg.py")]
    if FRESH:
        cmd.append("--fresh")
    try:
        p = subprocess.run(cmd, env=env, timeout=3600)
    except subprocess.TimeoutExpired:
        say("[失败] verify_pkg 超时")
        return 2
    if p.returncode != 0:
        say("\n[失败] verify_pkg 有失败项（rc=%d）" % p.returncode)
        return 2

    say("\n" + "=" * 62)
    say("安装版验收通过")
    say("界面验收接着跑（先手动起 %s，再）：" % exe)
    for t in ("verify_ui.py", "verify_ui_moments.py", "verify_ui_search.py"):
        say("    python tools/%s" % t)
    say("=" * 62)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
