#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验收：关掉界面窗口，后台进程一起退出（「关了就是关了」）

前置：被测目录里有 语义相册.exe，且本机装了 Edge/Chrome；**测试前先关掉所有
已有的语义相册窗口**（否则窗口计数会被干扰，脚本会先检测并退出）。

流程：起 exe → 等端口和窗口 → 再双击一次（应只多开一个窗口、不起第二份服务）
→ 关掉其中一个窗口（服务应仍在）→ 关掉剩下的（服务应自动退出）。

用法：
    python tools/verify_closeexit.py                 # 默认 release\\pkg
    AIWIN_PKG=D:\\某安装目录 python tools/verify_closeexit.py
"""
import ctypes
import json
import os
import subprocess
import sys
import time
import urllib.request
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_pkg import busy_ports, wait_port   # noqa: E402

HOME = Path(__file__).resolve().parent.parent
PKG = Path(os.environ.get("AIWIN_PKG") or (HOME / "release" / "pkg"))
EXE = PKG / "语义相册.exe"
TITLE = "语义相册"
WM_CLOSE = 0x0010

OK, BAD = [], []


def say(msg=""):
    print(msg, flush=True)


def chk(name, cond, extra=""):
    (OK if cond else BAD).append(name)
    say(("  [OK] " if cond else "  [!!] ") + name + (("  " + str(extra)) if extra else ""))
    return cond


def windows():
    """枚举标题为「语义相册」的可见顶层窗口"""
    u = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _cb(h, _l):
        n = u.GetWindowTextLengthW(h)
        if n:
            b = ctypes.create_unicode_buffer(n + 1)
            u.GetWindowTextW(h, b, n + 1)
            if b.value.strip() == TITLE and u.IsWindowVisible(h):
                found.append(h)
        return True

    u.EnumWindows(_cb, 0)
    return found


def close_win(h):
    """发 WM_CLOSE —— 等价于点窗口右上角的 ✕"""
    ctypes.windll.user32.PostMessageW(h, WM_CLOSE, 0, 0)


def alive(port):
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/api/ping" % port, timeout=1.5) as f:
            return json.loads(f.read().decode("utf-8")).get("app") == "aiwin"
    except Exception:
        return False


def wait_win(n, sec=40):
    t0 = time.time()
    while time.time() - t0 < sec:
        w = windows()
        if len(w) >= n:
            return w
        time.sleep(0.5)
    return windows()


def wait_gone(port, sec=40):
    t0 = time.time()
    while time.time() - t0 < sec:
        if not alive(port):
            return True
        time.sleep(1)
    return False


def main():
    if not EXE.is_file():
        say("[失败] 找不到 %s" % EXE)
        return 1

    left = windows()
    if left:
        say("[跳过] 本机已有 %d 个「%s」窗口，先关掉它们再跑本脚本" % (len(left), TITLE))
        return 2

    before = busy_ports()
    env = dict(os.environ, AIWIN_NO_UI="")      # 确保不受外部设置影响
    say("== 1. 启动 %s ==" % EXE)
    proc = subprocess.Popen([str(EXE)], cwd=str(PKG), env=env)
    port = wait_port(300, skip=before)
    if not chk("服务已起来", bool(port), "port=%s" % port):
        proc.kill()
        return 1

    w = wait_win(1)
    chk("界面窗口已打开", len(w) >= 1, "窗口数=%d" % len(w))
    if not w:
        proc.kill()
        return 1

    say("\n== 2. 再双击一次（单实例） ==")
    p2 = subprocess.Popen([str(EXE)], cwd=str(PKG), env=env)
    try:
        rc2 = p2.wait(timeout=45)
        chk("第二个进程自己退出了", rc2 == 0, "rc=%s" % rc2)
    except subprocess.TimeoutExpired:
        chk("第二个进程自己退出了", False, "45 秒还没退")
        p2.kill()
    w = wait_win(2)
    chk("界面多开了一个窗口（不是第二份服务）", len(w) >= 2, "窗口数=%d" % len(w))
    still = [p for p in busy_ports() if p not in before]
    chk("仍然只有一个服务实例", len(still) == 1, "端口=%s" % still)

    say("\n== 3. 只关一个窗口：服务应继续跑 ==")
    close_win(w[0])
    time.sleep(5)
    chk("还有窗口时服务不退出", alive(port), "窗口数=%d" % len(windows()))

    say("\n== 4. 关掉剩下的窗口：服务应自动退出 ==")
    for h in windows():
        close_win(h)
    ok = wait_gone(port, 40)
    chk("后台服务自动退出", ok)
    if ok:
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
    chk("进程真的结束了", proc.poll() is not None, "rc=%s" % proc.poll())
    chk("没有残留窗口", not windows(), "窗口数=%d" % len(windows()))
    if proc.poll() is None:
        proc.kill()

    say("\n===== 结果：%d 项通过，%d 项失败 =====" % (len(OK), len(BAD)))
    if BAD:
        say("失败项：" + "、".join(BAD))
    return 0 if not BAD else 1


if __name__ == "__main__":
    sys.exit(main())
