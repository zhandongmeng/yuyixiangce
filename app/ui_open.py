#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""打开界面窗口 / 打开目录

win_main（首次启动）和 server（界面里点「重新打开界面」）都要用，
放独立模块免得两边循环 import。

窗口用 Edge/Chrome 的 --app 模式（无地址栏，像原生应用）。为了让「关掉窗口 =
退出程序」成立，这里做了两件必须的事：

1. **返回 Popen**，调用方可以 wait() 到窗口结束；
2. **带独立 --user-data-dir**。否则 Edge 会把 --app 转交给已经在跑的实例，
   我们手里这个进程会立刻退出 —— 看起来就像「用户刚打开就关了」，
   会把后台服务一起带走。
"""
import os
import subprocess
import tempfile
import webbrowser

_EDGE_LIKE = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
]


def profile_dir() -> str:
    """窗口专用的浏览器配置目录（和用户日常用的那个分开）。

    放在 %LOCALAPPDATA% 下而不是程序目录：绿色包可能被放在只读位置，
    而且这样卸载/换版本都不影响它。
    """
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    return os.path.join(base, "aiwin_ui")


def app_window_proc(url: str):
    """开独立窗口，返回 Popen；一个浏览器都找不到时返回 None。

    返回的进程生命周期 = 窗口生命周期（因为用了独立配置目录，不会被复用）。
    """
    args = ["--app=%s" % url, "--window-size=1280,860",
            "--no-first-run", "--no-default-browser-check",
            "--user-data-dir=%s" % profile_dir()]
    for exe in _EDGE_LIKE:
        if not os.path.exists(exe):
            continue
        try:
            return subprocess.Popen([exe] + args, close_fds=True)
        except Exception:
            continue
    return None


def app_window(url: str) -> bool:
    """只用「能开就行」的场景（server 里的重新打开界面）"""
    if app_window_proc(url) is not None:
        return True
    try:
        return bool(webbrowser.open(url))
    except Exception:
        return False


def reveal(path: str) -> bool:
    """在资源管理器里打开某个目录"""
    try:
        os.startfile(path)          # noqa: S606 - 只用于本地目录
        return True
    except Exception:
        return False


def close_windows(title: str = "语义相册") -> int:
    """给标题匹配的可见窗口发 WM_CLOSE（等价于用户点 ✕）。

    界面上点「退出语义相册」时用：服务都要退了，留个连不上的窗口很难看。
    """
    try:
        import ctypes
        from ctypes import wintypes
        u = ctypes.windll.user32
    except Exception:
        return 0

    hits = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _cb(h, _l):
        n = u.GetWindowTextLengthW(h)
        if n:
            b = ctypes.create_unicode_buffer(n + 1)
            u.GetWindowTextW(h, b, n + 1)
            if b.value.strip() == title and u.IsWindowVisible(h):
                hits.append(h)
        return True

    try:
        u.EnumWindows(_cb, 0)
        for h in hits:
            u.PostMessageW(h, 0x0010, 0, 0)      # WM_CLOSE
    except Exception:
        return 0
    return len(hits)
