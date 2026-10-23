#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""语义相册 · Windows 版入口（后台运行，无控制台窗口）

双击 语义相册.exe 之后：
  1. 定位程序目录（models / data 都在这里）
  2. 把标准输出接到 data\\logs\\app.log（没有控制台了，日志必须落盘）
  3. 以窗口模式打开界面（Edge/Chrome 的 --app，没有地址栏）
  4. **关掉那个窗口，后台进程跟着一起退出**（用户要的就是「关了就是关了」）

窗口进程由本进程持有，关窗即触发 uvicorn 优雅收尾；界面上「⋯ → 退出语义相册」
保留为兜底（比如窗口被浏览器吃掉、拿不到进程句柄的机器）。

再次双击 exe 不会开第二个实例 —— 会扫一遍端口，发现已在跑就让它把窗口调出来，
监控也归那个实例管。

想看实时日志（排障用）：命令行执行
    语义相册.exe --console
这样会分配一个控制台窗口，同时日志照旧写文件。

验收脚本可以设 AIWIN_NO_UI=1：只起服务、不开窗、不跟随窗口退出。
"""
import json
import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

PORT_FROM, PORT_TO = 8765, 8799
APP_TAG = "aiwin"


# ================================================================ 环境准备
def _setup_path() -> Path:
    if getattr(sys, "frozen", False):
        home = Path(sys.executable).resolve().parent      # 安装目录\语义相册
        sys.path.insert(0, str(home / "_internal"))
    else:
        here = Path(__file__).resolve().parent
        home = here.parent
        sys.path.insert(0, str(here))
    os.environ["AIWIN_HOME"] = str(home)
    return home


def _attach_console() -> bool:
    """--console：给个控制台窗口，方便看即时日志"""
    try:
        import ctypes
        if not ctypes.windll.kernel32.AllocConsole():
            return False
        # AllocConsole 之后要重新打开 CONOUT$，原来的 stdout 还是空的
        out = open("CONOUT$", "w", encoding="utf-8", errors="replace", buffering=1)
        sys.stdout = out
        sys.stderr = out
        ctypes.windll.kernel32.SetConsoleTitleW("语义相册 · 日志（关掉此窗口不影响程序）")
        return True
    except Exception:
        return False


def _setup_io(home: Path) -> Path | None:
    """windowed 打包下 sys.stdout 是 None，任何 print 都会炸，所以第一件事就是接住它。

    日志落在 data\\logs\\app.log，超过 4MB 时把旧的改名成 app.old.log（只留一份）。
    """
    if "--console" in sys.argv:
        _attach_console()          # 成功就直接往控制台写；失败就只写日志文件

    logdir = home / "data" / "logs"
    try:
        logdir.mkdir(parents=True, exist_ok=True)
        target = logdir / "app.log"
        try:
            if target.exists() and target.stat().st_size > 4 * 1024 * 1024:
                old = logdir / "app.old.log"
                try:
                    old.unlink()
                except Exception:
                    pass
                target.rename(old)
        except Exception:
            pass
        logf = open(target, "a", encoding="utf-8", errors="replace", buffering=1)
        logf.write("\n===== %s 启动 =====\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
        logf.flush()
    except Exception:
        logf = None

    if logf is not None:
        # 控制台模式下 stdout 已经是 CONOUT$，再包一层同时写两边
        base_out = sys.stdout if sys.stdout is not None else None
        if base_out is not None:
            sys.stdout = _Tee(base_out, logf)
            sys.stderr = sys.stdout
        else:
            sys.stdout = logf
            sys.stderr = logf
    elif sys.stdout is None:
        # 既没控制台也没日志文件 —— 至少别让 print 崩
        sys.stdout = open(os.devnull, "w")
        sys.stderr = sys.stdout
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    return logf


class _Tee:
    """同时往控制台和日志文件写"""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def write(self, s):
        try:
            self.a.write(s)
        except Exception:
            pass
        try:
            self.b.write(s)
            self.b.flush()
        except Exception:
            pass

    def flush(self):
        for x in (self.a, self.b):
            try:
                x.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    def reconfigure(self, **kw):
        for x in (self.a, self.b):
            try:
                x.reconfigure(**kw)
            except Exception:
                pass


def _fatal_box(msg: str):
    """windowed 下启动就崩的话，用户什么都看不到 —— 弹个框告诉他"""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, msg, "语义相册 · 启动失败", 0x10)
    except Exception:
        pass


# ================================================================ 网络小工具
def free_port(start=PORT_FROM, end=PORT_TO) -> int:
    for p in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
    return start


def wait_up(port, timeout=90) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except OSError:
            time.sleep(0.4)
    return False


def find_running() -> int | None:
    """扫一遍端口，看是不是已经有一个语义相册在跑（认 /api/ping 返回的 app 标记）"""
    for p in range(PORT_FROM, PORT_TO + 1):
        try:
            with urllib.request.urlopen(
                    "http://127.0.0.1:%d/api/ping" % p, timeout=0.3) as r:
                j = json.loads(r.read().decode("utf-8", "replace"))
            if j.get("app") == APP_TAG:
                return p
        except Exception:
            continue
    return None


# ================================================================ 开窗口
def open_ui(port: int):
    """开界面窗口，返回窗口进程（一个浏览器都找不到时返回 None）"""
    import ui_open
    return ui_open.app_window_proc("http://127.0.0.1:%d/" % port)


def _ask_show(port: int) -> bool:
    """已有实例在跑：让「正在跑的那个」去开窗口。

    这样关窗退出的监控也归它管 —— 本进程马上就要退出了，盯不住窗口。
    老版本实例没有 /api/show-window，退回自己开一个（能看界面，但关窗不退出）。
    """
    try:
        req = urllib.request.Request("http://127.0.0.1:%d/api/show-window" % port,
                                     data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=20) as r:
            return bool(json.loads(r.read().decode("utf-8", "replace")).get("ok"))
    except Exception:
        pass
    open_ui(port)
    return False


# 窗口活不过这么久，判定为「浏览器把 --app 转交给已有实例了」，此时绝不能跟着退出
MIN_WINDOW_LIFE = 3.0
_WINDOW = {"proc": None}          # 正在被监控的那个窗口进程


def _watch_window(proc, srv):
    """窗口进程结束 = 用户关掉了程序 → 后台服务一起退出"""
    t0 = time.time()
    try:
        proc.wait()
    except Exception:
        return
    life = time.time() - t0
    if life < MIN_WINDOW_LIFE:
        print("   界面窗口启动后 %.1fs 就退出了（浏览器转交给了已有实例？），"
              "不跟随退出" % life, flush=True)
        return
    print("   界面窗口已关闭（存活 %.0fs），后台进程一起退出" % life, flush=True)
    try:
        srv.should_exit = True
    except Exception:
        os._exit(0)


def spawn_window(port, srv, why=""):
    """开窗口并挂上「关窗即退出」的监控；返回是否成功"""
    if os.environ.get("AIWIN_NO_UI"):
        return True                     # 验收脚本专用：只跑服务不开窗

    cur = _WINDOW["proc"]
    if cur is not None and cur.poll() is None:
        # 已有受监控的窗口，这次只是「再开一个」——
        # 那个浏览器进程还活着，关掉新窗口不影响它，不用再挂一次监控
        import ui_open
        ui_open.app_window("http://127.0.0.1:%d/" % port)
        return True

    p = open_ui(port)
    if p is None:
        print("   没找到 Edge/Chrome，已用默认浏览器打开%s；"
              "关掉页面不会退出程序，请点 ⋯ → 退出语义相册。" % why, flush=True)
        return False
    _WINDOW["proc"] = p
    print("   界面窗口进程 %d —— 关掉窗口，程序会自动退出。" % p.pid, flush=True)
    threading.Thread(target=_watch_window, args=(p, srv), daemon=True).start()
    return True


# ================================================================ 主流程
def main():
    home = _setup_path()
    _setup_io(home)

    print("=" * 56, flush=True)
    print("   语义相册 · 本地语义搜索（Windows 版）", flush=True)
    print("=" * 56, flush=True)
    print("   程序目录 : %s" % home, flush=True)
    print("   模型目录 : %s" % (home / "models"), flush=True)
    print("   数据目录 : %s" % (home / "data"), flush=True)
    print("   日志文件 : %s" % (home / "data" / "logs" / "app.log"), flush=True)

    # ---- 已经有实例在跑：让它把窗口调出来（监控也归它），别再起第二个 ----
    running = find_running()
    if running:
        print("   已有实例在 %d 端口，交给它打开窗口。" % running, flush=True)
        _ask_show(running)
        return

    port = free_port()
    os.environ["AIWIN_PORT"] = str(port)
    print("   本机地址 : http://127.0.0.1:%d" % port, flush=True)
    print("   局域网   : http://<这台电脑的IP>:%d（手机/平板也能用）" % port, flush=True)
    print("", flush=True)
    print("   界面会自动打开。关掉窗口，后台进程会一起退出。", flush=True)
    print("=" * 56, flush=True)

    import uvicorn
    import server

    # uvicorn.Server 实例：界面上的「退出程序」靠把它 should_exit 置真来优雅收尾
    cfg = uvicorn.Config(server.app, host="0.0.0.0", port=port,
                         log_level="info", access_log=False)
    srv = uvicorn.Server(cfg)
    # 第二个实例双击时、界面「重新打开界面」也走这里 —— 两条路都要挂上关窗监控
    server.set_window_hook(lambda: spawn_window(port, srv))

    def _open_soon():
        if not wait_up(port):
            print("   服务 90 秒内没起来，请查看日志：%s"
                  % (home / "data" / "logs" / "app.log"), flush=True)
            return
        spawn_window(port, srv)

    def _quit_all():
        """界面点「退出语义相册」：连窗口一起收掉，别留个连不上的空壳"""
        import ui_open
        try:
            ui_open.close_windows()
        except Exception:
            pass
        srv.should_exit = True

    server.set_shutdown_hook(_quit_all)

    threading.Thread(target=_open_soon, daemon=True).start()
    try:
        srv.run()
    finally:
        print("   已退出 %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as e:
        import traceback
        traceback.print_exc()
        try:
            log = os.environ.get("AIWIN_HOME", "")
            _fatal_box("语义相册 启动失败：\n%s: %s\n\n日志：%s\\data\\logs\\app.log"
                       % (type(e).__name__, e, log))
        except Exception:
            pass
        raise
