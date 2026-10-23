# -*- coding: utf-8 -*-
"""语义相册 安装程序 / 卸载程序（同一个 exe）

用法：
  语义相册安装程序.exe               → 图形安装向导
  语义相册安装程序.exe --silent      → 静默装到默认目录（脚本/测试用）
  语义相册安装程序.exe --check       → 只校验尾部 payload 可读（打包自检）
  Uninstall.exe                     → 图形卸载向导（安装时随包释放）
"""
import os
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import core  # noqa: E402

ACCENT = "#2563eb"
BG = "#f6f7fb"
FG = "#111827"
DIM = "#6b7280"
UNINSTALLER = core.UNINS_EXE


def resource(name):
    """PyInstaller 单文件模式下资源在 _MEIPASS"""
    base = Path(getattr(sys, "_MEIPASS", HERE))
    p = base / name
    return p if p.exists() else None


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return ("%.1f %s" % (n, u)) if u != "B" else ("%d B" % n)
        n /= 1024.0


def is_uninstall_mode() -> bool:
    if "--uninstall" in sys.argv:
        return True
    stem = Path(sys.executable).stem.lower()
    # 新版卸载器叫 Uninstall.exe，旧版是「卸载语义相册.exe」，两个都认
    return ("卸载" in stem) or stem.startswith("uninstall")


# ======================================================================
#  安装
# ======================================================================
class Installer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("%s 安装程序" % core.APP_NAME)
        self.configure(bg=BG)
        self.resizable(False, False)
        self.geometry("660x430")
        ico = resource("app.ico")
        if ico:
            try:
                self.iconbitmap(str(ico))
            except Exception:
                pass
        self.q = queue.Queue()
        self.busy = False
        self.ok = False

        # ---- 头部 ----
        head = tk.Frame(self, bg="#ffffff", height=96)
        head.pack(fill="x")
        head.pack_propagate(False)
        inner = tk.Frame(head, bg="#ffffff")
        inner.pack(side="left", padx=26)
        tk.Label(inner, text=core.APP_NAME, bg="#ffffff", fg="#111827",
                 font=("Microsoft YaHei UI", 20, "bold")).pack(anchor="w", pady=(18, 0))
        tk.Label(inner, text="本地语义搜索 · 照片 / 视频 / 音频   v%s" % core.APP_VERSION,
                 bg="#ffffff", fg=DIM, font=("Microsoft YaHei UI", 9)).pack(anchor="w")
        tk.Label(head, text="安装程序", bg="#ffffff", fg=DIM,
                 font=("Microsoft YaHei UI", 9)).pack(side="right", padx=26)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=26, pady=(16, 0))

        exist = core.read_install_info()
        if exist.get("InstallLocation"):
            note = "检测到已安装（%s），继续将覆盖更新。" % exist["InstallLocation"]
        else:
            note = "装到当前用户目录，无需管理员权限；可随时在「设置 → 应用」里卸载。"
        tk.Label(body, text=note, bg=BG, fg="#b45309" if exist else DIM,
                 font=("Microsoft YaHei UI", 9), wraplength=600,
                 justify="left").pack(anchor="w", pady=(0, 10))

        tk.Label(body, text="安装位置", bg=BG, fg=FG,
                 font=("Microsoft YaHei UI", 9, "bold")).pack(anchor="w")
        row = tk.Frame(body, bg=BG)
        row.pack(fill="x", pady=(4, 10))
        self.var_dir = tk.StringVar(value=str(exist.get("InstallLocation") or core.default_dir()))
        ent = tk.Entry(row, textvariable=self.var_dir, font=("Microsoft YaHei UI", 9),
                       relief="solid", bd=1, bg="#ffffff")
        ent.pack(side="left", fill="x", expand=True, ipady=5)
        self.btn_browse = tk.Button(row, text="浏览…", command=self.pick_dir,
                                    font=("Microsoft YaHei UI", 9), bg="#eef2ff",
                                    relief="flat", padx=12, cursor="hand2")
        self.btn_browse.pack(side="left", padx=(8, 0))

        self.var_desktop = tk.BooleanVar(value=True)
        self.var_menu = tk.BooleanVar(value=True)
        self.var_run = tk.BooleanVar(value=True)
        for v, t in ((self.var_desktop, "创建桌面快捷方式"),
                     (self.var_menu, "添加到开始菜单"),
                     (self.var_run, "安装完成后立即打开")):
            tk.Checkbutton(body, text=t, variable=v, bg=BG, fg=FG, activebackground=BG,
                           font=("Microsoft YaHei UI", 9), anchor="w",
                           highlightthickness=0).pack(anchor="w", pady=1)

        self.var_need = tk.StringVar()
        tk.Label(body, textvariable=self.var_need, bg=BG, fg=DIM,
                 font=("Microsoft YaHei UI", 8)).pack(anchor="w", pady=(8, 0))
        self.refresh_space()

        # ---- 进度 ----
        self.bar = ttk.Progressbar(body, mode="determinate", maximum=1000)
        self.bar.pack(fill="x", pady=(12, 4))
        self.var_status = tk.StringVar(value="准备就绪")
        tk.Label(body, textvariable=self.var_status, bg=BG, fg=DIM,
                 font=("Microsoft YaHei UI", 8), anchor="w").pack(fill="x")

        # ---- 底部按钮 ----
        foot = tk.Frame(self, bg=BG)
        foot.pack(fill="x", padx=26, pady=14)
        self.btn_go = tk.Button(foot, text="开始安装", command=self.start,
                                font=("Microsoft YaHei UI", 10, "bold"),
                                bg=ACCENT, fg="#ffffff", activebackground="#1d4ed8",
                                activeforeground="#ffffff", relief="flat",
                                padx=26, pady=7, cursor="hand2")
        self.btn_go.pack(side="right")
        self.btn_cancel = tk.Button(foot, text="取消", command=self._close,
                                    font=("Microsoft YaHei UI", 9), bg="#e5e7eb",
                                    relief="flat", padx=18, pady=7, cursor="hand2")
        self.btn_cancel.pack(side="right", padx=(0, 8))

        self.after(80, self.pump)

    # ---------- 小工具 ----------
    def refresh_space(self):
        try:
            p = Path(self.var_dir.get())
            base = p if p.exists() else p.parent
            while not base.exists() and base.parent != base:
                base = base.parent
            free = shutil.disk_usage(str(base)).free
            need = core.payload_size()
            self._need = need
            self.var_need.set("需要约 %s 磁盘空间 · 目标盘可用 %s"
                              % (human(need * 1.05), human(free)))
        except Exception:
            self._need = 0
            self.var_need.set("")

    def pick_dir(self):
        d = filedialog.askdirectory(title="选择安装位置", initialdir=self.var_dir.get())
        if d:
            self.var_dir.set(str(Path(d)))
            self.refresh_space()

    def _close(self):
        if self.busy:
            if not messagebox.askyesno(core.APP_NAME, "正在安装，确定要中断吗？"):
                return
        self.destroy()

    # ---------- 安装流程 ----------
    def start(self):
        if self.busy:
            return
        target = Path(self.var_dir.get().strip())
        if not target.is_absolute():
            messagebox.showerror(core.APP_NAME, "请选择有效的安装位置")
            return
        try:
            probe = target
            while not probe.exists() and probe.parent != probe:
                probe = probe.parent
            test = probe / ("_wb_test_%d" % os.getpid())
            test.write_bytes(b"1")
            test.unlink()
        except Exception as e:
            messagebox.showerror(core.APP_NAME, "该位置不可写：%s" % e)
            return
        if self._need and shutil.disk_usage(str(probe)).free < self._need * 1.05:
            messagebox.showerror(core.APP_NAME, "磁盘空间不足")
            return
        if core.app_running():
            if not messagebox.askyesno(core.APP_NAME,
                                       "语义相册正在运行，需要先关闭它才能覆盖安装。\n现在关闭吗？"):
                return
            core.kill_app()

        self.busy = True
        self.btn_go.configure(state="disabled", text="安装中…")
        self.btn_browse.configure(state="disabled")
        self.var_status.set("正在复制文件…")
        t = threading.Thread(target=self._work, args=(target,), daemon=True)
        t.start()

    def _work(self, target: Path):
        try:
            target.mkdir(parents=True, exist_ok=True)

            def cb(done, total, name):
                self.q.put(("p", done, total, name))

            total, failed = core.extract_payload(target, cb)
            if failed:
                self.q.put(("warn", "有 %d 个文件没能写入：\n%s\n\n"
                                    "多半是被杀毒软件或上次运行的进程锁住了，"
                                    "关掉杀软/重启后再装一次即可。"
                            % (len(failed), "\n".join(n for n, _ in failed[:6]))))
            self.q.put(("s", "正在创建快捷方式…"))
            exe = target / core.APP_EXE
            if self.var_desktop.get():
                core.make_shortcut(core.desktop_dir() / (core.APP_NAME + ".lnk"),
                                   exe, target, core.APP_NAME)
            if self.var_menu.get():
                core.make_shortcut(core.startmenu_dir() / (core.APP_NAME + ".lnk"),
                                   exe, target, core.APP_NAME)
            self.q.put(("s", "正在登记卸载信息…"))
            core.write_install_info(target, core.dir_size(target) // 1024)
            self.q.put(("done", target, exe, self.var_run.get()))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.q.put(("err", "%s: %s" % (type(e).__name__, e)))

    def pump(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "p":
                    _, done, total, name = msg
                    self.bar["value"] = done / max(total, 1) * 1000
                    self.var_status.set("正在复制：%s（%s / %s）"
                                        % (Path(name).name, human(done), human(total)))
                elif kind == "s":
                    self.var_status.set(msg[1])
                elif kind == "warn":
                    messagebox.showwarning(core.APP_NAME, msg[1])
                elif kind == "done":
                    _, target, exe, run = msg
                    self.bar["value"] = 1000
                    self.var_status.set("安装完成")
                    self.busy = False
                    self.ok = True
                    self.btn_go.configure(text="完成", state="normal", command=self.destroy)
                    self.btn_cancel.configure(text="关闭", command=self.destroy)
                    if run:
                        try:
                            subprocess.Popen([str(exe)], cwd=str(target),
                                             creationflags=0x00000010)
                        except Exception:
                            pass
                    messagebox.showinfo(core.APP_NAME,
                                        "安装完成！\n\n位置：%s\n\n双击桌面「语义相册」即可使用。"
                                        % target)
                elif kind == "err":
                    self.busy = False
                    self.btn_go.configure(state="normal", text="重试")
                    self.var_status.set("安装失败")
                    messagebox.showerror(core.APP_NAME, "安装失败：\n%s" % msg[1])
        except queue.Empty:
            pass
        self.after(80, self.pump)


# ======================================================================
#  卸载
# ======================================================================
class Uninstaller(tk.Tk):
    def __init__(self):
        super().__init__()
        info = core.read_install_info()
        loc = info.get("InstallLocation") or str(Path(sys.executable).resolve().parent)
        self.target = Path(loc)
        self.title("卸载 %s" % core.APP_NAME)
        self.configure(bg=BG)
        self.geometry("560x340")
        self.resizable(False, False)
        ico = resource("app.ico")
        if ico:
            try:
                self.iconbitmap(str(ico))
            except Exception:
                pass
        self.busy = False

        head = tk.Frame(self, bg="#ffffff", height=84)
        head.pack(fill="x")
        head.pack_propagate(False)
        inner = tk.Frame(head, bg="#ffffff")
        inner.pack(side="left", padx=26)
        tk.Label(inner, text="卸载 %s" % core.APP_NAME, bg="#ffffff",
                 font=("Microsoft YaHei UI", 18, "bold")).pack(anchor="w", pady=(16, 0))
        tk.Label(inner, text=str(self.target), bg="#ffffff", fg=DIM,
                 font=("Microsoft YaHei UI", 8)).pack(anchor="w")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=26, pady=16)
        size = human(core.dir_size(self.target)) if self.target.exists() else "0"
        tk.Label(body, text="将删除程序文件和快捷方式（当前占用 %s）。" % size,
                 bg=BG, fg=FG, font=("Microsoft YaHei UI", 9)).pack(anchor="w")
        self.var_keep = tk.BooleanVar(value=True)
        tk.Checkbutton(body, text="保留 data 目录（索引、缩略图、回收站，以后重装可继续用）",
                       variable=self.var_keep, bg=BG, fg=FG, activebackground=BG,
                       font=("Microsoft YaHei UI", 9), highlightthickness=0,
                       anchor="w").pack(anchor="w", pady=(14, 0))
        tk.Label(body, text="取消勾选 = 连数据和回收站一起删掉，删了找不回。",
                 bg=BG, fg="#b45309", font=("Microsoft YaHei UI", 8)).pack(anchor="w")

        self.bar = ttk.Progressbar(body, mode="determinate", maximum=1000)
        self.bar.pack(fill="x", pady=(18, 4))
        self.var_status = tk.StringVar(value="准备就绪")
        tk.Label(body, textvariable=self.var_status, bg=BG, fg=DIM,
                 font=("Microsoft YaHei UI", 8)).pack(anchor="w")

        foot = tk.Frame(self, bg=BG)
        foot.pack(fill="x", padx=26, pady=14)
        self.btn_go = tk.Button(foot, text="卸载", command=self.start,
                                font=("Microsoft YaHei UI", 10, "bold"),
                                bg="#dc2626", fg="#ffffff", activebackground="#b91c1c",
                                activeforeground="#ffffff", relief="flat",
                                padx=26, pady=7, cursor="hand2")
        self.btn_go.pack(side="right")
        tk.Button(foot, text="取消", command=self.destroy,
                  font=("Microsoft YaHei UI", 9), bg="#e5e7eb", relief="flat",
                  padx=18, pady=7, cursor="hand2").pack(side="right", padx=(0, 8))

    def start(self):
        if self.busy:
            return
        # ---- 安全护栏：目标目录必须真的是语义相册的安装目录 ----
        looks_installed = (self.target / core.APP_EXE).exists() or \
                          (self.target / "data").exists()
        if not looks_installed:
            messagebox.showerror(
                core.APP_NAME,
                "在以下位置没有找到语义相册：\n%s\n\n"
                "（可能已经卸载，或注册表信息缺失）\n不会删除任何文件。" % self.target)
            self.destroy()
            return
        if self.target.parent == self.target:
            messagebox.showerror(core.APP_NAME, "拒绝在磁盘根目录执行卸载。")
            self.destroy()
            return
        keep = self.var_keep.get()
        if not messagebox.askyesno(core.APP_NAME,
                                   "确定卸载 语义相册 吗？" + ("\n\n（保留 data 目录）" if keep else
                                                          "\n\n（将删除全部数据）")):
            return
        if core.app_running():
            if not messagebox.askyesno(core.APP_NAME, "语义相册正在运行，先关闭它？"):
                return
            core.kill_app()
        self.busy = True
        self.btn_go.configure(state="disabled", text="卸载中…")
        self.var_status.set("正在删除…")
        self.bar["value"] = 300
        self.update_idletasks()

        core.remove_shortcuts()
        core.delete_install_info()
        if keep:
            core.self_delete_tree(self.target, keep_sub="data")
            self._del_self()
            self.bar["value"] = 1000
            messagebox.showinfo(core.APP_NAME,
                                "已卸载。\n\n数据保留在：\n%s\\data" % self.target)
        else:
            core.self_delete_tree(self.target)
            self.bar["value"] = 1000
            messagebox.showinfo(core.APP_NAME, "已卸载，全部数据已删除。")
        self.destroy()

    def _del_self(self):
        exe = Path(sys.executable)
        script = 'ping -n 2 127.0.0.1 >nul & del /f /q "%s"' % str(exe)
        subprocess.Popen(["cmd", "/c", script],
                         creationflags=0x08000000 | 0x00000008)


# ======================================================================
#  CLI
# ======================================================================
def silent_install():
    # 支持 --dir=某目录（静默部署到自定义位置，便于脚本/多实例测试）
    target = core.default_dir()
    for a in sys.argv[1:]:
        if a.startswith("--dir=") and len(a) > 6:
            target = Path(a.split("=", 1)[1].strip('"'))
            break
    print("安装到：%s" % target, flush=True)
    if core.app_running():
        core.kill_app()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print("INSTALL_FAIL 建不了目标目录：%s → %s" % (target, e), flush=True)
        return 4
    # 先探一下能不能写：不可写的目录（受保护文件夹 / 权限 / 杀软）要立刻报错，
    # 否则会一个个文件重试到天荒地老
    try:
        probe = target / ("_wb_test_%d" % os.getpid())
        probe.write_bytes(b"1")
        probe.unlink()
    except Exception as e:
        print("INSTALL_FAIL 目标目录不可写：%s → %s" % (target, e), flush=True)
        return 4
    last = [0]

    def cb(done, total, name):
        pct = done * 100 // max(total, 1)
        if pct != last[0]:
            last[0] = pct
            print("  %3d%%  %s" % (pct, Path(name).name), flush=True)

    total, failed = core.extract_payload(target, cb)
    if failed:
        print("INSTALL_WARN failed=%d" % len(failed), flush=True)
        for n, e in failed[:10]:
            print("  没能写入 %s → %s" % (n, e), flush=True)
    exe = target / core.APP_EXE
    # 主程序都没解出来就别再往系统里写东西了：以前不管成功失败一律建快捷方式、
    # 写注册表，结果是"安装失败"但桌面留了个点不开的图标、控制面板也留了卸载项。
    if not exe.exists():
        print("INSTALL_FAIL 主程序没能写出（%s），已放弃创建快捷方式和注册项"
              % exe, flush=True)
        return 2
    if failed:
        print("INSTALL_WARN 有 %d 个文件没写进去，程序可能不完整" % len(failed), flush=True)
    core.make_shortcut(core.desktop_dir() / (core.APP_NAME + ".lnk"), exe, target, core.APP_NAME)
    core.make_shortcut(core.startmenu_dir() / (core.APP_NAME + ".lnk"), exe, target, core.APP_NAME)
    core.write_install_info(target, core.dir_size(target) // 1024)
    print("完成：%s" % exe, flush=True)
    return 1 if failed else 0


def _setup_stdio():
    """--windowed 打出来的 exe 没有控制台。

    优先写到自己的日志文件：曾经先 AttachConsole 再 open("CONOUT$")，
    当父进程把输出接了管道/重定向时，写 CONOUT$ 会一直阻塞
    （表现就是 --silent 装到一半再也不动），所以顺序反过来。
    """
    if sys.stdout is not None and sys.stderr is not None:
        return None
    try:
        d = Path(os.environ.get("TEMP", ".")) / ("setup_log_%d" % os.getpid())
        d.mkdir(parents=True, exist_ok=True)
        log = d / "setup.log"
        f = open(str(log), "a", encoding="utf-8", buffering=1)
        sys.stdout = f
        sys.stderr = f
        return log
    except Exception:
        pass
    try:
        import ctypes
        if ctypes.windll.kernel32.AttachConsole(-1):
            sys.stdout = open("CONOUT$", "w", encoding="utf-8", buffering=1)
            sys.stderr = sys.stdout
    except Exception:
        pass
    return None


def _check_file() -> Path:
    # 别用固定名：某些安全策略会把 %TEMP% 下同名文件锁定，导致 --check 拿不到结果。
    # 构建脚本可以用 SEMI_CHECK_FILE 指定位置。
    env = os.environ.get("SEMI_CHECK_FILE")
    if env:
        p = Path(env)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    d = Path(os.environ.get("TEMP", ".")) / ("setup_check_%d" % os.getpid())
    d.mkdir(parents=True, exist_ok=True)
    return d / "check.txt"


def silent_uninstall(keep=True) -> int:
    info = core.read_install_info()
    target = info.get("InstallLocation")
    if not target:
        # 注册表项没了（安全软件清理 / HKCU 被重置 / 用户手工拷走目录）时，
        # 默认用「卸载器自己所在目录」—— 和 GUI 卸载一致。
        # 绝不能直接回退到 core.default_dir()：Uninstall.exe 可能在 D:\某处，
        # 而默认目录里还装着一份好好的程序，那样会把它连根删掉。
        here = Path(sys.executable).resolve().parent
        target = str(here if ((here / core.APP_EXE).exists() or (here / "data").exists())
                     else core.default_dir())
    target = Path(target)
    ok = (target / core.APP_EXE).exists() or (target / "data").exists()
    if not ok:
        print("UNINSTALL_SKIP 未找到安装：%s" % target, flush=True)
        return 3
    if core.app_running():
        core.kill_app()
    core.remove_shortcuts()
    core.delete_install_info()
    core.self_delete_tree(target, keep_sub="data" if keep else None)
    if keep and Path(sys.executable).resolve().parent == target.resolve():
        # 保留数据时目录还在，得单独安排把自己的 exe 删掉
        core._handoff_delete(Path(sys.executable).resolve())
    print("UNINSTALL_OK keep_data=%s target=%s" % (keep, target), flush=True)
    return 0


def main():
    _setup_stdio()
    try:                       # 关掉 PyInstaller 启动画面（--splash）
        if "_PYI_SPLASH_IPC" in os.environ:      # 没有 splash 时别 import，会抛 KeyError
            import pyi_splash
            pyi_splash.close()
    except Exception:
        pass
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if "--check" in sys.argv:
        try:
            with core.payload_zip() as z:
                infos = [i for i in z.infolist() if not i.is_dir()]
                exe = [i for i in infos if i.filename.endswith(core.APP_EXE)]
                msg = ("PAYLOAD_OK entries=%d size=%.1f MB app_exe=%s"
                       % (len(infos), sum(i.file_size for i in infos) / 1048576,
                          "yes" if exe else "no"))
            with open(str(_check_file()), "w", encoding="utf-8") as f:
                f.write(msg)
            print(msg, flush=True)
            return 0
        except Exception as e:
            msg = "PAYLOAD_FAIL %s: %s" % (type(e).__name__, e)
            try:
                with open(str(_check_file()), "w", encoding="utf-8") as f:
                    f.write(msg)
            except Exception:
                pass
            print(msg, flush=True)
            return 2

    if is_uninstall_mode():
        if "--silent" in sys.argv:
            return silent_uninstall(keep="--no-keep-data" not in sys.argv)
        Uninstaller().mainloop()
        return 0

    if "--silent" in sys.argv:
        return silent_install()

    app = Installer()
    app.mainloop()
    return 0 if app.ok else 1


if __name__ == "__main__":
    sys.exit(main())
