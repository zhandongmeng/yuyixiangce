# -*- coding: utf-8 -*-
"""语义相册 安装/卸载 共用逻辑

安装包结构：PyInstaller onefile 生成的 stub exe + 尾部追加的 payload.zip
  运行安装程序  → 读自身尾部 zip，解压到目标目录，建快捷方式，写卸载注册表
  改名为 Uninstall.exe（旧版：卸载语义相册.exe）→ 进入卸载模式

关键约束：
  · 不写系统目录、不要求管理员（装到 %LOCALAPPDATA%\\Programs，注册表用 HKCU）
  · 所有路径含中文也没问题（内部一律 str + UTF-8）
"""
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

APP_NAME = "语义相册"
APP_VERSION = "1.0.0"
APP_EXE = "语义相册.exe"
# 卸载器用英文名：部分杀软对「卸载*.exe」有启发式，落盘会被拒（实测 PermissionError）。
# 旧版本装的是「卸载语义相册.exe」，卸载逻辑两个都认。
UNINS_EXE = "Uninstall.exe"
UNINS_EXE_OLD = "卸载%s.exe" % APP_NAME
PUBLISHER = "本地自用"

NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW


# ---------------- 路径 ----------------
def default_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "Programs" / APP_NAME


def desktop_dir() -> Path:
    return Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"


def startmenu_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs"


# ---------------- 注册表 ----------------
UNINST_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\%s" % APP_NAME


def read_install_info() -> dict:
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, UNINST_KEY) as k:
            out = {}
            for name in ("InstallLocation", "DisplayVersion", "EstimatedSize"):
                try:
                    out[name] = winreg.QueryValueEx(k, name)[0]
                except OSError:
                    out[name] = None
            return out
    except OSError:
        return {}


def uninstaller_path(install_dir: Path) -> Path:
    """优先新的 Uninstall.exe，兼容旧版的中文名"""
    new = Path(install_dir) / UNINS_EXE
    if new.exists():
        return new
    old = Path(install_dir) / UNINS_EXE_OLD
    return old if old.exists() else new


def write_install_info(install_dir: Path, size_kb: int):
    import winreg
    exe = install_dir / APP_EXE
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, UNINST_KEY) as k:
        vals = {
            "DisplayName": APP_NAME,
            "DisplayVersion": APP_VERSION,
            "Publisher": PUBLISHER,
            "DisplayIcon": str(exe),
            "InstallLocation": str(install_dir),
            "UninstallString": '"%s"' % uninstaller_path(install_dir),
            "NoModify": 1,
            "NoRepair": 1,
            "EstimatedSize": int(size_kb),
        }
        for name, v in vals.items():
            winreg.SetValueEx(k, name, 0, winreg.REG_SZ if isinstance(v, str) else winreg.REG_DWORD, v)


def delete_install_info():
    import winreg
    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, UNINST_KEY)
    except OSError:
        pass


# ---------------- 快捷方式 ----------------
def make_shortcut(lnk: Path, target: Path, workdir: Path, desc="", icon=None):
    """用 PowerShell 调 WScript.Shell 建 .lnk，失败退回 VBScript"""
    lnk.parent.mkdir(parents=True, exist_ok=True)
    icon = icon or target
    ps = (
        "$ws = New-Object -ComObject WScript.Shell; "
        "$s = $ws.CreateShortcut('%s'); "
        "$s.TargetPath = '%s'; $s.WorkingDirectory = '%s'; "
        "$s.Description = '%s'; $s.IconLocation = '%s,0'; $s.Save()"
        % (str(lnk), str(target), str(workdir), desc, str(icon))
    )
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive",
                    "-ExecutionPolicy", "Bypass", "-Command", ps],
                   capture_output=True, creationflags=NO_WINDOW)
    if lnk.exists():
        return True
    vbs = (
        'Set ws = CreateObject("WScript.Shell")\r\n'
        'Set s = ws.CreateShortcut("%s")\r\n'
        's.TargetPath = "%s"\r\n'
        's.WorkingDirectory = "%s"\r\n'
        's.IconLocation = "%s,0"\r\n'
        's.Description = "%s"\r\n'
        's.Save\r\n' % (str(lnk), str(target), str(workdir), str(icon), desc)
    )
    p = Path(os.environ.get("TEMP", ".")) / "_mklnk.vbs"
    try:
        p.write_text(vbs, encoding="mbcs")
        subprocess.run(["cscript", "//nologo", str(p)],
                       capture_output=True, creationflags=NO_WINDOW)
    except Exception:
        pass
    finally:
        try:
            p.unlink()
        except OSError:
            pass
    return lnk.exists()


def remove_shortcuts():
    for lnk in (desktop_dir() / (APP_NAME + ".lnk"),
                startmenu_dir() / (APP_NAME + ".lnk")):
        try:
            if lnk.exists():
                lnk.chmod(0o777)
                lnk.unlink()
        except OSError:
            pass


# ---------------- 进程 ----------------
# 用 ctypes 枚举而不用 tasklist：
#   tasklist /FI "IMAGENAME eq 语义相册.exe" 在中文进程名 + 非 936 代码页下会匹配不上，
#   结果就是「以为没在跑、其实在跑」，覆盖 _internal\*.pyd 时直接 PermissionError。
def list_processes():
    """返回 [(pid, exe_name)]"""
    import ctypes
    from ctypes import wintypes

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", ctypes.c_wchar * 260),
        ]

    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        snap = k32.CreateToolhelp32Snapshot(0x00000002, 0)   # TH32CS_SNAPPROCESS
        if not snap or snap == ctypes.c_void_p(-1).value:
            return []
        out = []
        try:
            e = PROCESSENTRY32W()
            e.dwSize = ctypes.sizeof(PROCESSENTRY32W)
            if k32.Process32FirstW(snap, ctypes.byref(e)):
                while True:
                    out.append((int(e.th32ProcessID), e.szExeFile))
                    if not k32.Process32NextW(snap, ctypes.byref(e)):
                        break
        finally:
            k32.CloseHandle(snap)
        return out
    except Exception:
        return []


def _app_pids():
    me = os.getpid()
    want = APP_EXE.lower()
    return [pid for pid, nm in list_processes()
            if pid != me and (nm or "").lower() == want]


def app_running() -> bool:
    return bool(_app_pids())


def kill_app(timeout=8.0):
    """结束正在运行的 语义相册.exe，并等文件句柄真正释放"""
    pids = _app_pids()
    if not pids:
        return True
    for pid in pids:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True, creationflags=NO_WINDOW)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not _app_pids():
            time.sleep(0.6)          # 再给 DLL 卸载留点时间
            return True
        time.sleep(0.3)
    return not _app_pids()


# ---------------- payload ----------------
def _res(name: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / name


def payload_zip():
    """payload.zip 两种来源都支持：
    ① 打包进 PyInstaller 归档（--add-data，启动时已在 _MEIPASS）—— 现行方案
    ② 追加在 exe 尾部（zip 中央目录在文件末尾，也能直接读）—— 兼容旧包"""
    p = _res("payload.zip")
    if p.exists():
        return zipfile.ZipFile(p)
    return zipfile.ZipFile(sys.executable)


def _safe_rel(name: str) -> bool:
    p = Path(name)
    return not (p.is_absolute() or ".." in p.parts)


def extract_payload(target: Path, cb=None):
    """解压到 target；cb(done_bytes, total_bytes, name) 用于进度。

    单个文件偶发占用（杀软正在扫描、上一版残留句柄）不该让整包装到一半就崩，
    这里重试几次；仍失败的收集起来返回，由调用方决定怎么提示。
    返回 (total_bytes, [(name, error), ...])
    """
    failed = []
    consec = 0
    with payload_zip() as z:
        infos = [i for i in z.infolist() if not i.is_dir() and _safe_rel(i.filename)]
        total = sum(i.file_size for i in infos) or 1
        done = 0
        for info in infos:
            dst = target / info.filename
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
            ok, err = False, None
            for attempt in range(4):
                try:
                    with z.open(info) as src, open(dst, "wb") as f:
                        shutil.copyfileobj(src, f, 1 << 20)
                    ok = True
                    break
                except OSError as e:          # PermissionError 也是它的子类
                    err = e
                    if attempt < 3:
                        time.sleep(0.25 * (attempt + 1))
            if ok:
                done += info.file_size
                consec = 0
            else:
                failed.append((info.filename, str(err)))
                consec += 1
                # 连续失败说明整个目标目录不可写（权限/受保护文件夹/杀软），
                # 再一个个试下去就是空转几小时，直接中止。
                if consec >= 20:
                    failed.append(("（已中止）", "连续 %d 个文件写入失败，"
                                                "目标目录可能不可写或受保护" % consec))
                    break
            if cb:
                cb(done, total, info.filename)
        return total, failed


def payload_size() -> int:
    with payload_zip() as z:
        return sum(i.file_size for i in z.infolist() if not i.is_dir())


def dir_size(p: Path) -> int:
    n = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                n += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return n


def self_delete_tree(path: Path, keep_sub=None):
    """卸载删除：能删的先在本进程删掉，自己（正在运行的 exe）交给 cmd 收尾。
    不整体 rmdir，避免 6000 个文件只靠一条命令、失败也没提示。"""
    p = Path(path).resolve()
    if p.parent == p or len(p.parts) <= 1:
        raise RuntimeError("拒绝删除根目录：%s" % p)
    me = Path(sys.executable).resolve() if getattr(sys, "frozen", False) else None
    keep = (p / keep_sub).resolve() if keep_sub else None

    for item in list(p.iterdir()):
        r = item.resolve()
        if keep and r == keep:
            continue
        if me and r == me:        # 自己删不掉自己
            continue
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink()
        except OSError:
            pass

    # 收尾：删掉自己的 exe；目录空了再顺手删目录
    if me and me.parent == p:
        _handoff_delete(me)
    if not keep_sub:
        subprocess.Popen(
            ["cmd", "/c", 'ping -n 3 127.0.0.1 >nul & rmdir /s /q "%s"' % p],
            creationflags=NO_WINDOW | 0x00000008)


def _handoff_delete(me: Path):
    """让别的进程在退出后删掉「正在运行的自己」。

    首选 cmd（ping 等一会儿，等本进程彻底退出再 del）。有些环境（安全策略、
    被父进程管着、无控制台会话）里 cmd 子进程会被直接掐掉，那就退一步：
    用 MoveFileEx 把文件挂到「重启后删除」队列，至少不会永久留个死文件。
    """
    try:
        subprocess.Popen(
            ["cmd", "/c", 'ping -n 2 127.0.0.1 >nul & del /f /q "%s"' % me],
            creationflags=NO_WINDOW | 0x00000008)
    except Exception:
        pass
    try:
        import ctypes
        MOVEFILE_DELAY_UNTIL_REBOOT = 0x4
        ctypes.windll.kernel32.MoveFileExW(str(me), None, MOVEFILE_DELAY_UNTIL_REBOOT)
    except Exception:
        pass
