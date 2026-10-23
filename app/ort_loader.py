#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""onnxruntime 加载器：DirectML 探测 + CPU 兜底

为什么要有这个文件
------------------
onnxruntime-directml 这个包，DirectML 执行器是**编译进 onnxruntime.dll** 的，
所以 `import onnxruntime` 的那一刻就会连带初始化 DirectML.dll。而 DirectML.dll
的 DllMain 需要 D3D12 设备，下面这些环境会直接失败：

  · 远程桌面（RDP）会话 —— 没有可用的 D3D12 适配器
  · 虚拟机 / 无独显且集显驱动过旧
  · 显卡驱动被回滚、或系统 D3D12 组件被精简过
  · Win7 / Win8（DirectML 要求 Win10 1903+）

失败时抛的是：

    ImportError: DLL load failed while importing onnxruntime_pybind11_state:
    动态链接库(DLL)初始化例程失败

注意关键词是「初始化例程失败」(ERROR_DLL_INIT_FAILED)，不是「找不到指定的模块」
—— DLL 找到了，是它自己的 DllMain 返回了 FALSE。这个错误发生在 import 阶段，
原来就导致整个程序起不来、什么都用不了。

这里的策略
----------
1. 先单独把 DirectML.dll LoadLibrary 一次做探测（不碰 onnxruntime）。
2. 探测通过 → 用包里的 DirectML 版，走 MX250/集显。
3. 探测不通过 → 直接改用内置的 CPU 版（_internal/ort_cpu/），**绝不 import
   DirectML 版**，避免它把进程里的同名 onnxruntime.dll 污染掉。
4. 万一探测通过但 import 仍然炸（D3D12 设备创建阶段才失败），清掉残留模块后
   再退到 CPU 版，这次能成功的概率很高。

两个版本是**同目录隔离**的：各在自己的文件夹里，靠 sys.path 顺序选择，
靠 pyd 所在目录（CPython 在 Windows 上以 LOAD_WITH_ALTERED_SEARCH_PATH 加载
扩展，优先搜 pyd 自己的目录）找到配套的 onnxruntime.dll。

可用环境变量强制指定：AIWIN_ORT=cpu | dml
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

MODE = ""          # "dml" / "cpu" / "cpu-fallback"
DIAG = []          # 探测过程的人类可读记录，界面「模型」那一行会用到
ort = None         # 最终暴露的 onnxruntime 模块


def _note(msg: str):
    DIAG.append(msg)


def _home() -> Path:
    """程序根目录（models/ data/ 的父目录）"""
    h = os.environ.get("AIWIN_HOME")
    if h:
        return Path(h)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _internal() -> Path:
    """PyInstaller 的 _internal 目录；开发态退化为项目根"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "_internal"
    return Path(__file__).resolve().parent.parent


def _dml_pack() -> Path:
    """DirectML 版 onnxruntime 包的父目录（含 onnxruntime/ 子目录）"""
    return _internal()


def _cpu_pack() -> Path:
    """CPU 兜底版的父目录（含 onnxruntime/ 子目录）"""
    return _internal() / "ort_cpu"


def _free(dll_path: Path):
    """加载完立刻卸载，探测不要留副作用"""
    try:
        h = ctypes.WinDLL(str(dll_path))
        hm = getattr(h, "_handle", None)
        if hm:
            ctypes.windll.kernel32.FreeLibrary(ctypes.c_void_p(hm))
    except Exception:
        pass


def _directml_dll() -> Path | None:
    """找 DirectML.dll：打包态在 _internal\\onnxruntime\\capi，开发态在 site-packages 里"""
    p = _dml_pack() / "onnxruntime" / "capi" / "DirectML.dll"
    if p.exists():
        return p
    try:
        import importlib.util
        spec = importlib.util.find_spec("onnxruntime")
        for loc in (getattr(spec, "submodule_search_locations", None) or []):
            q = Path(loc) / "capi" / "DirectML.dll"
            if q.exists():
                return q
    except Exception:
        pass
    return None


def _probe_directml() -> tuple[bool, str]:
    """只探 DirectML.dll 能不能完成初始化 —— 这一步就是复现用户那个报错"""
    dll = _directml_dll()
    if dll is None:
        return False, "缺少 DirectML.dll"
    try:
        h = ctypes.WinDLL(str(dll))
        hm = getattr(h, "_handle", None)
        if hm:
            ctypes.windll.kernel32.FreeLibrary(ctypes.c_void_p(hm))
        return True, ""
    except OSError as e:
        return False, "%s (winerror=%s)" % (e, getattr(e, "winerror", ""))


def _drop_modules():
    """把 import 失败的残壳清掉，否则第二次 import 直接命中缓存"""
    for k in [k for k in sys.modules if k == "onnxruntime" or k.startswith("onnxruntime.")]:
        sys.modules.pop(k, None)


def _use_pack(parent: Path) -> bool:
    """把某个 onnxruntime 包所在目录顶到 sys.path 最前面，并注册它的 DLL 目录"""
    pkg = parent / "onnxruntime"
    if not (pkg / "__init__.py").exists():
        return False
    capi = pkg / "capi"
    if hasattr(os, "add_dll_directory") and capi.is_dir():
        try:
            os.add_dll_directory(str(capi))
        except Exception:
            pass
    sp = str(parent)
    while sp in sys.path:
        sys.path.remove(sp)
    sys.path.insert(0, sp)          # 必须比 _internal 更靠前
    return True


def _do_import(tag: str):
    import importlib
    m = importlib.import_module("onnxruntime")
    return m


def load():
    """加载 onnxruntime，返回 (module, mode, diag_list)。

    mode: dml / cpu / cpu-fallback
    """
    global MODE, ort
    if ort is not None:
        return ort, MODE, DIAG

    force = (os.environ.get("AIWIN_ORT") or "").strip().lower()

    if force in ("cpu", "cpu-only"):
        _note("环境变量 AIWIN_ORT=cpu，跳过 DirectML")
        ok, why = False, "按要求强制 CPU"
    else:
        ok, why = _probe_directml()
        _note("DirectML 探测：%s%s" % ("可用" if ok else "不可用", "" if ok else " — " + why))

    if ok:
        try:
            _use_pack(_dml_pack())
            ort = _do_import("dml")
            MODE = "dml"
            _note("onnxruntime %s · DirectML 版" % getattr(ort, "__version__", "?"))
            return ort, MODE, DIAG
        except Exception as e:
            _note("DirectML 版 import 失败：%s: %s" % (type(e).__name__, e))
            _drop_modules()

    # ---- CPU 兜底 ----
    parent = _cpu_pack()
    if not _use_pack(parent):
        _note("CPU 兜底包缺失（应位于 %s）" % parent)
        # 没兜底包就把原始错误原样抛出去，报错信息比「包装过」的更有用
        _use_pack(_dml_pack())
        _drop_modules()
        ort = _do_import("dml-fail")
        MODE = "dml"
        return ort, MODE, DIAG

    try:
        ort = _do_import("cpu")
        MODE = "cpu-fallback" if not ok else "cpu"
        _note("onnxruntime %s · CPU 版（%s）"
              % (getattr(ort, "__version__", "?"),
                 "DirectML 不可用" if not ok else "DirectML 版加载失败，已回退"))
        # CPU 版没有 DML，主程序据此显示「CPU 模式」
        return ort, MODE, DIAG
    except Exception as e:
        _note("CPU 版也加载失败：%s: %s" % (type(e).__name__, e))
        raise


def human_error(diag=None):
    """把加载失败翻译成用户能照着做的话"""
    d = diag if diag is not None else DIAG
    txt = "；".join(d)
    return ("onnxruntime 加载失败（%s）。"
            "本机的 DirectML/DirectX 12 组件不可用，且 CPU 兜底版也没起来。"
            "如果是远程桌面里运行，请改用本机桌面；"
            "否则尝试更新显卡驱动，或重新安装本程序以恢复 CPU 兜底包。" % txt)
