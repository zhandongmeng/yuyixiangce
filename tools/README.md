# tools — 构建与维护脚本

全部在 venv（`C:\Users\48951\.workbuddy\binaries\python\envs\aiwin`）里跑。
venv 构建依赖：torch(CPU) / transformers / onnx / onnxscript（仅导出用，不打包）。

| 脚本 | 作用 |
|---|---|
| `pull_face.py` | 从手机（192.168.31.50）拉 `yunet.onnx` + `arcface.onnx` 到 `models/face/` |
| `export_clip.py` | 从 `models_src/clip_cn` 原始 PyTorch 权重导出 `models/clip/`：视觉塔 int8（DirectML batch=1）、文本塔 fp32（查询质量优先），并做 torch↔onnx 余弦一致性、分词一致性、真实照片抽查 |
| `smoke.py` | 不打包的全功能自检：建索引 → 文字搜索 → 相册（标签/城市/人脸）→ 语音转写 |
| `build.py` | PyInstaller onedir 打包 → `dist/语义相册/`（拷入 models、README、使用说明）→ 打 zip 到 `dist/out/` |
| `build_installer.py` | 一键安装包：stub（Tk 向导）+ `--add-data payload.zip` → `dist/out/语义相册安装程序.exe`，自带 `--check` 自检 |

### 验收脚本（改完代码必跑）

| 脚本 | 覆盖 |
|---|---|
| `t_face_cluster.py` | 人脸聚类单元测试：密度约束、锚点保护、手工拆分不被吸回、hints 认领。用合成向量，不依赖模型和图库 |
| `t_dups.py` | 重复照片算法：LSH 分段覆盖完整性、**阈值 2~7 零漏检**、dHash 压缩鲁棒性、有符号 64 位可逆 |
| `verify_ui.py` | 界面主流程：照片墙 → 分类卡 → 灯箱 → 搜索 → 容错 toast |
| `verify_ui_moments.py` | 「时刻」卡片 / 事件页 / 「其他面孔」入口与网格 / 认领选择器 |
| `verify_ui_search.py` | 搜索升级：文件名关键词加分（接口排序 + `_kw_bonus` 单测）、最近搜索 chip 的增删/上限/回填 |
| `verify_pkg.py` | 打包产物验收集：绿色包与安装版的文件完整性、接口可用、人脸挪动/命名、卸载残留 |
| `verify_closeexit.py` | 「关窗即退出」：起 exe → 再双击一次（应只多开窗口）→ 关一个窗口（服务仍在）→ 关完（服务退出、无残留窗口）。要求本机没有别的语义相册窗口，否则脚本会跳过 |
| `verify_install.py` | 安装版端到端：卸载旧版 → 静默装到 `_verify_install` → 校验主程序/无 `data` 残留 → 带 `AIWIN_PKG` 跑 `verify_pkg`。卸载器自删除超时只告警不中断 |

`verify_pkg.py` 认环境变量 `AIWIN_PKG` 指定安装目录；`verify_ui_search.py` 认 `AIWIN_PY`
指定带 numpy 的解释器（脚本本身可能跑在系统 python 上）。

> 验收脚本里的断言要盯住**语义**而不是**实现细节**。踩过的坑：用 `score - cos` 去反推
> 关键词加分，但 `score` 是模态校准后的排序分、`cos` 是原始余弦，两者天然有差，
> 这个差值根本不是加分量 —— 断言看似"失败"，其实是测试写错了。

> 为什么产物在 `dist/out/`：`dist/` 下已存在的 `.exe` / `.zip` 在沙箱里既不能删也不能改名
> （安全策略拦 safe-delete），PyInstaller / `shutil.copy2` 覆盖同名文件会失败。
> 同理重建前要先把 `dist/语义相册`、`build_pyi`、`build_installer` 整体 `mv` 成 `*_old_日期`
> —— 目录可以移，`--clean` 也不会再去删里面的 `warn-*.txt`。

## 已知坑（踩过）

1. **torch CPU 轮子**：download.pytorch.org 国内连不上，用 `mirrors.nju.edu.cn/pytorch/whl/cpu`。
2. **pip 默认源**超时，全程用 `-i https://mirrors.aliyun.com/pypi/simple/`。
3. **onnxruntime 被普通版覆盖**导致 `DmlExecutionProvider` 消失 —— 必须
   `pip install --force-reinstall --no-deps onnxruntime-directml`，并清掉残留的
   `onnxruntime-<ver>.dist-info` 和 `~nnxruntime*` 目录。
4. **ChineseCLIP 文本塔没有 pooler**：`get_text_features` 用的是
   `last_hidden_state[:, 0]`，导出包装器要照抄，不能用 `pooler_output`（是 None）。
5. **dynamo 导出的 value_info 会带错误形状**，`quantize_dynamic` 前置 shape inference
   直接报 768 vs 512 —— 量化前 `del graph.value_info[:]`。
6. **文本塔导出可能是 onnx + onnx.data 外置权重**，要 `onnx.load`（默认内联权重）
   再存成单文件；`*.onnx.data` 记得删，否则 330MB 会被打进安装包。
7. **文本塔 int8 余弦掉到 0.87**（视觉塔 0.979 没事）→ 只量化视觉塔。
8. **DirectML 对动态序列维的 BERT 图报 E_INVALIDARG** → 文本塔走 CPU。
9. **faster-whisper 本地目录加载要 4 个文件**：config.json / model.bin /
   tokenizer.json / vocabulary.txt（少最后一个报 "Cannot load the vocabulary"）。
10. **CLIP 文本塔向量彼此夹角小**，转写段和查询余弦天然偏高（0.55 vs 图文 0.35），
    混排会霸榜 → 搜索按「转写组/视觉组」均值对齐 + 尺度归一（`do_search` 里的模态校准）。

### DirectML 让独显真正干活的两个硬条件（2026-10 实测）

> 症状：`[model] vision.onnx on DmlExecutionProvider 失败：'utf-8' codec can't decode …`
> 或 stderr 里 `Non-zero status code … Reshape node … 0x8007023E`，随后**静默回落 CPU**
> （功能正常但独显白买）。极简 Add 模型在 DML 上正常 → 说明是图不兼容，不是驱动坏。

11. **torch 2.9 默认的 dynamo 导出器产的图 DML 编译不了**。必须
    `torch.onnx.export(..., dynamo=False)` 走经典 TorchScript 导出器。
    （这是最隐蔽的一条：session 能建起来，只在第一次 Run 时炸，靠 probe 才抓得到。）
12. **视觉塔必须固定 batch=1**，不能 `dynamic_axes={0:"batch"}`：
    DML 对自由维度的 Reshape 直接抛 0x8007023E。推理本来就是逐张，固定 1 零损失。

验证方法（venv 里）：

```python
import onnxruntime as ort, numpy as np
s = ort.InferenceSession("models/clip/vision.onnx", providers=["DmlExecutionProvider"])
print(s.get_providers())        # 必须是 ['DmlExecutionProvider', 'CPUExecutionProvider']
s.run(None, {"pixel_values": np.zeros((1,3,224,224), np.float32)})
# 实测 MX250：115 ms/张（DML） vs 430 ms/张（CPU）
```

打包后看黑窗口首行 `[model] CLIP 就绪 … 'vision': ['DmlExecutionProvider', …]`，
界面顶部「模型」标签会显示 **GPU 加速（DirectML）**。

### 安装包（build_installer.py）结构

```
语义相册Setup.exe
├─ PyInstaller onefile stub（windowed，Tk 安装向导，约 20MB）
└─ payload.zip（--add-data：应用全部文件 + Uninstall.exe）
```

- stub 是 windowed，没有控制台。**stdout 优先写自己的日志文件**（`%TEMP%\setup_log_<pid>\setup.log`），
  attach 控制台放最后 —— 先 `AttachConsole(-1)` 再 `open("CONOUT$")` 时，如果父进程把输出接了管道，
  写 CONOUT$ 会一直阻塞，表现就是「--silent 装到一半再也不动」。
- 安装到 `%LOCALAPPDATA%\Programs\语义相册`（HKCU 注册卸载信息，无需管理员）；
  `--silent` 可用 `--dir=路径` 指定别处（多实例/脚本测试）。
- 卸载器 = 同一个 stub 改名为 `Uninstall.exe`（无 payload），靠文件名进卸载模式
  （`is_uninstall_mode` 同时认旧名 `卸载语义相册.exe`）；
  卸载前会校验目标目录里确实有 `语义相册.exe` 或 `data`，避免误删。

### 安装器两个致命坑（2026-10-18 修）

1. **别用 `tasklist /FI "IMAGENAME eq 语义相册.exe"` 判断程序是否在跑。**
   Python 3.14+ 默认 UTF-8 模式，tasklist 输出是 GBK，`text=True` 解码直接抛
   `UnicodeDecodeError`，又被 `except: return False` 吞掉 → 永远认为没在运行 →
   不杀进程 → 覆盖 `_internal\*.pyd` 时 `PermissionError`。
   现在改用 `ctypes` 的 `CreateToolhelp32Snapshot`（Unicode API）枚举，实测 0.9 秒内能关掉。
2. **解包不能「一个文件失败就整包崩」，也不能无限重试。**
   单文件写入加重试 4 次（0.25/0.5/0.75s 退避）；连续 20 个失败就判定目标目录不可写并中止
   （否则 6000 个文件 × 2 秒 = 空转几小时）。另外解包前先写一个 `_wb_test_<pid>` 探针，
   不可写则 12 秒内直接报 `INSTALL_FAIL 目标目录不可写`。
3. 卸载器名字用英文 `Uninstall.exe`：部分杀软对「卸载*.exe」有启发式，落盘会被拒。

### windowed 打包 + onnxruntime 兜底（2026-10-18 晚）

**1. 从 `--console` 改 `--windowed` 不是加一个开关就完事。**
PyInstaller windowed 模式下 `sys.stdout` / `sys.stderr` 就是 `None`，任何 `print`
直接 `AttributeError`，而且用户看不到任何报错。所以 `win_main.py` 里
`_setup_io()` 必须在**第一个 print 之前**跑：把 stdout/stderr 接到
`data\logs\app.log`（超过 4MB 滚一份 `app.old.log`），控制台模式用 `_Tee` 同时写两边。
另一条命门是 `uvicorn`：它的 `StreamHandler` 在 import 时就绑 `sys.stderr`，
所以重定向必须早于 `import uvicorn`。
没有控制台之后「怎么退出」成了新问题 —— 界面 ⋯ 里加「退出语义相册」，
配 `/api/shutdown`（把 `uvicorn.Server.should_exit` 置真，实测日志会走完
`Shutting down → Finished server process`）；再配 `/api/ping` 做单实例检测，
第二次双击 exe 只把窗口拉出来，不起第二个进程。
排障时 `语义相册.exe --console` 用 `AllocConsole()` 叫出控制台（注意要重新
`open("CONOUT$")`，原来的 stdout 还是空的）。

### 关窗即退出（2026-10-23）

双击 exe 打开的是 Edge `--app` 窗口。之前关掉窗口后服务还在后台跑，得再去点
⋯ → 退出语义相册。现在改成：**关掉窗口，后台进程一起退**。

做法：`ui_open.app_window_proc()` 返回 Popen → `win_main.spawn_window()` 把它交给
`_watch_window()` 线程 → 进程结束就 `srv.should_exit = True`，走 uvicorn 优雅收尾。
⋯ →「退出语义相册」也改成先给窗口发 `WM_CLOSE` 再退出，不留一个连不上的空壳。

三条必须记住的：

1. **一定要带独立 `--user-data-dir`**（`%LOCALAPPDATA%\aiwin_ui`）。否则 Edge 会把
   `--app` 转交给已经在跑的实例，我们手里那个进程 0.1 秒就退出了 —— 看起来和
   「用户刚打开就关掉」一模一样，服务会被当场带走。另有 `MIN_WINDOW_LIFE=3s` 兜底：
   活不过 3 秒的判定为「转交掉了」，不跟随退出。
2. **第二次双击不能再挂监控**。已经有受监控窗口时 `spawn_window` 只调
   `ui_open.app_window()` 多开一个窗口，那个新 Popen 必然被转交、秒退；
   真正的浏览器进程还是原来那个 —— 关掉**所有**窗口它才退出，这正是想要的行为。
3. **拿不到进程时不挂监控**（机器上没有 Edge/Chrome，退回默认浏览器），
   日志里会写明「关掉页面不会退出程序，请点 ⋯ → 退出语义相册」。

验收脚本 `verify_closeexit.py` 覆盖这条链路：用 `WM_CLOSE` 模拟用户点 ✕
（比 taskkill 更接近真实操作），并检查「关一个窗口服务不倒、关完才倒、无残留窗口」。

**2. `ImportError: DLL load failed while importing onnxruntime_pybind11_state:
动态链接库(DLL)初始化例程失败` —— 不是打包漏文件。**
对过 `_internal\onnxruntime\capi` 与 site-packages，`DirectML.dll` /
`onnxruntime.dll` / `onnxruntime_providers_shared.dll` / `*.pyd` 字节数完全一致
（`onnxruntime-directml` 1.24 的 DML 执行器是**静态链进 onnxruntime.dll** 的，
本来就没有 `onnxruntime_providers_dml.dll`）。
关键词是「初始化例程失败」= `ERROR_DLL_INIT_FAILED`，DLL 找到了但 DllMain 返回 FALSE：
`DirectML.dll` 需要 D3D12 设备，**RDP 会话 / 虚拟机 / 老驱动 / Win7-8** 都会失败。
而 `import onnxruntime` 那一步就会初始化它，所以整个程序连界面都起不来。

修法是 `app/ort_loader.py`：
- 先 `ctypes.WinDLL` 单独探一次 `DirectML.dll`（加载完立刻 `FreeLibrary`，不留副作用）；
- 探测失败就**绝不 import DML 版**，直接把 `_internal\ort_cpu` 顶到 `sys.path[0]`，
  用打包进去的纯 CPU 版（31MB，`onnxruntime` 官方 wheel 的 `onnxruntime/` 目录原样解包）；
- 两版靠**目录隔离 + `sys.path` 顺序**共存：CPython 在 Windows 上以
  `LOAD_WITH_ALTERED_SEARCH_PATH` 加载扩展，优先搜 pyd 自己所在目录，
  所以各自的 `onnxruntime.dll` 不会串；
- 探测通过但 import 仍炸时还有一层：`_drop_modules()` 后重试 CPU 版。
- `AIWIN_ORT=cpu` 可强制走 CPU；界面「模型」那行会把降级原因写出来。

实测：默认 131ms/张（DmlExecutionProvider），`AIWIN_ORT=cpu` 402ms/张（CPU），
「先 import DML 版再切 CPU 包」这种最坏情况也能正常出推理。
