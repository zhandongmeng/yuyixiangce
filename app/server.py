#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""语义相册 · Windows 本地版
以文搜图 / 以图搜图 / 以话搜视频音频 + 智慧相册（人像 / 城市 / 场景 / 事物）

编码栈（全部本地、可离线）：
  图文  : chinese-clip-vit-base-patch16 → ONNX int8 → onnxruntime-directml（MX250/集显，自动回退 CPU）
  语音  : faster-whisper tiny (int8 CPU) → 文字 → 文本塔向量，同时保存转写文本
  人脸  : YuNet + ArcFace ONNX（与手机端同款），CPU
依赖：onnxruntime-directml / tokenizers / pillow / pillow-heif / av / faster-whisper
"""
import hashlib
import io
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import threading
import time
import traceback
import urllib.request
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

# Windows 控制台默认 GBK：中文路径/emoji 日志会直接 UnicodeEncodeError 崩掉
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
from PIL import Image

import ort_loader          # onnxruntime 的分级加载（DirectML 探测 + CPU 兜底）

# ---------------------------------------------------------------- 路径
def _app_home() -> Path:
    env = os.environ.get("AIWIN_HOME")
    if env:
        return Path(env)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent      # exe 所在目录
    return Path(__file__).resolve().parent.parent         # 开发态

import sys  # noqa: E402

APP_HOME = _app_home()
MODEL_DIR = APP_HOME / "models"          # models/clip  models/whisper  models/face
DATA_DIR = APP_HOME / "data"             # media.db + thumbs + trash
CLIP_DIR = MODEL_DIR / "clip"
WHISPER_DIR = MODEL_DIR / "whisper"
FACE_DIR = MODEL_DIR / "face"
THUMB_DIR = DATA_DIR / "thumbs"
DB_PATH = DATA_DIR / "media.db"
TRASH_DIR = DATA_DIR / "trash"
DIM = 512

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".heic", ".heif",
           ".tiff", ".avif"}
VID_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv", ".mpg",
           ".mpeg", ".3gp", ".ts", ".m2ts"}
AUD_EXT = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma",
           ".aiff", ".aif"}
IMG_BATCH = 4                              # CPU 上图片逐张编码最稳；DirectML 下批量更快
AUDIO_MAX_CHUNK = 30                       # 单个音频最多索引 30 段（每段 30 秒）
VIDEO_AUDIO_MAX_CHUNK = 20                 # 视频音轨最多索引 20 段
THUMB_MAX = 640
IGNORE_TH = 0.58                           # ArcFace 忽略名单阈值
FACE_CONF = 0.70
FACE_MINPX = 40
# 聚类三档阈值（都是余弦，人脸对齐后的典型分布：同一人 0.5~0.8，不同人 0.0~0.2）
FACE_TH = float(os.environ.get("FACE_TH", "0.42"))          # 单张脸归入某个簇
MERGE_CEN_TH = float(os.environ.get("MERGE_CEN", "0.52"))   # 两簇中心要多像才考虑合并
MERGE_AVG_TH = float(os.environ.get("MERGE_AVG", "0.40"))   # 两簇成员的平均相似度下限
TAG_MIN = float(os.environ.get("TAG_MIN", "0.16"))   # CLIP 余弦阈值（图文空间，量级远小于 EmbeddingGemma）
# 密度约束：一个「人物」至少要有几张脸。1 张脸也占一个分组时，误检（海报/卡通脸）、
# 背景路人会把人像页塞满；但设太大又会让小图库全是空白，所以库里脸少于 12 张时自动降为 1。
FACE_MIN_FACES = int(os.environ.get("FACE_MIN_FACES", "2"))
# 整个库人脸少于这个数时，密度约束自动失效（minf=1）。新用户只导入几十张照片，
# 要是按 2 张脸起步，人像页几乎全空，比"乱"更劝退。
FACE_SMALL_LIB = int(os.environ.get("FACE_SMALL_LIB", "12"))

for d in (DATA_DIR, THUMB_DIR, TRASH_DIR):
    d.mkdir(parents=True, exist_ok=True)

try:                                        # Pillow ≥ 9.1
    RES_LANCZOS = Image.Resampling.LANCZOS
    RES_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:                      # noqa: E722
    RES_LANCZOS = Image.LANCZOS
    RES_BILINEAR = Image.BILINEAR

MODEL_READY = False
MODEL_ERR = ""
MODEL_INFO = {"providers": [], "note": "", "ort": "", "diag": []}

# ================================================================ 编码层
class ClipEncoder:
    """中文 CLIP ONNX（视觉塔 int8 / 文本塔 fp32）。

    DirectML 可用时视觉塔跑 MX250/集显（带真实推理探测，失败自动回落 CPU）；
    文本塔固定 CPU —— 实测 DML 对动态序列维的 BERT 图直接报 E_INVALIDARG，
    而文本塔很小（一次查询/一次标签就几句），CPU fp32 又保精度又够快。
    """

    def __init__(self):
        from ort_loader import ort
        from tokenizers import BertWordPieceTokenizer

        meta = json.loads((CLIP_DIR / "meta.json").read_text(encoding="utf-8"))
        self.dim = int(meta["dim"])
        self.size = int(meta.get("image_size", 224))
        self.mean = np.array(meta["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(meta["std"], dtype=np.float32).reshape(1, 3, 1, 1)
        self.max_len = int(meta.get("max_text_len", 52))

        so = ort.SessionOptions()
        so.intra_op_num_threads = max(2, min(8, (os.cpu_count() or 4)))
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._dml = "DmlExecutionProvider" in ort.get_available_providers()

        def make_session(fname, prefer_dml, probe):
            """按候选 provider 依次尝试；probe 是真实推理输入，DML 的坑在这暴露"""
            cands = (["DmlExecutionProvider", "CPUExecutionProvider"]
                     if (prefer_dml and self._dml) else [])
            cands += ["CPUExecutionProvider"]
            last = None
            for i in range(len(cands)):
                prov = cands[i:]                 # 注意：是列表，不能是单个字符串
                try:
                    s = ort.InferenceSession(str(CLIP_DIR / fname), so,
                                             providers=prov)
                    if probe is not None:
                        s.run(None, probe)
                    return s, s.get_providers()
                except Exception as e:
                    last = e
                    print("[model] %s on %s 失败：%s" % (fname, prov[0], e), flush=True)
            raise last

        self.vis, pv = make_session(
            "vision.onnx", True,
            {"pixel_values": np.zeros((1, 3, self.size, self.size), np.float32)})
        self.txt, pt = make_session("text.onnx", False, None)
        self.providers = {"vision": pv, "text": pt}
        self.tok = BertWordPieceTokenizer(str(CLIP_DIR / "vocab.txt"),
                                          lowercase=True, handle_chinese_chars=True)
        self.tok.enable_truncation(self.max_len)
        self.lock = threading.Lock()

    def _vision_run(self, batch):
        """视觉推理：万一运行时炸了（显存不够/驱动抽风），自动回落 CPU 再试一次"""
        try:
            with self.lock:
                return self.vis.run(None, {"pixel_values": batch})[0]
        except Exception as e:
            print("[model] vision 运行失败，回落 CPU：%s" % e, flush=True)
            from ort_loader import ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = max(2, min(8, (os.cpu_count() or 4)))
            self.vis = ort.InferenceSession(
                str(CLIP_DIR / "vision.onnx"), so,
                providers=["CPUExecutionProvider"])
            self.providers["vision"] = self.vis.get_providers()
            return self.vis.run(None, {"pixel_values": batch})[0]

    # ---- 图片 ----
    def prep_image(self, path):
        im = Image.open(path)
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except Exception:
            pass
        im = im.convert("RGB").resize((self.size, self.size), RES_BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0
        a = np.transpose(a, (2, 0, 1))[None]        # (1,3,H,W)
        a = (a - self.mean) / self.std
        return a.astype(np.float32)

    def encode_images(self, paths):
        # vision.onnx 固定 batch=1（DirectML 不支持自由批维度的 Reshape）→ 逐张跑
        out = []
        for p in paths:
            v = self._vision_run(self.prep_image(p))
            out.append(np.asarray(v, dtype=np.float32)[0])
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        v = np.stack(out, 0)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    def encode_image_pil(self, im):
        im = im.convert("RGB").resize((self.size, self.size), RES_BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0
        a = np.transpose(a, (2, 0, 1))[None]
        a = ((a - self.mean) / self.std).astype(np.float32)
        v = self._vision_run(a)
        v = np.asarray(v, dtype=np.float32)
        return (v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9))[0]

    # ---- 文本 ----
    def encode_texts(self, texts, batch=16):
        out = []
        texts = list(texts)
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            encs = self.tok.encode_batch(chunk)
            L = min(max(len(e.ids) for e in encs), self.max_len)
            ids = np.zeros((len(encs), L), dtype=np.int64)
            am = np.zeros((len(encs), L), dtype=np.int64)
            for r, e in enumerate(encs):
                n = min(len(e.ids), L)
                ids[r, :n] = e.ids[:n]
                am[r, :n] = 1
            tt = np.zeros_like(ids)
            with self.lock:
                v = self.txt.run(None, {"input_ids": ids, "attention_mask": am,
                                        "token_type_ids": tt})[0]
            out.append(np.asarray(v, dtype=np.float32))
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        v = np.concatenate(out, 0)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    def encode_text(self, text):
        return self.encode_texts([text])[0]


CLIP = None
_WHISPER = None
WHISPER_LOCK = threading.Lock()


def load_model():
    global CLIP, MODEL_READY, MODEL_ERR
    t0 = time.time()
    try:
        if not (CLIP_DIR / "vision.onnx").exists():
            MODEL_ERR = "缺少模型文件：models\\clip\\vision.onnx（请从安装包完整复制 models 目录）"
            print("[model] " + MODEL_ERR, flush=True)
            return
        # 先定 onnxruntime 用哪一版。DirectML 在这台机器上起不来时，
        # 这里会自动切到内置的 CPU 版，而不是让 import 异常把整个程序带走。
        _, ort_mode, ort_diag = ort_loader.load()
        MODEL_INFO["ort"] = ort_mode
        MODEL_INFO["diag"] = list(ort_diag)
        for line in ort_diag:
            print("[ort] " + line, flush=True)

        CLIP = ClipEncoder()
        MODEL_INFO["providers"] = CLIP.providers
        gpu = any("DmlExecutionProvider" in (p or []) for p in CLIP.providers.values())
        if gpu:
            MODEL_INFO["note"] = "GPU 加速（DirectML）"
        elif ort_mode == "cpu-fallback":
            # 把探测结论带上，用户看到「CPU 模式」时能知道为什么
            why = [d for d in ort_diag if "DirectML" in d or "不可用" in d]
            MODEL_INFO["note"] = ("CPU 模式（DirectML 不可用，已自动降级%s）"
                                  % ("：" + why[-1] if why else ""))
        else:
            MODEL_INFO["note"] = "CPU 模式（未检测到可用独显/集显加速）"
        # 预热
        CLIP.encode_text("预热")
        CLIP.encode_image_pil(Image.new("RGB", (64, 64), (120, 130, 140)))
        MODEL_READY = True
        print("[model] CLIP 就绪 %.1fs · %s · %s"
              % (time.time() - t0, CLIP.providers, MODEL_INFO["note"]), flush=True)
    except Exception as e:
        if not ort_loader.MODE:
            # 连 onnxruntime 都没起来 —— 用「照着做」的话代替 traceback
            MODEL_ERR = ort_loader.human_error()
        else:
            MODEL_ERR = "%s: %s" % (type(e).__name__, e)
        print("[model] FAILED " + MODEL_ERR, flush=True)
        traceback.print_exc()


def whisper():
    """懒加载 whisper（只有真的遇到音频/视频才初始化，省内存）"""
    global _WHISPER
    if _WHISPER is not None:
        return _WHISPER
    with WHISPER_LOCK:
        if _WHISPER is not None:
            return _WHISPER
        if not (WHISPER_DIR / "model.bin").exists():
            return None
        from faster_whisper import WhisperModel
        t0 = time.time()
        _WHISPER = WhisperModel(str(WHISPER_DIR), device="cpu",
                                compute_type="int8",
                                cpu_threads=max(2, min(6, os.cpu_count() or 4)))
        print("[whisper] 就绪 %.1fs" % (time.time() - t0), flush=True)
        return _WHISPER


WHISPER_LANG = os.environ.get("WHISPER_LANG", "zh")   # 默认按中文转写（家庭相册场景）


def transcribe(y):
    """float32 16k 单声道 → 文本"""
    m = whisper()
    if m is None or y is None or len(y) < 1600:
        return ""
    try:
        kw = dict(beam_size=1, vad_filter=False, without_timestamps=True)
        if WHISPER_LANG:
            kw["language"] = WHISPER_LANG
        segs, _info = m.transcribe(y, **kw)
        return " ".join((s.text or "").strip() for s in segs).strip()
    except Exception as e:
        print("[whisper] 识别失败：%s" % e, flush=True)
        return ""


# 旧接口名兼容（移植时减少改动）
encode_texts = lambda texts, is_query=False: CLIP.encode_texts(texts)  # noqa: E731


def encode_media(batch):
    """phone 版接口兼容：batch 是 [{"image": path}]"""
    return CLIP.encode_images([b["image"] for b in batch])


threading.Thread(target=load_model, daemon=True).start()

# ================================================================ 向量库
class Store:
    def __init__(self):
        self.db = sqlite3.connect(DB_PATH, check_same_thread=False)
        # 必须是可重入锁：db() 是上下文管理器，业务代码里偶尔会出现
        # 「在 with db() 里再调一个自己开 db() 的函数」，普通 Lock 会直接
        # 死锁（整个服务所有接口挂住，没有任何报错）。RLock 让同线程重入无害。
        self.lock = threading.RLock()
        self.gen = 0        # 内容版本号：每次 reload 递增，给下游缓存失效用
        self.db.execute("""CREATE TABLE IF NOT EXISTS media(
            id INTEGER PRIMARY KEY,
            path TEXT, kind TEXT, mtime REAL, size INTEGER,
            w INTEGER, h INTEGER, seg REAL, dur REAL, vec BLOB,
            note TEXT DEFAULT '', src TEXT DEFAULT '')""")
        mcols = [r[1] for r in self.db.execute("PRAGMA table_info(media)")]
        if "ph" not in mcols:
            # ph = 64 位感知哈希（dHash），用来找连拍/重复副本
            self.db.execute("ALTER TABLE media ADD COLUMN ph INTEGER")
            self.db.commit()
        self.db.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS folders(path TEXT PRIMARY KEY, added REAL)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_media_path ON media(path)")
        self.db.commit()
        self.reload()
        self.load_folders()

    def reload(self):
        """重建内存里的向量表。

        整个重建过程必须在锁内、并且最后一次性换上去：
        以前是「锁里读一次数据库 → 出锁后逐个给 self.paths/self.n/self.V 赋值」，
        索引线程跑到一半时如果来个搜索请求，它可能看到新的 self.n 配旧的
        self.paths —— 直接 IndexError 500，或者更糟，读到张冠李戴的路径。
        """
        with self.lock:
            rows = self.db.execute(
                "SELECT id,path,kind,mtime,size,w,h,seg,dur,vec,note,src FROM media"
                " ORDER BY id").fetchall()
            ids, paths, kinds = [], [], []
            mtimes, sizes, segs, durs, wh = [], [], [], [], []
            notes, srcs = [], []
            vecs = []
            for r in rows:
                v = np.frombuffer(r[9], dtype=np.float32)
                if v.shape[0] != DIM:
                    continue
                ids.append(r[0]); paths.append(r[1]); kinds.append(r[2])
                mtimes.append(r[3]); sizes.append(r[4]); wh.append((r[5], r[6]))
                segs.append(r[7]); durs.append(r[8]); vecs.append(v)
                notes.append(r[10] or "")
                srcs.append(r[11] or "")
            V = (np.vstack(vecs).astype(np.float32)
                 if vecs else np.zeros((0, DIM), dtype=np.float32))
            # ---- 原子替换 ----
            self.ids, self.paths, self.kinds = ids, paths, kinds
            self.mtimes, self.sizes, self.segs, self.durs = mtimes, sizes, segs, durs
            self.wh, self.notes, self.srcs = wh, notes, srcs
            self.V, self.n = V, len(ids)
            # 内容版本号：下游（照片墙排序缓存等）靠它判断该不该重算。
            # 只看数量是不够的 —— 改名/换文件时条数不变，缓存却已经失效。
            self.gen = getattr(self, "gen", 0) + 1
        print("[store] loaded %d vectors" % self.n, flush=True)

    def load_folders(self):
        with self.lock:
            rows = self.db.execute("SELECT path FROM folders ORDER BY added").fetchall()
        global MEDIA_ROOTS
        MEDIA_ROOTS = [r[0] for r in rows]
        print("[store] 监控目录：%s" % (MEDIA_ROOTS or "（空）"), flush=True)

    def add_folder(self, path):
        path = str(Path(path).resolve())
        if not os.path.isdir(path):
            return False, "目录不存在：%s" % path
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO folders(path,added) VALUES(?,?)",
                            (path, time.time()))
            self.db.commit()
        self.load_folders()
        return True, path

    def remove_folder(self, path):
        with self.lock:
            self.db.execute("DELETE FROM folders WHERE path=?", (path,))
            self.db.commit()
        self.load_folders()
        return True

    def upsert(self, path, kind, mtime, size, w, h, seg, dur, vec, note="", src="",
               ph=None):
        b = np.asarray(vec, dtype=np.float32).tobytes()
        with self.lock:
            self.db.execute("DELETE FROM media WHERE path=? AND abs(seg-?)<0.01",
                            (path, seg))
            self.db.execute("INSERT INTO media(path,kind,mtime,size,w,h,seg,dur,vec,"
                            "note,src,ph) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                            (path, kind, mtime, size, w, h, seg, dur, b, note, src, ph))
            self.db.commit()

    def mark_failed(self, path, mtime):
        with self.lock:
            self.db.execute("DELETE FROM media WHERE path=?", (path,))
            self.db.execute("INSERT INTO media(path,kind,mtime,size,w,h,seg,dur,vec,note,src)"
                            " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                            (path, "failed", mtime, 0, 0, 0, 0.0, 0.0, b"", "", ""))
            self.db.commit()

    def has(self, path, mtime):
        with self.lock:
            r = self.db.execute("SELECT count(*) FROM media WHERE path=? AND"
                                " abs(mtime-?)<1", (path, mtime)).fetchone()
        return r[0] > 0

    def get_meta(self, k, d=None):
        with self.lock:
            r = self.db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return r[0] if r else d

    def set_meta(self, k, v):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO meta(k,v) VALUES(?,?)", (k, v))
            self.db.commit()


STORE = Store()
MEDIA_ROOTS = []
STORE.load_folders()


def dhash64(im):
    """64 位差异哈希：缩到 9×8 灰度，比较相邻像素亮度，连拍/副本几乎一样。

    和 CLIP 向量不同，它只关心"画面上是不是同一张"，不受语义影响 ——
    找重复照片用向量反而会误伤（同一个场景的十张不同照片也很像）。
    """
    try:
        g = im.convert("L").resize((9, 8), RES_BILINEAR)
        px = list(g.getdata())
    except Exception:
        return None
    bits = 0
    for r in range(8):
        row = r * 9
        for c in range(8):
            bits = (bits << 1) | (1 if px[row + c] > px[row + c + 1] else 0)
    return bits - (1 << 64) if bits >= (1 << 63) else bits     # SQLite 存有符号 64 位


PHASH_TH = int(os.environ.get("PHASH_TH", "6"))   # 汉明距离 <= 这个值算重复
# 单个 LSH 桶最多容纳多少个候选。纯色图/纯黑截图会把桶撑爆，
# 桶内两两比较是 O(k²)，太大就退化；超过就整个桶放弃（宁可漏检也不卡住）。
DUP_BUCKET_MAX = int(os.environ.get("DUP_BUCKET_MAX", "200"))
_PH = {"running": False, "done": 0, "total": 0}


def _lsh_segments(nseg):
    """把 64 位哈希切成 nseg 段，返回 [(shift, mask), ...]。

    鸽笼原理：两串哈希汉明距离为 d 时，切成 d+1 段必有一段完全相同。
    所以**段数必须 >= 阈值+1**，否则会漏掉「每个段都被改了几个位」的那类配对。
    最后一段吃掉除不尽的余位，保证 64 位全覆盖。
    """
    w = 64 // nseg
    out = []
    for s in range(nseg):
        lo = s * w
        hi = 64 if s == nseg - 1 else (s + 1) * w
        out.append((lo, (1 << (hi - lo)) - 1))
    return out


def phash_missing():
    with db() as c:
        return c.execute("SELECT count(*) FROM media WHERE kind='image'"
                         " AND ph IS NULL AND ifnull(vec,'')<>''").fetchone()[0]


_PH_LOCK = threading.Lock()


def build_phash():
    """给还没有指纹的图片补算 dHash（老库升级上来时用，后台跑）。

    用非阻塞抢锁而不是只看 _PH["running"]：running 是函数内部才置位的，
    两个 /api/dups 请求同时进来都能通过外部检查，然后起两个线程一起扫盘。
    """
    if not _PH_LOCK.acquire(blocking=False):
        return 0
    try:
        with db() as c:
            rows = c.execute("SELECT DISTINCT path FROM media WHERE kind='image'"
                             " AND ph IS NULL AND ifnull(vec,'')<>''").fetchall()
        _PH.update(running=True, done=0, total=len(rows))
        n = 0
        for i, (p,) in enumerate(rows):
            _PH["done"] = i + 1
            try:
                with Image.open(p) as im0:
                    h = dhash64(im0.convert("RGB"))
            except Exception:
                continue
            if h is None:
                continue
            with db() as c:
                c.execute("UPDATE media SET ph=? WHERE path=?", (h, p))
            n += 1
        print("[phash] 补算完成：%d/%d 张" % (n, len(rows)), flush=True)
        return n
    finally:
        # 必须放 finally：中途任何异常都不能把 running 留在 True，
        # 否则界面永远显示"正在补算"，而且再也不会重新触发
        _PH["running"] = False
        _PH_LOCK.release()


def find_dups(thresh=None):
    """返回重复组：[{n, cover, keep, waste, items:[{path,size,w,h,mtime}]}]

    先用「64 位哈希切段」做 LSH 粗筛（鸽笼原理：汉明距离 ≤d 的两张图，
    切成 d+1 段必有一段完全相同），再在候选对里算真实距离 ——
    十万张图的库也不用 O(n²) 全比。

    ⚠️ 段数必须跟着阈值走。这里以前是写死的 4 段 + 默认阈值 6，
    而 4 段只保证「距离 ≤3」不漏，距离 4..6 的配对大约有一半会掉段，
    实测漏检 41.9% —— 用户看到的就是"明明有重复却没找出来"。
    """
    th = PHASH_TH if thresh is None else int(thresh)
    th = max(0, min(th, 24))          # 再大就退化到没有粗筛的意义了
    with db() as c:
        rows = c.execute("SELECT path,ph,size,w,h,mtime FROM media"
                         " WHERE ph IS NOT NULL AND kind='image'"
                         " GROUP BY path").fetchall()
    n = len(rows)
    if n < 2:
        return []
    hs = [int(r[1]) & ((1 << 64) - 1) for r in rows]
    nseg = max(2, min(8, th + 1))
    segs = _lsh_segments(nseg)
    buckets = [{} for _ in segs]
    for i, h in enumerate(hs):
        for s, (sh, msk) in enumerate(segs):
            buckets[s].setdefault((h >> sh) & msk, []).append(i)
    cand = set()
    for bk in buckets:
        for idxs in bk.values():
            if len(idxs) > DUP_BUCKET_MAX:   # 纯色图会把桶撑爆，直接放弃
                continue
            for a in range(len(idxs)):
                ia = idxs[a]
                for b in range(a + 1, len(idxs)):
                    cand.add((ia, idxs[b]) if ia < idxs[b] else (idxs[b], ia))
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in cand:
        if bin(hs[a] ^ hs[b]).count("1") <= th:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for g in groups.values():
        if len(g) < 2:
            continue
        items = [{"path": rows[i][0], "size": int(rows[i][2] or 0),
                  "w": rows[i][3], "h": rows[i][4], "mtime": rows[i][5],
                  "name": os.path.basename(rows[i][0]),
                  "thumb": "thumb/%s" % thumb_key(rows[i][0])} for i in g]
        # 默认保留分辨率最高的（同分辨率保留文件大的，通常是原图而不是压缩过的）
        items.sort(key=lambda x: (-(x["w"] or 0) * (x["h"] or 0), -x["size"]))
        out.append({"n": len(items),
                    "cover": "thumb/%s" % thumb_key(items[0]["path"]),
                    "keep": items[0]["path"], "items": items,
                    "waste": sum(x["size"] for x in items[1:])})
    out.sort(key=lambda x: -x["waste"])
    return out


def thumb_key(path, seg=None):
    return hashlib.md5(("%s#%s" % (path, seg)).encode()).hexdigest()


def save_thumb(key, img, size=THUMB_MAX):
    try:
        im = img.convert("RGB")
        im.thumbnail((size, size), RES_LANCZOS)
        p = THUMB_DIR / ("%s.jpg" % key)
        im.save(p, "JPEG", quality=82, optimize=True)
        return p
    except Exception:
        return None


def cover_of(path, c=None):
    """给分组挑一张真正能打开的封面缩略图。

    视频/音频的缩略图是按 seg 存的（第一帧、波形图），
    md5(path#None) 并不存在，直接拼会 404、卡片一片空白。
    这里先试整文件缩略图，再退回到 media 表里任意一个存在的 seg。

    c: 可选的游标。已经在 with db() 里面时传进来可以省一次开事务。
       （早期 STORE.lock 是普通 Lock，不传会自己把自己锁死；现在是 RLock，
       不传也能跑，但没必要多开一层。）
    """
    k = thumb_key(path)
    if (THUMB_DIR / ("%s.jpg" % k)).exists():
        return "thumb/%s" % k
    segs = []
    try:
        sql = "SELECT seg FROM media WHERE path=? ORDER BY seg"
        if c is not None:
            segs = [r[0] for r in c.execute(sql, (path,))]
        else:
            with db() as c2:
                segs = [r[0] for r in c2.execute(sql, (path,))]
    except Exception:
        pass
    for s in segs:
        if s is None:
            continue
        kk = thumb_key(path, s)
        if (THUMB_DIR / ("%s.jpg" % kk)).exists():
            return "thumb/%s" % kk
    return ""


def root_of(p):
    """判断某个文件属于哪个监控目录。

    必须比到分隔符为止：光用 startswith 会把 `D:\\照片2\\x.jpg` 判给
    `D:\\照片`（纯字符串前缀），relpath 算出来的相对路径就是错的，
    拼出来的 /raw 链接 404 —— 缩略图能看、点开大图打不开。
    Windows 路径大小写不敏感，比较时统一 normcase。
    """
    if not p:
        return None
    np_ = os.path.normcase(os.path.normpath(p))
    best = None
    for r in MEDIA_ROOTS:
        nr = os.path.normcase(os.path.normpath(r))
        if np_ == nr or np_.startswith(nr + os.sep):
            # 取最长的那个：同时监控了 D:\照片 和 D:\照片\2020 时要归到更具体的
            if best is None or len(nr) > len(os.path.normcase(os.path.normpath(best))):
                best = r
    return best


def raw_url(p):
    return "raw/%s" % quote_path(os.path.relpath(p, root_of(p) or os.path.dirname(p)))


def quote_path(s):
    from urllib.parse import quote
    return quote(s)


# ================================================================ 工具
def media_info(path):
    ext = os.path.splitext(path)[1].lower()
    st = os.stat(path)
    if ext in IMG_EXT:
        try:
            with Image.open(path) as im:
                return "image", im.size[0], im.size[1], 0.0, st.st_size
        except Exception:
            return "image", 0, 0, 0.0, st.st_size
    if ext in VID_EXT:
        return "video", 0, 0, probe_duration(path), st.st_size
    if ext in AUD_EXT:
        return "audio", 0, 0, probe_duration(path), st.st_size
    return None, 0, 0, 0.0, st.st_size


def probe_duration(path):
    try:
        import av
        with av.open(path) as c:
            d = c.duration
            if d:
                return float(d) / 1e6
            for s in c.streams.video or c.streams.audio:
                if s.duration:
                    return float(s.duration * s.time_base)
    except Exception:
        pass
    return 0.0


def extract_frames(path, dur, want=6):
    import av
    out = []
    try:
        with av.open(path) as c:
            vs = c.streams.video
            if not vs:
                return out
            vs = vs[0]
            fps = float(vs.average_rate or 24)
            total = dur if dur > 0 else 60.0
            step = max(1.0, total / float(want))
            targets = [i * step + step / 2 for i in range(want)]
            vs.thread_type = "AUTO"
            idx, ti = 0, 0
            for frame in c.decode(video=0):
                ts = float(frame.pts * vs.time_base) if frame.pts is not None else idx / fps
                if ti < len(targets) and ts >= targets[ti]:
                    out.append((ts, frame.to_image()))
                    ti += 1
                    if ti >= len(targets):
                        break
                idx += 1
    except Exception:
        traceback.print_exc()
    return out


def write_wav16(p, y, sr=16000):
    import wave
    pcm = (np.clip(y, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setsamplerate(sr)
        w.writeframes(pcm.tobytes())


def load_audio_chunks(path, chunk=30.0, max_chunks=AUDIO_MAX_CHUNK):
    """[(start_sec, np.ndarray(float32 mono 16k))]，PyAV 解码 + 重采样"""
    import av
    SR = 16000
    out = []
    try:
        cont = av.open(str(path))
        astream = next((s for s in cont.streams if s.type == "audio"), None)
        if astream is None:
            cont.close()
            return out
        rs = av.AudioResampler(format="fltp", layout="mono", rate=SR)
        bufs = []
        for frame in cont.decode(astream):
            for f in rs.resample(frame):
                bufs.append(np.asarray(f.to_ndarray(), dtype=np.float32).reshape(-1))
        try:
            for f in rs.resample(None):
                bufs.append(np.asarray(f.to_ndarray(), dtype=np.float32).reshape(-1))
        except Exception:
            pass
        cont.close()
        if not bufs:
            return out
        y = np.concatenate(bufs)
    except Exception:
        traceback.print_exc()
        return out
    sr = SR
    n = int(chunk * sr)
    total = len(y)
    if total == 0:
        return out
    for i in range(0, total, n):
        out.append((i / sr, y[i:i + n]))
        if len(out) >= max_chunks:
            break
    return out


def waveform_img(data, w=640, h=360):
    from PIL import ImageDraw
    n = 1600
    if len(data) > n:
        step = len(data) // n
        a = np.abs(data[::step][:n])
    else:
        a = np.abs(data)
    a = a / (a.max() + 1e-9)
    im = Image.new("RGB", (w, h), (16, 18, 26))
    d = ImageDraw.Draw(im)
    pts = int(len(a))
    for i, v in enumerate(a):
        x = int(i * w / max(pts, 1))
        bh = max(2, int(v * (h - 60)))
        d.rectangle([x, (h - bh) // 2, x + max(1, w // pts), (h + bh) // 2],
                    fill=(96, 165, 250))
    return im


# ---------------------------------------------------------------- 索引对账
def _rename_index(old, new):
    with db() as c:
        for t in ("media", "faces", "tags", "gps", "face_scanned"):
            c.execute("UPDATE %s SET path=? WHERE path=?" % t, (new, old))
        segs = [r[0] for r in c.execute(
            "SELECT seg FROM media WHERE path=?", (new,)).fetchall()]
    for s in segs:
        a = THUMB_DIR / ("%s.jpg" % thumb_key(old, s if s else None))
        b = THUMB_DIR / ("%s.jpg" % thumb_key(new, s if s else None))
        if a.exists() and not b.exists():
            try:
                os.rename(a, b)
            except Exception:
                pass


def _purge_index(paths):
    if not paths:
        return 0
    keys = []
    with db() as c:
        for p in paths:
            for (s,) in c.execute("SELECT seg FROM media WHERE path=?", (p,)).fetchall():
                keys.append(thumb_key(p, s if s else None))
            c.execute("DELETE FROM media WHERE path=?", (p,))
            c.execute("DELETE FROM faces WHERE path=?", (p,))
            c.execute("DELETE FROM tags WHERE path=?", (p,))
            c.execute("DELETE FROM gps WHERE path=?", (p,))
            c.execute("DELETE FROM face_scanned WHERE path=?", (p,))
    for k in keys:
        f = THUMB_DIR / ("%s.jpg" % k)
        if f.exists():
            try:
                f.unlink()
            except Exception:
                pass
    return len(paths)


@contextmanager
def db():
    """数据库事务。成功才提交，出错必须回滚。

    ⚠️ 这里以前是 `try: yield; commit()  finally: pass` —— 异常时既不提交
    也不回滚，sqlite 连接就停在「有未完成事务」的状态，后面任何一次 commit
    都会把这一批半截操作（比如改了 3 个表前 2 个）一起提交上去，
    造成跨表数据不一致，而且从日志里完全看不出来。
    """
    with STORE.lock:
        try:
            yield STORE.db
        except BaseException:
            try:
                STORE.db.rollback()
            except Exception:
                pass
            raise
        else:
            STORE.db.commit()


def reconcile(disk_files):
    from collections import defaultdict
    disk = {}
    for p in disk_files:
        try:
            st = os.stat(p)
            disk[p] = (st.st_size, int(st.st_mtime))
        except Exception:
            pass
    with db() as c:
        rows = c.execute("SELECT DISTINCT path,size,mtime FROM media").fetchall()
    lib = {r[0]: (r[1] or 0, int(r[2] or 0)) for r in rows}

    gone = [p for p in lib if p not in disk]
    new = [p for p in disk if p not in lib]
    moved, dead = [], []
    if gone and new:
        bucket = defaultdict(list)
        for p in new:
            bucket[disk[p]].append(p)
        for p in gone:
            cands = bucket.get(lib[p])
            if cands:
                moved.append((p, cands.pop(0)))
            else:
                dead.append(p)
    else:
        dead = gone
    for old, np_ in moved:
        try:
            _rename_index(old, np_)
        except Exception as e:
            print("[reconcile] 改名失败 %s -> %s: %s" % (old, np_, e), flush=True)
    n_del = _purge_index(dead)
    if moved or dead:
        STORE.reload()
    return len(moved), n_del


# ================================================================ 索引
IDX = {"running": False, "done": 0, "total": 0, "file": "", "errors": 0,
       "skipped": 0, "started": 0, "log": []}
# 索引互斥锁。不能只靠 IDX["running"] 判断：那个标志是 do_index 内部才置位的，
# 两个 /api/index 请求（或用户手快连点两下）并发进来都能通过检查，
# 然后起两个线程同时写同一个库、同时调 CLIP —— 索引翻倍、进度条互相覆盖。
_IDX_LOCK = threading.Lock()


def push_log(msg):
    IDX["log"].append("%s %s" % (time.strftime("%H:%M:%S"), msg))
    IDX["log"] = IDX["log"][-200:]
    print("[index] %s" % msg, flush=True)


def walk(root):
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if not d.startswith(".")]
        for f in fn:
            if f.startswith("."):
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext in IMG_EXT or ext in VID_EXT or ext in AUD_EXT:
                yield os.path.join(dp, f)


def all_files():
    seen = []
    for r in list(MEDIA_ROOTS):
        if os.path.isdir(r):
            seen.extend(walk(r))
    return seen


def do_index(root=None):
    if not MODEL_READY:
        return
    if not _IDX_LOCK.acquire(blocking=False):
        print("[index] 已有一轮索引在跑，本次请求忽略", flush=True)
        return
    IDX.update(running=True, done=0, total=0, file="", errors=0, skipped=0,
               started=time.time(), log=[])
    try:
        files = sorted(all_files())
        try:
            n_mv, n_del = reconcile(files)
            if n_mv:
                push_log("有 %d 个文件改了名，索引已跟随（向量复用，无需重新编码）" % n_mv)
            if n_del:
                push_log("清理了 %d 个已删除文件的索引和缩略图" % n_del)
        except Exception as e:
            push_log("对账失败：%s" % e)

        IDX["total"] = len(files)
        push_log("扫描到 %d 个媒体文件" % len(files))
        pend = []

        def flush_images():
            if not pend:
                return
            items = list(pend)
            pend.clear()
            try:
                vecs = CLIP.encode_images([p for p, _, _ in items])
            except Exception as e:
                IDX["errors"] += len(items)
                push_log("批次失败：%s" % e)
                return
            if len(vecs) != len(items):
                push_log("批次返回向量数不符（%d/%d），按顺序对齐"
                         % (len(vecs), len(items)))
            for (p, mt, sz), v in zip(items, vecs):
                ww = hh = 0
                ph = None
                try:
                    # 必须用 with：批量几千张图，靠 GC 关文件会一直攒句柄
                    with Image.open(p) as im0:
                        im = im0.convert("RGB")
                    save_thumb(thumb_key(p), im)
                    ww, hh = im.size
                    # dHash 在这里一起算掉。以前这行漏了 ph，批量路径进来的
                    # 图片全是 NULL 指纹，重复照片功能只能靠 build_phash()
                    # 再全库重读一遍图片来补 —— 白扫一次磁盘。
                    ph = dhash64(im)
                except Exception:
                    pass
                STORE.upsert(p, "image", mt, sz, ww, hh, 0.0, 0.0, v, ph=ph)

        for path in files:
            if not IDX["running"]:
                break
            IDX["file"] = path
            try:
                st = os.stat(path)
                kind, w, h, dur, size = media_info(path)
                if kind is None:
                    IDX["done"] += 1     # 认不出类型也算处理过，否则进度条永远差一截
                    continue
                mtime = st.st_mtime
                if STORE.has(path, mtime):
                    IDX["skipped"] += 1
                elif kind == "image":
                    pend.append((path, mtime, size))
                    if len(pend) >= IMG_BATCH:
                        flush_images()
                else:
                    flush_images()
                    index_one(path, kind, mtime, size, w, h, dur)
            except Exception as e:
                IDX["errors"] += 1
                push_log("失败 %s: %s" % (os.path.basename(path), e))
                try:
                    STORE.mark_failed(path, os.stat(path).st_mtime)
                except Exception:
                    pass
            IDX["done"] += 1
            # 定期把内存向量表同步出去，让用户索引期间也能搜到已完成的部分。
            # 每 20 个重建一次太费：reload 是 O(n) 的全量重建，几千个文件
            # 等于白做几百次 vstack。100 个一次，体感上没区别。
            if IDX["done"] % 100 == 0:
                STORE.reload()
        flush_images()
        STORE.reload()
        push_log("索引完成：总量 %d 条向量（跳过 %d，失败 %d）"
                 % (STORE.n, IDX["skipped"], IDX["errors"]))
    finally:
        IDX["running"] = False
        IDX["file"] = ""
        _IDX_LOCK.release()


def index_one(path, kind, mtime, size, w, h, dur):
    if kind == "image":
        v = CLIP.encode_images([path])[0]
        ph = None
        try:
            im = Image.open(path).convert("RGB")
            w, h = im.size
            save_thumb(thumb_key(path), im)
            ph = dhash64(im)
        except Exception:
            pass
        STORE.upsert(path, "image", mtime, size, w, h, 0.0, 0.0, v, ph=ph)
        return 1
    if kind == "video":
        n = 0
        frames = extract_frames(path, dur)
        for i, (ts, im) in enumerate(frames):
            v = CLIP.encode_image_pil(im)
            save_thumb(thumb_key(path, ts), im)
            if i == 0:
                # 额外存一张“整文件”封面，供相册分组/列表兜底使用
                save_thumb(thumb_key(path), im)
            STORE.upsert(path, "video", mtime, size, im.size[0], im.size[1],
                         ts, dur, v)
            n += 1
        # 视频音轨：转写成文字 → 文本塔向量（可以“搜视频里说的话”）
        chunks = load_audio_chunks(path, max_chunks=VIDEO_AUDIO_MAX_CHUNK)
        for ts, y in chunks:
            txt = transcribe(y)
            if not txt:
                continue
            v = CLIP.encode_text(txt)
            save_thumb(thumb_key(path, "a%s" % ts), waveform_img(y))
            STORE.upsert(path, "video", mtime, size, 0, 0, ts, dur, v,
                         note=txt, src="ta")
            n += 1
        return n
    if kind == "audio":
        chunks = load_audio_chunks(path)
        n = 0
        for ts, y in chunks:
            txt = transcribe(y)
            save_thumb(thumb_key(path, ts), waveform_img(y))
            if txt:
                v = CLIP.encode_text(txt)
                STORE.upsert(path, "audio", mtime, size, 0, 0, ts, dur, v,
                             note=txt, src="asr")
            else:
                v = CLIP.encode_text("（无语音内容）")
                STORE.upsert(path, "audio", mtime, size, 0, 0, ts, dur, v,
                             note="", src="asr")
            n += 1
        return n
    return 0


# ================================================================ 检索
ASR_SRC = ("asr", "ta")     # 语音转写段（文本塔向量）


KW_EXACT = 0.22      # 整串落进文件名：用户就是在找这个名字，直接压过语义分
KW_PATH = 0.12       # 只是路径（目录名）里有：有用但不是铁证
KW_TOK = 0.05        # 分词命中一个词
KW_TOK_CAP = 0.12    # 分词通道总封顶


def _kw_bonus(qtext, paths):
    """文件名 / 路径命中查询词时的加分。

    纯 CLIP 有个硬伤：搜「IMG_2025」「D 盘那个旅行目录」这类"字面上的东西"
    它基本看不懂（向量空间里没有文件名的概念）。所以补一条关键词通道。

    加分分三档，不是一刀切（实测踩出来的）：
      · 查询**整串**出现在文件名里 → 0.22。这是最强的用户意图信号。
        反例：搜「IMG_20251207」，目标文件的语义余弦只有 0.36，比一张
        只是碰巧构图相似的图（0.38）还低，只加 0.05 的话它排到第 5 名去，
        用户会觉得"我搜自己文件的名字都搜不到"。0.22 足够跨过语义分歧，
        又不会让无关文件跟着浮上来。
      · 只出现在路径（目录名）里 → 0.12。
      · 分词命中 → 每个词 0.05，封顶 0.12，避免文件名巧合盖过语义主体。
    """
    if not qtext:
        return None
    q = qtext.strip().lower()
    toks = [t for t in re.split(r"[\s,，、/\\]+", qtext) if len(t) >= 2]
    if len(qtext) >= 2 and qtext not in toks:
        toks.append(qtext)                  # 整串也算一个词（中文没有空格）
    if not toks:
        return None
    bonus = np.zeros(len(paths), np.float32)
    for i, p in enumerate(paths):
        low = p.lower()
        # 整串精确命中文件名（ASCII 至少 3 字符，中文两字整串也认）
        if len(q) >= 3 or (len(q) >= 2 and not q.isascii()):
            if q in os.path.basename(low):
                bonus[i] = KW_EXACT
                continue
            if q in low:
                bonus[i] = KW_PATH
                continue
        n = sum(1 for t in toks if t.lower() in low)
        if n:
            bonus[i] = min(KW_TOK_CAP, KW_TOK * n)
    return bonus


def do_search(qvec, kind=None, k=24, qtext=None):
    S = STORE
    if S.n == 0:
        return []
    sims = S.V @ qvec
    mask = np.ones(S.n, dtype=bool)
    if kind and kind != "all":
        mask = np.array([x == kind for x in S.kinds])
    sims = np.where(mask, sims, -2.0)
    # ---- 模态校准 ----
    # 文本塔出来的向量（语音转写）彼此夹角小，和查询的余弦系统性地偏高，
    # 直接图文混排会让转写段霸榜（实测 0.55 vs 图文 0.35）。
    # 把「转写组 / 视觉组」各自的均值对齐到全局均值、尺度归一到全局标准差，
    # 任何组大小都适用（1 个成员也能对齐），排序和展示都用校准后的分数。
    rank = sims.copy()
    if mask.sum() >= 4:
        gmu = float(sims[mask].mean())
        gsd = float(sims[mask].std()) or 1e-6
        asr_m = np.array([s in ASR_SRC for s in S.srcs])
        for grp in (asr_m & mask, (~asr_m) & mask):
            idx = np.where(grp)[0]
            if len(idx) == 0:
                continue
            mu = float(sims[idx].mean())
            sd = float(sims[idx].std())
            if sd > 1e-6:
                rank[idx] = (sims[idx] - mu) / sd * gsd + gmu
            else:
                rank[idx] = sims[idx] - mu + gmu
    else:
        rank = sims
    if qtext:
        kb = _kw_bonus(qtext, S.paths)
        if kb is not None:
            rank = rank + kb
    best = {}
    for i in np.argsort(-rank):
        if not mask[i]:          # 类型筛选：不匹配的直接跳过，别让它混进结果里
            continue
        p = S.paths[i]
        if p in best:
            continue
        best[p] = i
        if len(best) >= k:
            break
    out = []
    for p, i in best.items():
        out.append({
            "path": p, "kind": S.kinds[i], "score": float(rank[i]),
            # cos 是未校准的原始余弦，给界面显示用：校准分可能为负，
            # 排序看 score，人看 cos，两个都不误导。
            "cos": float(sims[i]),
            "_r": float(rank[i]),
            "w": S.wh[i][0], "h": S.wh[i][1], "seg": S.segs[i], "dur": S.durs[i],
            "size": S.sizes[i], "mtime": S.mtimes[i],
            "note": S.notes[i], "src": S.srcs[i],
            "thumb": "thumb/%s" % thumb_key(p, S.segs[i] if S.kinds[i] != "image" else None),
            "name": os.path.basename(p),
            "rel": os.path.relpath(p, root_of(p) or os.path.dirname(p)),
            "open": raw_url(p),
        })
    out.sort(key=lambda x: -x["_r"])
    for r in out:
        r.pop("_r", None)
    return out


# ================================================================ API
from fastapi import FastAPI, File, UploadFile  # noqa: E402
from fastapi.responses import (FileResponse, HTMLResponse,  # noqa: E402
                               JSONResponse, Response)  # noqa: E402

app = FastAPI(title="语义相册 · 本地版")
# index.html 的位置：打包后在 _internal/app/，也可能在 _internal/ 或 exe 同级
_UI_CANDIDATES = [
    Path(__file__).resolve().parent / "index.html",
    Path(__file__).resolve().parent / "app" / "index.html",
    APP_HOME / "index.html",
]
UI_PATH = next((p for p in _UI_CANDIDATES if p.exists()), _UI_CANDIDATES[0])


@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse(str(UI_PATH))


@app.get("/raw/{rel:path}")
def api_raw(rel: str):
    for r in MEDIA_ROOTS:
        root_ = os.path.abspath(r)
        p = os.path.normpath(os.path.join(root_, rel))
        if p.startswith(root_ + os.sep) and os.path.isfile(p):
            return FileResponse(p, headers={
                "Cache-Control": "public, max-age=86400",
                "Content-Disposition": "inline"})
    return Response(status_code=404)


@app.get("/api/status")
def status():
    return {
        "ready": MODEL_READY, "error": MODEL_ERR, "info": MODEL_INFO,
        "indexed": STORE.n, "indexing": IDX["running"],
        "progress": {k: IDX[k] for k in
                     ("done", "total", "file", "errors", "skipped", "started")},
        "log": IDX["log"][-20:],
        "roots": MEDIA_ROOTS,
        "whisper": _WHISPER is not None or (WHISPER_DIR / "model.bin").exists(),
        "auto": {"on": AUTO["on"], "every": AUTO["every"],
                 "last": AUTO["last"], "index": AUTO["index"]},
    }


@app.get("/api/ping")
def ping():
    """给「再次双击 exe」用：已经有一个实例在跑就直接开窗口，别起第二个"""
    return {"app": "aiwin", "pid": os.getpid(),
            "port": int(os.environ.get("AIWIN_PORT") or 0)}


# 桌面模式（没有控制台窗口）下，界面上必须有个能真的退出的口子
SHUTDOWN_HOOK = None
# 开界面窗口的钩子：由 win_main 注入，这样「关窗即退出」的监控才在服务进程手里
WINDOW_HOOK = None


def set_shutdown_hook(fn):
    global SHUTDOWN_HOOK
    SHUTDOWN_HOOK = fn


def set_window_hook(fn):
    global WINDOW_HOOK
    WINDOW_HOOK = fn


@app.post("/api/shutdown")
async def shutdown():
    if SHUTDOWN_HOOK is None:
        return JSONResponse(
            {"ok": False, "msg": "当前不是桌面应用模式，请在终端里按 Ctrl+C 结束"}, 400)
    threading.Timer(0.5, SHUTDOWN_HOOK).start()   # 先把响应发回去，再退
    return {"ok": True}


@app.post("/api/show-window")
async def api_show_window():
    """第二个实例双击 exe 时走这里：由正在跑的这个实例开窗，并负责「关窗即退出」"""
    if WINDOW_HOOK is not None:
        ok = WINDOW_HOOK()
        return {"ok": bool(ok), "msg": "" if ok else "没找到可用的浏览器"}
    import ui_open
    port = int(os.environ.get("AIWIN_PORT") or 8765)
    return {"ok": ui_open.app_window("http://127.0.0.1:%d/" % port)}


@app.get("/api/reveal")
def api_reveal(what: str = "log"):
    """界面上的「打开日志」「重新打开界面」（后台运行时没有控制台，得给个出口）"""
    import ui_open
    if what == "log":
        d = DATA_DIR / "logs"
        d.mkdir(parents=True, exist_ok=True)
        ok = ui_open.reveal(str(d))
        return {"ok": ok, "msg": "" if ok else "打开目录失败"}
    if what == "home":
        return {"ok": ui_open.reveal(str(APP_HOME))}
    if what == "window":
        if WINDOW_HOOK is not None:
            ok = WINDOW_HOOK()
            return {"ok": bool(ok), "msg": "" if ok else "没找到可用的浏览器"}
        port = int(os.environ.get("AIWIN_PORT") or 8765)
        return {"ok": ui_open.app_window("http://127.0.0.1:%d/" % port)}
    return {"ok": False, "msg": "未知目标"}


@app.get("/api/stats")
def stats():
    k = {}
    for x in STORE.kinds:
        k[x] = k.get(x, 0) + 1
    return {"files": len(set(STORE.paths)), "segments": STORE.n, "by_kind": k}


@app.get("/api/folders")
def api_folders_get():
    return {"ok": True, "roots": MEDIA_ROOTS}


@app.post("/api/folders")
def api_folders_add(payload: dict):
    path = (payload.get("path") or "").strip().strip('"')
    if not path:
        return JSONResponse({"ok": False, "msg": "路径不能为空"}, 400)
    ok, msg = STORE.add_folder(path)
    if not ok:
        return JSONResponse({"ok": False, "msg": msg}, 400)
    return {"ok": True, "roots": MEDIA_ROOTS}


@app.delete("/api/folders")
def api_folders_del(payload: dict):
    STORE.remove_folder((payload.get("path") or "").strip())
    return {"ok": True, "roots": MEDIA_ROOTS}


@app.post("/api/index")
def api_index(payload: dict = None):
    if not MODEL_READY:
        return JSONResponse({"ok": False, "msg": "模型尚未加载：%s" % MODEL_ERR}, 400)
    if not MEDIA_ROOTS:
        return JSONResponse({"ok": False, "msg": "还没有添加任何文件夹"}, 400)
    if IDX["running"]:
        return {"ok": False, "msg": "正在索引中"}
    threading.Thread(target=do_index, daemon=True).start()
    return {"ok": True, "msg": "已启动索引"}


@app.post("/api/index/stop")
def api_stop():
    IDX["running"] = False
    return {"ok": True}


@app.post("/api/search")
def api_search(payload: dict):
    if not MODEL_READY:
        return JSONResponse({"ok": False, "msg": "模型尚未加载：%s" % MODEL_ERR}, 400)
    q = (payload.get("q") or "").strip()
    if not q:
        return JSONResponse({"ok": False, "msg": "请输入搜索内容"}, 400)
    kind = payload.get("kind") or "all"
    # 上限 300：k 直接决定内存和响应体积，不封顶的话一个 k=10**9 就能把服务拖死
    k = max(1, min(int(payload.get("k") or 24), 300))
    t0 = time.time()
    qv = CLIP.encode_text(q)
    res = do_search(qv, kind, k, qtext=q)
    return {"ok": True, "ms": int((time.time() - t0) * 1000), "results": res}


@app.post("/api/similar")
async def api_similar(file: UploadFile = File(...), k: int = 24, kind: str = "all"):
    if not MODEL_READY:
        return JSONResponse({"ok": False, "msg": "模型尚未加载"}, 400)
    t0 = time.time()
    data = await file.read()
    try:
        with Image.open(io.BytesIO(data)) as im0:
            im = im0.convert("RGB")
    except Exception as e:
        # 上传的不是图片 / 格式不支持：给个 400，别让它变成未捕获的 500
        return JSONResponse({"ok": False, "msg": "这张图读不了：%s" % e}, 400)
    qv = CLIP.encode_image_pil(im)
    res = do_search(qv, kind, max(1, min(int(k), 300)))
    return {"ok": True, "ms": int((time.time() - t0) * 1000), "results": res}


@app.get("/api/similar/path")
def api_similar_path(path: str, k: int = 120, kind: str = "all"):
    """库内照片直接找相似：不必让前端再上传一遍，按路径编码一次即可。"""
    if not MODEL_READY:
        return JSONResponse({"ok": False, "msg": "模型尚未加载"}, 400)
    p = os.path.normpath(path or "")
    # 只允许照片目录里的文件。这个接口会真的去打开路径读图，
    # 不校验的话，同局域网的任何页面都能借它探测/读取本机任意图片
    # （甚至用一条 UNC 路径把服务卡死在网络 IO 上）。
    if not p or not root_of(p) or not os.path.isfile(p):
        return JSONResponse({"ok": False, "msg": "路径不在已添加的照片目录里"}, 400)
    t0 = time.time()
    try:
        qv = CLIP.encode_images([p])[0]
    except Exception as e:
        return JSONResponse({"ok": False, "msg": "读不了这张图：%s" % e}, 400)
    res = do_search(qv, kind, max(1, min(int(k), 300)))
    return {"ok": True, "ms": int((time.time() - t0) * 1000), "results": res}


@app.post("/api/voice")
async def api_voice(file: UploadFile = File(...), k: int = 24, kind: str = "all"):
    """语音搜索：录音 → whisper 转写 → 文本检索"""
    if not MODEL_READY:
        return JSONResponse({"ok": False, "msg": "模型尚未加载"}, 400)
    if whisper() is None:
        return JSONResponse({"ok": False, "msg": "语音模型缺失（models\\whisper）"}, 400)
    t0 = time.time()
    data = await file.read()
    tmp = DATA_DIR / ("_v_%d.bin" % int(time.time() * 1000))
    tmp.write_bytes(data)
    try:
        chunks = load_audio_chunks(str(tmp))
        y = np.concatenate([c[1] for c in chunks]) if chunks else np.zeros(0, np.float32)
        if len(y) < 1600:
            return JSONResponse({"ok": False, "msg": "录音太短"}, 400)
        txt = transcribe(y)
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass
    if not txt:
        return JSONResponse({"ok": False, "msg": "没有识别到语音内容"}, 400)
    qv = CLIP.encode_text(txt)
    res = do_search(qv, kind, k)
    return {"ok": True, "ms": int((time.time() - t0) * 1000),
            "query": txt, "results": res}


@app.get("/api/latest")
def api_latest(k: int = 24, kind: str = "all"):
    S = STORE
    order = np.argsort(-np.array(S.mtimes)) if S.n else []
    out, seen = [], set()
    for i in order:
        p = S.paths[i]
        if p in seen or (kind != "all" and S.kinds[i] != kind):
            continue
        seen.add(p)
        out.append({
            "path": p, "kind": S.kinds[i], "w": S.wh[i][0], "h": S.wh[i][1],
            "seg": S.segs[i], "dur": S.durs[i], "size": S.sizes[i],
            "mtime": S.mtimes[i], "note": S.notes[i], "src": S.srcs[i],
            "thumb": "thumb/%s" % thumb_key(p, S.segs[i] if S.kinds[i] != "image" else None),
            "name": os.path.basename(p),
            "rel": os.path.relpath(p, root_of(p) or os.path.dirname(p)),
            "open": raw_url(p),
        })
        if len(out) >= k:
            break
    return {"ok": True, "results": out}


@app.get("/thumb/{key}")
def api_thumb(key: str):
    p = THUMB_DIR / ("%s.jpg" % (key.replace("/", ""),))
    if not p.exists():
        return Response(status_code=404)
    return FileResponse(p, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=864000"})


@app.get("/api/selftest")
def selftest():
    if not MODEL_READY:
        return JSONResponse({"ok": False, "msg": MODEL_ERR}, 400)
    out = {"providers": CLIP.providers}
    try:
        v = CLIP.encode_text("一只猫躺在草地上")
        out["text"] = {"dim": int(len(v)), "norm": round(float(np.linalg.norm(v)), 4)}
    except Exception as e:
        out["text"] = {"err": str(e)}
    try:
        from PIL import Image as _I
        im = _I.new("RGB", (224, 224), (200, 120, 40))
        v = CLIP.encode_image_pil(im)
        out["image"] = {"dim": int(len(v)), "norm": round(float(np.linalg.norm(v)), 4)}
    except Exception as e:
        out["image"] = {"err": "%s: %s" % (type(e).__name__, e)}
    return {"ok": True, "res": out}


@app.on_event("startup")
def _boot_album_tables():
    _init_album()
    _maybe_upgrade_faces()


def _maybe_upgrade_faces():
    """人脸编码方式变了（比如这次加 5 点对齐）就得整批重扫。

    旧向量和新向量根本不在一个空间里，留着只会把聚类搅乱。
    重扫放后台，不挡启动；用户起的名字按 (照片路径, 人脸框) 认领回来。
    """
    if not face_upgrade_needed():
        return
    with db() as c:
        has = c.execute("SELECT count(*) FROM faces").fetchone()[0]
    if not has:                       # 还没扫过脸，交给正常的建索引流程
        face_ver_set(FACE_VER)
        return

    def _work():
        for _ in range(1200):         # 最多等 10 分钟模型就绪
            if MODEL_READY:
                break
            time.sleep(0.5)
        if not MODEL_READY:
            return
        alog("人脸识别算法已升级（%s），正在后台重新扫描人脸，旧分组的名字会保留 …" % FACE_VER)
        run_album("faces")

    threading.Thread(target=_work, daemon=True).start()
    alog("检测到人脸算法升级，稍后自动重扫（不影响其它功能）")



# ================================================================ 智慧相册
TAG_SETS = {
    "thing": [
        ("人物", "人物 人像 一个人的照片"),
        ("美食", "美食 食物 菜肴 饭菜 甜点"),
        ("动物", "动物 猫 狗 宠物 鸟"),
        ("花卉植物", "花 植物 树木 绿植 草地"),
        ("车辆", "汽车 车辆 摩托车 自行车 交通"),
        ("建筑", "建筑 楼房 高楼 房屋 桥梁"),
        ("夜景", "夜景 夜晚 灯光 夜色"),
        ("天空云彩", "天空 云彩 蓝天 晚霞"),
        ("雪景", "雪 雪景 冬天 雪地"),
        ("水面", "水 湖 河 海 江 水面"),
        ("运动", "运动 健身 球场 跑步 球赛"),
        ("聚会", "聚会 聚餐 朋友合影 庆典"),
        ("宝宝", "婴儿 小孩 宝宝 儿童"),
        ("自拍", "自拍 前置摄像头 半身照"),
        ("文档截图", "文档 截图 文字 表格 页面 便签"),
        ("商品", "商品 购物 物品 包装"),
        ("电子产品", "手机 电脑 键盘 屏幕 电子设备"),
        ("室内", "室内 房间 居家 家具"),
    ],
    "place": [
        ("海边", "海边 沙滩 海岸 海浪"),
        ("山区", "山 山脉 高原 登山 山顶"),
        ("森林", "森林 树林 树木 林地"),
        ("城市街道", "城市 街道 马路 高楼大厦"),
        ("居家室内", "家里 客厅 卧室 家中室内"),
        ("办公室", "办公室 工位 会议室 写字楼"),
        ("公园", "公园 草地 游乐场 广场"),
        ("商场", "商场 超市 店铺 商业街"),
        ("餐厅", "餐厅 饭店 咖啡馆 食堂"),
        ("学校", "学校 教室 校园 操场"),
        ("交通枢纽", "机场 车站 地铁 高铁 码头"),
        ("乡村田野", "乡村 田野 农田 村庄 农家"),
        ("沙漠", "沙漠 戈壁 沙丘 荒漠"),
        ("湖泊河流", "湖 河 溪流 水塘"),
        ("日落", "日落 夕阳 黄昏 晚霞"),
        ("夜市街景", "夜市 夜景 霓虹灯 夜间街道"),
        ("名胜古迹", "古迹 寺庙 古建筑 景区"),
    ],
}


def _init_album():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS faces(
            id INTEGER PRIMARY KEY, path TEXT, box TEXT, score REAL,
            person INTEGER DEFAULT -1, vec BLOB)""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_faces_path ON faces(path)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person)")
        c.execute("""CREATE TABLE IF NOT EXISTS people(
            id INTEGER PRIMARY KEY, name TEXT DEFAULT '', cover TEXT DEFAULT '',
            center BLOB)""")
        c.execute("""CREATE TABLE IF NOT EXISTS tags(
            path TEXT, cat TEXT, tag TEXT, score REAL,
            PRIMARY KEY(path, cat, tag))""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tags ON tags(cat, tag)")
        c.execute("""CREATE TABLE IF NOT EXISTS gps(
            path TEXT PRIMARY KEY, lat REAL, lon REAL,
            city TEXT DEFAULT '', sub TEXT DEFAULT '')""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_gps_city ON gps(city)")
        c.execute("""CREATE TABLE IF NOT EXISTS geocache(
            k TEXT PRIMARY KEY, city TEXT, sub TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS ignored(
            id INTEGER PRIMARY KEY, vec BLOB, note TEXT, path TEXT DEFAULT '')""")
        cols = [r[1] for r in c.execute("PRAGMA table_info(ignored)")]
        if "path" not in cols:
            c.execute("ALTER TABLE ignored ADD COLUMN path TEXT DEFAULT ''")
        pcols = [r[1] for r in c.execute("PRAGMA table_info(people)")]
        if "center" not in pcols:
            c.execute("ALTER TABLE people ADD COLUMN center BLOB")
        c.execute("CREATE TABLE IF NOT EXISTS face_scanned(path TEXT PRIMARY KEY)")
        # 事件时间轴：一次活动 = 一段连续拍摄（相邻间隔 <= EVENT_GAP 小时）
        c.execute("""CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY, start REAL, end REAL, n INTEGER,
            title TEXT DEFAULT '', city TEXT DEFAULT '', cover TEXT DEFAULT '')""")
        c.execute("""CREATE TABLE IF NOT EXISTS event_items(
            event INTEGER, path TEXT, ts REAL)""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_event_items ON event_items(event)")
        c.execute("""INSERT OR IGNORE INTO face_scanned(path)
                     SELECT DISTINCT path FROM faces""")
        # q     = 人脸质量分（检测分 × 清晰度 × 正脸程度），选封面、排「未归类」都用它
        # manual= 用户手动纠正过的脸，重聚类时不许被自动算法打散
        fcols = [r[1] for r in c.execute("PRAGMA table_info(faces)")]
        if "q" not in fcols:
            c.execute("ALTER TABLE faces ADD COLUMN q REAL DEFAULT 0")
        if "manual" not in fcols:
            c.execute("ALTER TABLE faces ADD COLUMN manual INTEGER DEFAULT 0")
        # ---- 一次性清理：旧版删人物分组时会给每张照片写一条 (vec=NULL, path) 的
        # 忽略记录，而重扫逻辑把「ignored 里有这个 path」理解成「整张照片不扫脸」。
        # 结果就是删过一个人物之后，那张合影里**所有人**的脸都不再被检测，
        # 人像页莫名其妙地少人。现在只管向量，这些记录一律清掉。
        if not c.execute("SELECT 1 FROM meta WHERE k='ign_clean_v1'").fetchone():
            c.execute("DELETE FROM ignored WHERE ifnull(vec,'')=''")
            c.execute("INSERT OR REPLACE INTO meta(k,v) VALUES('ign_clean_v1','1')")


_init_album()  # 模块导入即建表（startup 事件之外再兜底一次）

ALBUM = {"running": False, "stage": "", "done": 0, "total": 0, "faces": 0,
         "people": 0, "log": []}
_ALBUM_LOCK = threading.Lock()      # 相册构建互斥，见 run_album


def alog(m):
    ALBUM["log"].append("%s %s" % (time.strftime("%H:%M:%S"), m))
    ALBUM["log"] = ALBUM["log"][-200:]
    print("[album] %s" % m, flush=True)


# ---------------------------------------------------------------- 人脸检测
_FACE_SESS = None
_ARC = None

# 人脸向量的"版本"。改了编码方式（比如这次加了对齐）就要换这个值，
# 旧库里的向量跟新向量不在一个空间里，混着用只会更乱 —— 换值后自动整批重扫。
FACE_VER = "4-quality"

# ArcFace 的标准 5 点（112x112）。眼睛必须在固定的高度、脸宽固定，
# 这是它训练时看到的输入分布。
#
# YuNet 吐的 5 个关键点顺序是「右眼 / 左眼 / 鼻尖 / 右嘴角 / 左嘴角」
# （按被摄者的说法命名），在图像上正好是从左到右，与下面这组同序对应。
_ARC_SRC = np.array([[38.2946, 51.6963],
                     [73.5318, 51.5014],
                     [56.0252, 71.7366],
                     [41.5493, 92.3655],
                     [70.7299, 92.2041]], np.float32)


def face_ver_get() -> str:
    with db() as c:
        r = c.execute("SELECT v FROM meta WHERE k='face_ver'").fetchone()
    return r[0] if r else ""


def face_ver_set(v: str):
    with db() as c:
        c.execute("INSERT OR REPLACE INTO meta(k,v) VALUES('face_ver',?)", (v,))


def face_upgrade_needed() -> bool:
    return face_ver_get() != FACE_VER


def face_session():
    global _FACE_SESS
    if _FACE_SESS is None:
        from ort_loader import ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        _FACE_SESS = ort.InferenceSession(
            str(FACE_DIR / "yunet.onnx"), so,
            providers=["CPUExecutionProvider"])
    return _FACE_SESS


def _priors():
    P = []
    for s in (8, 16, 32):
        f = 640 // s
        for gy in range(f):
            for gx in range(f):
                P.append(((gx + 0.5) * s, (gy + 0.5) * s, s))
    return np.array(P, dtype=np.float32)


_PRIORS = _priors()


def detect_faces(im, conf=FACE_CONF):
    """返回 [[x1,y1,x2,y2,score,kps(5x2 原图坐标)], ...]"""
    im = im.convert("RGB")
    w, h = im.size
    sc = 640.0 / max(h, w)
    nh, nw = max(1, int(h * sc)), max(1, int(w * sc))
    canvas = np.zeros((640, 640, 3), np.uint8)
    canvas[:nh, :nw] = np.asarray(im.resize((nw, nh), RES_BILINEAR))[:, :, ::-1]
    x = np.transpose(canvas, (2, 0, 1))[None].astype(np.float32).copy()
    s = face_session()
    names = [o.name for o in s.get_outputs()]
    d = dict(zip(names, s.run(names, {"input": x})))
    cls = np.concatenate([d["cls_8"], d["cls_16"], d["cls_32"]], 1)[0, :, 0]
    obj = np.concatenate([d["obj_8"], d["obj_16"], d["obj_32"]], 1)[0, :, 0]
    bb = np.concatenate([d["bbox_8"], d["bbox_16"], d["bbox_32"]], 1)[0]
    # 5 个关键点 × (dx,dy)，之前这里被整个忽略了 —— 对齐全靠它
    kp = np.concatenate([d["kps_8"], d["kps_16"], d["kps_32"]], 1)[0]
    sco = np.sqrt(np.clip(cls * obj, 0, 1))
    idx = np.where(sco > conf)[0]
    out = []
    for i in idx:
        cx = _PRIORS[i][0] + bb[i, 0] * _PRIORS[i][2]
        cy = _PRIORS[i][1] + bb[i, 1] * _PRIORS[i][2]
        hw = math.exp(bb[i, 2]) * _PRIORS[i][2] / 2
        hh = math.exp(bb[i, 3]) * _PRIORS[i][2] / 2
        pts = np.array([[_PRIORS[i][0] + kp[i, 2 * k] * _PRIORS[i][2],
                         _PRIORS[i][1] + kp[i, 2 * k + 1] * _PRIORS[i][2]]
                        for k in range(5)], np.float32) / sc
        out.append([(cx - hw) / sc, (cy - hh) / sc, (cx + hw) / sc,
                    (cy + hh) / sc, float(sco[i]), pts])
    return _nms(out)


def _umeyama(src, dst):
    """最小二乘相似变换：求 s·R·src + t ≈ dst（只有旋转、缩放、平移，不拉伸）

    用它而不是「直接按框裁正方形再 resize」的原因：后者把脸拉变形，
    同一人不同角度会长得不一样，不同人反而因为构图相同而靠近。
    """
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc_, dc_ = src - mu_s, dst - mu_d
    C = dc_.T @ sc_ / n
    U, S, Vt = np.linalg.svd(C)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:      # 防镜像
        D[1, 1] = -1
    R = U @ D @ Vt
    var_s = (sc_ ** 2).sum() / n
    s = float((S * np.diag(D)).sum() / max(var_s, 1e-9))
    M = np.zeros((2, 3), np.float32)
    M[:, :2] = s * R
    M[:, 2] = mu_d - s * (R @ mu_s)
    return M


def align_face(im, kps, size=112):
    """把脸摆到 ArcFace 的标准位（眼睛水平、脸居中、占满画幅）"""
    src = np.asarray(kps, np.float32)
    dst = _ARC_SRC * (size / 112.0)
    M = _umeyama(src, dst)
    inv = np.linalg.inv(np.vstack([M, [0.0, 0.0, 1.0]]))[:2]
    # PIL 的 AFFINE 参数是「输出坐标 → 输入坐标」，所以要传逆矩阵
    return im.convert("RGB").transform(
        (size, size), Image.AFFINE, tuple(inv.ravel()), resample=RES_BILINEAR)



def square_crop(im, box, pad=0.25):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    s = max(x2 - x1, y2 - y1) * (1 + 2 * pad)
    x1, y1 = max(0, int(cx - s / 2)), max(0, int(cy - s / 2))
    x2, y2 = min(im.width, int(cx + s / 2)), min(im.height, int(cy + s / 2))
    return im.crop((x1, y1, x2, y2))


def _nms(b, iou=0.35):
    b = sorted(b, key=lambda z: -z[4])
    out = []
    for c in b:
        for k in out:
            x1, y1 = max(c[0], k[0]), max(c[1], k[1])
            x2, y2 = min(c[2], k[2]), min(c[3], k[3])
            if max(0, x2 - x1) * max(0, y2 - y1) > iou * (k[2] - k[0]) * (k[3] - k[1]):
                break
        else:
            out.append(c)
    return out


def arc_session():
    global _ARC
    if _ARC is None:
        from ort_loader import ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        _ARC = ort.InferenceSession(str(FACE_DIR / "arcface.onnx"), so,
                                    providers=["CPUExecutionProvider"])
    return _ARC


def face_embed(im):
    a = np.asarray(im.convert("RGB").resize((112, 112), RES_BILINEAR),
                   dtype=np.float32)
    a = (a[:, :, ::-1] - 127.5) / 127.5
    s = arc_session()
    v = np.asarray(s.run(None, {s.get_inputs()[0].name:
                                np.transpose(a, (2, 0, 1))[None]})[0]).ravel()
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


def encode_faces(ims):
    return np.stack([face_embed(im) for im in ims])


def _face_quality(crop, score, kps=None):
    """0~1 的人脸质量分：清晰度为主、检测分为辅，再看是不是正脸。

    两处用得上：挑人物封面（糊脸/大侧脸当封面最难看），
    以及「未归类」列表默认按质量倒序（先看能认出来的）。
    """
    try:
        g = np.asarray(crop.convert("L"), dtype=np.float32)
        lap = (4.0 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
               - g[1:-1, :-2] - g[1:-1, 2:])
        sharp = float(lap.var())
    except Exception:
        sharp = 0.0
    # 112×112 的对齐人脸：清楚的 Laplacian 方差 300~2500，糊脸一般在 120 以下
    sharp01 = min(1.0, max(0.0, (sharp - 40.0) / 600.0))
    front01 = 0.6
    if kps is not None and np.all(np.isfinite(kps)):
        dx = abs(float(kps[1][0]) - float(kps[0][0]))
        dy = abs(float(kps[1][1]) - float(kps[0][1]))
        front01 = max(0.0, 1.0 - min(1.0, (dy / (dx + 1e-6)) * 3.0))
    return round(0.42 * sharp01 + 0.38 * float(score) + 0.20 * front01, 4)


def faces_of_image(im, maxn=6):
    """一张照片 → [(box_str, 检测分, 质量分, 向量)]

    检测 → 按 5 点做仿射对齐 → ArcFace 编码。对齐是 ArcFace 能用的前提，
    之前直接方框裁剪再拉伸，同一个人的向量会散、不同人反而靠近。
    """
    boxes = detect_faces(im)
    boxes = [b for b in boxes
             if (b[2] - b[0]) >= FACE_MINPX and (b[3] - b[1]) >= FACE_MINPX]
    if not boxes:
        return []
    rgb = im.convert("RGB")
    crops, metas = [], []
    for b in boxes[:maxn]:
        x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
        x2, y2 = min(im.width, int(b[2])), min(im.height, int(b[3]))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        kps = b[5] if len(b) > 5 else None
        try:
            if kps is not None and np.all(np.isfinite(kps)):
                crop = align_face(rgb, kps)
            else:                       # 关键点异常时退回方框裁剪，别整张图丢掉
                crop = square_crop(rgb, (x1, y1, x2, y2)).resize(
                    (112, 112), RES_BILINEAR)
        except Exception:
            crop = square_crop(rgb, (x1, y1, x2, y2)).resize(
                (112, 112), RES_BILINEAR)
        crops.append(crop)
        metas.append(("%d,%d,%d,%d" % (x1, y1, x2, y2), float(b[4]),
                      _face_quality(crop, b[4], kps)))
    if not crops:
        return []
    vecs = encode_faces(crops)
    return [(m[0], m[1], m[2], v) for m, v in zip(metas, vecs)]


def cluster_faces(hints=None):
    """把所有人脸向量聚成人物分组。

    参数 hints 可选：{(照片路径, 人脸框字符串): 名字}，
    用于重扫后把用户之前起过的名字认领回来（按位置匹配，检测框没变）。

    算法换过一次（2026-10-21），原因是老的「连通分量」会把不相干的人缝在一起：
    只要 A≈B、B≈C 各自过阈值，A 和 C 哪怕完全不像，也会因为连在一条链上而归成一簇。
    人脸多、照片质量参差时，一条由低质量脸组成的链足以把两拨人串成一坨。

    现在分两步：
      1. 质心聚类 —— 按检测质量从高到低处理，每张脸归到最像的簇，簇心随成员滚动更新；
      2. 平均连接合并 —— 两个簇要合并，不光要求「簇心像」，
         还要求「两簇成员两两之间的平均相似度」也够高。这一步是杜绝链式传染的关键。
    """
    with db() as c:
        rows = c.execute("SELECT id,path,box,vec,score,q,manual,person FROM faces"
                         " ORDER BY id").fetchall()
        old = dict(c.execute(
            "SELECT f.id,p.name FROM faces f JOIN people p ON p.id=f.person"
            " WHERE ifnull(p.name,'')<>''").fetchall())
        # ⚠️ 这里**不能** DELETE FROM people。
        # 以前是在这里清库、聚类算完了再重建，两段之间要跑几十秒。
        # 中途只要抛一次异常（内存不够、向量损坏…），people 就空了、
        # faces.person 还指着一批已经不存在的 id，用户所有命名瞬间全丢。
        # 改成和重建写在同一个事务里（见函数末尾），失败了整体回滚，原样保留。
    for f in THUMB_DIR.glob("face_*.jpg"):
        try:
            f.unlink()
        except Exception:
            pass
    if not rows:
        return 0
    V = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    # 质量优先用 q（检测分×清晰度×正脸度）；老库还没算过 q 就退回检测分
    Q = np.array([(float(r[5]) if (r[5] or 0) > 0 else float(r[4] or 0))
                  for r in rows], np.float32)
    MAN = np.array([int(r[6] or 0) for r in rows], np.int32)
    with db() as c:
        ig = c.execute("SELECT vec FROM ignored WHERE vec IS NOT NULL").fetchall()
    skip = np.zeros(len(rows), dtype=bool)
    if ig:
        try:
            IV = np.stack([np.frombuffer(x[0], dtype=np.float32) for x in ig])
            IV = IV / (np.linalg.norm(IV, axis=1, keepdims=True) + 1e-9)
            skip = (IV @ V.T).max(axis=0) >= IGNORE_TH
        except Exception:
            pass

    keep = np.array([i for i in range(len(rows)) if not skip[i]], dtype=np.int64)
    clusters = []
    loose = []          # 脸太少、不足以单独成「人物」的（进「其他面孔」）
    minf = 1
    if len(keep):
        Vk = V[keep]
        Qk = Q[keep]
        cents, members, sums = [], [], []
        # 每个簇"归属哪个人物"，只对锚点簇有值；用来禁止自动合并两个用户确认过的人物
        cpids = []
        # ---- 锚点：用户手动归过的脸 ----
        # 用户说"这张脸就是人物 3"，那就必须留在人物 3 里。做法是把这个人物
        # **当前的全部成员**（不只是那张 manual 脸）建成初始簇，其余的脸再往
        # 锚点上靠；否则下次聚类又会把它算到别的组去，用户改一次白改一次。
        pinned_pids = {int(rows[int(i)][7]) for i in range(len(rows))
                       if MAN[i] and rows[int(i)][7] is not None
                       and int(rows[int(i)][7]) >= 0}
        anchor_of = {}
        for kk in range(len(keep)):
            i = int(keep[kk])
            p_ = rows[i][7]
            if p_ is not None and int(p_) in pinned_pids:
                anchor_of.setdefault(int(p_), []).append(kk)
        fixed = set()
        sealed = []       # 与 cents 同步：True = 纯人工簇，不再吸收别的脸
        for p_, idxs in anchor_of.items():
            vs = Vk[idxs]
            s = vs.sum(axis=0)
            cents.append((s / (np.linalg.norm(s) + 1e-9)).astype(np.float32))
            members.append(list(idxs))
            sums.append(s.astype(np.float32))
            cpids.append({p_})
            sealed.append(all(bool(MAN[keep[i]]) for i in idxs))
            fixed.update(idxs)
        if fixed:
            print("[album] 锚定 %d 张脸（用户确认过的分组，重扫时不会被拆散）" % len(fixed),
                  flush=True)
        # 质量高的脸先立簇：低质量脸跟着走，避免一张糊脸把簇心带偏
        order = [int(t) for t in np.argsort(-(Qk + 0.001 * np.arange(len(keep))))
                 if int(t) not in fixed]
        for t in order:
            v = Vk[t]
            if cents:
                sims = np.stack(cents) @ v
                # 纯人工簇（整个组都是用户亲手放进去的）不再吸收别的脸：
                # 用户把一张脸拆成独立一组，就是"它不属于原来那帮人"，
                # 不能因为长得像又被自动吸回去。
                for si in range(len(cents)):
                    if sealed[si]:
                        sims[si] = -2.0
                j = int(np.argmax(sims))
                if sims[j] >= FACE_TH:
                    members[j].append(int(t))
                    sums[j] = sums[j] + v
                    c = sums[j]
                    cents[j] = (c / (np.linalg.norm(c) + 1e-9)).astype(np.float32)
                    continue
            cents.append(v.astype(np.float32).copy())
            members.append([int(t)])
            sums.append(v.astype(np.float32).copy())
            cpids.append(set())
            sealed.append(False)

        # 平均连接合并
        merges = 0
        while len(cents) > 1:
            C = np.stack(cents)
            S2 = C @ C.T
            np.fill_diagonal(S2, -2.0)
            hit = None
            for _ in range(len(cents)):
                i, j = np.unravel_index(int(np.argmax(S2)), S2.shape)
                i, j = int(i), int(j)
                if S2[i, j] < MERGE_CEN_TH:
                    break
                if cpids[i] and cpids[j] and not (cpids[i] & cpids[j]):
                    S2[i, j] = -2.0      # 两个人都是用户亲手分的，不许算法合并
                    continue
                ai, bj = Vk[members[i]], Vk[members[j]]
                if float((ai @ bj.T).mean()) >= MERGE_AVG_TH:
                    hit = (i, j)
                    break
                S2[i, j] = -2.0          # 这两簇不合格，换下一对
            if hit is None:
                break
            i, j = hit
            members[i] = members[i] + members[j]
            sums[i] = sums[i] + sums[j]
            c = sums[i]
            cents[i] = (c / (np.linalg.norm(c) + 1e-9)).astype(np.float32)
            cpids[i] = cpids[i] | cpids[j]
            sealed[i] = sealed[i] and sealed[j]
            del members[j], sums[j], cents[j], cpids[j], sealed[j]
            merges += 1

        # 只保留有成员的分组；按簇大小排序，界面上人多的排前面
        clusters = [sorted(m) for m in members if m]
        # ---- 密度约束（照搬 immich 的 core point 思路）----
        # 只有一两张脸的小簇，绝大多数是误检（海报/动画截图）、背景路人、或者模特图，
        # 让它们各自占一个「人物」只会把列表搞乱。两条例外永远保留：
        #   ① 簇里有用户手动归过的脸（manual）—— 人工判断优先于算法；
        #   ② 整个库的脸本来就很少（< SMALL_LIB 张），这时候全留着，
        #      否则新用户刚导入几张照片会看到人像页一片空白，比乱还吓人。
        SMALL_LIB = FACE_SMALL_LIB
        minf = FACE_MIN_FACES if len(keep) >= SMALL_LIB else 1
        big, tiny = [], []
        for m in clusters:
            if len(m) >= minf or any(MAN[keep[i]] for i in m):
                big.append(m)
            else:
                tiny.append(m)
        clusters = big
        loose = sorted(i for m in tiny for i in m)
        if minf > 1:
            print("[album] 密度约束：%d 张脸不足以成组（< %d 张）→ 归入「其他面孔」"
                  % (len(loose), minf), flush=True)
        clusters.sort(key=lambda m: (-len(m), m[0]))

    print("[album] 质心聚类 %d 张脸 → %d 个人物 + %d 张未归类"
          "（平均连接合并 %d 次，阈值 %.2f/%.2f/%.2f）"
          % (len(keep), len(clusters), len(loose), locals().get("merges", 0),
             FACE_TH, MERGE_CEN_TH, MERGE_AVG_TH), flush=True)
    with db() as c:
        # 清空 + 重建在同一个事务里：要么整个换新，要么原样不动（见上面注释）
        c.execute("DELETE FROM people")
        c.execute("UPDATE faces SET person=-1")
        for i in loose:                 # person=-3：有脸、但不够格单独成人物
            c.execute("UPDATE faces SET person=-3 WHERE id=?",
                      (int(rows[keep[i]][0]),))
        for pid, idxs in enumerate(clusters):
            # 封面挑质量最高的：检测分高不等于看得清，糊脸/大侧脸当封面最难看
            best = max(idxs, key=lambda i: (float(rows[keep[i]][5] or 0) or
                                            float(rows[keep[i]][4] or 0)))
            cover = rows[keep[best]][1] + "|" + str(rows[keep[best]][2])
            names = [old.get(int(rows[keep[i]][0]), "") for i in idxs]
            if hints:
                names += [hints.get((rows[keep[i]][1], rows[keep[i]][2]), "")
                          for i in idxs]
            names = [x for x in names if x]
            nm = Counter(names).most_common(1)[0][0] if names else ""
            cen = V[keep[idxs]].mean(axis=0)
            cen = cen / (np.linalg.norm(cen) + 1e-9)
            c.execute("INSERT INTO people(id,name,cover,center) VALUES(?,?,?,?)",
                      (pid, nm, cover, cen.astype(np.float32).tobytes()))
            for i in idxs:
                c.execute("UPDATE faces SET person=? WHERE id=?",
                          (pid, int(rows[keep[i]][0])))

        # 同一张照片里，一个人只保留分数最高的那张脸。
        # 合影里同一个人出现两次很少见；反倒是海报、动画截图里的卡通脸
        # 经常被连着检出来、还长得都差不多，一起塞进分组会让人以为识别错了。
        dups = 0
        for (pid,) in c.execute("SELECT DISTINCT person FROM faces WHERE person>=0"):
            seen = set()
            for fid, p_ in c.execute(
                    "SELECT id,path FROM faces WHERE person=?"
                    " ORDER BY ifnull(manual,0) DESC, ifnull(q,0) DESC, score DESC",
                    (pid,)).fetchall():
                if p_ in seen:
                    c.execute("UPDATE faces SET person=-1 WHERE id=?", (fid,))
                    dups += 1
                else:
                    seen.add(p_)
        if dups:
            print("[album] 同一张照片里同一人只留一张，剔除 %d 张重复检测" % dups,
                  flush=True)
    merge_same_name()
    return len(clusters)


def merge_same_name():
    with db() as c:
        dup = c.execute("""SELECT name, group_concat(id) FROM people
                           WHERE ifnull(name,'')<>'' GROUP BY name
                           HAVING count(*)>1""").fetchall()
    n = 0
    for name, ids in dup:
        ids = [int(x) for x in str(ids).split(",")]
        cnt = []
        with db() as c:
            for i in ids:
                cnt.append((c.execute("SELECT count(*) FROM faces WHERE person=?",
                                      (i,)).fetchone()[0], i))
        if len(ids) < 2:
            continue
        keep = max(cnt)[1]
        for i in ids:
            if i == keep:
                continue
            with db() as c:
                c.execute("UPDATE faces SET person=? WHERE person=?", (keep, i))
                c.execute("DELETE FROM people WHERE id=?", (i,))
            n += 1
        if n:
            # 合并后存活的那个组，成员变了：封面和中心向量都得重算，头像缓存作废
            with db() as c:
                _person_stats(keep, c)
            _drop_face_cache(keep)
    return n


def incremental_faces():
    with db() as c:
        known = set(x[0] for x in c.execute("SELECT path FROM face_scanned"))
        ign_p = set(x[0] for x in
                    c.execute("SELECT DISTINCT path FROM ignored WHERE path<>''"))
    imgs = [p for p, k in zip(STORE.paths, STORE.kinds)
            if k == "image" and p not in known and p not in ign_p]
    if not imgs:
        return 0, _people_count()
    ALBUM["stage"] = "找人脸"
    ALBUM["total"] = len(imgs)
    ALBUM["done"] = 0
    new = []
    seen = []
    for i, p in enumerate(imgs):
        if not ALBUM["running"]:
            break
        ALBUM["done"] = i + 1
        try:
            # 必须 with，理由同全量重扫：增量也可能一次处理几千张新图
            with Image.open(p) as im0:
                im = im0.convert("RGB")
        except Exception:
            continue
        seen.append(p)
        try:
            got = faces_of_image(im)
        except Exception:
            continue
        if not got:
            continue
        for box, sc_, q_, v in got:
            new.append((p, box, sc_, q_, v.astype(np.float32).tobytes()))
        if (i + 1) % 20 == 0:
            alog("新照片扫脸 %d/%d，已找到 %d 张脸" % (i + 1, len(imgs), len(new)))
    if not new:
        if seen:
            with db() as c:
                for p in seen:
                    c.execute("INSERT OR IGNORE INTO face_scanned(path) VALUES(?)", (p,))
        return 0, _people_count()
    with db() as c:
        for p in seen:
            c.execute("INSERT OR IGNORE INTO face_scanned(path) VALUES(?)", (p,))
        for pp, box, sc, q_, vb in new:
            c.execute("INSERT INTO faces(path,box,score,q,person,vec)"
                      " VALUES(?,?,?,?,-1,?)", (pp, box, sc, q_, vb))
    ALBUM["stage"] = "人脸归类"
    np_ = cluster_faces()
    alog("增量归类：%d 张新脸 → 现共 %d 个人物" % (len(new), np_))
    return len(new), np_


def _people_count():
    with db() as c:
        return c.execute("SELECT count(*) FROM people").fetchone()[0]


# ---------------------------------------------------------------- 后台自动更新
AUTO_SCAN_SEC = int(os.environ.get("AUTO_SCAN_SEC", "300"))
AUTO = {"on": os.environ.get("AUTO_INDEX", "1") == "1",
        "every": AUTO_SCAN_SEC, "fp": None, "last": 0.0, "index": 0}


def _disk_fp():
    n = tot = 0
    latest = 0.0
    for p in all_files():
        try:
            st = os.stat(p)
        except OSError:
            continue
        n += 1
        tot += st.st_size
        if st.st_mtime > latest:
            latest = st.st_mtime
    return (n, tot, latest)


def auto_watch():
    time.sleep(20)
    AUTO["fp"] = _disk_fp()
    while True:
        try:
            time.sleep(AUTO_SCAN_SEC)
            if not AUTO["on"] or not MODEL_READY:
                continue
            if IDX["running"] or ALBUM["running"]:
                continue
            fp = _disk_fp()
            if fp == AUTO["fp"]:
                continue
            AUTO["fp"] = fp
            AUTO["last"] = time.time()
            print("[auto] 目录有新文件，自动建立索引…", flush=True)
            do_index()
            AUTO["index"] += 1
            run_album("new")
        except Exception as e:
            print("[auto] 出错：%s" % e, flush=True)


threading.Thread(target=auto_watch, daemon=True).start()


# ---------------------------------------------------------------- 标签分类
def classify_tags():
    S = STORE
    if S.n == 0:
        return 0
    V = S.V
    n = 0
    with db() as c:
        for cat, items in TAG_SETS.items():
            c.execute("DELETE FROM tags WHERE cat=?", (cat,))
            T = CLIP.encode_texts([it[1] for it in items])
            sims = V @ T.T
            for i, p in enumerate(S.paths):
                if S.srcs[i] in ASR_SRC:      # 转写段不参与画面标签（余弦天然偏高，是噪声）
                    continue
                order = np.argsort(-sims[i])
                top = float(sims[i][order[0]])
                if top < TAG_MIN:
                    continue
                keep = [order[0]]
                if len(order) > 1 and float(sims[i][order[1]]) >= top - 0.03:
                    keep.append(order[1])
                for j in keep:
                    c.execute("INSERT OR REPLACE INTO tags VALUES(?,?,?,?)",
                              (p, cat, items[j][0], float(sims[i][j])))
                    n += 1
    return n


# ---------------------------------------------------------------- GPS → 城市
GEO_URL = os.environ.get(
    "GEO_URL", "https://api.bigdatacloud.net/data/reverse-geocode-client")
GEO_UA = "photo-album/1.0"
GEO_TIMEOUT = 12.0
GEO_ROUND = 3
GEO_SLEEP = 0.25

_SIMP_MAP = {"區": "区", "縣": "县", "鄉": "乡", "鎮": "镇", "東": "东", "龍": "龙",
             "門": "门", "華": "华", "馬": "马", "烏": "乌", "蘇": "苏", "廣": "广",
             "內": "内", "舊": "旧", "樓": "楼", "興": "兴", "陽": "阳", "陰": "阴",
             "嶺": "岭", "島": "岛", "峽": "峡", "灣": "湾", "澤": "泽", "濱": "滨",
             "溝": "沟", "環": "环", "頭": "头", "麗": "丽", "寶": "宝", "雲": "云",
             "鳳": "凤", "鶴": "鹤", "鹽": "盐", "邊": "边", "裡": "里", "園": "园"}


def _simp(s):
    if not s:
        return ""
    return "".join(_SIMP_MAP.get(ch, ch) for ch in s)


def _dms(v):
    try:
        d, m, s = v
        return float(d) + float(m) / 60.0 + float(s) / 3600.0
    except Exception:
        try:
            return float(v)
        except Exception:
            return None


def read_gps(path):
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            g = ex.get_ifd(0x8825) if ex else None
    except Exception:
        g = None
    if g:
        lat, lon = _dms(g.get(2)), _dms(g.get(4))
        if lat is not None and lon is not None:
            if str(g.get(1, "N")).upper().startswith("S"):
                lat = -lat
            if str(g.get(3, "E")).upper().startswith("W"):
                lon = -lon
            if -90 <= lat <= 90 and -180 <= lon <= 180 and (lat or lon):
                return lat, lon
    return None


def read_gps_video(path):
    try:
        import av
        with av.open(path) as c:
            loc = ""
            for k in ("location", "com.apple.quicktime.location.ISO6709", "Location"):
                if c.metadata.get(k):
                    loc = c.metadata[k]
                    break
        if not loc:
            return None
        m = re.match(r"([+-]?\d+(?:\.\d+)?)([+-]?\d+(?:\.\d+)?)", loc.replace(" ", ""))
        return (float(m.group(1)), float(m.group(2))) if m else None
    except Exception:
        return None


def geocode(lat, lon):
    key = "%.*f,%.*f" % (GEO_ROUND, lat, GEO_ROUND, lon)
    with db() as c:
        r = c.execute("SELECT city,sub FROM geocache WHERE k=?", (key,)).fetchone()
    if r:
        return r[0], r[1]
    url = "%s?latitude=%.6f&longitude=%.6f&localityLanguage=zh" % (GEO_URL, lat, lon)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": GEO_UA})
        with urllib.request.urlopen(req, timeout=GEO_TIMEOUT) as fp:
            d = json.loads(fp.read().decode("utf-8", "replace"))
        city = _simp(d.get("city") or d.get("locality")
                     or d.get("principalSubdivision") or "")
        sub = _simp(d.get("locality") or "")
        if sub == city:
            sub = ""
        if city:
            with db() as c:
                c.execute("INSERT OR REPLACE INTO geocache VALUES(?,?,?)",
                          (key, city, sub))
        return city, sub
    except Exception as e:
        alog("查城市名失败 %.3f,%.3f：%s" % (lat, lon, e))
        return "", ""


def build_cities():
    S = STORE
    if S.n == 0:
        return 0
    items = list(zip(S.paths, S.kinds))
    with db() as c:
        c.execute("DELETE FROM gps")
        c.execute("DELETE FROM tags WHERE cat='city'")
    spots, n = {}, 0
    ALBUM["stage"] = "读定位"
    ALBUM["total"] = len(items)
    for i, (p, k) in enumerate(items):
        ALBUM["done"] = i + 1
        g = read_gps(p) if k == "image" else read_gps_video(p)
        if not g:
            continue
        lat, lon = g
        key = "%.*f,%.*f" % (GEO_ROUND, lat, GEO_ROUND, lon)
        spots.setdefault(key, (lat, lon))
        with db() as c:
            c.execute("INSERT OR REPLACE INTO gps(path,lat,lon,city,sub)"
                      " VALUES(?,?,?,?,?)", (p, lat, lon, "", ""))
        n += 1
    alog("%d 个文件带定位，去重后 %d 个地点，开始联网查城市名 …" % (n, len(spots)))
    ALBUM["stage"] = "查城市名"
    ALBUM["total"] = max(1, len(spots))
    ALBUM["done"] = 0
    got = 0
    for i, (key, (lat, lon)) in enumerate(spots.items()):
        ALBUM["done"] = i + 1
        city, sub = geocode(lat, lon)
        if not city:
            continue
        rlat, rlon = round(lat, GEO_ROUND), round(lon, GEO_ROUND)
        with db() as c:
            rows = c.execute("SELECT path FROM gps WHERE round(lat,?)=? AND"
                             " round(lon,?)=?", (GEO_ROUND, rlat, GEO_ROUND, rlon)).fetchall()
            for (p,) in rows:
                c.execute("UPDATE gps SET city=?,sub=? WHERE path=?", (city, sub, p))
                c.execute("INSERT OR REPLACE INTO tags VALUES(?,?,?,?)",
                          (p, "city", city, 1.0))
        got += 1
        time.sleep(GEO_SLEEP)
    with db() as c:
        ncity = c.execute("SELECT count(DISTINCT city) FROM gps WHERE city<>''").fetchone()[0]
    alog("城市归类完成：%d 个城市，覆盖 %d 个文件" % (ncity, n))
    return ncity


# ---------------------------------------------------------------- 事件时间轴
# 照 google photos / photoprism 的 moments 做：按拍摄时间把照片切成「一次活动」。
# 相邻两张间隔超过 EVENT_GAP 小时就断开，段内少于 EVENT_MIN 张的不要
# （零零散散的截图、随手拍不该占一个事件位）。
EVENT_GAP = float(os.environ.get("EVENT_GAP", "6"))
EVENT_MIN = int(os.environ.get("EVENT_MIN", "3"))


def _event_title(seg, city):
    t0, t1 = seg[0][1], seg[-1][1]
    d0, d1 = time.localtime(t0), time.localtime(t1)
    now = time.localtime()
    if (d0.tm_year, d0.tm_yday) == (d1.tm_year, d1.tm_yday):
        base = "%d月%d日" % (d0.tm_mon, d0.tm_mday)
    elif d0.tm_year == d1.tm_year:
        base = "%d月%d日–%d月%d日" % (d0.tm_mon, d0.tm_mday, d1.tm_mon, d1.tm_mday)
    else:
        base = "%d月%d日 – %d年%d月%d日" % (d0.tm_mon, d0.tm_mday,
                                            d1.tm_year, d1.tm_mon, d1.tm_mday)
    if d0.tm_year != now.tm_year:
        base = "%d年%s" % (d0.tm_year, base)
    return base + (" · " + city if city else "")


def build_events():
    """重建事件表，返回事件数。几十万张也就几秒，不必做增量。"""
    with db() as c:
        rows = c.execute("SELECT path, mtime FROM media WHERE kind<>'failed'"
                         " AND ifnull(mtime,0)>0 GROUP BY path"
                         " ORDER BY mtime ASC").fetchall()
        city_of = {r[0]: r[1] for r in
                   c.execute("SELECT path,city FROM gps WHERE ifnull(city,'')<>''")}
    if not rows:
        with db() as c:
            c.execute("DELETE FROM events")
            c.execute("DELETE FROM event_items")
        return 0
    gap = EVENT_GAP * 3600.0
    segs, cur = [], [rows[0]]
    for p, t in rows[1:]:
        if t - cur[-1][1] > gap:
            segs.append(cur)
            cur = [(p, t)]
        else:
            cur.append((p, t))
    segs.append(cur)
    segs = [s for s in segs if len(s) >= EVENT_MIN]
    with db() as c:
        c.execute("DELETE FROM events")
        c.execute("DELETE FROM event_items")
        for seg in segs:
            t0, t1 = seg[0][1], seg[-1][1]
            cities = [city_of.get(p, "") for p, _ in seg]
            cities = [x for x in cities if x]
            city = Counter(cities).most_common(1)[0][0] if cities else ""
            eid = int(t0)
            cov = cover_of(seg[0][0], c) or ""
            c.execute("INSERT OR REPLACE INTO events(id,start,end,n,title,city,cover)"
                      " VALUES(?,?,?,?,?,?,?)",
                      (eid, t0, t1, len(seg), _event_title(seg, city), city, cov))
            for p, t in seg:
                c.execute("INSERT INTO event_items(event,path,ts) VALUES(?,?,?)", (eid, p, t))
    alog("时间轴：%d 个事件（%d 张照片/视频，间隔 > %.0f 小时断开）"
         % (len(segs), len(rows), EVENT_GAP))
    return len(segs)


# ---------------------------------------------------------------- 一键构建
def run_album(only=""):
    if not MODEL_READY:
        return
    # 和 do_index 同一个道理：ALBUM["running"] 是函数内部才置位的，
    # 光靠它拦不住同时进来的两个构建请求，两轮 cluster_faces 会把
    # faces 表删了又建、互相踩。
    if not _ALBUM_LOCK.acquire(blocking=False):
        return
    ALBUM.update(running=True, stage="", done=0, total=0, faces=0, people=0, log=[])
    try:
        if only != "faces":
            ALBUM["stage"] = "打标签"
            alog("开始生成「事物 / 地点」标签 …")
            n = classify_tags()
            alog("标签完成，共 %d 条" % n)
        if only == "tags":
            ALBUM["stage"] = "完成"
            return
        if only in ("", "city", "quick", "new"):
            ALBUM["stage"] = "定位城市"
            alog("开始读取照片定位信息 …")
            try:
                build_cities()
            except Exception:
                traceback.print_exc()
        if only in ("", "city", "quick", "new"):
            ALBUM["stage"] = "整理时间轴"
            try:
                build_events()
            except Exception:
                traceback.print_exc()
        if only in ("city", "quick"):
            ALBUM["stage"] = "完成"
            return
        if only == "new":
            ALBUM["stage"] = "更新相册"
            nf, np_ = incremental_faces()
            ALBUM["stage"] = "完成"
            alog("自动更新完成：新增 %d 张脸，现共 %d 个人物" % (nf, np_))
            return
        if only == "cluster":
            ALBUM["stage"] = "人脸归类"
            ALBUM["people"] = cluster_faces()
            with db() as c:
                ALBUM["faces"] = c.execute("SELECT count(*) FROM faces").fetchone()[0]
            ALBUM["stage"] = "完成"
            return
        # 全量：重扫人脸
        ALBUM["stage"] = "找人脸"
        S = STORE
        with db() as c:
            ign_p = set(x[0] for x in
                        c.execute("SELECT DISTINCT path FROM ignored WHERE path<>''"))
        imgs = [p for p, k in zip(S.paths, S.kinds)
                if k == "image" and p not in ign_p]
        ALBUM["total"] = len(imgs)
        # 清库之前先把用户起过的名字记下来：检测框没变，靠 (路径, 框) 就能认回来
        hints = {}
        with db() as c:
            for a, b, nm in c.execute(
                    "SELECT f.path,f.box,p.name FROM faces f"
                    " JOIN people p ON p.id=f.person"
                    " WHERE ifnull(p.name,'')<>''"):
                hints[(a, b)] = nm
        with db() as c:
            c.execute("DELETE FROM faces")
            c.execute("DELETE FROM face_scanned")
        total_faces = 0
        seen = []
        for i, p in enumerate(imgs):
            if not ALBUM["running"]:
                break
            ALBUM["done"] = i + 1
            try:
                # 必须 with：这里要连开几千张图，不关句柄迟早 WinError 32。
                # convert 出来的是独立副本，with 退出后照样能用。
                with Image.open(p) as im0:
                    im = im0.convert("RGB")
            except Exception:
                continue
            seen.append(p)
            try:
                got = faces_of_image(im)
            except Exception:
                continue
            if not got:
                continue
            with db() as c:
                for box, sc_, q_, v in got:
                    c.execute("INSERT INTO faces(path,box,score,q,person,vec)"
                              " VALUES(?,?,?,?,-1,?)",
                              (p, box, sc_, q_, v.astype(np.float32).tobytes()))
            total_faces += len(got)
            ALBUM["faces"] = total_faces
            if (i + 1) % 10 == 0:
                alog("已扫 %d/%d 张，找到 %d 张脸" % (i + 1, len(imgs), total_faces))
        with db() as c:
            for p in seen:
                c.execute("INSERT OR IGNORE INTO face_scanned(path) VALUES(?)", (p,))
        face_ver_set(FACE_VER)          # 记下这批向量是用哪版算法生成的
        ALBUM["stage"] = "人脸归类"
        alog("开始把人脸按人归类 …")
        np_ = cluster_faces(hints)
        ALBUM["people"] = np_
        alog("归类完成：%d 张脸，%d 个人物" % (total_faces, np_))
        ALBUM["stage"] = "完成"
    except Exception:
        traceback.print_exc()
        alog("构建失败：%s" % traceback.format_exc().splitlines()[-1])
    finally:
        ALBUM["running"] = False
        _ALBUM_LOCK.release()


# ---------------------------------------------------------------- 相册 API
@app.post("/api/album/build")
def api_album_build(only: str = ""):
    if not MODEL_READY:
        return JSONResponse({"ok": False, "msg": "模型尚未加载"}, 400)
    if ALBUM["running"]:
        return {"ok": False, "msg": "正在构建中"}
    threading.Thread(target=run_album, args=(only,), daemon=True).start()
    return {"ok": True, "msg": "已开始构建智慧相册"}


@app.get("/api/album/status")
def api_album_status():
    return {"ok": True, **{k: ALBUM[k] for k in
                           ("running", "stage", "done", "total", "faces", "people", "log")}}


@app.get("/api/album/tree")
def api_album_tree():
    people, place, thing, city = [], [], [], []
    with db() as c:
        q = c.execute("""SELECT p.id,p.name,p.cover,
                (SELECT count(DISTINCT f.path) FROM faces f WHERE f.person=p.id) AS c
                FROM people p WHERE c>0 ORDER BY c DESC""")
        for i, (pid, name, cover, cnt) in enumerate(q):
            people.append({"id": pid, "name": name or ("人物 %d" % (i + 1)),
                           "named": bool(name), "count": cnt,
                           "cover": "facecut/%d" % pid})
        for cat, bucket in (("city", city), ("place", place), ("thing", thing)):
            for tag, cnt in c.execute(
                    "SELECT tag,count(*) FROM tags WHERE cat=? GROUP BY tag"
                    " ORDER BY count(*) DESC", (cat,)):
                row = c.execute("SELECT path FROM tags WHERE cat=? AND tag=?"
                                " ORDER BY score DESC LIMIT 1", (cat, tag)).fetchone()
                one = {"tag": tag, "count": cnt,
                       "cover": cover_of(row[0], c) if row else ""}
                if cat == "city":
                    subs = c.execute("SELECT sub,count(*) FROM gps WHERE city=?"
                                     " AND sub<>'' GROUP BY sub"
                                     " ORDER BY count(*) DESC LIMIT 3",
                                     (tag,)).fetchall()
                    one["subs"] = [s[0] for s in subs]
                bucket.append(one)
    located = 0
    moments = []
    with db() as c:
        located = c.execute("SELECT count(*) FROM gps").fetchone()[0]
        unassigned = c.execute("SELECT count(*) FROM faces WHERE person=-3").fetchone()[0]
        uf = c.execute("SELECT id FROM faces WHERE person=-3"
                       " ORDER BY ifnull(q,0) DESC, score DESC LIMIT 1").fetchone()
        for r in c.execute("SELECT id,start,n,title,cover FROM events"
                           " ORDER BY start DESC LIMIT 120"):
            moments.append({"id": r[0], "start": r[1], "count": r[2],
                            "title": r[3] or "一段回忆", "cover": r[4] or ""})
    nfile = len(STORE.paths)
    return {"ok": True, "people": people, "city": city, "place": place,
            "thing": thing, "moments": moments, "unassigned": unassigned,
            "unassigned_cover": uf[0] if uf else 0,
            "geo": {"located": located, "total": nfile}}


_PATH_IDX = {"gen": -1, "d": {}}


def _path_index():
    """路径 → STORE 下标的字典。

    每个分页请求都遍历几万条路径重建一遍是纯浪费；用 STORE.gen 做缓存键，
    内容一变（改名/增删）自动重算。
    """
    if _PATH_IDX["gen"] != STORE.gen:
        d = {}
        for i, p in enumerate(STORE.paths):
            if p not in d:
                d[p] = i
        _PATH_IDX.update(gen=STORE.gen, d=d)
    return _PATH_IDX["d"]


def _item(i):
    """STORE 下标 → 前端通用卡片结构"""
    S = STORE
    p = S.paths[i]
    return {
        "path": p, "kind": S.kinds[i], "w": S.wh[i][0], "h": S.wh[i][1],
        "seg": S.segs[i], "dur": S.durs[i], "size": S.sizes[i],
        "mtime": S.mtimes[i], "note": S.notes[i], "src": S.srcs[i],
        "thumb": "thumb/%s" % thumb_key(p, S.segs[i] if S.kinds[i] != "image" else None),
        "name": os.path.basename(p),
        "rel": os.path.relpath(p, root_of(p) or os.path.dirname(p)),
        "open": raw_url(p),
    }


@app.get("/api/album/items")
def api_album_items(cat: str = "thing", key: str = "", k: int = 60, offset: int = 0):
    off, lim = max(0, offset), max(1, min(k, 300))
    faces_of = {}
    with db() as c:
        if cat in ("person", "people"):
            # 一张合影可能有多张脸属于同一个人，按照片去重再分页，
            # 同时把每张脸的框带上 —— 前端要靠它指定「调哪一张脸」
            rs = c.execute("SELECT path,box,score FROM faces WHERE person=?"
                           " ORDER BY score DESC", (int(key),)).fetchall()
            for p_, b_, s_ in rs:
                faces_of.setdefault(p_, []).append(
                    {"box": b_, "score": round(float(s_ or 0), 3)})
            order = list(faces_of.keys())
            total = len(order)
            rows = [(p_,) for p_ in order[off:off + lim]]
        elif cat in ("event", "moment"):
            try:
                eid = int(key)
            except (TypeError, ValueError):
                return {"ok": True, "total": 0, "offset": off, "has_more": False, "items": []}
            total = c.execute("SELECT count(*) FROM event_items WHERE event=?",
                              (eid,)).fetchone()[0]
            rows = c.execute("SELECT path FROM event_items WHERE event=?"
                             " ORDER BY ts LIMIT ? OFFSET ?",
                             (eid, lim, off)).fetchall()
        else:
            total = c.execute("SELECT count(*) FROM tags WHERE cat=? AND tag=?",
                              (cat, key)).fetchone()[0]
            rows = c.execute("SELECT path FROM tags WHERE cat=? AND tag=?"
                             " ORDER BY score DESC LIMIT ? OFFSET ?",
                             (cat, key, lim, off)).fetchall()
    idx = _path_index()
    items = []
    for r in rows:
        if r[0] not in idx:
            continue
        it = _item(idx[r[0]])
        if faces_of.get(r[0]):
            it["faces"] = faces_of[r[0]]
        items.append(it)
    return {"ok": True, "total": total, "offset": off,
            "has_more": off + len(items) < total, "items": items}


# ---------------- 相册首页：照片墙（按文件去重、时间倒序分页） ----------------
# 缓存键必须用 STORE.gen（内容版本号）而不是 STORE.n：
# 改名、替换文件时条数不变，但下标指向的东西全变了，只比 n 会拿到过期索引。
_ORDER_CACHE = {"gen": -1, "kind": None, "idx": []}


def _file_order(kind="all"):
    """同一视频多段只保留首段；返回按 mtime 倒序的 STORE 索引"""
    if _ORDER_CACHE["gen"] != STORE.gen or _ORDER_CACHE["kind"] != kind:
        seen, idx = set(), []
        for i, p in enumerate(STORE.paths):
            if p in seen:
                continue
            seen.add(p)
            if kind != "all" and STORE.kinds[i] != kind:
                continue
            idx.append(i)
        idx.sort(key=lambda i: (STORE.mtimes[i] or 0), reverse=True)
        _ORDER_CACHE.update(gen=STORE.gen, kind=kind, idx=idx)
    return _ORDER_CACHE["idx"]


@app.get("/api/photos")
def api_photos(offset: int = 0, limit: int = 80, kind: str = "all"):
    order = _file_order(kind)
    total = len(order)
    off, lim = max(0, offset), max(1, min(limit, 300))
    chunk = order[off:off + lim]
    return {"ok": True, "total": total, "offset": off,
            "has_more": off + len(chunk) < total,
            "items": [_item(i) for i in chunk]}


@app.get("/api/photo/detail")
def api_photo_detail(path: str):
    p = os.path.normpath(path or "")
    i = _path_index().get(p)
    if i is None:
        return JSONResponse({"ok": False, "msg": "文件不在索引里"}, 404)
    out = {"ok": True, "item": _item(i)}
    with db() as c:
        out["tags"] = [{"cat": a, "tag": b, "score": round(float(s), 3)}
                       for a, b, s in c.execute(
                "SELECT cat,tag,score FROM tags WHERE path=?"
                " ORDER BY score DESC", (p,))]
        g = c.execute("SELECT lat,lon,city,sub FROM gps WHERE path=?", (p,)).fetchone()
        out["gps"] = ({"lat": g[0], "lon": g[1], "city": g[2] or "", "sub": g[3] or ""}
                      if g else None)
        out["people"] = [{"id": pid, "name": (nm or ("人物 %d" % pid))}
                         for pid, nm in c.execute(
                "SELECT DISTINCT p.id,p.name FROM faces f JOIN people p ON p.id=f.person"
                " WHERE f.path=? AND f.person>=0", (p,))]
    return out


@app.post("/api/person/rename")
def api_person_rename(payload: dict):
    pid = int(payload.get("id", -1))
    name = (payload.get("name") or "").strip()
    with db() as c:
        c.execute("UPDATE people SET name=? WHERE id=?", (name, pid))
    merged = merge_same_name() if name else 0
    return {"ok": True, "merged": merged}


@app.post("/api/item/delete")
def api_item_delete(payload: dict):
    """删除本地文件：先挪到程序目录下的 trash 回收站，可手工找回"""
    src = os.path.normpath((payload.get("path") or "").strip())
    if not any(src.startswith(os.path.abspath(r) + os.sep) for r in MEDIA_ROOTS):
        return JSONResponse({"ok": False, "msg": "路径不合法"}, 400)
    if not os.path.exists(src):
        _purge_index([src])
        return JSONResponse({"ok": False, "msg": "文件已经不在了"}, 404)
    try:
        rel, dst = _to_trash(src)
    except Exception as e:
        return JSONResponse({"ok": False, "msg": "移动失败：%s" % e}, 500)
    _purge_index([src])
    STORE.reload()
    alog("已删除（存到回收站）：%s" % rel)
    return {"ok": True, "rel": rel, "trash": dst}


def _to_trash(src):
    """把文件挪进程序自己的回收站，返回 (相对路径, 目标路径)"""
    rel = os.path.relpath(src, root_of(src) or os.path.dirname(src))
    day = time.strftime("%Y%m%d")
    dst = os.path.join(str(TRASH_DIR), day, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        stem, ext = os.path.splitext(dst)
        dst = "%s_%s%s" % (stem, time.strftime("%H%M%S"), ext)
    try:
        os.rename(src, dst)
    except OSError:
        shutil.move(src, dst)
    return rel, dst


@app.get("/api/dups")
def api_dups(thresh: int = 0):
    """重复 / 连拍照片。第一次调用会自动在后台补算指纹（老库没有 ph）。"""
    miss = phash_missing()
    if miss and not _PH["running"]:
        threading.Thread(target=build_phash, daemon=True).start()
    groups = find_dups(thresh or None)
    return {"ok": True, "groups": groups, "count": len(groups),
            "waste": sum(g["waste"] for g in groups),
            "phash": {"missing": miss, "running": _PH["running"],
                      "done": _PH["done"], "total": _PH["total"]}}


@app.post("/api/dups/clean")
def api_dups_clean(payload: dict):
    """每组只留 keep（默认分辨率最高那张），其余挪进回收站。

    dry=true 时只统计不真删，前端可以先用它确认"能省多少"。
    """
    dry = bool((payload or {}).get("dry"))
    groups = find_dups((payload or {}).get("thresh") or None)
    moved, freed = 0, 0
    for g in groups:
        for it in g["items"]:
            if it["path"] == g["keep"]:
                continue
            if dry:
                moved += 1
                freed += it["size"]
                continue
            try:
                _to_trash(it["path"])
                _purge_index([it["path"]])
                moved += 1
                freed += it["size"]
            except Exception as e:
                print("[dups] 没删掉 %s → %s" % (it["path"], e), flush=True)
    if moved and not dry:
        STORE.reload()
        alog("清理重复照片：移除 %d 个文件，释放 %.1f MB"
             % (moved, freed / 1048576.0))
    return {"ok": True, "moved": moved, "freed": freed, "dry": dry}


@app.post("/api/person/delete")
def api_person_delete(payload: dict):
    pid = int(payload.get("id", -1))
    with db() as c:
        r = c.execute("SELECT name FROM people WHERE id=?", (pid,)).fetchone()
        name = r[0] if r else ""
        vecs = c.execute("SELECT vec FROM faces WHERE person=?", (pid,)).fetchall()
        n = len(vecs)
        # 只按**人脸向量**进忽略名单，不按照片路径。
        # 以前这里还额外插了一条 (vec=NULL, path=照片) 记录，但 cluster_faces
        # 只读 vec IS NOT NULL，那条记录从来没生效过；万一将来真按 path 忽略，
        # 就会把这张合影里**别人**的脸一起忽略掉 —— 比不做更糟。
        for (v,) in vecs:
            if v:
                c.execute("INSERT INTO ignored(vec,note,path) VALUES(?,?,?)",
                          (v, name or "", ""))
        c.execute("DELETE FROM faces WHERE person=?", (pid,))
        c.execute("DELETE FROM people WHERE id=?", (pid,))
    _drop_face_cache(pid)
    f = THUMB_DIR / ("face_%d.jpg" % pid)
    if f.exists():
        try:
            f.unlink()
        except Exception:
            pass
    alog("已删除人物分组 #%d「%s」（%d 张脸，已加入忽略名单）" % (pid, name or "未命名", n))
    return {"ok": True, "faces": n}


def _drop_face_cache(*pids):
    """分组结构变了就删掉对应的头像裁图缓存。

    `face_{pid}.jpg` 是按**编号**缓存的，而 pid 会在重聚类后被复用：
    用户把某张脸从 1 号组挪走、或把两个组合并之后，1 号组的封面还是老那张脸，
    界面上看起来就是"改了没用"。所以任何动到分组归属的接口都必须清一次。
    """
    for pid in pids:
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            continue
        if pid < 0:
            continue
        f = THUMB_DIR / ("face_%d.jpg" % pid)
        try:
            if f.exists():
                f.unlink()
        except OSError:
            pass


def _person_stats(pid, c):
    """重算某个分组的封面 / 中心向量，返回剩下的脸数"""
    rows = c.execute("SELECT path,box,vec FROM faces WHERE person=?",
                     (pid,)).fetchall()
    if not rows:
        return 0
    try:
        V = np.stack([np.frombuffer(r[2], np.float32) for r in rows])
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        cen = V.mean(axis=0)
        cen = cen / (np.linalg.norm(cen) + 1e-9)
        blob = cen.astype(np.float32).tobytes()
    except Exception:
        blob = None
    c.execute("UPDATE people SET cover=?,center=? WHERE id=?",
              (rows[0][0] + "|" + str(rows[0][1]), blob, pid))
    return len(rows)


def _new_person(c) -> int:
    nid = int(c.execute("SELECT ifnull(max(id),0)+1 FROM people").fetchone()[0])
    c.execute("INSERT INTO people(id,name,cover,center) VALUES(?,?,?,?)",
              (nid, "", "", None))
    return nid


@app.post("/api/face/move")
def api_face_move(payload: dict):
    """手动纠正：把某张脸挪到别的人物分组。

    to >= 0  移到这个人物
    to == -2 单独成一组（"这根本不是他" —— 拆出来，之后爱怎么处理怎么处理）
    to == -1 移出所有人像分类（不再出现在人像页）
    """
    path = (payload.get("path") or "").strip()
    box = str(payload.get("box") or "").strip()
    src_pid = payload.get("from")
    to = int(payload.get("to", -2))
    if not path:
        return JSONResponse({"ok": False, "msg": "缺少照片路径"}, 400)
    with db() as c:
        if box:
            # 精确到某一张脸
            rows = c.execute("SELECT id,person FROM faces WHERE path=? AND box=?",
                             (path, box)).fetchall()
        elif src_pid is not None and int(src_pid) >= 0:
            # 「这张照片里的他不对」：只取属于当前这个分组的脸，
            # 别把合影里其他人的脸也一起挪走
            rows = c.execute("SELECT id,person FROM faces WHERE path=? AND person=?",
                             (path, int(src_pid))).fetchall()
        else:
            rows = c.execute("SELECT id,person FROM faces WHERE path=?", (path,)).fetchall()
        if not rows:
            return JSONResponse({"ok": False, "msg": "这张照片里没有可调整的人脸"}, 404)
        fids = [int(r[0]) for r in rows]
        src = sorted({int(r[1]) for r in rows})
        if to == -2:
            to = _new_person(c)
        elif to >= 0:
            if not c.execute("SELECT 1 FROM people WHERE id=?", (to,)).fetchone():
                return JSONResponse({"ok": False, "msg": "目标人物不存在"}, 404)
        if to == -1:
            # 「别出现在人像里」必须持久。以前只是 UPDATE faces SET person=-1，
            # 但重聚类时 person=-1 只意味着"未归类"，那张脸会照常参与聚类、
            # 又被算回某个组 —— 用户点完一次，下次重建又冒出来。
            # 真正的做法和删除人物分组一致：人脸向量进忽略名单，以后连重扫都跳过。
            for f in fids:
                r = c.execute("SELECT vec FROM faces WHERE id=?", (f,)).fetchone()
                if r and r[0]:
                    # path 一律留空：ignored 里带 path 的记录会被
                    # incremental_faces / 全量重扫当成"整张照片不扫脸"，
                    # 那张合影里**别人**的脸也会跟着消失。只按向量忽略。
                    c.execute("INSERT INTO ignored(vec,note,path) VALUES(?,?,?)",
                              (r[0], "手动移出人像", ""))
                c.execute("DELETE FROM faces WHERE id=?", (f,))
        else:
            for f in fids:
                # manual=1：人已经亲口说"这张脸该归这儿"，以后重聚类不许把它打散
                c.execute("UPDATE faces SET person=?, manual=1 WHERE id=?", (to, f))
        gone = []
        for s in src:
            if s < 0 or s == to:
                continue
            if _person_stats(s, c) == 0:
                c.execute("DELETE FROM people WHERE id=?", (s,))
                gone.append(s)
        left = _person_stats(to, c) if to >= 0 else 0
    _drop_face_cache(to, *src)
    alog("调整人脸归属：%d 张脸 → 分组 %s" % (len(fids), to if to >= 0 else "未归类"))
    return {"ok": True, "faces": len(fids), "to": to, "gone": gone, "left": left}


@app.post("/api/person/merge_into")
def api_person_merge_into(payload: dict):
    """把分组 from 整个并进 to（不用先改成同名）"""
    a = int(payload.get("from", -1))
    b = int(payload.get("to", -1))
    if a < 0 or b < 0 or a == b:
        return JSONResponse({"ok": False, "msg": "参数不对"}, 400)
    with db() as c:
        ra = c.execute("SELECT name FROM people WHERE id=?", (a,)).fetchone()
        rb = c.execute("SELECT name FROM people WHERE id=?", (b,)).fetchone()
        if not ra or not rb:
            return JSONResponse({"ok": False, "msg": "分组不存在"}, 404)
        na, nb = ra[0] or "", rb[0] or ""
        n = c.execute("SELECT count(*) FROM faces WHERE person=?", (a,)).fetchone()[0]
        c.execute("UPDATE faces SET person=?, manual=1 WHERE person=?", (b, a))
        c.execute("DELETE FROM people WHERE id=?", (a,))
        if not nb and na:                     # 目标没名字就沿用来源的名字
            c.execute("UPDATE people SET name=? WHERE id=?", (na, b))
        _person_stats(b, c)
    _drop_face_cache(a, b)
    alog("合并人物分组：%d「%s」→ %d「%s」（%d 张脸）" % (a, na or "未命名", b, nb or na or "未命名", n))
    return {"ok": True, "faces": n, "to": b}


@app.post("/api/face/remove")
def api_face_remove(payload: dict):
    """「这张不是他」：从当前分组拆出来，单独成一组。

    注意这里**不删人脸向量**（老版本是直接 DELETE，等于把证据扔了，之后连纠正都没法纠正）。
    拆出来的那张脸还留在库里，可以再归给别人，也可以直接删掉那个新分组。
    """
    pid = int(payload.get("id", -1))
    path = (payload.get("path") or "").strip()
    box = str(payload.get("box") or "").strip()
    with db() as c:
        if box:
            rows = c.execute("SELECT id FROM faces WHERE person=? AND path=? AND box=?",
                             (pid, path, box)).fetchall()
        else:
            rows = c.execute("SELECT id FROM faces WHERE person=? AND path=?",
                             (pid, path)).fetchall()
        if not rows:
            return JSONResponse({"ok": False, "msg": "没找到这张脸"}, 404)
        nid = _new_person(c)
        for (fid,) in rows:
            c.execute("UPDATE faces SET person=?, manual=1 WHERE id=?", (nid, fid))
        left = _person_stats(pid, c) if pid >= 0 else 0
        gone = False
        if pid >= 0 and left == 0:
            c.execute("DELETE FROM people WHERE id=?", (pid,))
            gone = True
        _person_stats(nid, c)
    _drop_face_cache(pid, nid)
    alog("「不是他」：%d 张脸从分组 %d 拆出 → 新分组 %d" % (len(rows), pid, nid))
    return {"ok": True, "faces": len(rows), "left": left, "gone": gone, "new": nid}


@app.post("/api/person/merge")
def api_person_merge():
    n = merge_same_name()
    return {"ok": True, "merged": n}


def _face_crop(path, box, out: Path, pad_ratio=0.15):
    """按框裁一张脸存成 jpg；失败返回 False。人物封面和「其他面孔」网格共用。"""
    try:
        with Image.open(path) as im0:
            im = im0.convert("RGB")
        x1, y1, x2, y2 = [int(v) for v in box.split(",")]
        pad = int(max(x2 - x1, y2 - y1) * pad_ratio)
        im.crop((max(0, x1 - pad), max(0, y1 - pad),
                 min(im.width, x2 + pad), min(im.height, y2 + pad))
                ).save(out, "JPEG", quality=88)
        return True
    except Exception:
        return False


@app.get("/facecut/{pid}")
def api_facecut(pid: int):
    p = THUMB_DIR / ("face_%d.jpg" % pid)
    if not p.exists():
        with db() as c:
            # 用质量分挑封面：检测分高但糊的、大侧脸的，都不适合当人物头像
            r = c.execute("SELECT path,box FROM faces WHERE person=?"
                          " ORDER BY ifnull(q,0) DESC, score DESC LIMIT 1",
                          (pid,)).fetchone()
        if not r or not _face_crop(r[0], r[1], p):
            return Response(status_code=404)
    # 不能给长缓存：pid 是编号会被复用，改分组后同一个 URL 会指向另一个人，
    # 浏览器拿着 10 天的缓存就会一直显示旧头像。no-cache 让它每次都验一下，
    # 没变就是 304，本地服务这点开销可以忽略。
    return FileResponse(p, media_type="image/jpeg",
                        headers={"Cache-Control": "no-cache"})


@app.get("/facecut/f/{fid}")
def api_facecut_one(fid: int):
    """单张脸的裁剪：给「其他面孔」网格用（不依赖它属于哪个人物）。"""
    p = THUMB_DIR / ("face_f%d.jpg" % fid)
    if not p.exists():
        with db() as c:
            r = c.execute("SELECT path,box FROM faces WHERE id=?", (fid,)).fetchone()
        if not r or not _face_crop(r[0], r[1], p):
            return Response(status_code=404)
    return FileResponse(p, media_type="image/jpeg",
                        headers={"Cache-Control": "no-cache"})


@app.get("/api/faces/unassigned")
def api_faces_unassigned(offset: int = 0, k: int = 60):
    """「其他面孔」：检测到了脸，但同一个人的脸太少，不够单独成为一个「人物」。

    这是 immich 那套密度约束的另一半 —— 不建人物不等于把人脸扔掉，
    用户想认领的时候随时能捞回来（归到某人 / 单独建组 / 标记为不是人）。
    """
    lim = max(1, min(k, 200))
    off = max(0, offset)
    with db() as c:
        total = c.execute("SELECT count(*) FROM faces WHERE person=-3").fetchone()[0]
        rows = c.execute("SELECT id,path,box,score,q FROM faces WHERE person=-3"
                         " ORDER BY ifnull(q,0) DESC, score DESC LIMIT ? OFFSET ?",
                         (lim, off)).fetchall()
    items = [{"fid": r[0], "path": r[1], "box": r[2],
              "score": round(float(r[3] or 0), 3), "q": round(float(r[4] or 0), 3),
              "name": os.path.basename(r[1]), "thumb": "facecut/f/%d" % r[0]}
             for r in rows]
    return {"ok": True, "total": total, "offset": off, "items": items,
            "has_more": off + len(items) < total}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8765")))
