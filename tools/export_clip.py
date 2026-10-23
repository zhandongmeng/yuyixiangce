#!/usr/bin/env python3
"""把中文 CLIP (chinese-clip-vit-base-patch16) 导出为 ONNX 并 int8 量化。

产出（models/clip/）：
  vision.onnx   int8  图片塔  输入 pixel_values (N,3,224,224) → (N,512)
  text.onnx     int8  文本塔  输入 input_ids/attention_mask → (N,512)
  vocab.txt            BERT 中文词表（tokenizers.BertWordPieceTokenizer 用）
  meta.json            预处理参数 + 校验信息
"""
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

_HOME = Path(__file__).resolve().parent.parent
# 原始 PyTorch 权重放哪、产物写到哪，都可以用环境变量覆盖
SRC = os.environ.get("CLIP_SRC") or str(_HOME / "models_src" / "clip_cn")
OUT = os.environ.get("CLIP_OUT") or str(_HOME / "models" / "clip")
os.makedirs(OUT, exist_ok=True)

print("== 加载 torch 权重 ==", flush=True)
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

m = ChineseCLIPModel.from_pretrained(SRC)
m.eval()
proc = ChineseCLIPProcessor.from_pretrained(SRC)
print("vision:", type(m.vision_model).__name__, "| text:", type(m.text_model).__name__, flush=True)
for fn in ("get_image_features", "get_text_features"):
    assert hasattr(m, fn), "模型缺少 " + fn


class VisWrap(nn.Module):
    def __init__(self, mm):
        super().__init__()
        self.m = mm

    def forward(self, pixel_values):
        o = self.m.vision_model(pixel_values=pixel_values)
        pooled = o.pooler_output if hasattr(o, "pooler_output") else o[1]
        return self.m.visual_projection(pooled)


class TxtWrap(nn.Module):
    def __init__(self, mm):
        super().__init__()
        self.m = mm

    def forward(self, input_ids, attention_mask, token_type_ids):
        # chinese-clip 文本塔没有 pooler：pooled = last_hidden_state[:, 0]（与 get_text_features 一致）
        o = self.m.text_model(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
        first = o.last_hidden_state[:, 0, :]
        return self.m.text_projection(first)


def cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


IMG_EXT_SET = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".heic", ".heif"}


def _pil_to_input(path, size=224):
    """与 app/server.py 运行时一致的预处理"""
    from PIL import Image as _I
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except Exception:
        pass
    im = _I.open(path).convert("RGB").resize((size, size), _I.BILINEAR)
    a = np.transpose(np.asarray(im, dtype=np.float32) / 255.0, (2, 0, 1))[None]
    mean = np.array([0.48145466, 0.4578275, 0.40821073], np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.26862954, 0.26130258, 0.27577711], np.float32).reshape(1, 3, 1, 1)
    return ((a - mean) / std).astype(np.float32)


def _toks(texts):
    from tokenizers import BertWordPieceTokenizer
    tok = BertWordPieceTokenizer(os.path.join(SRC, "vocab.txt"), lowercase=True,
                                 handle_chinese_chars=True)
    tok.enable_truncation(52)
    encs = tok.encode_batch(texts)
    L = min(max(len(e.ids) for e in encs), 52)
    ids = np.zeros((len(encs), L), dtype=np.int64)
    am = np.zeros((len(encs), L), dtype=np.int64)
    for r, e in enumerate(encs):
        n = min(len(e.ids), L)
        ids[r, :n] = e.ids[:n]
        am[r, :n] = 1
    return ids, am, np.zeros_like(ids)


def _tok_ids(texts):
    return _toks(texts)[0]


def _tok_am(texts):
    return _toks(texts)[1]


def _tok_tt(texts):
    return _toks(texts)[2]


# ---------------- 1. fp32 导出 ----------------
print("== 导出 vision ONNX ==", flush=True)
vw = VisWrap(m).eval()
dummy_img = torch.randn(1, 3, 224, 224)
t0 = time.time()
torch.onnx.export(
    vw, (dummy_img,), os.path.join(OUT, "vision_fp32.onnx"),
    input_names=["pixel_values"], output_names=["image_embeds"],
    # 固定 batch=1（不打 dynamic_axes）：DirectML 对自由维度支持很差，
    # int8 模型里的 Reshape 节点遇到符号维直接 kernel 报 0x8007023E。
    # 视觉推理本来就是逐张进行，固定 1 零损失；文本塔是 CPU EP 跑，动态维无害。
    # ⚠️ dynamo=False：torch 2.9 起新版 dynamo 导出器是默认值，它产的图
    # DirectML 编译不过（EP Error 后自动回落 CPU，独显等于白买）。
    # 经典 TorchScript 导出器实测 DML 直接跑通，两个塔都用它。
    opset_version=17, do_constant_folding=True, dynamo=False)
print("  %.1fs" % (time.time() - t0), flush=True)

print("== 导出 text ONNX ==", flush=True)
tw = TxtWrap(m).eval()
ids = torch.randint(5, 20000, (1, 16))
am = torch.ones(1, 16, dtype=torch.long)
tt = torch.zeros(1, 16, dtype=torch.long)
t0 = time.time()
torch.onnx.export(
    tw, (ids, am, tt), os.path.join(OUT, "text_fp32.onnx"),
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["text_embeds"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                  "attention_mask": {0: "batch", 1: "seq"},
                  "token_type_ids": {0: "batch", 1: "seq"},
                  "text_embeds": {0: "batch"}},
    opset_version=17, do_constant_folding=True, dynamo=False)
print("  %.1fs" % (time.time() - t0), flush=True)

# ---------------- 2. int8 量化 ----------------
# 只量化视觉塔：图片是批处理大头（int8 收益最大）；
# 文本塔保持 fp32 —— 实测 int8 会把文本向量余弦打到 0.87，查询向量失真直接伤搜索效果，
# 而文本塔很小（一次查询/标签就编码几句），fp32 速度完全够。
print("== int8 动态量化（仅视觉塔，文本塔保持 fp32）==", flush=True)
import onnx  # noqa: E402
from onnxruntime.quantization import quantize_dynamic, QuantType  # noqa: E402


def _strip_value_info(path):
    """ dynamo 导出器写的 value_info 有时和真实形状冲突（768 vs 512），
        quantize_dynamic 前置的 shape inference 会直接报错 —— 全清掉再量化。"""
    m = onnx.load(path)
    del m.graph.value_info[:]
    onnx.save(m, path)


_strip_value_info(os.path.join(OUT, "vision_fp32.onnx"))
quantize_dynamic(os.path.join(OUT, "vision_fp32.onnx"),
                 os.path.join(OUT, "vision.onnx"), weight_type=QuantType.QInt8)
# 文本塔：dynamo 导出器把 fp32 权重写成了 text_fp32.onnx + text_fp32.onnx.data（外置），
# onnx.load 默认会把外置权重内联进模型，再存就是单文件
_m = onnx.load(os.path.join(OUT, "text_fp32.onnx"))
onnx.save(_m, os.path.join(OUT, "text.onnx"))
del _m
for junk in ("text_fp32.onnx", "text_fp32.onnx.data"):
    p = os.path.join(OUT, junk)
    if os.path.exists(p):
        os.remove(p)
print("  完成", flush=True)

# ---------------- 3. 一致性校验 ----------------
print("== 校验 torch vs onnx（视觉塔 int8 / 文本塔 fp32）==", flush=True)
import onnxruntime as ort

opt = ort.SessionOptions()
opt.intra_op_num_threads = 4
vs = ort.InferenceSession(os.path.join(OUT, "vision.onnx"), opt,
                          providers=["CPUExecutionProvider"])
ts = ort.InferenceSession(os.path.join(OUT, "text.onnx"), opt,
                          providers=["CPUExecutionProvider"])

# 图片：processor 预处理 vs torch 参考
img = Image.new("RGB", (300, 200), (80, 140, 200))
# 画点花样，别用纯色
import random
random.seed(0)
px = img.load()
for i in range(300):
    for j in range(200):
        px[i, j] = (i % 255, j % 255, (i * j) % 255)

inp = proc(images=img, return_tensors="pt")
with torch.no_grad():
    t_img = m.get_image_features(pixel_values=inp["pixel_values"]).pooler_output.numpy()[0]
o_img = vs.run(None, {"pixel_values": inp["pixel_values"].numpy()})[0][0]
c_img = cos(t_img, o_img)
print("  图片向量余弦 = %.5f (dim=%d)" % (c_img, len(t_img)), flush=True)
assert c_img > 0.975, "图片塔量化掉点太多！"

texts = ["一只猫躺在草地上", "海边的日落", "公司会议室的白板", "a dog running on the beach"]
tin = proc(text=texts, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    t_txt = m.get_text_features(input_ids=tin["input_ids"],
                                attention_mask=tin["attention_mask"]).pooler_output.numpy()
o_txt = ts.run(None, {"input_ids": tin["input_ids"].numpy(),
                      "attention_mask": tin["attention_mask"].numpy(),
                      "token_type_ids": tin.get("token_type_ids",
                                                torch.zeros_like(tin["input_ids"])).numpy()})[0]
for k, txt in enumerate(texts):
    c = cos(t_txt[k], o_txt[k])
    print("  文本 %d 余弦 = %.5f  %s" % (k, c, txt), flush=True)
    assert c > 0.995, "文本塔（fp32）掉点太多！"

# 文本塔和图片塔必须在同一空间：拿真实照片抽查图文匹配合理性（只打印不硬断言）
CAPS = ["一只猫", "一只狗", "海边沙滩", "山间森林", "城市高楼", "餐桌美食",
        "室内房间", "道路上汽车", "婴儿小孩", "花草植物", "电脑屏幕", "运动球场"]
cap_vecs = ts.run(None, {"input_ids": _tok_ids(CAPS), "attention_mask": _tok_am(CAPS),
                         "token_type_ids": _tok_tt(CAPS)})[0]
import glob as _glob
REAL = []
for _pat in (r"E:/相册/相机照片/*.*", r"E:/相册/备份/**/*.*", r"E:/相册/*.*"):
    REAL += [p for p in _glob.glob(_pat, recursive=True)
             if os.path.splitext(p)[1].lower() in IMG_EXT_SET]
REAL = REAL[:6]
print("\n== 真实照片图文匹配抽查（int8 onnx）==", flush=True)
for p in REAL:
    try:
        v = vs.run(None, {"pixel_values": _pil_to_input(p)})[0][0]
    except Exception as e:
        print("  跳过 %s: %s" % (os.path.basename(p), e), flush=True)
        continue
    v = v / (np.linalg.norm(v) + 1e-9)
    sims = cap_vecs @ v
    top = np.argsort(-sims)[:3]
    print("  %-26s → %s" % (os.path.basename(p)[:26],
          "  ".join("%s(%.2f)" % (CAPS[i], sims[i]) for i in top)), flush=True)
if not REAL:
    print("  （本机没有找到 E:\\相册 照片，跳过抽查）", flush=True)

# ---------------- 4. tokenizer 一致性（tokenizers vs transformers） ----------------
print("== 校验 tokenizers.BertWordPieceTokenizer 与官方一致 ==", flush=True)
from tokenizers import BertWordPieceTokenizer  # noqa: E402

tw_tok = BertWordPieceTokenizer(os.path.join(SRC, "vocab.txt"), lowercase=True,
                                handle_chinese_chars=True)
tw_tok.enable_truncation(52)
for txt in texts + ["智慧相册", "2026 年春节全家福", "照片里有三只小猫在草地上打滚"]:
    # 用官方 processor 编码作为参照（padding 对齐到本批最长）
    tin_ref = proc(text=[txt], return_tensors="pt", padding="max_length",
                   truncation=True, max_length=52)
    ids_ref = [int(x) for x in tin_ref["input_ids"][0] if int(x) != 0]
    ids_new = [int(x) for x in tw_tok.encode(txt).ids if int(x) != 0]
    assert ids_new == ids_ref, "分词不一致：%s\n%s\n%s" % (txt, ids_new, ids_ref)
    print("  %-28s -> %s" % (txt, ids_new[:12]), flush=True)
print("  分词一致 ✓", flush=True)

# ---------------- 5. 收尾：词表 + meta ----------------
shutil.copy(os.path.join(SRC, "vocab.txt"), os.path.join(OUT, "vocab.txt"))
meta = {
    "arch": "chinese-clip-vit-base-patch16",
    "dim": 512,
    "image_size": 224,
    "mean": [0.48145466, 0.4578275, 0.40821073],
    "std": [0.26862954, 0.26130258, 0.27577711],
    "max_text_len": 52,
    "cos_image": round(c_img, 5),
}
with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
for junk in ("vision_fp32.onnx", "vision_fp32.onnx.data", "text_fp32.onnx",
             "text_fp32.onnx.data"):
    p = os.path.join(OUT, junk)
    if os.path.exists(p):
        os.remove(p)
print("\n== 产出 ==", flush=True)
for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    print("  %-16s %8.1f MB" % (f, os.path.getsize(p) / 1048576), flush=True)
print("EXPORT_DONE", flush=True)
