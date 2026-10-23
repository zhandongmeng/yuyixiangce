#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""开发模式全功能自检：不打包，直接跑 server 的核心流程"""
import os
import sys
import time
from pathlib import Path

_HOME = Path(__file__).resolve().parent.parent
os.environ["AIWIN_HOME"] = str(_HOME)
sys.path.insert(0, str(_HOME / "app"))

# 自检用的素材目录，可以用 AIWIN_TEST_MEDIA 指到别处
TEST_DIR = os.environ.get("AIWIN_TEST_MEDIA") or str(_HOME.parent / "aiwin_test_media")

import server  # noqa: E402

# 1) 等模型
t0 = time.time()
while not server.MODEL_READY and time.time() - t0 < 120:
    if server.MODEL_ERR:
        print("!! 模型加载失败:", server.MODEL_ERR); sys.exit(1)
    time.sleep(1)
print("== 模型就绪 %.1fs providers=%s ==" % (time.time() - t0, server.MODEL_INFO), flush=True)

# 2) 编码健全性
v = server.CLIP.encode_text("一只猫躺在草地上")
print("文本向量 norm=%.3f dim=%d" % (float(v @ v), len(v)), flush=True)
import glob
p0 = sorted(glob.glob(TEST_DIR + "/*.jpg"))[0]
vi = server.CLIP.encode_images([p0])[0]
print("图片向量 norm=%.3f dim=%d" % (float(vi @ vi), len(vi)), flush=True)

# 3) 加目录 + 建索引
ok, msg = server.STORE.add_folder(TEST_DIR)
print("== add_folder:", ok, msg, flush=True)
server.do_index()
print("== 索引后 STORE.n =", server.STORE.n, flush=True)

# 4) 搜索
for q in ("海边的日落", "一只猫", "路边停放的汽车", "人们在一起吃饭"):
    res = server.do_search(server.CLIP.encode_text(q), k=5)
    print("\n搜索「%s」→ %d 条:" % (q, len(res)), flush=True)
    for r in res[:5]:
        print("   %5.1f%% %-28s %s %s" % (r["score"] * 100, r["name"][:28], r["kind"],
              ("🎤" + r["note"][:30]) if r.get("note") else ""), flush=True)

# 5) 相册：标签 + 城市 + 增量人脸
server.run_album("new")
print("\n== 相册树 ==", flush=True)
# 直接调函数，不走 HTTP
tree = server.api_album_tree()
print("人像 %d：%s" % (len(tree["people"]),
      [(p["name"], p["count"]) for p in tree["people"]]), flush=True)
print("城市 %d · 场景 %d · 事物 %d" % (len(tree["city"]), len(tree["place"]), len(tree["thing"])), flush=True)
print("事物分类：%s" % [t["tag"] for t in tree["thing"]], flush=True)
print("场景分类：%s" % [t["tag"] for t in tree["place"]], flush=True)

# 6) 语音转写结果（视频音轨）
asr = [(p, n) for p, k, n in zip(server.STORE.paths, server.STORE.kinds, server.STORE.notes) if n]
print("\n== 语音转写 ==", flush=True)
for p, n in asr[:6]:
    print("  %s → %s" % (os.path.basename(p)[:30], n[:50]), flush=True)
print("\nSMOKE_DONE", flush=True)
