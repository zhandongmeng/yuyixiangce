#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""界面验收（第三批）：搜索升级 —— 关键词融合 + 历史记录

前置：语义相册在跑。
用法：python tools/verify_ui_search.py [端口]
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cdp  # noqa: E402

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
SHOT = Path(__file__).resolve().parent.parent / "_shots"
SHOT.mkdir(parents=True, exist_ok=True)
OK, BAD = [], []


def chk(name, cond, extra=""):
    (OK if cond else BAD).append(name)
    print(("  [OK] " if cond else "  [!!] ") + name + (("  " + str(extra)) if extra else ""),
          flush=True)
    return cond


def post(path, payload):
    req = urllib.request.Request(
        "http://127.0.0.1:%d%s" % (PORT, path),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as f:
        return json.loads(f.read().decode("utf-8"))


def test_api():
    """接口层：文件名命中必须被加分抬升排名"""
    print("== 接口：关键词融合 ==")
    d = post("/api/search", {"q": "IMG_20251207", "k": 8})
    chk("搜索返回 ok", d.get("ok") is True)
    rs = d.get("results") or []
    chk("有结果", len(rs) > 0, "%d 项" % len(rs))
    if not rs:
        return

    # 前两项必须都是文件名含 IMG_20251207 的
    top2 = [r.get("name", "") for r in rs[:2]]
    chk("文件名命中排前两名", all("IMG_20251207" in n for n in top2), top2)

    # 注意：score 是「模态校准后」的排序分，cos 是原始余弦，两者天然不等，
    # 所以不能用 score-cos 反推关键词加分（那会把校准量算进来）。
    # 加分逻辑本身放到下面 _kw_bonus 单测里精确验证。
    hit_idx = [i for i, r in enumerate(rs) if "IMG_20251207" in r.get("name", "")]
    miss_idx = [i for i, r in enumerate(rs) if "img" not in r.get("path", "").lower()]
    chk("命中项全部排在未命中项之前",
        not hit_idx or not miss_idx or max(hit_idx) < min(miss_idx),
        "hit=%s miss=%s" % (hit_idx, miss_idx[:3]))


def _py():
    """找带 numpy/onnxruntime 的解释器：验收脚本自己可能跑在系统 python 上"""
    import os
    cand = [os.environ.get("AIWIN_PY", "")]
    cand.append(str(Path.home() / ".workbuddy/binaries/python/envs/aiwin/Scripts/python.exe"))
    for c in cand:
        if c and Path(c).exists():
            return c
    return sys.executable


def test_bonus():
    """单元层：直接测加分函数，不依赖 CLIP 模型"""
    print("== 单元：加分函数 ==")
    import subprocess
    code = r'''
import os, sys
sys.path.insert(0, r"%s")
import server
res = {}
paths = ["D:/p/IMG_20251207_130540.jpg", "D:/p/海边日落.jpg", "D:/p/AQ_IMG_1.jpg"]
b = server._kw_bonus("IMG_20251207", paths)
res["hit"]   = [round(float(x), 4) for x in b]
b2 = server._kw_bonus("IMG", paths)          # 短词整串命中文件名
res["short"] = None if b2 is None else [round(float(x), 4) for x in b2]
b3 = server._kw_bonus("IMG 海边 日落", ["D:/p/IMG_海边_日落.jpg", "D:/p/IMG_a.jpg"])
res["multi"] = None if b3 is None else [round(float(x), 4) for x in b3]
b5 = server._kw_bonus("20251207", ["D:/旅行/20251207/x.jpg", "D:/p/IMG_20251207_1.jpg"])
res["path"] = None if b5 is None else [round(float(x), 4) for x in b5]
b4 = server._kw_bonus("", paths)
res["empty"] = b4 is None
import json as _j
print("@@" + _j.dumps(res))
os._exit(0)
''' % str(Path(__file__).resolve().parent.parent / "app")
    p = subprocess.run([_py(), "-u", "-c", code], capture_output=True,
                       text=True, timeout=180)
    out = p.stdout
    line = [l for l in out.splitlines() if l.startswith("@@")]
    if not chk("加分函数可调用", bool(line),
               (p.stderr or out)[-300:] if not line else ""):
        return
    r = json.loads(line[0][2:])

    chk("整串命中文件名 → 0.22（压过语义分）", r["hit"][0] == 0.22, r["hit"])
    chk("未命中项不加分", r["hit"][2] == 0.0, r["hit"])
    chk("短词整串命中文件名 → 0.22", r["short"] and r["short"][2] == 0.22, r["short"])
    chk("仅路径命中 → 0.12", r["path"] and r["path"][0] == 0.12, r["path"])
    chk("文件名命中优先于路径命中", r["path"] and r["path"][1] == 0.22, r["path"])
    chk("空查询返回 None", r["empty"] is True)
    if r["multi"]:
        chk("多词全命中封顶 0.12", r["multi"][0] == 0.12, r["multi"])
        chk("部分命中按词数计分", r["multi"][1] == 0.05, r["multi"])


def test_ui():
    print("== 界面：搜索历史 ==")
    pg = cdp.Page("http://127.0.0.1:%d/" % PORT, 1440, 940)
    try:
        time.sleep(2.5)
        # 清空历史，确保起点干净
        pg.js("try{localStorage.removeItem('qhist')}catch{}")
        pg.js("location.hash='#/photos'")
        time.sleep(1.5)

        # 1. 无历史时，打开搜索面板不应显示历史行
        pg.js("el('btnSearch').click()")
        time.sleep(0.9)
        vis = pg.js("getComputedStyle(el('qHist')).display")
        chk("无历史时历史行隐藏", vis == "none", vis)
        pg.shot(str(SHOT / "s1_no_hist.png"))

        # 2. 用真实搜索跑两次（走 UI 的 runSearch，会写历史）
        pg.js("el('q').value='海边日落'; el('btnGo').click()")
        time.sleep(3.0)
        n1 = pg.js("qHistGet().length")
        chk("第一次搜索写入历史", n1 == 1, "%s" % n1)

        pg.js("el('btnSearch').click()")
        time.sleep(0.6)
        pg.js("el('q').value='生日蛋糕'; el('btnGo').click()")
        time.sleep(3.0)
        n2 = pg.js("qHistGet().length")
        chk("第二次搜索历史累加", n2 == 2, "%s" % n2)
        first = pg.js("qHistGet()[0]")
        chk("最新搜索在最前", first == "生日蛋糕", first)

        # 3. 重新打开面板，历史 chip 应可见
        pg.js("el('btnSearch').click()")
        time.sleep(0.9)
        vis2 = pg.js("getComputedStyle(el('qHist')).display")
        chk("有历史时历史行显示", vis2 != "none", vis2)
        chips = pg.js("document.querySelectorAll('#qHist .chip[data-q]').length")
        chk("chip 数 = 历史数 + 清除", chips == 3, "%d" % chips)
        txt = pg.js("(document.querySelector('#qHist .chip[data-q]')||{}).textContent||''")
        chk("首个 chip 是最近搜索", txt == "生日蛋糕", txt)
        pg.shot(str(SHOT / "s2_hist.png"))

        # 4. 点 chip 应回填并直接搜索
        pg.js("document.querySelector('#qHist .chip[data-q]').click()")
        time.sleep(3.5)
        qv = pg.js("el('q').value")
        chk("点 chip 回填输入框", qv == "生日蛋糕", qv)
        hd = pg.js("(document.querySelector('#main .pagehead h1')||{}).textContent||''")
        chk("点 chip 触发搜索", "搜索结果" in (hd or ""), hd)

        # 5. 历史条数上限 8
        pg.js("for(let i=0;i<12;i++) qHistAdd('填充词'+i)")
        time.sleep(0.4)
        n = pg.js("qHistGet().length")
        chk("历史上限 8 条", n == 8, "%d" % n)

        # 6. 清除按钮生效
        pg.js("el('btnSearch').click()")
        time.sleep(0.6)
        pg.js("document.querySelector('#qHist .chip[data-q=\"__clear\"]').click()")
        time.sleep(0.5)
        n3 = pg.js("qHistGet().length")
        chk("清除后历史为空", n3 == 0, "%d" % n3)
        vis3 = pg.js("getComputedStyle(el('qHist')).display")
        chk("清除后历史行隐藏", vis3 == "none", vis3)
        pg.shot(str(SHOT / "s3_cleared.png"))
    finally:
        pg.close()


def main():
    test_api()
    test_bonus()
    test_ui()
    print("\n通过 %d 项，失败 %d 项" % (len(OK), len(BAD)), flush=True)
    if BAD:
        print("失败：" + " / ".join(BAD))
        return 1
    print("搜索升级验收全部通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
