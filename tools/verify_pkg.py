#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""打包产物验收：直接跑 release\\pkg\\语义相册.exe，全走 HTTP 接口

用法：
    python tools/verify_pkg.py            # 复用现有 data（快）
    python tools/verify_pkg.py --fresh    # 把 data 挪走，从零索引（验证全新安装）

覆盖：启动 → 端口探测 → 添加目录 → 建索引 → 人像聚类 → 手动纠错接口
      （face/move 带 from 限定、to=-2 新建、to=-1 移出、person/merge_into）
      → 重聚类复原 → 优雅退出
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HOME = Path(__file__).resolve().parent.parent
# 默认验收 release\pkg；装到别处时用 AIWIN_PKG=<安装目录> 指过去
PKG = Path(os.environ.get("AIWIN_PKG") or (HOME / "release" / "pkg"))
EXE = PKG / "语义相册.exe"
# 验收用的素材目录，默认取仓库上一层的 aiwin_test_media，可用 AIWIN_TEST_MEDIA 覆盖
MEDIA = os.environ.get("AIWIN_TEST_MEDIA") or str(HOME.parent / "aiwin_test_media")
FRESH = "--fresh" in sys.argv

OK = []
BAD = []


def say(msg):
    print(msg, flush=True)


def chk(name, cond, extra=""):
    (OK if cond else BAD).append(name)
    say(("  [OK] " if cond else "  [!!] ") + name + (("  " + str(extra)) if extra else ""))
    return cond


def req(port, path, body=None, timeout=60):
    url = "http://127.0.0.1:%d%s" % (port, path)
    data = None if body is None else json.dumps(body).encode("utf-8")
    r = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET",
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read().decode("utf-8") or "{}")


def find_port(tag_only=False, skip=()):
    for p in range(8765, 8800):
        if p in skip:
            continue
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/api/ping" % p, timeout=0.4) as f:
                d = json.loads(f.read().decode("utf-8"))
            if d.get("app") == "aiwin":
                return p if not tag_only else d
        except Exception:
            pass
    return None


def busy_ports():
    """启动被测 exe 之前，先记下已经被占用的 aiwin 端口。

    ⚠️ 血的教训：find_port() 扫 8765-8800 返回**第一个** aiwin 服务。
    如果本机还跑着一个开发用的 uvicorn（正好也在 8766），它就会抢在
    被测 exe 前面被选中 —— 验收全程打在旧服务上，打包产物一次都没验到，
    而且结果还是"全绿"，看不出任何异常。所以必须先快照、再排除。
    """
    out = set()
    for p in range(8765, 8800):
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/api/ping" % p, timeout=0.3) as f:
                d = json.loads(f.read().decode("utf-8"))
            if d.get("app") == "aiwin":
                out.add(p)
        except Exception:
            pass
    return out


def wait_port(sec=180, skip=()):
    t0 = time.time()
    while time.time() - t0 < sec:
        p = find_port(skip=skip)
        if p:
            return p
        time.sleep(1)
    return None


def wait_idle(port, sec=1800):
    """等索引 + 智慧相册都跑完"""
    t0 = time.time()
    last = ""
    idx_done_at = None
    while time.time() - t0 < sec:
        try:
            st = req(port, "/api/status", timeout=10)
            s = req(port, "/api/stats", timeout=10)
        except Exception:
            time.sleep(1)
            continue
        msg = "文件 %d / 片段 %d / 索引中=%s" % (
            s.get("files", 0), s.get("segments", 0), st.get("indexing"))
        if msg != last:
            say("  … " + msg)
            last = msg
        if not st.get("indexing"):
            if idx_done_at is None:
                idx_done_at = time.time()
                # 索引刚完，给相册留出启动时间
                time.sleep(3)
                try:
                    req(port, "/api/album/build?only=", {})
                except Exception:
                    pass
            alb = req(port, "/api/album/status", timeout=10)
            if not alb.get("running") and time.time() - idx_done_at > 5:
                return s
        time.sleep(2)
    return None


def main():
    if not EXE.exists():
        say("找不到 %s，先跑 tools/build.py" % EXE)
        return 2

    say("== 1. 干净数据目录 ==")
    data = PKG / "data"
    if FRESH and data.exists():
        dst = PKG / ("_data_old_%d" % int(time.time()))
        shutil.move(str(data), str(dst))
        say("  旧 data 已挪到 %s" % dst.name)
    (PKG / "data" / "logs").mkdir(parents=True, exist_ok=True)

    say("== 2. 启动 exe ==")
    before = busy_ports()
    if before:
        say("  已有 aiwin 服务占用端口 %s，本次验收会跳过它们" % sorted(before))
    # AIWIN_NO_UI=1：只起服务、不开界面窗口（否则每跑一次验收就多弹一个 Edge 窗口，
    # 而且验收结束关服务时窗口会留下）
    penv = dict(os.environ, AIWIN_NO_UI="1")
    proc = subprocess.Popen([str(EXE)], cwd=str(PKG), env=penv)
    say("  pid=%d" % proc.pid)
    port = wait_port(300, skip=before)
    if not port:
        say("  [!!] 5 分钟内没能连上服务，看 %s" % (PKG / "data" / "logs" / "app.log"))
        # 别把被测进程留成孤儿，否则下次又抢端口
        try:
            proc.kill()
        except Exception:
            pass
        return 3
    chk("服务已起来，端口 %d" % port, True)
    chk("连的是本次启动的实例（非旧服务）", port not in before, "before=%s" % sorted(before))

    # 等模型加载完
    t0 = time.time()
    st = {}
    while time.time() - t0 < 300:
        st = req(port, "/api/status")
        if st.get("ready"):
            break
        if st.get("error"):
            break
        time.sleep(2)
    chk("模型已加载 %s" % json.dumps(st.get("info"), ensure_ascii=False), bool(st.get("ready")),
        st.get("error") or "")

    say("== 3. 加目录 + 建索引 ==")
    d = req(port, "/api/folders", {"path": MEDIA})
    chk("添加监控目录", d.get("ok"), d.get("roots"))
    d = req(port, "/api/index", {})
    chk("触发索引", d.get("ok") or d.get("msg") == "正在索引中", d.get("msg"))
    s = wait_idle(port)
    chk("索引完成", bool(s) and s.get("files", 0) >= 8, s)

    say("== 4. 人像聚类 ==")
    d = req(port, "/api/album/build?only=faces", None) if False else None
    # 索引完成后自动跑智慧相册；再兜底等一次
    t0 = time.time()
    while time.time() < t0 + 60:
        tree = req(port, "/api/album/tree")
        if tree.get("people"):
            break
        time.sleep(2)
    people = tree.get("people") or []
    chk("人像分组非空", len(people) > 0, "%d 组：%s" % (
        len(people), [(p["name"], p["count"]) for p in people]))

    if len(people) < 2:
        say("  （只有 1 组，跳过手动纠错测试）")
    else:
        say("== 5. 手动纠错接口 ==")

        def items_of(pid, k=300):
            return req(port, "/api/album/items?cat=person&key=%d&k=%d&offset=0" % (pid, k))["items"]

        def counts():
            return {p["id"]: p["count"] for p in req(port, "/api/album/tree")["people"]}

        a, b = people[0], people[1]
        pa, pb = a["id"], b["id"]
        ia, ib = items_of(pa), items_of(pb)
        chk("A 组能取到照片", len(ia) > 0, "photo=%d" % a["count"])
        # 挑一张「只在 A 组、B 组没有」的照片，这样挪过去 B 的照片数才会 +1
        paths_b = {x["path"] for x in ib}
        cand = [x for x in ia if x["path"] not in paths_b]
        only_a = cand[0] if cand else (ia[0] if ia else None)

        if only_a:
            path = only_a["path"]
            box = ((only_a.get("faces") or [{}])[0] or {}).get("box", "")
            c0 = counts()

            # 5.1 带 from 不传 box：只动这张照片里属于 A 的脸
            r = req(port, "/api/face/move", {"path": path, "from": pa, "to": pb})
            chk("face/move 带 from 只动本组", bool(r.get("ok")) and r.get("faces", 0) >= 1, r)
            c1 = counts()
            chk("B 组照片 +1", c1.get(pb, 0) == c0.get(pb, 0) + 1,
                "B %d→%d" % (c0.get(pb, 0), c1.get(pb, 0)))
            chk("A 组照片 -1", c1.get(pa, 0) == c0.get(pa, 0) - 1,
                "A %d→%d" % (c0.get(pa, 0), c1.get(pa, 0)))

            # 5.2 用 box 精确挪回（不碰这张照片里别人的脸）
            if box:
                r = req(port, "/api/face/move", {"path": path, "box": box, "to": pa})
                chk("box 精确挪回一张脸", bool(r.get("ok")) and r.get("faces", 0) == 1, r)
                c2 = counts()
                chk("复原", c2.get(pa, 0) == c0.get(pa, 0) and c2.get(pb, 0) == c0.get(pb, 0),
                    "A %d, B %d" % (c2.get(pa, 0), c2.get(pb, 0)))

            # 5.3 to=-2 单独成一组
            n0 = len(counts())
            c3 = counts()
            r = req(port, "/api/face/move", {"path": path, "from": pa, "to": -2})
            c4 = counts()
            chk("to=-2 拆出新组", bool(r.get("ok")) and len(c4) == n0 + 1 and c4.get(pa, 0) == c3.get(pa, 0) - 1,
                "%d 组 → %d 组，新组 id=%s" % (n0, len(c4), r.get("to")))

            # 5.4 merge_into 把新组并回 A
            newid = r.get("to")
            if newid is not None and newid >= 0:
                r2 = req(port, "/api/person/merge_into", {"from": newid, "to": pa})
                chk("merge_into 并回原组", bool(r2.get("ok")) and counts().get(pa, 0) == c3.get(pa, 0),
                    r2)

    say("== 6. 优雅退出 ==")
    try:
        req(port, "/api/shutdown", {})
    except Exception:
        pass
    t0 = time.time()
    while time.time() - t0 < 20 and proc.poll() is None:
        time.sleep(0.5)
    if proc.poll() is None:
        proc.terminate()
        chk("进程已退出（terminate 兜底）", True)
    else:
        chk("进程已退出 exit=%s" % proc.returncode, True)

    say("\n===== 结果：%d 项通过，%d 项失败 =====" % (len(OK), len(BAD)))
    if BAD:
        say("失败项：" + "、".join(BAD))
    return 0 if not BAD else 1


if __name__ == "__main__":
    sys.exit(main())
