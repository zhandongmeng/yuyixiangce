#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""界面验收：headless Edge + CDP 走一遍「人像分组 → 不是他 → 纠正」全流程

前置：语义相册已在跑（开发模式 python -m uvicorn 或 release\\pkg\\语义相册.exe）
用法：python tools/verify_ui.py [端口] [截图目录]
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cdp  # noqa: E402

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
SHOT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "_shots"
SHOT.mkdir(parents=True, exist_ok=True)

OK, BAD = [], []


def chk(name, cond, extra=""):
    (OK if cond else BAD).append(name)
    print(("  [OK] " if cond else "  [!!] ") + name + (("  " + str(extra)) if extra else ""), flush=True)
    return cond


def api(path):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (PORT, path), timeout=20) as f:
        return json.loads(f.read().decode("utf-8"))


def main():
    tree = api("/api/album/tree")
    people = tree.get("people") or []
    print("接口层人像组数：%d %s" % (len(people), [(p["name"], p["count"]) for p in people]), flush=True)
    if len(people) < 2:
        print("  [!!] 人像组少于 2，界面纠错流程无法完整验证")
        return 1

    pg = cdp.Page("http://127.0.0.1:%d/" % PORT, 1440, 940)
    try:
        time.sleep(2.5)
        chk("页面已加载", pg.js("!!document.querySelector('#main')"))

        # 1. 进人像分类
        pg.js("location.hash='#/people'")
        time.sleep(2.5)
        n = pg.js("document.querySelectorAll('.gc.person').length")
        chk("人像卡片数量与接口一致", n == len(people), "%s vs %d" % (n, len(people)))
        nm = pg.js("document.querySelectorAll('.gc.person [data-act=\"merge\"]').length")
        chk("每张人物卡都有「合并」", nm == n, nm)
        pg.shot(str(SHOT / "ui1_people.png"))

        # 2. 点进第一个分组
        pg.js("document.querySelector('.gc.person').click()")
        time.sleep(3)
        nf = pg.js("document.querySelectorAll('.ph .fix').length")
        chk("分组页有人像照片", (pg.js("document.querySelectorAll('.ph').length") or 0) > 0)
        chk("每张照片都有「✕ 不是他」", nf > 0, nf)
        op = pg.js("getComputedStyle(document.querySelector('.ph .fix')).opacity")
        chk("「不是他」常显（不靠 hover）", float(op or 0) >= 0.6, "opacity=%s" % op)
        pg.shot(str(SHOT / "ui2_group.png"))

        # 3. 点「不是他」→ 选择器
        pg.js("document.querySelector('.ph .fix').click()")
        time.sleep(1.2)
        chk("选择器弹出", pg.js("document.querySelector('#pkMask').classList.contains('on')"))
        title = pg.js("document.querySelector('#pkTitle').textContent")
        chk("标题是「这张脸其实是谁？」", "这张脸其实是谁" in (title or ""), title)
        cand = pg.js("document.querySelectorAll('#pkList .pkitem').length")
        chk("候选人物 = 其它组数", cand == len(people) - 1, "%s vs %d" % (cand, len(people) - 1))
        chk("底部有「单独新建」", pg.js("!!document.querySelector('#pkNew')"))
        chk("底部有「这些不是人」", pg.js("!!document.querySelector('#pkHide')"))
        pg.shot(str(SHOT / "ui3_picker.png"))

        # 4. 单独新建 → 出现一个新组
        ids_before = {p["id"] for p in api("/api/album/tree")["people"]}
        pg.js("document.querySelector('#pkNew').click()")
        time.sleep(0.8)                     # toast 只活 2.2s，得赶在它消失前读
        toast = pg.js("(document.querySelector('#toast')||{}).textContent||''")
        chk("toast 提示已拆组", "单独一组" in (toast or ""), toast)
        time.sleep(2.5)
        after = api("/api/album/tree")
        ids_after = {p["id"] for p in after["people"]}
        new_ids = sorted(ids_after - ids_before)
        # 原组只有这一张照片时会被自动删掉，总数可能不变，所以看「有没有新 id」
        chk("拆出来的新组已建立", len(new_ids) >= 1,
            "新组 id=%s，共 %d 组" % (new_ids, len(after["people"])))
        pg.shot(str(SHOT / "ui4_after.png"))

        # 5. 复原：把新组并回原组
        # ⚠️ 不能直接用开头缓存下来的 people[0]["id"]：如果原组只有这一张脸，
        # 拆出去之后原组会被自动删掉（这是产品预期行为），merge_into 就会 404。
        # 所以并之前重新取一次，挑一个「还活着、且不是刚拆出来的」组当目标。
        if new_ids:
            cur = api("/api/album/tree")
            alive = [p["id"] for p in cur["people"] if p["id"] not in new_ids]
            if not alive:
                chk("并回原组", False, "原组已被拆空删除，没有可并回去的目标")
            else:
                req = urllib.request.Request(
                    "http://127.0.0.1:%d/api/person/merge_into" % PORT,
                    data=json.dumps({"from": new_ids[0], "to": alive[0]}).encode(),
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=20) as f:
                    r = json.loads(f.read().decode("utf-8"))
                chk("并回原组", r.get("ok"), r)
    finally:
        pg.close()

    print("\n===== 界面验收：%d 项通过，%d 项失败 =====" % (len(OK), len(BAD)))
    if BAD:
        print("失败项：" + "、".join(BAD))
    print("截图：%s" % SHOT)
    return 0 if not BAD else 1


if __name__ == "__main__":
    sys.exit(main())
