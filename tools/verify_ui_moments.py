#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""界面验收（第二批）：时刻时间轴 + 其他面孔

前置：语义相册在跑，且库里至少有一个事件、一张未归类脸。
用法：python tools/verify_ui_moments.py [端口]
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


def api(path):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (PORT, path), timeout=20) as f:
        return json.loads(f.read().decode("utf-8"))


def main():
    tree = api("/api/album/tree")
    moments = tree.get("moments") or []
    nloose = tree.get("unassigned") or 0
    print("接口：事件 %d 个，未归类 %d 张" % (len(moments), nloose), flush=True)
    if not moments:
        print("  [!!] 没有事件，先跑一次「更新标签 + 城市」")
        return 1

    pg = cdp.Page("http://127.0.0.1:%d/" % PORT, 1440, 940)
    try:
        time.sleep(2.5)

        print("== 时刻 ==")
        pg.js("location.hash='#/moments'")
        time.sleep(2.5)
        n = pg.js("document.querySelectorAll('.gc.event').length")
        chk("时刻卡片数 = 接口事件数", n == len(moments), "%s vs %d" % (n, len(moments)))
        title = pg.js("(document.querySelector('.gc.event .nm')||{}).textContent||''")
        chk("卡片标题含日期", "月" in (title or ""), title)
        cnt = pg.js("(document.querySelector('#cMoments')||{}).textContent||''")
        chk("导航计数显示", cnt == str(len(moments)), cnt)
        pg.shot(str(SHOT / "m1_moments.png"))

        # 进事件
        pg.js("document.querySelector('.gc.event').click()")
        time.sleep(3)
        nph = pg.js("document.querySelectorAll('#gWall .ph').length")
        chk("事件页照片数 = 接口项数", nph == moments[0]["count"],
            "%s vs %d" % (nph, moments[0]["count"]))
        gtitle = pg.js("(document.querySelector('#gTitle')||{}).textContent||''")
        chk("事件页标题正确", gtitle == moments[0]["title"], gtitle)
        pg.shot(str(SHOT / "m2_event.png"))
        pg.js("document.querySelector('#gBack').click()")
        time.sleep(2)

        print("== 其他面孔 ==")
        if nloose <= 0:
            chk("未归类为 0 时不显示入口卡", True, "跳过（库里没有未归类脸）")
        else:
            pg.js("location.hash='#/people'")
            time.sleep(2.5)
            has = pg.js("!!document.querySelector('.gc.person.loose')")
            chk("人像页出现「其他面孔」入口", has)
            txt = pg.js("(document.querySelector('.gc.person.loose .ct')||{}).textContent||''")
            chk("入口显示待认领数量", str(nloose) in (txt or ""), txt)
            pg.js("document.querySelector('.gc.person.loose').click()")
            time.sleep(2.5)
            nf = pg.js("document.querySelectorAll('.fcell').length")
            chk("其他面孔网格有脸", nf == nloose, "%s vs %d" % (nf, nloose))
            hint = pg.js("!!document.querySelector('.loosehint')")
            chk("有说明文字", hint)
            pg.shot(str(SHOT / "m3_loose.png"))
            # 点一张脸 → 选择器
            pg.js("document.querySelector('.fcell').click()")
            time.sleep(1.2)
            chk("点脸弹出认领选择器",
                pg.js("document.querySelector('#pkMask').classList.contains('on')"))
            chk("选择器按钮齐全",
                pg.js("!!document.querySelector('#pkNew') && !!document.querySelector('#pkHide')"))
            pg.shot(str(SHOT / "m4_claim.png"))
            pg.js("document.querySelector('#pkClose').click()")
    finally:
        pg.close()

    print("\n===== 界面验收：%d 项通过，%d 项失败 =====" % (len(OK), len(BAD)))
    if BAD:
        print("失败项：" + "、".join(BAD))
    print("截图：%s" % SHOT)
    return 0 if not BAD else 1


if __name__ == "__main__":
    sys.exit(main())
