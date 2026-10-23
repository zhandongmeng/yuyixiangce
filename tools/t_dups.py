#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重复照片（dHash + LSH）单元测试

不依赖数据库和模型，只验证算法本身：
  ① 分段必须覆盖完整的 64 位（漏一位就等于丢掉一半判据）
  ② 汉明距离 <= 阈值时，LSH 粗筛必须至少命中一段（否则真重复会被漏掉）
  ③ dHash 对同一张图稳定、对轻微压缩不敏感、对不同图有区分度
  ④ 有符号 64 位转换可逆（SQLite 的 INTEGER 存不下无符号）

用法：python tools/t_dups.py
"""
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import server  # noqa: E402

OK, BAD = [], []


def chk(name, cond, extra=""):
    (OK if cond else BAD).append(name)
    print(("  [OK] " if cond else "  [!!] ") + name + (("  " + str(extra)) if extra else ""),
          flush=True)
    return cond


def test_segments():
    print("== 1. 分段覆盖 ==")
    for nseg in range(2, 9):
        segs = server._lsh_segments(nseg)
        # 逐位检查每个位恰好被一段覆盖
        covered = [0] * 64
        for sh, msk in segs:
            w = bin(msk).count("1")
            for b in range(sh, sh + w):
                covered[b] += 1
        chk("%d 段覆盖全部 64 位且不重叠" % nseg, all(c == 1 for c in covered),
            "缺位=%s" % [i for i, c in enumerate(covered) if c != 1][:5])


def test_recall():
    print("== 2. LSH 召回（关键：以前 4 段配阈值 6 会漏 41.9%）==")
    random.seed(11)
    for th in (2, 3, 4, 5, 6, 7):
        nseg = max(2, min(8, th + 1))
        segs = server._lsh_segments(nseg)
        miss = 0
        N = 3000
        for _ in range(N):
            a = random.getrandbits(64)
            b = a
            for bit in random.sample(range(64), th):
                b ^= (1 << bit)
            ka = [(a >> sh) & msk for sh, msk in segs]
            kb = [(b >> sh) & msk for sh, msk in segs]
            if not any(x == y for x, y in zip(ka, kb)):
                miss += 1
        chk("阈值 %d（%d 段）零漏检" % (th, nseg), miss == 0, "漏 %d/%d" % (miss, N))
    chk("默认阈值 6 用的是 7 段",
        max(2, min(8, server.PHASH_TH + 1)) == 7, server.PHASH_TH)


def test_dhash():
    print("== 3. dHash 性质 ==")
    from PIL import Image
    base = Image.new("RGB", (320, 240))
    px = base.load()
    for y in range(240):
        for x in range(320):
            px[x, y] = ((x * 3) % 256, (y * 5) % 256, ((x + y) * 2) % 256)

    h1 = server.dhash64(base)
    h2 = server.dhash64(base.copy())
    chk("同一张图哈希稳定", h1 == h2 and h1 is not None, h1)

    # 轻微压缩/缩放后仍应很接近（模拟"同一张图被存了两份"）
    import io
    buf = io.BytesIO()
    base.save(buf, "JPEG", quality=92)
    buf.seek(0)
    with Image.open(buf) as im0:
        h3 = server.dhash64(im0.convert("RGB"))
    d = bin((h1 & ((1 << 64) - 1)) ^ (h3 & ((1 << 64) - 1))).count("1")
    chk("JPEG 有损压缩后距离很小（<= %d）" % server.PHASH_TH, d <= server.PHASH_TH, "距离 %d" % d)

    # 完全不同的图距离应该明显更大
    other = Image.new("RGB", (320, 240), (10, 10, 10))
    h4 = server.dhash64(other)
    d2 = bin((h1 & ((1 << 64) - 1)) ^ (h4 & ((1 << 64) - 1))).count("1")
    chk("不同内容距离更大（> %d）" % server.PHASH_TH, d2 > server.PHASH_TH, "距离 %d" % d2)

    # 有符号转换必须可逆
    v = -1234567890123456789
    back = int(v) & ((1 << 64) - 1)
    chk("有符号 64 位可逆", (back - (1 << 64) if back >= (1 << 63) else back) == v)
    chk("哈希落在有符号范围内",
        -2 ** 63 <= h1 < 2 ** 63, h1)


def main():
    test_segments()
    test_recall()
    test_dhash()
    print("\n通过 %d 项，失败 %d 项" % (len(OK), len(BAD)))
    if BAD:
        print("失败：" + " / ".join(BAD))
        return 1
    print("重复照片算法验收全部通过 ✓")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # onnxruntime 的线程池在 Windows 上不肯自己退，直接杀掉进程
    os._exit(rc)
