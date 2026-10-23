#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""人脸聚类单元测试（合成向量，不碰真实库）

覆盖 2026-10-21 加的两件事：
  1) 密度约束：只有一两张脸的小簇不立「人物」，进 -3（其他面孔）
  2) 锚点保护：用户手动归过/拆过的脸，重聚类不许改它的归属

用法：python tools/t_face_cluster.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np

HOME = tempfile.mkdtemp(prefix="aiwin_facetest_")
os.environ["AIWIN_HOME"] = HOME
os.environ["FACE_SMALL_LIB"] = "0"      # 强制密度约束生效（真实默认 12）
os.environ["FACE_MIN_FACES"] = "2"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import server as S  # noqa: E402

OK, BAD = [], []


def chk(name, cond, extra=""):
    (OK if cond else BAD).append(name)
    print(("  [OK] " if cond else "  [!!] ") + name + (("  " + str(extra)) if extra else ""),
          flush=True)
    return cond


rng = np.random.default_rng(3)
DIM = 512


def unit(v):
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


def base():
    return unit(rng.normal(size=DIM))


def near(a, k=0.6):
    return unit(a + k * rng.normal(size=DIM) / np.sqrt(DIM))


def add_faces(spec):
    """spec: [(path, vec, q)]"""
    with S.db() as c:
        for path, v, q in spec:
            c.execute("INSERT INTO faces(path,box,score,q,person,vec)"
                      " VALUES(?,?,?,?,-1,?)",
                      (path, "0,0,10,10", q, q, v.astype(np.float32).tobytes()))


def people():
    with S.db() as c:
        return {r[0]: r[1] for r in c.execute(
            "SELECT p.id, (SELECT count(*) FROM faces f WHERE f.person=p.id)"
            " FROM people p")}


def person_of(path):
    with S.db() as c:
        return c.execute("SELECT person FROM faces WHERE path=?", (path,)).fetchone()[0]


def loose():
    with S.db() as c:
        return c.execute("SELECT count(*) FROM faces WHERE person=-3").fetchone()[0]


def move(path, to):
    with S.db() as c:
        if to == -2:
            to = S._new_person(c)
        c.execute("UPDATE faces SET person=?, manual=1 WHERE path=?", (to, path))
    return to


def main():
    A, B, C, D = base(), base(), base(), base()
    add_faces([("a0.jpg", near(A), 0.9), ("a1.jpg", near(A), 0.88),
               ("a2.jpg", near(A), 0.86), ("b0.jpg", near(B), 0.85),
               ("c0.jpg", near(C), 0.8), ("d0.jpg", near(D), 0.75)])

    print("== 1. 密度约束 ==")
    n = S.cluster_faces()
    ppl = people()
    chk("只建出 1 个人物（3 张脸的 A）", n == 1 and len(ppl) == 1, "%s" % ppl)
    chk("另外 3 张脸进「其他面孔」", loose() == 3, "loose=%d" % loose())

    print("== 2. 手工认领：B 归到 A 那个人物 ==")
    pid_a = list(ppl)[0]
    move("b0.jpg", pid_a)
    S.cluster_faces()
    ppl2 = people()
    pb = person_of("b0.jpg")
    chk("B 仍在 A 那个人物里（没有被分出去）",
        pb == person_of("a0.jpg"), "b=%s a0=%s 组=%s" % (pb, person_of("a0.jpg"), ppl2))

    print("== 3. 手工拆分：把 A 里一张脸拆成独立一组 ==")
    move("a1.jpg", -2)
    S.cluster_faces()
    pa1, pa0 = person_of("a1.jpg"), person_of("a0.jpg")
    chk("拆出去的 A1 重聚类后仍然独立", pa1 != pa0, "a1=%s a0=%s" % (pa1, pa0))
    chk("原组剩下 A0（可能含 B）", people().get(pa0, 0) >= 1, people())

    print("== 4. 重扫同名认领（hints 不影响结构）==")
    before = people()
    S.cluster_faces(hints={("a0.jpg", "0,0,10,10"): "老王"})
    with S.db() as c:
        nm = c.execute("SELECT name FROM people WHERE id=?",
                       (person_of("a0.jpg"),)).fetchone()[0]
    chk("hints 起到名字", nm == "老王", nm)
    chk("分组结构没变", people().keys() == before.keys(), "%s vs %s" % (people(), before))

    print("\n===== 人脸聚类：%d 项通过，%d 项失败 =====" % (len(OK), len(BAD)))
    if BAD:
        print("失败项：" + "、".join(BAD))
    return 0 if not BAD else 1


if __name__ == "__main__":
    rc = main()
    shutil.rmtree(HOME, ignore_errors=True)
    sys.stdout.flush()
    # import server 会把 onnxruntime 拖进来，它的线程池在 Windows 上不肯退出，
    # 正常 sys.exit 会卡住；这里直接结束进程（清理已经做完了）。
    os._exit(rc)
