#!/usr/bin/env python3
"""Pick a balanced video subset for the agile-deepsquat console entry.

R2/R3: from a list of available captioned filenames (stdin or --avail file),
keep up to N ok + N fail per test -> prints chosen basenames (one per line).
Arena: from local arena jsonl, pick a stratified set of variants (per H tier
for M1, spread of deepest-stage for M2) and print "agile-deepsquat/<file>"
video_files lines for merge_arena_model.py --videos-list.

Usage:
  python3 select_ds_videos.py r2r3 --avail avail.txt [--per 3]
  python3 select_ds_videos.py arena --arena-root <results/arena> [--m1 3 --m2 8]
"""
import argparse
import collections
import glob
import json
import os
import re

MODEL = "agile-deepsquat"
VRE = re.compile(r"^%s_(.+)_t(\d+)_(ok|fail)_cap\.mp4$" % re.escape(MODEL))


def pick_r2r3(avail, per):
    by_test = collections.defaultdict(lambda: {"ok": [], "fail": []})
    for name in avail:
        name = name.strip()
        mm = VRE.match(name)
        if not mm:
            continue
        by_test[mm.group(1)][mm.group(3)].append((int(mm.group(2)), name))
    chosen = []
    for test in sorted(by_test):
        for tag in ("ok", "fail"):
            items = sorted(by_test[test][tag])[:per]
            chosen += [n for _, n in items]
    return chosen


def _g(row, key):
    if row.get(key) is not None:
        return row.get(key)
    return (row.get("metrics") or {}).get(key)


def load_arena(root, test):
    out = {}
    for fp in glob.glob(os.path.join(root, MODEL, test, "v*", "*.jsonl")):
        try:
            v = int(fp.split("/v")[-1].split("/")[0])
            out[v] = json.loads(open(fp).read().splitlines()[0])
        except (ValueError, IndexError, json.JSONDecodeError):
            pass
    return out


def vid_name(test, v, row):
    tag = "ok" if row.get("success") else "fail"
    return "%s/%s_%s_v%02d_t00_%s_cap.mp4" % (MODEL, MODEL, test, v, tag)


def pick_arena(root, m1n, m2n):
    out = []
    m1 = load_arena(root, "arena_M1")
    by_h = collections.defaultdict(list)
    for v, r in m1.items():
        by_h["%.2f" % _g(r, "H_pick")].append(v)
    for h in sorted(by_h):
        vs = sorted(by_h[h])
        # prefer a mix: take some grasped + some failed
        ok = [v for v in vs if m1[v].get("success")]
        bad = [v for v in vs if not m1[v].get("success")]
        sel = (ok[:max(1, m1n // 2)] + bad[:m1n - max(1, m1n // 2)])[:m1n]
        for v in sel:
            out.append(vid_name("arena_M1", v, m1[v]))
    m2 = load_arena(root, "arena_M2")
    # spread by deepest reached stage
    def depth(r):
        for i, k in enumerate(("place_ok_store", "regrasp_ok", "cube_transfer_ok",
                               "place_ok_relay", "grasp_ok")):
            if _g(r, k):
                return 5 - i
        return 0
    ranked = sorted(m2.items(), key=lambda kv: (-depth(kv[1]), kv[0]))
    for v, r in ranked[:m2n]:
        out.append(vid_name("arena_M2", v, r))
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_argument
    ap.add_argument("mode", choices=["r2r3", "arena"])
    ap.add_argument("--avail")
    ap.add_argument("--per", type=int, default=3)
    ap.add_argument("--arena-root")
    ap.add_argument("--m1", type=int, default=3)
    ap.add_argument("--m2", type=int, default=8)
    ap.add_argument("--model", default="agile-deepsquat")
    a = ap.parse_args()
    global MODEL, VRE
    MODEL = a.model
    VRE = re.compile(r"^%s_(.+)_t(\d+)_(ok|fail)_cap\.mp4$" % re.escape(MODEL))
    if a.mode == "r2r3":
        avail = open(a.avail).read().splitlines() if a.avail else []
        print("\n".join(pick_r2r3(avail, a.per)))
    else:
        print("\n".join(pick_arena(a.arena_root, a.m1, a.m2)))


if __name__ == "__main__":
    main()
