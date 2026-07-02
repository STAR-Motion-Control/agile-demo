#!/usr/bin/env python3
"""Aggregate ManipArena M1/M2 sweep results by model + table-height tier.

Reads <root>/<model>/<test>/v*/*.jsonl (one mission row each). Uses the real
metric keys: mission_status, grasp_ok, place_ok_store/relay, cube_transfer_ok,
regrasp_ok, n_stages. Top-level `success` is gated on strict 5cm/5deg nav-cal
convergence (a known per-model locomotion limit), so we report stage flags
instead as the meaningful task signal.

Usage: python aggregate_arena.py <root_dir> [--md]
"""
import json, glob, os, sys, collections

def load(root, model, test):
    rows = {}
    for f in glob.glob(os.path.join(root, model, test, "v*", "*.jsonl")):
        try:
            v = int(f.split("/v")[-1].split("/")[0])
            rows[v] = json.loads(open(f).read().splitlines()[0])
        except Exception:
            pass
    return rows

def m1_by_height(rows):
    byh = collections.defaultdict(collections.Counter)
    for x in rows.values():
        m = x.get("metrics", {})
        c = byh[m.get("H_pick")]
        c["n"] += 1
        c["grasp"] += bool(m.get("grasp_ok"))
        c["place"] += bool(m.get("place_ok_store"))
        c["done"] += (m.get("mission_status") == "done")
        c["fall"] += (x.get("fall_time") is not None)
    return byh

def m2_chain(rows):
    cc = collections.Counter()
    for x in rows.values():
        m = x.get("metrics", {})
        cc["n"] += 1
        for k, mk in [("grasp", "grasp_ok"), ("relay", "place_ok_relay"),
                      ("cube", "cube_transfer_ok"), ("regrasp", "regrasp_ok"),
                      ("store", "place_ok_store")]:
            cc[k] += bool(m.get(mk))
        cc["fall"] += (x.get("fall_time") is not None)
    return cc

def main(root, md=False):
    models = sorted(d for d in os.listdir(root)
                    if os.path.isdir(os.path.join(root, d)))
    for model in models:
        print("\n## %s" % model.upper())
        r1 = load(root, model, "arena_M1")
        if r1:
            print(" M1 (n=%d) by table height — grasp / place_store / done / fall:" % len(r1))
            for h in sorted(k for k in m1_by_height(r1) if k is not None):
                c = m1_by_height(r1)[h]
                print("   H=%.2f: n=%2d grasp=%2d place=%2d done=%2d fall=%2d"
                      % (h, c["n"], c["grasp"], c["place"], c["done"], c["fall"]))
        r2 = load(root, model, "arena_M2")
        if r2:
            c = m2_chain(r2)
            print(" M2 relay chain (n=%d) — stage completion counts:" % c["n"])
            print("   grasp=%d -> relay_place=%d -> cube_xfer=%d -> regrasp=%d -> store_place=%d  (fall=%d)"
                  % (c["grasp"], c["relay"], c["cube"], c["regrasp"], c["store"], c["fall"]))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
