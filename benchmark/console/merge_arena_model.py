#!/usr/bin/env python3
"""Merge one model's ManipArena (R6) results into console/arena_agg.json.

Reads <jsonl_root>/<model>/arena_M{1,2}/v*/*.jsonl (one mission row each, the
bench_agile/falcon arena layout), computes the by_h / chain aggregates + per-
variant rows in the arena_agg schema, and merges them in-place. Video files for
the sampled captioned clips are added from --videos-list (one "<model>/<file>"
per line) if given.

Usage:
  python3 merge_arena_model.py <model> <jsonl_root> [--videos-list FILE]
"""
import argparse
import glob
import json
import os

CONSOLE_DIR = os.path.dirname(os.path.abspath(__file__))
AGG_PATH = os.path.join(CONSOLE_DIR, "arena_agg.json")
ROW_KEYS = ("H_pick", "fall_time", "fall_phase", "success", "mission_status",
            "grasp_ok", "place_ok_store", "place_ok_relay", "cube_transfer_ok",
            "regrasp_ok", "max_tilt_rad", "nav_arrival_err_last")


def g(row, key):
    """Field from row top-level, else from row['metrics']."""
    if row.get(key) is not None:
        return row.get(key)
    return (row.get("metrics") or {}).get(key)


def load_variant_rows(root, model, test):
    rows = {}
    pat = os.path.join(root, model, test, "v*", "*.jsonl")
    for fp in glob.glob(pat):
        try:
            v = int(fp.split("/v")[-1].split("/")[0])
            line = open(fp).read().splitlines()[0]
            rows[v] = json.loads(line)
        except (ValueError, IndexError, json.JSONDecodeError):
            pass
    return rows


def slim(row):
    out = {}
    for k in ROW_KEYS:
        out[k] = g(row, k)
    return out


def hk(h):
    return ("%.2f" % h) if isinstance(h, (int, float)) else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("jsonl_root")
    ap.add_argument("--videos-list", default=None)
    args = ap.parse_args()
    model = args.model

    with open(AGG_PATH) as fp:
        agg = json.load(fp)
    agg.setdefault("models", {})
    agg.setdefault("rows", {})
    agg.setdefault("video_files", [])

    model_entry = {}

    # M1
    m1 = load_variant_rows(args.jsonl_root, model, "arena_M1")
    if m1:
        by_h = {}
        n_fall = 0
        for v, r in m1.items():
            s = slim(r)
            agg["rows"]["%s:arena_M1:%d" % (model, v)] = s
            h = hk(s["H_pick"])
            b = by_h.setdefault(h, {"n": 0, "grasp": 0, "place": 0, "done": 0, "fall": 0})
            b["n"] += 1
            b["grasp"] += bool(s["grasp_ok"])
            b["place"] += bool(s["place_ok_store"])
            b["done"] += (s["mission_status"] == "done")
            fell = s["fall_time"] is not None
            b["fall"] += fell
            n_fall += fell
        model_entry["arena_M1"] = {"by_h": by_h, "n": len(m1), "fall": n_fall}

    # M2
    m2 = load_variant_rows(args.jsonl_root, model, "arena_M2")
    if m2:
        chain = {"grasp": 0, "relay": 0, "cube": 0, "regrasp": 0, "store": 0}
        n_fall = 0
        for v, r in m2.items():
            s = slim(r)
            agg["rows"]["%s:arena_M2:%d" % (model, v)] = s
            chain["grasp"] += bool(s["grasp_ok"])
            chain["relay"] += bool(s["place_ok_relay"])
            chain["cube"] += bool(s["cube_transfer_ok"])
            chain["regrasp"] += bool(s["regrasp_ok"])
            chain["store"] += bool(s["place_ok_store"])
            fell = s["fall_time"] is not None
            n_fall += fell
        model_entry["arena_M2"] = {"chain": chain, "n": len(m2), "fall": n_fall}

    agg["models"][model] = model_entry

    if args.videos_list and os.path.exists(args.videos_list):
        existing = set(agg["video_files"])
        for line in open(args.videos_list):
            rel = line.strip()
            if rel and rel not in existing:
                agg["video_files"].append(rel)
                existing.add(rel)

    with open(AGG_PATH, "w") as fp:
        json.dump(agg, fp, ensure_ascii=False, indent=1)

    print("merged model=%s  M1=%d M2=%d  total models=%d  video_files=%d" % (
        model, len(m1), len(m2), len(agg["models"]), len(agg["video_files"])))
    if "arena_M1" in model_entry:
        print("  M1 by_h:", {k: v for k, v in model_entry["arena_M1"]["by_h"].items()})
    if "arena_M2" in model_entry:
        print("  M2 chain:", model_entry["arena_M2"]["chain"], "fall", model_entry["arena_M2"]["fall"])


if __name__ == "__main__":
    main()
