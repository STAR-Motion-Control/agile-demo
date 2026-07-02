#!/usr/bin/env python3
"""Aggregate R3 final-queue results across models (defensive about metric key aliases)."""
import json, glob, os, statistics as st, collections, sys

ALIAS = {
    "mean_vx": ["mean_vx_last8s"],
    "radial": ["radial_err_mean", "radial_err_mean_m"],
    "pick": ["pick_success"],
    "box_kept": ["box_kept"],
    "h_target": ["target_height"],
    "rate": ["ramp_speed"],
    "depth": ["achieved_depth"],
    "drift_h": ["root_drift_hold"],
    "drift_t": ["root_drift_total"],
    "e_pos_nav": ["err_pos_nav"],
    "e_pos_cal": ["err_pos_cal"],
    "e_yaw_cal": ["err_yaw_cal"],
    "fine": ["success_fine"],
    "coarse": ["success_coarse"],
    "place": ["box_place_ok"],
    "t_cal": ["t_cal"],
    "land": ["box_land_err", "box_land_dist_from_c", "box_land_dist"],
}

def g(m, key, default=None):
    for k in ALIAS.get(key, [key]):
        if k in m and m[k] is not None:
            return m[k]
    return default

def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None

def fmt(x, p=3):
    return ("%."+str(p)+"f") % x if x is not None else "-"

def load(path):
    return [json.loads(l) for l in open(path)]

def main(root):
    for model in sorted(os.listdir(root)):
        mdir = os.path.join(root, model)
        if not os.path.isdir(mdir):
            continue
        print("#### %s" % model.upper())
        for test in ["walk_speed_psi0", "squat_box_psi0", "circle_pillar_psi0",
                     "squat_sweep", "goto_ab", "pipeline_abc"]:
            p = os.path.join(mdir, test + ".jsonl")
            if not os.path.exists(p):
                print("  %s: MISSING" % test); continue
            d = load(p)
            n = len(d)
            succ = sum(1 for x in d if x.get("success"))
            falls = sum(1 for x in d if x.get("fall_time") is not None)
            herr = sum(1 for x in d if x.get("harness_error"))
            M = [x.get("metrics", {}) for x in d]
            line = "  %-20s %2d/%2d falls=%d herr=%d" % (test, succ, n, falls, herr)
            if test == "walk_speed_psi0":
                line += "  vx=%s box_kept=%d" % (fmt(mean([g(m,"mean_vx") for m in M])), sum(1 for m in M if g(m,"box_kept")))
            elif test == "squat_box_psi0":
                line += "  pick=%d" % sum(1 for m in M if g(m,"pick"))
            elif test == "circle_pillar_psi0":
                line += "  radial=%s" % fmt(mean([g(m,"radial") for m in M]))
            elif test == "goto_ab":
                co = sum(1 for m in M if g(m,"coarse")); fi = sum(1 for m in M if g(m,"fine"))
                line += "  coarse=%d fine=%d e_nav=%s e_cal=%sm/%sdeg t_cal=%s" % (
                    co, fi, fmt(mean([g(m,"e_pos_nav") for m in M])),
                    fmt(mean([g(m,"e_pos_cal") for m in M])),
                    fmt((mean([g(m,"e_yaw_cal") for m in M]) or 0)*57.3, 1),
                    fmt(mean([g(m,"t_cal") for m in M]), 1))
            elif test == "pipeline_abc":
                co = sum(1 for m in M if g(m,"coarse")); fi = sum(1 for m in M if g(m,"fine"))
                pl = sum(1 for m in M if g(m,"place"))
                line += "  coarse=%d fine=%d place_ok=%d land_err=%s" % (
                    co, fi, pl, fmt(mean([g(m,"land") for m in M])))
            print(line)
            if test == "squat_sweep":
                byh = collections.defaultdict(list)
                bys = collections.defaultdict(list)
                for x, m in zip(d, M):
                    h, r = g(m,"h_target"), g(m,"rate")
                    rec = (x.get("fall_time") is not None, g(m,"depth"), g(m,"drift_h"), g(m,"drift_t"))
                    if r == 0.2 and h is not None: byh[h].append(rec)
                    if h == 0.45 and r is not None: bys[r].append(rec)
                print("      depth grid (rate=0.2): H -> falls | achieved | drift_hold | drift_total")
                for h in sorted(byh, reverse=True):
                    v = byh[h]
                    print("      H=%.2f: %d/%d | %s | %s | %s" % (h, sum(1 for r in v if r[0]), len(v),
                          fmt(mean([r[1] for r in v])), fmt(mean([r[2] for r in v])), fmt(mean([r[3] for r in v]))))
                print("      speed grid (H=0.45): rate -> falls | drift_hold")
                for r in sorted(bys):
                    v = bys[r]
                    print("      rate=%.1f: %d/%d | %s" % (r, sum(1 for q in v if q[0]), len(v), fmt(mean([q[2] for q in v]))))
        print()

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "final")
