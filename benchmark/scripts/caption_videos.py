#!/usr/bin/env python3
"""Burn experiment-setting banners into benchmark videos (R2+ visualization).

Usage:
  python caption_videos.py --pairs DIR1:JSONL1 DIR2:JSONL2 ... --out OUT_DIR \
      [--max-ok 2] [--max-fail 2] [--all]

Matches <anything>_tNN_{ok,fail}.mp4 in each DIR to the JSONL row with
trial == NN, composes a caption from the row (model/test/settings/result)
and re-encodes with a top banner. Deterministic videos => captions accurate.
"""
import argparse, glob, json, os, re, sys

import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont


def font(size):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


F1, F2 = font(17), font(14)


def fmt(v, p=3):
    if isinstance(v, bool):
        return "Y" if v else "N"
    if isinstance(v, float):
        return ("%."+str(p)+"f") % v
    return str(v)


def caption_lines(row):
    m = row.get("metrics", {})
    fw = row.get("framework", "?").upper()
    test = row.get("test", "?")
    trial = row.get("trial")
    if row.get("fall_time") is not None:
        outcome = "FALL@%s t=%.1fs" % (row.get("fall_phase") or "?", row["fall_time"])
    elif row.get("harness_error"):
        outcome = "HARNESS_ERR"
    else:
        outcome = "SUCCESS" if row.get("success") else "FAIL(criteria)"
    l1 = "%s | %s | trial %02d | %s" % (fw, test, trial, outcome)

    bits = []
    if "squat_sweep" in test:
        bits += ["squat_H=%sm" % fmt(m.get("target_height"), 2),
                 "ramp=%sm/s" % fmt(m.get("ramp_speed"), 1),
                 "achieved=%sm" % fmt(m.get("achieved_depth")),
                 "drift_hold=%sm" % fmt(m.get("root_drift_hold"))]
    elif "squat_limit" in test:
        bits += ["descent=%sm/s" % fmt(m.get("descent_rate", 0.05), 2)]
        if row.get("fall_time") is not None or m.get("fall_cmd_height") is not None:
            bits.append("fall@cmd=%sm" % fmt(m.get("fall_cmd_height"), 2))
        else:
            bits.append("floor=%sm" % fmt(m.get("floor_height",
                                                m.get("achieved_floor")), 2))
        d5 = m.get("drift5cm_cmd_height", m.get("drift5cm_height"))
        if d5 is not None:
            bits.append("drift5cm@%sm" % fmt(d5, 2))
    elif "squat_place_psi0" in test:
        bits += ["ep053 reach-fwd place (Psi0)",
                 "h_rmse=%sm" % fmt(m.get("height_rmse", m.get("height_rmse_m"))),
                 "drift_place=%sm" % fmt(m.get("drift_place",
                                               m.get("root_drift_place"))),
                 "land_dx=%sm" % fmt(m.get("land_dx", m.get("box_land_dx")))]
    elif "squat_box" in test:
        bits += ["squat 0.75->0.45m @0.2m/s x20", "box 2kg wrist-weld",
                 "cycles=%s" % fmt(m.get("cycles_completed")),
                 "h_rmse=%sm" % fmt(m.get("height_rmse", m.get("height_rmse_m")))]
        if "pick_success" in m:
            bits.append("table-pick=%s" % fmt(m.get("pick_success")))
    elif "squat_track" in test:
        bits += ["tracking qpos squat",
                 "variant=%s" % m.get("variant"),
                 "target_z=%sm" % fmt(m.get("target_min_base_z"), 2),
                 "min_z=%sm" % fmt(m.get("min_base_z"), 2),
                 "z_rmse=%sm" % fmt(m.get("root_z_rmse"))]
        if m.get("root_xy_drift_max") is not None:
            bits.append("xy_drift=%sm" % fmt(m.get("root_xy_drift_max"), 2))
    elif "speed_sweep" in test:
        bits += ["cmd_vx=%sm/s" % fmt(m.get("cmd_vx", m.get("vx_target", m.get("target_vx"))), 1),
                 "actual=%sm/s" % fmt(m.get("mean_vx_last8s"))]
    elif "walk_speed" in test:
        cmd_vx = m.get("cmd_vx", m.get("vx_target", m.get("target_vx", 1.0)))
        bits += ["cmd vx=%sm/s (2s ramp,10s hold)" % fmt(cmd_vx, 1),
                 "actual=%sm/s" % fmt(m.get("mean_vx_last8s"))]
    elif "circle" in test:
        c_vx = m.get("circle_vx", 0.4)
        c_wz = m.get("circle_wz", 0.4)
        bits += ["vx=%s wz=%s%s r=%sm" % (
                     fmt(c_vx, 1), "-" if m.get("direction") == "cw" else "+",
                     fmt(c_wz, 1), fmt(c_vx / c_wz, 1) if c_wz else "?"),
                 "dir=%s" % m.get("direction"),
                 "radial_err=%sm" % fmt(m.get("radial_err_mean", m.get("radial_err_mean_m")))]
        col = m.get("pillar_collision", m.get("collision"))
        if col is not None:
            bits.append("hit_pillar=%s" % fmt(col))
    elif test == "goto_ab":
        bits += ["A->B=(3,1,+90deg)",
                 "nav_err=%sm" % fmt(m.get("err_pos_nav")),
                 "cal_err=%sm/%sdeg" % (fmt(m.get("err_pos_cal")),
                                        fmt((m.get("err_yaw_cal") or 0) * 57.3, 1)),
                 "fine(5cm/5deg)=%s" % fmt(m.get("success_fine"))]
    elif test == "pipeline_abc":
        bits += ["C=(0,-2.5,-90deg) carry 2kg",
                 "cal_err=%sm" % fmt(m.get("err_pos_cal")),
                 "squat 0.45 place",
                 "place_ok=%s" % fmt(m.get("box_place_ok")),
                 "land=%sm" % fmt(m.get("box_land_dist_from_c", m.get("box_land_err")))]
    elif "arena_M1" in test:
        h_pick = m.get("H_pick", row.get("H_pick"))
        bits += ["table H=%sm" % fmt(h_pick, 2),
                 "grasp=%s" % fmt(m.get("grasp_ok")),
                 "place_store=%s" % fmt(m.get("place_ok_store")),
                 "mission=%s" % fmt(m.get("mission_status"))]
        if row.get("fall_time") is not None:
            bits.append("fall@%s" % (row.get("fall_phase") or "?"))
    elif "arena_M2" in test:
        h_pick = m.get("H_pick", row.get("H_pick"))
        # relay chain: grasp -> relay-place -> cube-transfer -> regrasp -> store-place
        reached = "none"
        if fmt(m.get("grasp_ok")) == "Y":
            reached = "grasp✓"
        if fmt(m.get("place_ok_relay")) == "Y":
            reached = "relay✓"
        if fmt(m.get("cube_transfer_ok")) == "Y":
            reached = "cube_xfer✓"
        if fmt(m.get("regrasp_ok")) == "Y":
            reached = "regrasp✓"
        if fmt(m.get("place_ok_store")) == "Y":
            reached = "store✓"
        # show the first failed link after the deepest reached one
        chain = [("grasp", m.get("grasp_ok")), ("relay", m.get("place_ok_relay")),
                 ("cube_xfer", m.get("cube_transfer_ok")),
                 ("regrasp", m.get("regrasp_ok")), ("store", m.get("place_ok_store"))]
        failed = next((nm for nm, v in chain if fmt(v) == "N"), None)
        stage_txt = "stage: %s" % reached
        if failed:
            stage_txt += " %s✗" % failed
        bits += [stage_txt, "H=%sm" % fmt(h_pick, 2)]
        if row.get("fall_time") is not None:
            bits.append("fall@%s" % (row.get("fall_phase") or "?"))
    l2 = "  ".join(bits)
    l3 = None
    if row.get("upper_mode"):
        mode = row["upper_mode"]
        if str(mode).startswith("box_demo"):
            detail = "box_demo_2 IK upper"
        elif str(mode).startswith("psi0_replay"):
            detail = "Psi0 traj replay"
        else:
            detail = "upper-body override"
        l3 = "upper-body: %s (%s)" % (mode, detail)
    return [l for l in (l1, l2, l3) if l]


def burn(src, dst, lines):
    reader = imageio.get_reader(src)
    fps = (reader.get_meta_data() or {}).get("fps", 25) or 25
    h_band = 22 + 18 * len(lines)
    writer = imageio.get_writer(dst, fps=fps, codec="libx264",
                                quality=7, macro_block_size=None)
    try:
        for frame in reader:
            img = Image.fromarray(frame)
            d = ImageDraw.Draw(img, "RGBA")
            d.rectangle([0, 0, img.width, h_band], fill=(0, 0, 0, 175))
            d.text((8, 4), lines[0], font=F1, fill=(255, 255, 120, 255))
            for i, ln in enumerate(lines[1:]):
                d.text((8, 26 + 18 * i), ln, font=F2, fill=(255, 255, 255, 255))
            writer.append_data(np.asarray(img))
    finally:
        writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="VIDEODIR:JSONL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-ok", type=int, default=2)
    ap.add_argument("--max-fail", type=int, default=2)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n_done = 0
    for pair in a.pairs:
        vdir, jpath = pair.rsplit(":", 1)
        if not os.path.exists(jpath):
            print("[skip] no jsonl:", jpath); continue
        rows = {}
        for line in open(jpath):
            try:
                r = json.loads(line)
                rows[int(r.get("trial"))] = r
            except Exception:
                pass
        vids = sorted(glob.glob(os.path.join(vdir, "**", "*.mp4"), recursive=True))
        vids = [v for v in vids if "_cap" not in v]
        n_ok = n_fail = 0
        for v in vids:
            mm = re.search(r"_t(\d+)_(ok|fail)\.mp4$", v)
            if not mm:
                continue
            trial, tag = int(mm.group(1)), mm.group(2)
            if not a.all:
                if tag == "ok" and n_ok >= a.max_ok:
                    continue
                if tag == "fail" and n_fail >= a.max_fail:
                    continue
            row = rows.get(trial)
            # single-row jsonl (one trial per file, e.g. arena per-variant where
            # trial==variant_seed but video filename is always _t00) -> use it.
            if row is None and len(rows) == 1:
                row = next(iter(rows.values()))
            if row is None:
                print("[skip] no row for", v); continue
            dst = os.path.join(a.out, os.path.basename(v).replace(".mp4", "_cap.mp4"))
            try:
                burn(v, dst, caption_lines(row))
                n_done += 1
                n_ok, n_fail = n_ok + (tag == "ok"), n_fail + (tag == "fail")
                print("[ok]", os.path.basename(dst))
            except Exception as e:
                print("[err]", v, repr(e))
    print("DONE captioned=%d -> %s" % (n_done, a.out))


if __name__ == "__main__":
    main()
