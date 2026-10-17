"""
detect_clip.py — Milestone 1 of the Defensive Coverage Analyzer.

Run pretrained YOLOv8 (COCO "person" class) over a single broadcast clip and
measure how many players are detected, and how stably, before the snap.

This deliberately does NOT do tracking, offense/defense separation, or
coverage classification. Its only job is to answer: on this clip, is the
pre-snap formation visible enough to be worth building alignment features on?

Usage:
    python coverage_analyzer/detect_clip.py clip.mp4 --snap-time 6.5
    python coverage_analyzer/detect_clip.py clip.mp4 --snap-time 6.5 --start 2 --end 9
    python coverage_analyzer/detect_clip.py clip.mp4 --model yolov8s.pt --imgsz 1280 --conf 0.2

Outputs (in --out-dir, default coverage_analyzer/out/):
    <clip>_annotated.mp4   kept boxes in green, filtered-out boxes in grey, count HUD
    <clip>_detections.csv  one row per kept box: frame, t_sec, x1, y1, x2, y2, conf
    <clip>_summary.json    detection stats overall and for the pre-snap window
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ultralytics not installed. Run: pip install -r coverage_analyzer/requirements.txt")


PERSON_CLASS = 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip", type=Path, help="path to a single-play video clip")
    p.add_argument("--snap-time", type=float, default=None,
                   help="seconds into the clip when the ball is snapped (eyeball it). "
                        "Enables the pre-snap window report.")
    p.add_argument("--start", type=float, default=0.0, help="start processing at this many seconds")
    p.add_argument("--end", type=float, default=None, help="stop processing at this many seconds")
    p.add_argument("--stride", type=int, default=1, help="process every Nth frame (speeds up CPU runs)")

    p.add_argument("--model", default="yolov8n.pt", help="ultralytics weights (n/s/m/l/x or a fine-tuned .pt)")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="inference size. Broadcast players are small; 1280 finds more than the 640 default.")
    p.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    p.add_argument("--device", default=None, help="'cpu', '0', 'mps', ... (default: auto)")

    p.add_argument("--min-box-height", type=float, default=0.04,
                   help="drop boxes shorter than this fraction of frame height (crowd, distant sideline)")
    p.add_argument("--ignore-top", type=float, default=0.12,
                   help="drop boxes whose center is in the top fraction of the frame (stands, upper crowd)")
    p.add_argument("--ignore-bottom", type=float, default=0.0,
                   help="drop boxes whose center is in the bottom fraction of the frame (score bug, sideline)")

    p.add_argument("--formation-min", type=int, default=22,
                   help="kept-detection count that counts as 'full formation visible' (22 = both units)")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "out")
    p.add_argument("--no-video", action="store_true", help="skip writing the annotated video")
    return p.parse_args()


def keep_box(xyxy, frame_h, args):
    """Cheap geometric filters to drop obvious non-players on a broadcast frame."""
    x1, y1, x2, y2 = xyxy
    h = y2 - y1
    cy = (y1 + y2) / 2.0
    if h < args.min_box_height * frame_h:
        return False
    if cy < args.ignore_top * frame_h:
        return False
    if args.ignore_bottom > 0 and cy > (1.0 - args.ignore_bottom) * frame_h:
        return False
    return True


def longest_run(flags):
    """Length of the longest consecutive run of True in a list."""
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def window_stats(rows, args, fps_eff):
    """Summarize kept-detection counts for a list of per-frame rows."""
    if not rows:
        return None
    counts = [r["n_kept"] for r in rows]
    diffs = [abs(b - a) for a, b in zip(counts, counts[1:])]
    hold_frames = longest_run([c >= args.formation_min for c in counts])
    return {
        "frames": len(rows),
        "seconds": round(len(rows) / fps_eff, 2),
        "kept_mean": round(statistics.fmean(counts), 2),
        "kept_median": statistics.median(counts),
        "kept_min": min(counts),
        "kept_max": max(counts),
        "kept_stdev": round(statistics.pstdev(counts), 2),
        "mean_frame_to_frame_change": round(statistics.fmean(diffs), 2) if diffs else 0.0,
        "pct_frames_ge_11": round(100 * sum(c >= 11 for c in counts) / len(counts), 1),
        "pct_frames_ge_18": round(100 * sum(c >= 18 for c in counts) / len(counts), 1),
        f"pct_frames_ge_{args.formation_min}": round(100 * sum(c >= args.formation_min for c in counts) / len(counts), 1),
        f"longest_hold_ge_{args.formation_min}_frames": hold_frames,
        f"longest_hold_ge_{args.formation_min}_seconds": round(hold_frames / fps_eff, 2),
    }


def draw(frame, kept, dropped, n_kept, t_sec, snap_time):
    for (x1, y1, x2, y2), _ in dropped:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (120, 120, 120), 1)
    for (x1, y1, x2, y2), conf in kept:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
        cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1, cv2.LINE_AA)
    phase = ""
    if snap_time is not None:
        phase = "  PRE-SNAP" if t_sec < snap_time else "  POST-SNAP"
    hud = f"t={t_sec:6.2f}s  players={n_kept:2d}  (grey={len(dropped)}){phase}"
    cv2.rectangle(frame, (0, 0), (max(560, 12 * len(hud)), 30), (0, 0, 0), -1)
    cv2.putText(frame, hud, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main():
    args = parse_args()
    if not args.clip.exists():
        sys.exit(f"clip not found: {args.clip}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.clip.stem

    cap = cv2.VideoCapture(str(args.clip))
    if not cap.isOpened():
        sys.exit(f"could not open video: {args.clip}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_eff = fps / args.stride

    start_f = int(args.start * fps)
    end_f = int(args.end * fps) if args.end is not None else total
    end_f = min(end_f, total) if total > 0 else end_f
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    print(f"\n  clip:   {args.clip}  ({width}x{height} @ {fps:.2f} fps, {total} frames)")
    print(f"  model:  {args.model}  imgsz={args.imgsz}  conf={args.conf}")
    print(f"  range:  frames {start_f}-{end_f}  stride={args.stride}"
          + (f"  snap @ {args.snap_time}s" if args.snap_time is not None else "  (no --snap-time given)"))

    model = YOLO(args.model)

    writer = None
    if not args.no_video:
        out_video = args.out_dir / f"{stem}_annotated.mp4"
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps_eff, (width, height))

    csv_path = args.out_dir / f"{stem}_detections.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "t_sec", "x1", "y1", "x2", "y2", "conf"])

    rows = []
    frame_idx = start_f
    while frame_idx < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        if (frame_idx - start_f) % args.stride != 0:
            frame_idx += 1
            continue
        t_sec = frame_idx / fps

        res = model.predict(frame, classes=[PERSON_CLASS], conf=args.conf, imgsz=args.imgsz,
                            device=args.device, verbose=False)[0]
        kept, dropped = [], []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for box, conf in zip(xyxy, confs):
                (kept if keep_box(box, height, args) else dropped).append((box, float(conf)))

        for (x1, y1, x2, y2), conf in kept:
            csv_writer.writerow([frame_idx, f"{t_sec:.3f}", f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}", f"{conf:.3f}"])
        rows.append({"frame": frame_idx, "t_sec": t_sec, "n_raw": len(kept) + len(dropped), "n_kept": len(kept)})

        if writer is not None:
            writer.write(draw(frame, kept, dropped, len(kept), t_sec, args.snap_time))

        if len(rows) % 30 == 0:
            print(f"    frame {frame_idx:6d}  t={t_sec:6.2f}s  players={len(kept):2d}", end="\r")
        frame_idx += 1

    cap.release()
    csv_file.close()
    if writer is not None:
        writer.release()
    print(" " * 70, end="\r")

    if not rows:
        sys.exit("no frames processed — check --start/--end against the clip length")

    summary = {
        "clip": str(args.clip),
        "model": args.model,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "fps": fps,
        "stride": args.stride,
        "snap_time": args.snap_time,
        "filters": {
            "min_box_height_frac": args.min_box_height,
            "ignore_top_frac": args.ignore_top,
            "ignore_bottom_frac": args.ignore_bottom,
        },
        "raw_mean_per_frame": round(statistics.fmean(r["n_raw"] for r in rows), 2),
        "overall": window_stats(rows, args, fps_eff),
    }
    if args.snap_time is not None:
        pre = [r for r in rows if r["t_sec"] < args.snap_time]
        post = [r for r in rows if r["t_sec"] >= args.snap_time]
        summary["pre_snap"] = window_stats(pre, args, fps_eff)
        summary["post_snap"] = window_stats(post, args, fps_eff)

    summary_path = args.out_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    def show(title, s):
        if s is None:
            print(f"\n  {title}: no frames")
            return
        print(f"\n  {title}  ({s['frames']} frames, {s['seconds']}s)")
        print(f"    players/frame   mean {s['kept_mean']:5.1f}  median {s['kept_median']:4}  "
              f"min {s['kept_min']:2d}  max {s['kept_max']:2d}  stdev {s['kept_stdev']}")
        print(f"    frame-to-frame  mean |delta| {s['mean_frame_to_frame_change']}")
        print(f"    % frames >=11: {s['pct_frames_ge_11']:5.1f}   >=18: {s['pct_frames_ge_18']:5.1f}   "
              f">={args.formation_min}: {s[f'pct_frames_ge_{args.formation_min}']:5.1f}")
        print(f"    longest hold >={args.formation_min}: "
              f"{s[f'longest_hold_ge_{args.formation_min}_frames']} frames "
              f"({s[f'longest_hold_ge_{args.formation_min}_seconds']}s)")

    print(f"\n  raw person detections/frame (before filters): {summary['raw_mean_per_frame']}")
    show("OVERALL", summary["overall"])
    if args.snap_time is not None:
        show("PRE-SNAP", summary["pre_snap"])
        show("POST-SNAP", summary["post_snap"])

    print(f"\n  wrote {csv_path}")
    print(f"  wrote {summary_path}")
    if writer is not None:
        print(f"  wrote {args.out_dir / (stem + '_annotated.mp4')}")
    print()


if __name__ == "__main__":
    main()
