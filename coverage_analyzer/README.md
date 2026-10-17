# Defensive Coverage Analyzer

Status: scope decided. Goal is learning CV (YOLOv8 + OpenCV + tracking), not a
reliable coverage-tendency report.

This directory is independent of the NBA predictor in the rest of the repo.

## Decisions made

- **Footage**: broadcast TV film (YouTube full-game replays / ESPN+, etc.).
- **Why**: primary goal is learning YOLOv8 + OpenCV + tracking, not building a
  research-grade coverage classifier.
- **What this costs**: camera pans, zoom, and cuts mean players go off-screen
  constantly, especially boundary defenders. Post-snap alignment in particular
  is unreliable on the broadcast angle. This is accepted, not a problem to
  solve away.

## What the project is

Not: "classify coverage and study team tendencies."

Is: a CV pipeline that detects and tracks defensive players on broadcast film,
and attempts pre-snap formation/alignment features on plays where the full
formation happens to be visible before the snap (more often true pre-snap than
post-snap, since the camera hasn't started following the ball yet).

Coverage classification and tendency analysis are a stretch goal / v2, not the
deliverable.

## Problem breakdown (v1 scope)

1. Player detection (YOLOv8) — bounding boxes per frame, pretrained-plus-filtering
   first, fine-tune only if needed.
2. Tracking across frames — consistent player ID within a play.
3. Play segmentation — find where a play starts/ends in a full game video
   (pre-snap set, snap, whistle), since broadcast film isn't pre-cut.
4. Pre-snap alignment features — DB count, box count, spacing, depth, ONLY on
   plays where the formation is fully visible pre-snap.
5. (Stretch) Post-snap movement — likely unreliable on this angle.
6. (Stretch/v2) Coverage classification — rule-based first, only if alignment
   features are clean enough.

## Milestone 1 (this is what exists right now)

Run pretrained YOLOv8 on a single clip and measure how much of the formation is
detected, and how stably, before the snap. The result decides whether step 4 is
worth building. No tracking, no offense/defense split, no classification yet.

### Setup

```bash
pip install -r coverage_analyzer/requirements.txt
```

Weights (`yolov8n.pt`, etc.) download automatically on first run.

### Run

Cut a single play out of a broadcast recording (a few seconds before the snap
to a few seconds after), note roughly when the snap happens, then:

```bash
python coverage_analyzer/detect_clip.py play01.mp4 --snap-time 6.5
```

Useful knobs:

```bash
--start 2 --end 10        # only process this time range
--stride 2                # every 2nd frame (CPU runs are slow at imgsz 1280)
--model yolov8s.pt        # bigger model, more small-player recall
--imgsz 1280              # default; broadcast players are small, 640 misses them
--conf 0.2                # lower = more detections, more junk
--min-box-height 0.04     # drop boxes shorter than 4% of frame height (crowd)
--ignore-top 0.12         # drop boxes centered in the top 12% (stands)
--ignore-bottom 0.08      # drop boxes centered in the bottom 8% (score bug)
--formation-min 22        # kept count that means "both units visible"
--no-video                # skip the annotated mp4
```

### Output

In `coverage_analyzer/out/`:

- `<clip>_annotated.mp4` — green boxes are kept detections, grey boxes were
  filtered out, HUD shows per-frame count and PRE-SNAP / POST-SNAP.
- `<clip>_detections.csv` — one row per kept box (frame, time, xyxy, conf).
- `<clip>_summary.json` — count statistics overall, pre-snap, and post-snap.

The console prints the same summary. The numbers that matter for the go/no-go
on step 4 are in the PRE-SNAP block:

- `players/frame mean / median / max` — how many people the detector finds.
  22 means both units on screen; realistically broadcast angles show 15-20
  with boundary players cut off.
- `% frames >= 22` and `longest hold >= 22` — how often, and for how long
  contiguously, the full formation is on screen pre-snap. If the longest hold
  is under ~0.5s on most plays, there is no stable window to extract alignment
  features from and step 4 should be rethought (e.g. use the widest-angle frame
  only, or accept partial formations).
- `stdev` and `mean |delta|` — count stability. A high frame-to-frame delta
  means detections flicker, which will make tracking (step 2) harder.

### Known limits of this milestone

- COCO `person` detects everyone: both teams, refs, sideline staff. The count
  is a ceiling on "formation visible", not a defender count. Offense/defense
  separation (jersey color clustering is the cheap first attempt) is part of
  step 4, not this step.
- Filters are geometric heuristics; tune them per broadcast feed by watching the
  grey boxes in the annotated video.
- `yolov8n` on CPU at `imgsz=1280` runs at roughly 1-3 fps. Use `--stride` or a
  GPU for anything longer than a single play.

### Next step after this

Run it on 5-10 clips from the same broadcast. If the pre-snap hold is
consistently long enough, move to step 2 (tracking; ultralytics has ByteTrack
built in via `model.track()`), then step 4. If not, revisit what "alignment
features" can mean on partial formations before writing any more code.
