#!/usr/bin/env python3
# =============================================================================
#  main.py — PROYA Livestream Clip Automation Pipeline
#  
#  REQUIREMENTS (install before first run):
#    pip install faster-whisper ultralytics moviepy opencv-python openai
#    pip install pillow streamlit tqdm
#
#  LM STUDIO SETUP:
#    1. Download LM Studio: https://lmstudio.ai
#    2. Download a model (recommended: Gemma 3 12B Instruct Q4)
#    3. Go to "Local Server" tab → Start Server
#    4. Make sure the model is loaded (green indicator)
#
#  USAGE:
#    # Full pipeline (transcribe → detect moments → scan video → cut → edit):
#    python main.py --video livestream.mp4
#
#    # Skip to a specific stage (if previous stages are cached):
#    python main.py --video livestream.mp4 --skip-transcribe
#    python main.py --video livestream.mp4 --skip-transcribe --skip-moments
#
#    # Only cut clips, skip editing (faster for testing):
#    python main.py --video livestream.mp4 --cut-only
#
#    # Train YOLO on your product dataset (one-time):
#    python main.py --train-yolo
#
#    # Launch Streamlit web UI:
#    streamlit run app.py
# =============================================================================

import argparse
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Configure logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("proya.main")


def _build_clip_job(moment: dict, index: int, output_dir: str, raw_dir: Path) -> dict:
    clip_id = moment.get("clip_id", f"clip_{index+1:04d}")
    start = moment["start"]
    end = moment["end"]
    score = moment["score"]
    product = moment.get("product", "general")
    clip_type = moment.get("clip_type", "general")
    safe_hook = _safe_filename(moment.get("hook", clip_id))[:40]
    output_filename = f"{clip_id}_score{int(score)}_{safe_hook}.mp4"
    return {
        "index": index,
        "clip_id": clip_id,
        "start": start,
        "end": end,
        "score": score,
        "product": product,
        "clip_type": clip_type,
        "moment": moment,
        "output_filename": output_filename,
        "output_path": str(Path(output_dir) / output_filename),
        "raw_path": str(raw_dir / f"{clip_id}_raw.mp4"),
    }


def _process_clip_job(job: dict, video_path: str, transcript_words: list, product_events: list, cut_only: bool, cfg) -> dict:
    from ffmpeg_editor import cut_raw_clip, edit_clip, get_words_for_clip
    from vision_scanner import get_events_for_clip

    output_path = job["output_path"]
    raw_path    = job["raw_path"]

    if Path(output_path).exists():
        return {
            "clip_id": job["clip_id"],
            "status": "skipped",
            "output_filename": job["output_filename"],
            "manifest": None,
        }

    # Variant-aware cut — applies mirror/speed/grade/crop at cut time via FFmpeg
    variant = job["moment"].get("_variant", None)
    variant_baked = False
    try:
        from variation_engine import cut_raw_clip_with_variant
        cut_ok = cut_raw_clip_with_variant(
            video_path, job["start"], job["end"], raw_path, variant, cfg
        )
        variant_baked = variant is not None
    except ImportError:
        cut_ok = cut_raw_clip(video_path, job["start"], job["end"], raw_path, cfg=cfg)

    if not cut_ok:
        return {
            "clip_id": job["clip_id"],
            "status": "failed",
            "output_filename": job["output_filename"],
            "manifest": _build_manifest_row(job, 0, "failed"),
        }

    if cut_only:
        shutil.copy2(raw_path, output_path)
        if Path(raw_path).exists():
            os.remove(raw_path)
        return {
            "clip_id": job["clip_id"],
            "status": "ok",
            "output_filename": job["output_filename"],
            "manifest": _build_manifest_row(job, 0, "ok"),
        }

    # Apply variant style overrides (font/color/zoom/y-pos) to cfg
    if variant is not None:
        try:
            from variation_engine import apply_variant_to_cfg
            edit_cfg = apply_variant_to_cfg(cfg, variant)
            setattr(edit_cfg, "_variant_transforms_baked", variant_baked)
        except ImportError:
            edit_cfg = cfg
    else:
        edit_cfg = cfg

    clip_words          = get_words_for_clip(transcript_words, job["start"], job["end"])
    clip_product_events = get_events_for_clip(product_events, job["start"], job["end"])

    mirror = bool(getattr(variant, "mirror", False)) if variant is not None else False
    crop_x_offset = float(getattr(variant, "crop_x_offset", 0.0)) if variant is not None else 0.0
    if mirror or abs(crop_x_offset) > 0.005:
        clip_product_events = _remap_events_for_spatial_variant(
            clip_product_events,
            mirror=mirror,
            crop_x_offset=crop_x_offset,
        )

    speed_ramp = getattr(variant, "speed_ramp", 1.0) if variant is not None else 1.0
    if abs(speed_ramp - 1.0) > 0.02:
        clip_words = _remap_words_for_speed_ramp(clip_words, speed_ramp)
        clip_product_events = _remap_events_for_speed_ramp(clip_product_events, speed_ramp)

    edit_ok = edit_clip(
        raw_clip_path=raw_path,
        output_path=output_path,
        moment=job["moment"],
        clip_words=clip_words,
        product_events=clip_product_events,
        cfg=edit_cfg,
    )

    if Path(raw_path).exists():
        os.remove(raw_path)

    return {
        "clip_id": job["clip_id"],
        "status": "ok" if edit_ok else "failed",
        "output_filename": job["output_filename"],
        "manifest": _build_manifest_row(job, len(clip_product_events), "ok" if edit_ok else "failed"),
    }


def _build_manifest_row(job: dict, product_event_count: int, status: str) -> dict:
    moment = job["moment"]
    return {
        "clip_id": job["clip_id"],
        "output_file": job["output_filename"],
        "start": job["start"],
        "end": job["end"],
        "duration": round(job["end"] - job["start"], 1),
        "score": job["score"],
        "hook": moment.get("hook", ""),
        "product": job["product"],
        "clip_type": job["clip_type"],
        "reason": moment.get("reason", ""),
        "product_events": product_event_count,
        "status": status,
    }


def _remap_words_for_speed_ramp(words: list, speed_ramp: float) -> list:
    """Map clip-relative word timestamps onto a speed-ramped output timeline."""
    if abs(speed_ramp - 1.0) <= 0.02:
        return words

    remapped = []
    for word in words:
        mapped = dict(word)
        mapped["start"] = round(float(word["start"]) / speed_ramp, 6)
        mapped["end"] = round(float(word["end"]) / speed_ramp, 6)
        remapped.append(mapped)
    return remapped


def _remap_events_for_spatial_variant(events: list, mirror: bool, crop_x_offset: float) -> list:
    """Map product bbox coordinates into the rendered clip's spatial coordinate system."""
    if not mirror and abs(crop_x_offset) <= 0.005:
        return events

    remapped = []
    for event in events:
        mapped = dict(event)
        frame_w = float(event.get("frame_w") or 0)
        frame_h = float(event.get("frame_h") or 0)

        def remap_bbox(bbox):
            return _remap_bbox_for_variant(bbox, frame_w, frame_h, mirror, crop_x_offset)

        for key in ("best_bbox", "start_bbox", "end_bbox"):
            if event.get(key):
                mapped[key] = remap_bbox(event.get(key))

        if event.get("relative_track"):
            mapped["relative_track"] = [
                {
                    **sample,
                    "bbox": remap_bbox(sample.get("bbox")),
                }
                for sample in event["relative_track"]
            ]

        remapped.append(mapped)
    return remapped


def _remap_bbox_for_variant(bbox, frame_w: float, frame_h: float, mirror: bool, crop_x_offset: float):
    if not bbox or frame_w <= 0 or frame_h <= 0:
        return bbox

    x1, y1, x2, y2 = [float(v) for v in bbox]
    out_w = frame_w
    out_h = frame_h

    if abs(crop_x_offset) > 0.005:
        crop_w = frame_w * (1.0 - abs(crop_x_offset))
        crop_x = frame_w * crop_x_offset if crop_x_offset > 0 else 0.0
        if crop_w > 1.0:
            scale_x = frame_w / crop_w
            x1 = (x1 - crop_x) * scale_x
            x2 = (x2 - crop_x) * scale_x
            x1 = max(0.0, min(out_w, x1))
            x2 = max(0.0, min(out_w, x2))

    if mirror:
        x1, x2 = out_w - x2, out_w - x1

    y1 = max(0.0, min(out_h, y1))
    y2 = max(0.0, min(out_h, y2))
    x1 = max(0.0, min(out_w, x1))
    x2 = max(0.0, min(out_w, x2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]


def _remap_events_for_speed_ramp(events: list, speed_ramp: float) -> list:
    """Map clip-relative product event timestamps onto a speed-ramped output timeline."""
    if abs(speed_ramp - 1.0) <= 0.02:
        return events

    remapped = []
    for event in events:
        mapped = dict(event)
        rel_start = event.get("relative_start")
        rel_end = event.get("relative_end")
        if rel_start is not None:
            mapped["relative_start"] = round(float(rel_start) / speed_ramp, 6)
        if rel_end is not None:
            mapped["relative_end"] = round(float(rel_end) / speed_ramp, 6)
        if rel_start is not None and rel_end is not None:
            mapped["duration"] = round((float(rel_end) - float(rel_start)) / speed_ramp, 6)
        if event.get("relative_track"):
            mapped["relative_track"] = [
                {
                    **sample,
                    "relative_time": round(float(sample["relative_time"]) / speed_ramp, 6),
                }
                for sample in event["relative_track"]
                if sample.get("relative_time") is not None
            ]
        remapped.append(mapped)
    return remapped


def run_pipeline(
    video_path: str,
    skip_transcribe: bool = False,
    skip_moments: bool = False,
    skip_vision: bool = False,
    cut_only: bool = False,
    max_clips: int = None,
    min_score: float = None,
    output_tag: str | None = None,
    working_tag: str | None = None,
    progress_callback=None,   # optional: fn(stage, pct, message)
):
    """
    Full pipeline orchestrator. All stages cache their results so you can
    safely re-run after a crash — it picks up where it left off.
    """
    import config as cfg

    # Allow runtime overrides
    if min_score is not None:
        cfg.MIN_SCORE = min_score

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    base_stem = Path(video_path).stem
    working_stem = base_stem
    if working_tag:
        safe_working_tag = _safe_filename(working_tag)
        working_stem = f"{base_stem}__{safe_working_tag}"
    working_dir = str(Path(cfg.WORKING_DIR) / working_stem)

    output_stem = base_stem
    if output_tag:
        safe_tag = _safe_filename(output_tag)
        output_stem = f"{output_stem}__{safe_tag}"
    output_dir = str(Path(cfg.OUTPUT_DIR) / output_stem)
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    _report(progress_callback, "init", 0, f"Pipeline started for: {video_path}")
    log.info("=" * 70)
    log.info("PROYA LIVESTREAM CLIP PIPELINE")
    log.info(f"  Input:      {video_path}")
    log.info(f"  Working:    {working_dir}")
    log.info(f"  Output:     {output_dir}")
    if working_tag:
        log.info(f"  Working tag:{working_tag}")
    if output_tag:
        log.info(f"  Rerun tag:   {output_tag}")
    log.info(f"  LM Studio:  {cfg.LM_STUDIO_BASE_URL}")
    log.info("=" * 70)

    pipeline_start = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1: TRANSCRIPTION
    # ══════════════════════════════════════════════════════════════════════════
    if not skip_transcribe:
        _report(progress_callback, "transcribe", 5, "Transcribing audio (Whisper)...")
        log.info("\n── STAGE 1: TRANSCRIPTION ─────────────────────────────────────────")

        from transcriber import transcribe, build_text_chunks
        t0 = time.time()
        transcript = transcribe(video_path, working_dir, cfg)
        log.info(f"Transcription done in {_fmt_time(time.time()-t0)}")
    else:
        log.info("Skipping transcription (using cached)")
        from transcriber import (
            build_text_chunks,
            load_cached_transcript,
            transcript_cache_is_compatible,
            transcribe,
        )
        transcript_path = Path(working_dir) / "transcript.json"
        transcript = load_cached_transcript(working_dir)
        if transcript is None:
            raise FileNotFoundError(f"No cached transcript at {transcript_path}. Run without --skip-transcribe first.")
        if not transcript_cache_is_compatible(transcript, cfg):
            log.info("Cached transcript is outdated or uses raw word timings; rebuilding aligned transcript")
            transcript = transcribe(video_path, working_dir, cfg)

    _report(progress_callback, "transcribe", 20, f"Transcript: {len(transcript['words'])} words")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2: LLM MOMENT DETECTION (LM Studio)
    # ══════════════════════════════════════════════════════════════════════════
    if not skip_moments:
        _report(progress_callback, "moments", 22, "Detecting moments with LLM (LM Studio)...")
        log.info("\n── STAGE 2: MOMENT DETECTION (LM Studio) ─────────────────────────")

        from transcriber import build_text_chunks
        from moment_detector import detect_moments

        chunks = build_text_chunks(transcript, cfg.CHUNK_DURATION, cfg.CHUNK_OVERLAP)
        t0 = time.time()
        moments = detect_moments(chunks, working_dir, cfg)
        log.info(f"Moment detection done in {_fmt_time(time.time()-t0)}")
    else:
        log.info("Skipping moment detection (using cached)")
        import json
        moments_path = Path(working_dir) / "moments.json"
        if not moments_path.exists():
            raise FileNotFoundError(f"No cached moments at {moments_path}. Run without --skip-moments first.")
        with open(moments_path, "r", encoding="utf-8") as f:
            moments = json.load(f)

    if not moments:
        log.warning("No moments detected! Check your LM Studio connection and transcript quality.")
        return {"clips_created": 0, "clips_failed": 0, "moments_found": 0}

    # Apply max_clips limit (takes highest scored first since list is sorted)
    if max_clips and len(moments) > max_clips:
        log.info(f"Limiting to top {max_clips} clips (from {len(moments)} total)")
        moments = moments[:max_clips]

    log.info(f"Moments to process: {len(moments)}")
    _report(progress_callback, "moments", 35, f"Found {len(moments)} clip moments")

    # ── Variation expansion ───────────────────────────────────────────────────
    n_variants = getattr(cfg, "VARIANTS_PER_CLIP", 1)
    if n_variants > 1:
        try:
            from variation_engine import expand_moments_with_variants
            variant_seed = getattr(cfg, "VARIANT_SEED", 42)
            log.info(f"\n── VARIATION ENGINE ──────────────────────────────────────────────")
            log.info(f"  Base moments: {len(moments)} | Variants per clip: {n_variants}")
            moments = expand_moments_with_variants(moments, cfg, n_variants=n_variants, seed=variant_seed)
            log.info(f"  Total clip jobs after expansion: {len(moments)}")
        except ImportError:
            log.warning("variation_engine.py not found — skipping variations")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3: VISION SCAN (YOLO)
    # ══════════════════════════════════════════════════════════════════════════
    product_events = []

    if not skip_vision:
        yolo_available = Path(cfg.YOLO_WEIGHTS).exists()
        if not yolo_available:
            log.warning(f"YOLO weights not found at {cfg.YOLO_WEIGHTS}. Skipping vision scan.")
            log.warning("Run 'python main.py --train-yolo' to train your product detector first.")
        else:
            _report(progress_callback, "vision", 37, "Scanning for products (YOLO)...")
            log.info("\n── STAGE 3: VISION SCAN (YOLOv8) ────────────────────────────────")

            from vision_scanner import build_scan_ranges_from_moments, scan_video_for_products
            t0 = time.time()
            scan_ranges = build_scan_ranges_from_moments(moments, cfg)
            product_events = scan_video_for_products(
                video_path,
                working_dir,
                cfg,
                scan_ranges=scan_ranges,
            )
            log.info(f"Vision scan done in {_fmt_time(time.time()-t0)}")
            log.info(f"Product events found: {len(product_events)}")
    else:
        log.info("Skipping vision scan (using cached or disabled)")
        import json
        detections_path = Path(working_dir) / "product_detections.json"
        if detections_path.exists():
            with open(detections_path, "r") as f:
                cached_product_events = json.load(f)
            try:
                from vision_scanner import _is_valid_cached_events
                if _is_valid_cached_events(cached_product_events):
                    product_events = cached_product_events
                    log.info(f"Loaded {len(product_events)} cached product events")
                else:
                    log.warning("Cached product events are outdated; rerun without --skip-vision to rebuild them")
                    product_events = []
            except Exception:
                product_events = cached_product_events
                log.info(f"Loaded {len(product_events)} cached product events")

    _report(progress_callback, "vision", 50, f"{len(product_events)} product events loaded")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 4: CUT + EDIT CLIPS
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n── STAGE 4: CUT & EDIT CLIPS ─────────────────────────────────────")

    from vision_scanner import get_events_for_clip

    raw_dir = Path(working_dir) / "raw_cuts"
    raw_dir.mkdir(exist_ok=True)

    clips_created = 0
    clips_failed  = 0
    clips_skipped = 0
    manifest      = []

    jobs = [_build_clip_job(moment, i, output_dir, raw_dir) for i, moment in enumerate(moments)]
    max_workers = max(1, int(getattr(cfg, "MAX_PARALLEL_CLIPS", 6)))
    log.info(f"  Total jobs: {len(jobs)} | Parallel workers: {max_workers}")

    for job in jobs:
        log.info(
            f"  [{job['index']+1:03d}/{len(jobs):03d}] {job['clip_id']} | "
            f"t={job['start']:.1f}s-{job['end']:.1f}s | score={job['score']} | "
            f"type={job['clip_type']} | product={job['product']}"
        )

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _process_clip_job,
                job,
                video_path,
                transcript["words"],
                product_events,
                cut_only,
                cfg,
            ): job
            for job in jobs
        }

        for future in as_completed(future_map):
            job = future_map[future]
            completed += 1
            pct = 50 + int((completed / len(jobs)) * 45)
            _report(
                progress_callback, "editing", pct,
                f"[{completed}/{len(jobs)}] {job['clip_id']} | score={job['score']} | {job['product']}",
            )

            try:
                result = future.result()
            except Exception as e:
                clips_failed += 1
                log.error(f"    Worker failed for {job['clip_id']}: {e}")
                manifest.append(_build_manifest_row(job, 0, "failed"))
                continue

            if result["status"] == "skipped":
                clips_skipped += 1
                clips_created += 1
                log.info(f"    Already exists, skipping: {result['output_filename']}")
                continue

            if result["status"] == "ok":
                clips_created += 1
                log.info(f"    Created: {result['output_filename']}")
            else:
                clips_failed += 1
                log.error(f"    Edit failed for {job['clip_id']}")

            if result["manifest"]:
                manifest.append(result["manifest"])

    # ══════════════════════════════════════════════════════════════════════════
    # DONE
    # ══════════════════════════════════════════════════════════════════════════
    total_time = time.time() - pipeline_start

    # Save manifest
    import json
    manifest_path = Path(output_dir) / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    log.info("\n" + "=" * 70)
    log.info("PIPELINE COMPLETE")
    log.info(f"  Total time:     {_fmt_time(total_time)}")
    log.info(f"  Moments found:  {len(moments)}")
    log.info(f"  Clips created:  {clips_created}")
    log.info(f"  Clips failed:   {clips_failed}")
    log.info(f"  Clips skipped:  {clips_skipped} (already existed)")
    log.info(f"  Output dir:     {output_dir}")
    log.info(f"  Manifest:       {manifest_path}")
    log.info("=" * 70)

    _report(progress_callback, "done", 100, f"Done! {clips_created} clips created in {_fmt_time(total_time)}")

    return {
        "clips_created": clips_created,
        "clips_failed": clips_failed,
        "clips_skipped": clips_skipped,
        "moments_found": len(moments),
        "total_time": total_time,
        "output_dir": output_dir,
        "manifest_path": str(manifest_path),
    }


# ── Utility functions ──────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _safe_filename(text: str) -> str:
    import re
    text = re.sub(r'[<>:"/\\|?*\n\r]', '', text)
    text = re.sub(r'\s+', '_', text.strip())
    return text or "clip"


def _report(callback, stage: str, pct: int, message: str):
    if callback:
        callback(stage, pct, message)


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PROYA Livestream → TikTok Clips Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", type=str, help="Path to input livestream video")
    parser.add_argument("--skip-transcribe", action="store_true", help="Use cached transcript")
    parser.add_argument("--skip-moments", action="store_true", help="Use cached moments")
    parser.add_argument("--skip-vision", action="store_true", help="Skip YOLO product scan")
    parser.add_argument("--cut-only", action="store_true", help="Cut clips without editing")
    parser.add_argument("--max-clips", type=int, default=None, help="Max number of clips to process")
    parser.add_argument("--min-score", type=float, default=None, help="Minimum LLM score (1-10)")
    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help="Write clips to a new output folder suffix while reusing cached working data",
    )
    parser.add_argument(
        "--working-tag",
        type=str,
        default=None,
        help="Write caches to a new working folder suffix so transcript/moments/YOLO redo from scratch",
    )
    parser.add_argument(
        "--redo-tag",
        type=str,
        default=None,
        help="Convenience tag that applies to both working and output folders for a true full redo",
    )
    parser.add_argument("--train-yolo", action="store_true", help="Train YOLO on your product dataset")
    parser.add_argument("--test-lm-studio", action="store_true", help="Test LM Studio connection")
    parser.add_argument(
        "--preview-corrections", action="store_true",
        help="Show what word corrections would be applied to a cached transcript (use with --video)"
    )
    parser.add_argument(
        "--preview-ba", action="store_true",
        help="List before/after images found in BEFORE_AFTER_DIR"
    )
    parser.add_argument(
        "--setup-sfx", action="store_true",
        help="Create SFX folders and show their status"
    )

    args = parser.parse_args()

    # ── Train YOLO ────────────────────────────────────────────────────────────
    if args.train_yolo:
        import config as cfg
        from vision_scanner import train_model
        log.info("Starting YOLO training...")
        train_model(cfg)
        return

    # ── Test LM Studio ────────────────────────────────────────────────────────
    if args.test_lm_studio:
        _test_lm_studio()
        return

    # ── Preview word corrections ──────────────────────────────────────────────
    if args.preview_corrections:
        if not args.video:
            print("Error: --preview-corrections requires --video")
            sys.exit(1)
        import config as cfg
        import json
        from word_corrector import preview_corrections
        working_dir = str(Path(cfg.WORKING_DIR) / Path(args.video).stem)
        transcript_path = Path(working_dir) / "transcript.json"
        if not transcript_path.exists():
            print(f"No cached transcript found at {transcript_path}")
            print("Run the pipeline once (or just transcription) first.")
            sys.exit(1)
        with open(transcript_path, encoding="utf-8") as f:
            transcript = json.load(f)
        examples = preview_corrections(transcript, cfg, max_examples=30)
        if not examples:
            print("✓ No corrections needed — transcript looks clean!")
        else:
            print(f"\n{'='*60}")
            print(f"WORD CORRECTIONS PREVIEW ({len(examples)} examples found)")
            print(f"{'='*60}")
            for ex in examples:
                t = ex['time']
                print(f"\n  t={t:.1f}s")
                print(f"  BEFORE: {ex['original']}")
                print(f"  AFTER:  {ex['corrected']}")
        return

    # ── Preview before/after images ───────────────────────────────────────────
    if args.preview_ba:
        import config as cfg
        from pathlib import Path
        ba_dir = Path(cfg.BEFORE_AFTER_DIR)
        if not ba_dir.exists():
            print(f"Folder not found: {ba_dir}")
            print(f"Create it and put your before/after images there.")
        else:
            exts = {".jpg", ".jpeg", ".png", ".webp"}
            imgs = [p for p in ba_dir.iterdir() if p.suffix.lower() in exts]
            print(f"\n✓ Found {len(imgs)} images in {ba_dir}:")
            for img in sorted(imgs):
                size_kb = img.stat().st_size // 1024
                print(f"  {img.name}  ({size_kb} KB)")
        return

    # ── Setup SFX folders ─────────────────────────────────────────────────────
    if args.setup_sfx:
        import config as cfg
        from sfx_player import create_sfx_folders
        create_sfx_folders(cfg)
        return
        parser.print_help()
        print("\nError: --video is required unless using --train-yolo or --test-lm-studio")
        sys.exit(1)

    output_tag = args.output_tag
    working_tag = args.working_tag
    if args.redo_tag:
        output_tag = args.redo_tag
        working_tag = args.redo_tag

    run_pipeline(
        video_path=args.video,
        skip_transcribe=args.skip_transcribe,
        skip_moments=args.skip_moments,
        skip_vision=args.skip_vision,
        cut_only=args.cut_only,
        max_clips=args.max_clips,
        min_score=args.min_score,
        output_tag=output_tag,
        working_tag=working_tag,
    )


def _test_lm_studio():
    """Quick test to verify LM Studio is running and responding."""
    import config as cfg

    log.info(f"Testing LM Studio at {cfg.LM_STUDIO_BASE_URL}...")
    try:
        from openai import OpenAI
        client = OpenAI(base_url=cfg.LM_STUDIO_BASE_URL, api_key=cfg.LM_STUDIO_API_KEY)

        response = client.chat.completions.create(
            model=cfg.LM_STUDIO_MODEL,
            messages=[
                {"role": "user", "content": "Balas hanya dengan 'OK' jika kamu bisa mendengar saya."}
            ],
            max_tokens=10,
            timeout=30,
        )
        reply = response.choices[0].message.content.strip()
        log.info(f"✓ LM Studio is working! Response: '{reply}'")
        log.info(f"  Model: {response.model}")
    except Exception as e:
        log.error(f"✗ LM Studio connection failed: {e}")
        log.error("Make sure LM Studio is running and a model is loaded in the Local Server tab.")
        sys.exit(1)


if __name__ == "__main__":
    main()
