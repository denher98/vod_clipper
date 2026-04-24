# =============================================================================
#  variation_engine.py — Clip variation generator for PROYA Clipper
#
#  Turns 1 raw clip moment into N styled variants using purely parameter-level
#  mutations (no re-transcription, no re-detection).
#
#  Variation axes (mix-and-match per variant):
#    1. Mirror / horizontal flip
#    2. Subtitle font + color palette
#    3. Subtitle Y position (top / mid / bottom)
#    4. Zoom timing offset  (+/- seconds from original trigger)
#    5. Zoom scale magnitude
#    6. Color grade (brightness / contrast / saturation via FFmpeg filter)
#    7. Speed ramp (0.9×, 1.0×, 1.1× — slight slow/fast)
#    8. Crop offset (re-frame slightly left/right within 9:16)
#    9. Hook text display (show / hide, different duration)
#   10. Karaoke active word highlight colour
#
#  Usage in main.py:
#    from variation_engine import expand_moments_with_variants
#    moments = expand_moments_with_variants(moments, cfg)
#    # then proceed with the normal clip-editing loop — each variant is its own job
# =============================================================================

from __future__ import annotations

import copy
import hashlib
import logging
import random
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("proya.variation")


# ─────────────────────────────────────────────────────────────────────────────
#  STYLE PALETTE LIBRARY
#  Each palette defines a complete visual identity for one variant.
#  Add more palettes to increase variety without any code changes.
# ─────────────────────────────────────────────────────────────────────────────

SUBTITLE_PALETTES = [
    # name, font, active_color, inactive_opacity, stroke_color, stroke_w
    ("tiktok_classic",   "assets/fonts/Montserrat-ExtraBold.ttf",  "#FFD600", 1.0, "#000000", 3),
    ("tiktok_white",     "assets/fonts/Montserrat-ExtraBold.ttf",  "#FFFFFF", 0.6, "#000000", 4),
    ("neon_green",       "assets/fonts/Anton-Regular.ttf",         "#00FF7F", 1.0, "#003300", 3),
    ("hot_pink",         "assets/fonts/Anton-Regular.ttf",         "#FF2D78", 1.0, "#1A0008", 4),
    ("ice_blue",         "assets/fonts/Montserrat-ExtraBold.ttf",  "#00D4FF", 1.0, "#001A2E", 3),
    ("orange_punch",     "assets/fonts/Anton-Regular.ttf",         "#FF6B00", 1.0, "#1A0E00", 3),
    ("purple_glow",      "assets/fonts/Montserrat-ExtraBold.ttf",  "#C77DFF", 1.0, "#0D0020", 3),
    ("cream_soft",       "assets/fonts/Montserrat-ExtraBold.ttf",  "#FFF5D7", 0.7, "#2B1A00", 2),
    ("red_alarm",        "assets/fonts/Anton-Regular.ttf",         "#FF3B30", 1.0, "#000000", 4),
    ("playful_yellow",   "assets/fonts/PlayfairDisplay-Italic-VariableFont_wght.ttf", "#FFE500", 1.0, "#222200", 3),
]

HOOK_PALETTES = [
    # name, color, stroke_color, stroke_w, fontsize_multiplier
    ("bold_white",  "white",   "black",   5,  1.0),
    ("bold_yellow", "#FFD600", "black",   5,  1.0),
    ("bold_pink",   "#FF2D78", "black",   4,  0.95),
    ("big_white",   "white",   "#333333", 6,  1.1),
    ("neon_cyan",   "#00D4FF", "black",   4,  1.0),
]

# Y-position presets (fraction of frame height)
SUBTITLE_Y_POSITIONS = [0.72, 0.78, 0.83, 0.88]

# Zoom scale variants
ZOOM_SCALES = [1.25, 1.35, 1.45, 1.55, 1.65]

# Zoom timing offsets in seconds (applied to the detected trigger time)
ZOOM_OFFSETS = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]

# Speed ramp multipliers
SPEED_RAMPS = [0.90, 0.95, 1.00, 1.05, 1.10]

# Color grade presets (FFmpeg vf filter strings)
COLOR_GRADES = [
    ("natural",    ""),                                                           # no filter
    ("vivid",      "eq=saturation=1.3:contrast=1.05"),
    ("warm",       "colortemperature=temperature=7500"),
    ("cool",       "colortemperature=temperature=5000"),
    ("bright",     "eq=brightness=0.05:contrast=1.1"),
    ("cinematic",  "eq=saturation=0.85:contrast=1.15:brightness=-0.02"),
    ("punch",      "eq=saturation=1.5:contrast=1.2"),
    ("matte",      "eq=saturation=0.7:contrast=0.95:brightness=0.03"),
]

# Crop offset as fraction of width (shifts the 9:16 reframe left/right)
CROP_X_OFFSETS = [-0.04, -0.02, 0.0, 0.02, 0.04]


# ─────────────────────────────────────────────────────────────────────────────
#  VARIANT CONFIG DATACLASS
#  A VariantConfig patches cfg values at render time.
#  It's passed as a thin override layer on top of the main config.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VariantConfig:
    """Per-variant style overrides. All fields are optional."""
    variant_id: str = ""
    variant_index: int = 0          # 0 = original, 1+ = variant

    # Mirror
    mirror: bool = False

    # Subtitles
    font_subtitle: str = ""
    karaoke_active_color: str = ""
    karaoke_inactive_opacity: float = 1.0
    subtitle_stroke: str = "#000000"
    subtitle_stroke_w: int = 3
    subtitle_y_pos: float = 0.80

    # Hook
    hook_color: str = "white"
    hook_stroke_color: str = "black"
    hook_stroke_w: int = 5
    hook_fontsize_mult: float = 1.0
    hook_duration: float = 0.0     # 0 = disabled

    # Zoom
    zoom_scale: float = 1.45
    zoom_trigger_offset: float = 0.0   # seconds relative to detected trigger

    # Speed (1.0 = no change; applies during FFmpeg raw cut)
    speed_ramp: float = 1.0

    # Color grade (empty string = no filter)
    color_grade_filter: str = ""

    # Crop X offset
    crop_x_offset: float = 0.0


def apply_variant_to_cfg(base_cfg, variant: VariantConfig):
    """
    Return a lightweight object that looks like cfg but with variant overrides.
    Does NOT mutate base_cfg. Uses __dict__ copy + override.
    """
    class PatchedCfg:
        pass

    patched = PatchedCfg()
    # Copy all base cfg attributes
    for k, v in vars(base_cfg).items():
        setattr(patched, k, v)
    # Also copy module-level attributes (config.py is a module, not a class)
    import types
    if isinstance(base_cfg, types.ModuleType):
        for k in dir(base_cfg):
            if not k.startswith("_"):
                setattr(patched, k, getattr(base_cfg, k))

    # Apply overrides
    if variant.font_subtitle:
        patched.FONT_SUBTITLE = variant.font_subtitle
    if variant.karaoke_active_color:
        patched.KARAOKE_ACTIVE_COLOR = variant.karaoke_active_color
    patched.KARAOKE_INACTIVE_OPACITY = variant.karaoke_inactive_opacity
    patched.SUBTITLE_STROKE = variant.subtitle_stroke
    patched.SUBTITLE_STROKE_W = variant.subtitle_stroke_w
    patched.SUBTITLE_Y_POS = variant.subtitle_y_pos
    patched.HOOK_COLOR = variant.hook_color
    patched.HOOK_STROKE_COLOR = variant.hook_stroke_color
    patched.HOOK_STROKE_W = variant.hook_stroke_w
    patched.HOOK_FONTSIZE = int(getattr(base_cfg, "HOOK_FONTSIZE", 130) * variant.hook_fontsize_mult)
    patched.HOOK_DURATION = variant.hook_duration
    patched.ZOOM_SCALE = variant.zoom_scale
    patched._zoom_trigger_offset = variant.zoom_trigger_offset  # read by edit_clip
    patched._mirror = variant.mirror
    patched._speed_ramp = variant.speed_ramp
    patched._color_grade_filter = variant.color_grade_filter
    patched._crop_x_offset = variant.crop_x_offset
    patched._variant_id = variant.variant_id
    patched._variant_index = variant.variant_index

    return patched


# ─────────────────────────────────────────────────────────────────────────────
#  VARIANT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_variants(base_cfg, n_variants: int, seed: int | None = None) -> list[VariantConfig]:
    """
    Generate `n_variants` VariantConfig objects.

    Variant 0 is always the "original" (no mutations) so the base clip is
    always produced. Variants 1..N each randomly sample from the axes above,
    using a deterministic seed so re-runs produce the same set.

    Args:
        base_cfg:    The main config module/object.
        n_variants:  Total variants including original (so 1 = no extra variants).
        seed:        RNG seed for reproducibility.

    Returns:
        List of VariantConfig objects, length == n_variants.
    """
    rng = random.Random(seed)
    variants = []

    hook_dur_base = getattr(base_cfg, "HOOK_DURATION", 0.0)

    for i in range(n_variants):
        if i == 0:
            # Original — keep everything default
            vc = VariantConfig(
                variant_id="v0_original",
                variant_index=0,
                zoom_scale=getattr(base_cfg, "ZOOM_SCALE", 1.45),
                subtitle_y_pos=getattr(base_cfg, "SUBTITLE_Y_POS", 0.80),
                font_subtitle=getattr(base_cfg, "FONT_SUBTITLE", ""),
                karaoke_active_color=getattr(base_cfg, "KARAOKE_ACTIVE_COLOR", "#FFD600"),
                karaoke_inactive_opacity=getattr(base_cfg, "KARAOKE_INACTIVE_OPACITY", 1.0),
                hook_duration=hook_dur_base,
            )
        else:
            palette_name, font, active_color, inactive_op, stroke_c, stroke_w = rng.choice(SUBTITLE_PALETTES)
            hook_name, hook_col, hook_stroke_c, hook_stroke_w, hook_fs_mult = rng.choice(HOOK_PALETTES)

            vc = VariantConfig(
                variant_id=f"v{i}_{palette_name}",
                variant_index=i,
                mirror=rng.random() < 0.30,                      # 30% chance flip
                font_subtitle=font,
                karaoke_active_color=active_color,
                karaoke_inactive_opacity=inactive_op,
                subtitle_stroke=stroke_c,
                subtitle_stroke_w=stroke_w,
                subtitle_y_pos=rng.choice(SUBTITLE_Y_POSITIONS),
                hook_color=hook_col,
                hook_stroke_color=hook_stroke_c,
                hook_stroke_w=hook_stroke_w,
                hook_fontsize_mult=hook_fs_mult,
                hook_duration=hook_dur_base,
                zoom_scale=rng.choice(ZOOM_SCALES),
                zoom_trigger_offset=rng.choice(ZOOM_OFFSETS),
                speed_ramp=rng.choice(SPEED_RAMPS),
                color_grade_filter=rng.choice(COLOR_GRADES)[1],
                crop_x_offset=rng.choice(CROP_X_OFFSETS),
            )

        variants.append(vc)

    log.info(f"Generated {len(variants)} variant configs (seed={seed})")
    return variants


# ─────────────────────────────────────────────────────────────────────────────
#  MOMENT EXPANSION
#  Takes the LLM moments list and clones each moment N times (one per variant).
#  Each clone carries variant metadata so the editor can apply the right style.
# ─────────────────────────────────────────────────────────────────────────────

def expand_moments_with_variants(
    moments: list[dict],
    base_cfg,
    n_variants: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """
    Expand the moments list so each moment appears N times — once per variant.

    Args:
        moments:     Original moments from detect_moments().
        base_cfg:    Config module.
        n_variants:  How many variants per clip. Defaults to cfg.VARIANTS_PER_CLIP (or 4).
        seed:        RNG seed for variant generation.

    Returns:
        Expanded list with (len(moments) * n_variants) entries.
        Each entry has a "_variant" key containing the VariantConfig.
    """
    if n_variants is None:
        n_variants = getattr(base_cfg, "VARIANTS_PER_CLIP", 4)

    if n_variants <= 1:
        # No expansion — just tag every moment as v0_original
        for m in moments:
            m["_variant"] = VariantConfig(variant_id="v0_original", variant_index=0)
        return moments

    variants = generate_variants(base_cfg, n_variants, seed=seed)
    expanded = []

    for moment in moments:
        base_clip_id = moment.get("clip_id", "clip_unknown")
        for vc in variants:
            m = copy.deepcopy(moment)
            m["_variant"] = vc
            # Give variant its own clip_id so files don't collide
            m["clip_id"] = f"{base_clip_id}_{vc.variant_id}"
            expanded.append(m)

    log.info(
        f"Expanded {len(moments)} moments × {n_variants} variants "
        f"= {len(expanded)} total clip jobs"
    )
    return expanded


# ─────────────────────────────────────────────────────────────────────────────
#  FFmpeg VARIANT HELPERS
#  Called inside cut_raw_clip (or a wrapper) to bake mirror, speed, grade, crop
#  into the raw cut stage (before MoviePy editing) for maximum throughput.
#  Python-level image ops (mirror) on MoviePy clips are slow; FFmpeg is fast.
# ─────────────────────────────────────────────────────────────────────────────

def build_ffmpeg_vf_chain(variant: VariantConfig | None, frame_w: int = 1080, frame_h: int = 1920) -> str:
    """
    Compose an FFmpeg -vf filter chain string for a variant.
    Returns empty string if no filters needed.

    Filters applied in order:
      1. crop offset (reframe X)
      2. scale back to target resolution
      3. hflip (mirror)
      4. setpts (speed ramp)
      5. color grade (eq / colortemperature)
    """
    if variant is None:
        return ""

    filters = []

    # 1. Crop X offset — shift horizontal slice before other transforms
    ox = getattr(variant, "crop_x_offset", 0.0)
    if abs(ox) > 0.005:
        # Crop a slightly narrower strip then scale back up
        crop_w = int(frame_w * (1.0 - abs(ox)))
        crop_x = int(frame_w * (ox if ox > 0 else 0))
        filters.append(f"crop={crop_w}:{frame_h}:{crop_x}:0")
        filters.append(f"scale={frame_w}:{frame_h}")

    # 2. Mirror
    if getattr(variant, "mirror", False):
        filters.append("hflip")

    # 3. Speed ramp — setpts changes presentation timestamps
    speed = getattr(variant, "speed_ramp", 1.0)
    if abs(speed - 1.0) > 0.02:
        pts = round(1.0 / speed, 4)
        filters.append(f"setpts={pts}*PTS")

    # 4. Color grade
    grade = getattr(variant, "color_grade_filter", "")
    if grade:
        filters.append(grade)

    return ",".join(filters) if filters else ""


def build_ffmpeg_atempo(speed: float) -> list[str]:
    """
    Build FFmpeg -af atempo arguments for audio speed matching.
    atempo only supports 0.5–2.0 per filter; chain filters for extremes.
    """
    if abs(speed - 1.0) <= 0.02:
        return []
    # clamp to reasonable range for TikTok
    speed = max(0.75, min(1.25, speed))
    return ["-af", f"atempo={round(speed, 4)}"]


def cut_raw_clip_with_variant(
    input_video: str,
    start: float,
    end: float,
    output_path: str,
    variant,
    cfg,
) -> bool:
    import os
    import subprocess
    from pathlib import Path

    os.makedirs(Path(output_path).parent, exist_ok=True)

    # Guard: skip zero or negative duration clips
    duration = end - start
    if duration <= 0.5:
        log.error(f"Skipping clip with invalid duration: start={start} end={end}")
        return False

    raw_codec  = getattr(cfg, "RAW_CUT_CODEC", "h264_nvenc")
    raw_preset = getattr(cfg, "RAW_CUT_PRESET", "p1")

    # Use -ss AFTER -i (output seek) — slower but accurate, avoids empty output
    # Also add -avoid_negative_ts make_zero to handle edge cases
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{max(0.0, start):.3f}",   # input seek — BEFORE -i
        "-i", input_video,
        "-t", f"{duration:.3f}",
        "-c:v", raw_codec, "-preset", raw_preset,
        "-c:a", "aac", "-avoid_negative_ts", "make_zero",
    ]

    if raw_codec == "libx264":
        cmd += ["-crf", "28"]
    elif raw_codec.endswith("_nvenc"):
        cmd += ["-cq", str(getattr(cfg, "OUTPUT_CQ", 35))]

    # Variant filters
    if variant is not None:
        vf = build_ffmpeg_vf_chain(variant, frame_w=1080, frame_h=1920)
        if vf:
            cmd += ["-vf", vf]
        af = build_ffmpeg_atempo(getattr(variant, "speed_ramp", 1.0))
        if af:
            cmd += af

    cmd.append(output_path)

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            log.error(f"FFmpeg variant cut error: {r.stderr[-300:]}")
            return False
        # Extra guard: check output file actually has content
        if Path(output_path).exists() and Path(output_path).stat().st_size < 1024:
            log.error(f"FFmpeg produced empty/tiny file (<1KB): {output_path}")
            Path(output_path).unlink(missing_ok=True)
            return False
        return True
    except subprocess.TimeoutExpired:
        log.error(f"FFmpeg timed out: {output_path}")
        return False
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found")

# ─────────────────────────────────────────────────────────────────────────────
#  THROUGHPUT MATH HELPER
# ─────────────────────────────────────────────────────────────────────────────

def estimate_throughput(
    livestream_duration_hours: float,
    avg_clip_duration_s: float = 20.0,
    avg_edit_time_s: float = 8.0,
    n_variants: int = 6,
    parallel_workers: int = 4,
) -> dict:
    """
    Estimate how many clips can be produced in 24 hours.

    Returns dict with key metrics.
    """
    total_seconds = 24 * 3600
    moments_per_hour = 60  # conservative LLM detection estimate
    base_moments = int(livestream_duration_hours * moments_per_hour)
    total_clips = base_moments * n_variants

    # Time to edit all clips with N parallel workers
    total_edit_time = (total_clips * avg_edit_time_s) / parallel_workers
    clips_per_24h = int((total_seconds / avg_edit_time_s) * parallel_workers)

    return {
        "base_moments_detected": base_moments,
        "variants_per_clip": n_variants,
        "total_clips_to_render": total_clips,
        "estimated_edit_time_hours": round(total_edit_time / 3600, 1),
        "clips_per_24h_at_current_settings": clips_per_24h,
        "bottleneck": "editing" if total_edit_time > total_seconds else "moment_detection",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG ADDITIONS (paste these into config.py)
# ─────────────────────────────────────────────────────────────────────────────

SUGGESTED_CONFIG_ADDITIONS = """
# ── Variation Engine ──────────────────────────────────────────────────────────
# How many style variants to render per detected moment.
# 1 = no variation (just the original). 6 = 6x clip output.
# With a ~1h livestream → ~60 moments → 6 variants = ~360 clips.
# For 8–18k clips target across multiple VODs, set to 8–12.
VARIANTS_PER_CLIP = 6

# Seed for variant randomisation. Change to get a different style mix.
VARIANT_SEED = 42

# Whether to bake mirror/speed/grade into the FFmpeg raw cut (recommended True).
# False = these transforms are done in MoviePy (slower).
VARIANT_FFMPEG_BAKE = True
"""
