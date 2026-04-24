# =============================================================================
#  moment_detector.py — LLM-based moment scoring via LM Studio
#  LM Studio exposes an OpenAI-compatible API at localhost:1234/v1
#  Make sure LM Studio is running and a model is loaded before running this.
# =============================================================================

import json
import logging
import re
import time
from pathlib import Path

from hook_text import build_hook_payload

log = logging.getLogger("proya.moment_detector")

# ── System prompt (Bahasa Indonesia + English fallback) ──────────────────────
SYSTEM_PROMPT = """Kamu adalah editor video TikTok profesional untuk brand skincare PROYA 5X Vitamin C.

TUJUAN UTAMA:
Pilih momen yang PALING KUAT dalam menjelaskan MANFAAT PRODUK (product benefits).
Fokus utama adalah membuat clip yang membuat orang ingin membeli karena hasil/manfaatnya jelas.

PRIORITAS WAJIB (HARUS DIPATUHI):
1. Setiap clip HARUS mengandung penjelasan manfaat produk yang jelas.
2. Jika tidak ada manfaat produk → JANGAN dipilih, walaupun lucu atau menarik.
3. Utamakan kalimat yang:
   - Menjelaskan hasil (kulit jadi apa)
   - Menjelaskan perubahan (sebelum vs sesudah)
   - Menjelaskan efek nyata (cerah, glowing, jerawat berkurang, dll)

FILTER KERAS (WAJIB):
- Clip WAJIB mengandung minimal 1 "attention_benefits"
- Clip tanpa manfaat langsung → SKIP
- Promo TANPA manfaat → SKIP
- Humor TANPA manfaat → SKIP
- QnA TANPA manfaat → SKIP

PRIORITAS SKOR (RE-WEIGHTED):
- Penjelasan manfaat produk jelas → 10 (WAJIB ADA)
- Demo + menjelaskan manfaat → 9-10
- Testimoni / hasil nyata → 9-10
- Before-after / perubahan → 9-10
- Penjelasan Vitamin C + efek ke kulit → 8-9

SECONDARY (hanya jika ada benefit):
- Promo + manfaat → 8-9
- QnA + manfaat → 7-8
- Tips + manfaat → 7-8

PENALTI:
- Tidak fokus ke produk → -3
- Tidak ada hasil/manfaat jelas → -5 (AUTO REJECT)
- Terlalu umum / tidak spesifik → -2

DEFINISI "MANFAAT PRODUK" (WAJIB PAHAM):
Manfaat = hasil POSITIF yang dirasakan user setelah pakai produk

Contoh VALID:
- "bikin kulit glowing"
- "jerawat cepat kering"
- "kulit jadi cerah dalam 3 hari"
- "pori-pori terlihat lebih kecil"
- "kulit jadi lembap dan halus"

Contoh TIDAK VALID:
- "ini bagus banget"
- "produk ini viral"
- "lagi promo"

STRATEGI PEMILIHAN:
- Cari kalimat yang mengandung PERUBAHAN (before → after)
- Cari kata kerja hasil: bikin, membuat, membantu, menghilangkan, meredakan
- Cari klaim hasil nyata / cepat

OUTPUT RULE TAMBAHAN:
- Pastikan 80%+ clip yang dipilih adalah "attention_benefits"
- Jika dalam 1 chunk tidak ada benefit kuat → return []

KATEGORI KEYWORD — untuk setiap momen, identifikasi kata kunci dominan dari 3 kategori.

PENTING: Kategorisasi berdasarkan KONTEKS kalimat, bukan hanya kata itu sendiri.
- "jerawat" dalam "meredakan jerawat" atau "ampuh untuk jerawat" = attention_benefits (manfaat)
- "jerawat" dalam "kulit berjerawat parah" atau "masalah jerawat saya" = pain_problem
- "kusam" dalam "bikin kusam hilang" = attention_benefits
- "kusam" dalam "kulit saya kusam banget" = pain_problem
- Selalu baca frasa lengkap sebelum dan sesudah kata kunci sebelum mengkategorikan.

1. "attention_benefits" — manfaat produk, hasil positif, atau masalah yang SUDAH diatasi:
   Contoh: cerah, glowing, putih, bersih, vitamin C, antioksidan, terbaik, rekomendasi,
           meredakan jerawat, menghilangkan flek, kulit lembap, tidak kusam lagi

2. "result_proof" — klaim kecepatan, bukti nyata, atau perbandingan sebelum/sesudah:
   Contoh: 3 hari, 7 hari, langsung, instan, terbukti, hasil nyata, sebelum sesudah, efektif,
           sudah terbukti, bisa dilihat, nyata hasilnya

3. "pain_problem" — masalah kulit yang BELUM diatasi, kondisi negatif yang sedang dialami:
   Contoh: kulit berjerawat, masalah flek, kulit kusam banget, pori-pori besar,
           kering parah, berminyak terus, kulit tidak merata

FORMAT OUTPUT:
Kembalikan HANYA JSON array yang valid. Tidak ada teks lain, tidak ada markdown, tidak ada penjelasan.
Format setiap objek:
{
  "start": <float, detik dari awal video — awal segmen PERTAMA>,
  "end": <float, detik dari awal video — akhir segmen TERAKHIR>,
  "segments": [
    {"start": <float>, "end": <float>, "description": "<isi segmen singkat>"},
    {"start": <float>, "end": <float>, "description": "<isi segmen singkat>"}
  ],
  "score": <float 1-10>,
  "hook": "<headline besar untuk top text TikTok, fokus hasil/perubahan, max 8 kata, dalam Bahasa Indonesia>",
  "reason": "<alasan singkat dalam Bahasa Indonesia>",
  "product": "<nama produk jika disebutkan, atau 'general' jika tidak spesifik>",
  "clip_type": "<demo|testimoni|tips|promo|qna|humor>",
  "keyword_category": "<attention_benefits|result_proof|pain_problem>",
  "keywords_found": [{"word": "<kata kunci>", "category": "<attention_benefits|result_proof|pain_problem>", "context": "<frasa 3-5 kata sekitar kata kunci>"}]
}

PENTING:
- Minimum durasi clip: 15 detik
- Maksimum durasi clip: 60 detik
- Hanya sertakan momen dengan skor >= 6
- Boleh ada beberapa momen yang tumpang tindih jika berbeda konteksnya
- Jika tidak ada momen bagus di chunk ini, kembalikan array kosong: []
- keyword_category: pilih kategori yang PALING dominan di momen tersebut
"""


def _call_lm_studio(client, messages: list, cfg) -> str:
    """Make a single call to LM Studio and return the text response."""
    response = client.chat.completions.create(
        model=cfg.LM_STUDIO_MODEL,
        messages=messages,
        temperature=0.2,       # low temperature = more consistent JSON output
        max_tokens=8192,
        timeout=cfg.LM_STUDIO_TIMEOUT,
    )
    return response.choices[0].message.content.strip()


def _parse_moments_json(raw: str) -> list:
    """Safely parse LLM JSON output. Handles common formatting issues."""
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    # Try direct parse first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "moments" in parsed:
            return parsed["moments"]
    except json.JSONDecodeError:
        pass

    # Try extracting JSON array with regex
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    log.warning(f"Could not parse LLM output as JSON. Raw output:\n{raw[:300]}...")
    return []


def _validate_moment(m: dict, chunk_start: float, chunk_end: float, cfg) -> dict | None:
    """Validate and clean a single moment dict."""
    try:
        start = float(m.get("start", 0))
        end = float(m.get("end", 0))
        score = float(m.get("score", 0))

        # Basic sanity checks
        if end <= start:
            return None
        if score < cfg.MIN_SCORE:
            return None

        duration = end - start
        if duration < cfg.MIN_CLIP_DURATION:
            # Try to expand the clip
            end = start + cfg.MIN_CLIP_DURATION
            duration = cfg.MIN_CLIP_DURATION
        if duration > cfg.MAX_CLIP_DURATION:
            end = start + cfg.MAX_CLIP_DURATION

        # Timestamps must be within or near the chunk
        # Allow ±30s outside chunk (LLM sometimes drifts slightly)
        if start < chunk_start - 30 or start > chunk_end + 30:
            return None

        validated = {
            "start": round(max(0, start - cfg.PAD_START), 2),
            "end": round(end + cfg.PAD_END, 2),
            "score": round(score, 1),
            "hook": str(m.get("hook", "Momen menarik dari livestream PROYA"))[:80],
            "reason": str(m.get("reason", ""))[:150],
            "product": str(m.get("product", "general")),
            "clip_type": str(m.get("clip_type", "general")),
            "keyword_category": str(m.get("keyword_category", "attention_benefits")),
            "keywords_found": m.get("keywords_found", []) if isinstance(m.get("keywords_found"), list) else [],
        }
        hook_overlay = build_hook_payload(validated)
        validated["hook_overlay"] = hook_overlay
        validated["hook"] = hook_overlay["headline"]
        return validated
    except (TypeError, ValueError):
        return None


def detect_moments(chunks: list, working_dir: str, cfg) -> list:
    """
    Run all transcript chunks through LM Studio to find good clip moments.
    Saves results to JSON cache to avoid re-running on crash.
    """
    moments_path = Path(working_dir) / "moments.json"

    if moments_path.exists():
        log.info(f"Loading cached moments from {moments_path}")
        with open(moments_path, "r", encoding="utf-8") as f:
            return json.load(f)

    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    # ── Connect to LM Studio ─────────────────────────────────────────────────
    client = OpenAI(
        base_url=cfg.LM_STUDIO_BASE_URL,
        api_key=cfg.LM_STUDIO_API_KEY,
    )

    log.info(f"Connected to LM Studio at {cfg.LM_STUDIO_BASE_URL}")
    log.info(f"Model: {cfg.LM_STUDIO_MODEL}")
    log.info(f"Processing {len(chunks)} transcript chunks...")

    all_moments = []
    failed_chunks = 0

    for i, chunk in enumerate(chunks):
        log.info(f"  Chunk {i+1}/{len(chunks)} | t={chunk['chunk_start']:.0f}s–{chunk['chunk_end']:.0f}s")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Ini adalah transkrip dari segmen livestream "
                    f"(t={chunk['chunk_start']:.1f}s hingga t={chunk['chunk_end']:.1f}s):\n\n"
                    f"{chunk['text']}\n\n"
                    f"Identifikasi momen bagus dan kembalikan JSON array."
                ),
            },
        ]

        try:
            raw = _call_lm_studio(client, messages, cfg)
            raw_moments = _parse_moments_json(raw)

            valid = 0
            for m in raw_moments:
                validated = _validate_moment(m, chunk["chunk_start"], chunk["chunk_end"], cfg)
                if validated:
                    all_moments.append(validated)
                    valid += 1

            log.info(f"    → {len(raw_moments)} detected, {valid} valid (score≥{cfg.MIN_SCORE})")

        except Exception as e:
            log.error(f"    ✗ LM Studio error on chunk {i+1}: {e}")
            failed_chunks += 1
            if failed_chunks > 5:
                log.error("Too many LM Studio failures. Check that LM Studio is running and a model is loaded.")
                raise
            time.sleep(3)
            continue

    # ── Deduplicate overlapping moments ──────────────────────────────────────
    all_moments = _deduplicate_moments(all_moments)

    # ── Sort by score descending ──────────────────────────────────────────────
    all_moments.sort(key=lambda m: m["score"], reverse=True)

    # ── Assign clip IDs ───────────────────────────────────────────────────────
    for idx, m in enumerate(all_moments):
        m["clip_id"] = f"clip_{idx+1:04d}"

    log.info(f"Total moments found: {len(all_moments)} (from {len(chunks)} chunks)")

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    with open(moments_path, "w", encoding="utf-8") as f:
        json.dump(all_moments, f, ensure_ascii=False, indent=2)

    log.info(f"Moments saved to {moments_path}")
    return all_moments


def _deduplicate_moments(moments: list, overlap_threshold: float = 0.6) -> list:
    """
    Remove moments that overlap too much with a higher-scored moment.
    Uses Intersection over Union (IoU) on time ranges.
    """
    if not moments:
        return []

    moments_sorted = sorted(moments, key=lambda m: m["score"], reverse=True)
    kept = []

    for candidate in moments_sorted:
        c_start, c_end = candidate["start"], candidate["end"]
        c_dur = c_end - c_start

        is_duplicate = False
        for existing in kept:
            e_start, e_end = existing["start"], existing["end"]

            # Calculate overlap
            overlap_start = max(c_start, e_start)
            overlap_end = min(c_end, e_end)
            overlap = max(0, overlap_end - overlap_start)

            union = max(c_end, e_end) - min(c_start, e_start)
            iou = overlap / union if union > 0 else 0

            if iou > overlap_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(candidate)

    log.info(f"Dedup: {len(moments)} → {len(kept)} moments after removing overlaps")
    return kept
