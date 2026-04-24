from __future__ import annotations

import re
from typing import Any


PAIN_PATTERNS: list[tuple[str, str]] = [
    (r"\bmata panda\b|\blingkaran hitam\b", "MATA PANDA"),
    (r"\bflek hitam\b|\bnoda hitam\b|\bflek\b", "FLEK HITAM"),
    (r"\bbekas jerawat\b", "BEKAS JERAWAT"),
    (r"\bjerawat\b", "JERAWAT"),
    (r"\bberuntusan\b", "BERUNTUSAN"),
    (r"\bkemerahan\b|\bmerah\b", "KEMERAHAN"),
    (r"\bkering\b|\bflaky\b", "KULIT KERING"),
    (r"\bkusam\b", "KUSAM"),
    (r"\bgelap\b", "WAJAH GELAP"),
    (r"\bberminyak\b|\bminyakan\b|\boily\b", "MINYAKAN"),
    (r"\bpori\b", "PORI BESAR"),
    (r"\bkerutan\b|\bgaris halus\b", "GARIS HALUS"),
]

BENEFIT_PATTERNS: list[tuple[str, str]] = [
    (r"\bglow(?:ing)?\b", "GLOWING"),
    (r"\bcerah(?:kan)?\b|\bbright(?:ening)?\b", "CERAH"),
    (r"\bputih\b", "CERAH"),
    (r"\blembap\b|\bmelembap(?:kan)?\b|\bhydrat(?:e|ing)\b", "LEMBAP"),
    (r"\bhalus\b|\bsmooth\b", "HALUS"),
    (r"\bbersih(?:kan)?\b|\bclean\b", "BERSIH"),
    (r"\bkalem\b|\btenang\b|\breda\b", "KALEM"),
    (r"\bsegar\b|\bfresh\b", "FRESH"),
    (r"\bpudar\b|\bmemudar(?:kan)?\b", "LEBIH PUDAR"),
]

PROOF_PATTERNS: list[tuple[str, str]] = [
    (r"\b1x sehari\b", "Dipakai cuma 1x sehari"),
    (r"\b3 hari\b", "Hasil keliatan 3 hari"),
    (r"\b7 hari\b", "Hasil keliatan 7 hari"),
    (r"\b10 hari\b", "Hasil keliatan 10 hari"),
    (r"\blangsung\b|\binstan\b", "Sekali pakai langsung keliatan"),
    (r"\btanpa klinik\b|\btanpa treatment\b", "Tanpa klinik, tanpa treatment mahal"),
    (r"\blow budget\b|\bmurah\b|\bhemat\b", "Low budget"),
]

DEFAULT_SUBTEXTS = [
    "Pakai ini doang",
    "Tanpa treatment mahal",
    "Dipakai rutin tiap hari",
    "Cuma 1 produk ini",
]

DEFAULT_CTAS = [
    "Gimana caranya??",
    "Aku pakai apa??",
    "Serius cuma ini??",
    "Rahasianya di sini??",
]

TIPS_CTAS = [
    "Pakenya gimana??",
    "Step-nya gimana??",
    "Gimana caranya??",
]

PRODUCT_CTAS = [
    "Produknya apa??",
    "Aku pakai apa??",
    "Serius cuma ini??",
]


def _normalize(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def _seed_from_moment(moment: dict[str, Any]) -> str:
    parts = [
        _normalize(moment.get("clip_id")),
        _normalize(moment.get("product")),
        _normalize(moment.get("hook")),
        _normalize(moment.get("reason")),
        _normalize(moment.get("keyword_category")),
    ]
    return "|".join(parts)


def _stable_pick(options: list[str], seed: str) -> str:
    clean_options = [opt for opt in options if _normalize(opt)]
    if not clean_options:
        return ""
    idx = sum(ord(ch) for ch in seed) % len(clean_options)
    return clean_options[idx]


def _find_label(text: str, patterns: list[tuple[str, str]]) -> str | None:
    for pattern, label in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label
    return None


def _dedupe_keep_order(options: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for option in options:
        clean = _normalize(option)
        if not clean:
            continue
        key = clean.upper()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(clean)
    return ordered


def _collect_context(moment: dict[str, Any]) -> str:
    chunks = [
        _normalize(moment.get("hook")),
        _normalize(moment.get("reason")),
        _normalize(moment.get("product")),
        _normalize(moment.get("clip_type")),
        _normalize(moment.get("keyword_category")),
    ]
    for keyword in moment.get("keywords_found", []) or []:
        if isinstance(keyword, dict):
            chunks.append(_normalize(keyword.get("word")))
            chunks.append(_normalize(keyword.get("context")))
    return " | ".join(part for part in chunks if part)


def _fallback_problem(benefit: str) -> str:
    mapping = {
        "GLOWING": "KUSAM",
        "CERAH": "KUSAM",
        "LEMBAP": "KULIT KERING",
        "HALUS": "KASAR",
        "BERSIH": "KUSAM",
        "KALEM": "KEMERAHAN",
        "FRESH": "WAJAH CAPEK",
        "LEBIH PUDAR": "FLEK HITAM",
    }
    return mapping.get(benefit, "KUSAM")


def _infer_problem(moment: dict[str, Any], context: str) -> str:
    direct = _find_label(context, PAIN_PATTERNS)
    if direct:
        return direct

    category = _normalize(moment.get("keyword_category")).lower()
    if category == "pain_problem":
        return "KUSAM"
    return ""


def _infer_benefit(moment: dict[str, Any], context: str) -> str:
    direct = _find_label(context, BENEFIT_PATTERNS)
    if direct:
        return direct

    category = _normalize(moment.get("keyword_category")).lower()
    if category in {"attention_benefits", "result_proof"}:
        return "CERAH"
    return "GLOWING"


def _extract_day_claim(context: str) -> int | None:
    match = re.search(r"\b(\d{1,2})\s*hari\b", context, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _headline_benefit_word(benefit: str) -> str:
    mapping = {
        "LEBIH PUDAR": "SAMAR",
        "FRESH": "SEGAR",
        "KALEM": "TENANG",
    }
    return mapping.get(benefit, benefit)


def _infer_subtext(moment: dict[str, Any], context: str, seed: str) -> str:
    direct = _find_label(context, PROOF_PATTERNS)
    if direct:
        return direct

    if re.search(r"\bdipakai rutin\b|\brutin\b|\bsetiap hari\b", context, flags=re.IGNORECASE):
        return "Dipakai rutin tiap hari"

    clip_type = _normalize(moment.get("clip_type")).lower()
    if clip_type == "demo":
        return "Pakai ini doang"
    if clip_type in {"tips", "qna"}:
        return "Tanpa treatment mahal"
    if _normalize(moment.get("product")).lower() not in {"", "general"}:
        return "Cuma 1 produk ini"
    return _stable_pick(DEFAULT_SUBTEXTS, seed + "|sub")


def _infer_cta(moment: dict[str, Any], seed: str) -> str:
    clip_type = _normalize(moment.get("clip_type")).lower()
    product = _normalize(moment.get("product")).lower()

    if clip_type in {"tips", "qna"}:
        return _stable_pick(TIPS_CTAS, seed + "|cta_tips")
    if product not in {"", "general"}:
        return _stable_pick(PRODUCT_CTAS, seed + "|cta_product")
    return _stable_pick(DEFAULT_CTAS, seed + "|cta_default")


def _build_headline(moment: dict[str, Any], problem: str, benefit: str, seed: str) -> str:
    problem = _normalize(problem)
    benefit = _normalize(benefit)
    base_hook = _normalize(moment.get("hook"))
    context = _collect_context(moment)
    clip_type = _normalize(moment.get("clip_type")).lower()
    product = _normalize(moment.get("product"))
    category = _normalize(moment.get("keyword_category")).lower()
    day_claim = _extract_day_claim(context)
    benefit_word = _headline_benefit_word(benefit)

    if not problem:
        problem = _fallback_problem(benefit)

    headline_options = [
        f"{problem} -> {benefit}",
        f"DARI {problem} JADI {benefit}",
        f"{problem} AUTO {benefit}",
        f"{problem} PARAH JADI {benefit_word}",
        f"AWALNYA {problem}, SEKARANG {benefit_word}",
        f"SEBELUMNYA {problem}, SEKARANG {benefit_word}",
        f"GILA SIH, {problem} JADI {benefit_word}",
        f"KOK BISA {benefit_word} GINI?",
        f"INI BUKAN FILTER, {benefit_word}",
    ]

    if category == "result_proof":
        headline_options.extend([
            f"HASILNYA JADI {benefit}",
            f"KOK BISA SECEPAT INI?",
            f"{benefit_word} CUMA DALAM {day_claim} HARI" if day_claim else "",
            f"{day_claim} HARI BERUBAH TOTAL" if day_claim else "BERUBAH TOTAL",
        ])

    if category == "pain_problem":
        headline_options.extend([
            f"{problem} BISA JADI {benefit}?",
            f"STOP {problem}, JADI {benefit_word}",
        ])

    if clip_type in {"demo", "testimoni"}:
        headline_options.extend([
            f"CUMA PAKAI INI, JADI {benefit}",
            f"AWALNYA GA PERCAYA, EH {benefit}",
        ])

    if clip_type in {"tips", "qna"}:
        headline_options.extend([
            f"TERNYATA {problem} BISA {benefit}",
            f"RAHASIA {benefit} TANPA RIBET",
        ])

    if product and product.lower() not in {"general", ""}:
        headline_options.extend([
            "CUMA 1 PRODUK INI...",
            f"PAKAI {product}, JADI {benefit}",
        ])

    if benefit in {"CERAH", "GLOWING"}:
        headline_options.extend([
            f"{problem} HILANG, {benefit_word} DATANG",
            "MALU KARENA KUSAM? CEK INI",
        ])

    if benefit in {"LEMBAP", "HALUS", "KALEM", "FRESH"}:
        headline_options.extend([
            f"{problem} HILANG, KULIT JADI {benefit}",
            f"BARU PAHAM KENAPA BISA {benefit}",
        ])

    if benefit == "LEBIH PUDAR":
        headline_options.extend([
            f"{problem} MULAI MEMUDAR",
            "NODANYA MAKIN SAMAR",
        ])

    if base_hook:
        cleaned = re.sub(r"[^\w\s?]", " ", base_hook, flags=re.UNICODE)
        cleaned = " ".join(cleaned.upper().split())
        strong_pattern = (
            r"->|^DARI\b|^AWALNYA\b|^SEBELUMNYA\b|^INI BUKAN FILTER\b|"
            r"^KOK BISA\b|^GILA\b|^CUMA\b|^GA PERCAYA\b"
        )
        if 4 <= len(cleaned) <= 34 and re.search(strong_pattern, cleaned, flags=re.IGNORECASE):
            headline_options.append(cleaned)

    headline_options = _dedupe_keep_order([
        option
        for option in headline_options
        if 4 <= len(_normalize(option)) <= 38
    ])

    return _stable_pick(headline_options, seed + "|headline")


def build_hook_payload(moment: dict[str, Any]) -> dict[str, str]:
    seed = _seed_from_moment(moment)
    context = _collect_context(moment)
    benefit = _infer_benefit(moment, context)
    problem = _infer_problem(moment, context)
    headline = _build_headline(moment, problem, benefit, seed)
    subtext = _infer_subtext(moment, context, seed)
    cta = _infer_cta(moment, seed)

    return {
        "headline": _normalize(headline).upper(),
        "subtext": _normalize(subtext),
        "cta": _normalize(cta).upper(),
    }


def ensure_hook_payload(moment: dict[str, Any]) -> dict[str, str]:
    existing = moment.get("hook_overlay")
    if isinstance(existing, dict):
        headline = _normalize(existing.get("headline")).upper()
        subtext = _normalize(existing.get("subtext"))
        cta = _normalize(existing.get("cta")).upper()
        if headline and subtext and cta:
            return {
                "headline": headline,
                "subtext": subtext,
                "cta": cta,
            }

    payload = build_hook_payload(moment)
    moment["hook_overlay"] = payload
    if not _normalize(moment.get("hook")):
        moment["hook"] = payload["headline"]
    return payload
