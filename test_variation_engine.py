import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from variation_engine import (
    _apply_variant_timeline_offsets,
    apply_variant_to_cfg,
    expand_moments_with_variants,
    generate_variants,
    VariantConfig,
)
from variation_profile import default_profile, save_active_profile


class VariationGeneratorTests(unittest.TestCase):
    def test_variant_cfg_preserves_base_settings_from_runtime_wrapper(self):
        base_cfg = SimpleNamespace(
            HOOK_DURATION=1.5,
            ZOOM_SCALE=1.45,
            SUBTITLE_Y_POS=0.80,
            FONT_SUBTITLE="assets/fonts/Montserrat-ExtraBold.ttf",
            FONT_HOOK="assets/fonts/Poppins-Bold.ttf",
            BEFORE_AFTER_ENABLED=True,
            BEFORE_AFTER_DIR="assets/before_after",
            KARAOKE_ACTIVE_COLOR="#FFD600",
            KARAOKE_INACTIVE_OPACITY=1.0,
            BROLL_INTRO_ENABLED=False,
        )

        class RuntimeCfg:
            def __init__(self, base):
                self._base = base

            def __getattr__(self, name):
                return getattr(self._base, name)

        runtime_cfg = RuntimeCfg(base_cfg)
        variant = generate_variants(runtime_cfg, 6, seed=42)[1]
        patched = apply_variant_to_cfg(runtime_cfg, variant)

        self.assertTrue(patched.BEFORE_AFTER_ENABLED)
        self.assertEqual(patched.BEFORE_AFTER_DIR, "assets/before_after")
        self.assertEqual(patched.FONT_HOOK, "assets/fonts/Poppins-Bold.ttf")
        self.assertEqual(patched.HOOK_COLOR, variant.hook_color)
        self.assertEqual(patched._variant_archetype, variant.archetype)
        self.assertEqual(patched._hook_layout_mode, variant.hook_layout_mode)
        self.assertEqual(patched._before_after_variant_mode, variant.before_after_variant_mode)

    def test_seeded_six_pack_uses_distinct_visible_styles(self):
        base_cfg = SimpleNamespace(
            HOOK_DURATION=1.5,
            ZOOM_SCALE=1.45,
            SUBTITLE_Y_POS=0.80,
            FONT_SUBTITLE="assets/fonts/Montserrat-ExtraBold.ttf",
            KARAOKE_ACTIVE_COLOR="#FFD600",
            KARAOKE_INACTIVE_OPACITY=1.0,
            BROLL_INTRO_ENABLED=False,
        )

        variants = generate_variants(base_cfg, 6, seed=42)
        repeat_variants = generate_variants(base_cfg, 6, seed=42)

        self.assertEqual(variants, repeat_variants)
        self.assertEqual(variants[0].variant_id, "v0_original")

        mutated = variants[1:]
        self.assertEqual(len(mutated), 5)
        self.assertEqual(len({variant.variant_id for variant in variants}), 6)
        self.assertEqual(
            len({variant.variant_id.split("_", 1)[1] for variant in mutated}),
            len(mutated),
        )

        subtitle_styles = {
            (
                variant.font_subtitle,
                variant.karaoke_active_color,
                variant.karaoke_inactive_opacity,
                variant.subtitle_stroke,
                variant.subtitle_stroke_w,
            )
            for variant in mutated
        }
        hook_styles = {
            (
                variant.hook_color,
                variant.hook_stroke_color,
                variant.hook_stroke_w,
                variant.hook_fontsize_mult,
            )
            for variant in mutated
        }

        self.assertEqual(len(subtitle_styles), len(mutated))
        self.assertEqual(len(hook_styles), len(mutated))

    def test_seeded_six_pack_uses_named_archetype_slots(self):
        base_cfg = SimpleNamespace(
            HOOK_DURATION=1.5,
            ZOOM_SCALE=1.45,
            SUBTITLE_Y_POS=0.80,
            FONT_SUBTITLE="assets/fonts/Montserrat-ExtraBold.ttf",
            KARAOKE_ACTIVE_COLOR="#FFD600",
            KARAOKE_INACTIVE_OPACITY=1.0,
            BROLL_INTRO_ENABLED=False,
        )

        variants = generate_variants(base_cfg, 6, seed=42)

        self.assertEqual(
            [variant.archetype for variant in variants],
            [
                "original",
                "product_broll_open",
                "tight_product_focus",
                "result_overlay",
                "host_focus_fast",
                "clean_commerce",
            ],
        )
        self.assertEqual(
            [variant.variant_id for variant in variants],
            [
                "v0_original",
                "v1_product_broll_open",
                "v2_tight_product_focus",
                "v3_result_overlay",
                "v4_host_focus_fast",
                "v5_clean_commerce",
            ],
        )
        self.assertGreaterEqual(
            len({variant.hook_layout_mode for variant in variants[1:]}),
            4,
        )
        self.assertGreaterEqual(
            len({variant.before_after_variant_mode for variant in variants[1:]}),
            4,
        )

    def test_timeline_offsets_clamp_near_zero_and_keep_valid_duration(self):
        base_cfg = SimpleNamespace(
            HOOK_DURATION=1.5,
            ZOOM_SCALE=1.45,
            SUBTITLE_Y_POS=0.80,
            FONT_SUBTITLE="assets/fonts/Montserrat-ExtraBold.ttf",
            KARAOKE_ACTIVE_COLOR="#FFD600",
            KARAOKE_INACTIVE_OPACITY=1.0,
            BROLL_INTRO_ENABLED=False,
        )
        variant = generate_variants(base_cfg, 6, seed=42)[1]
        moment = {"clip_id": "clip_0001", "start": 0.1, "end": 20.1}

        adjusted = _apply_variant_timeline_offsets(moment, variant)

        self.assertEqual(adjusted["start"], 0.0)
        self.assertGreater(adjusted["end"], adjusted["start"])
        self.assertAlmostEqual(adjusted["end"] - adjusted["start"], 20.0, places=3)

    def test_expanded_moments_include_timeline_offsets(self):
        base_cfg = SimpleNamespace(
            HOOK_DURATION=1.5,
            ZOOM_SCALE=1.45,
            SUBTITLE_Y_POS=0.80,
            FONT_SUBTITLE="assets/fonts/Montserrat-ExtraBold.ttf",
            KARAOKE_ACTIVE_COLOR="#FFD600",
            KARAOKE_INACTIVE_OPACITY=1.0,
            BROLL_INTRO_ENABLED=False,
        )
        moments = [{
            "clip_id": "clip_0001",
            "start": 10.0,
            "end": 40.0,
            "score": 9,
            "hook": "Serum best seller",
            "product": "Serum",
            "selected_text": "pakai serum proya ini",
        }]

        expanded = expand_moments_with_variants(moments, base_cfg, n_variants=6, seed=42)
        by_archetype = {moment["_variant"].archetype: moment for moment in expanded}

        self.assertEqual(by_archetype["original"]["start"], 10.0)
        self.assertLess(by_archetype["product_broll_open"]["start"], 10.0)
        self.assertGreater(by_archetype["host_focus_fast"]["start"], 10.0)
        self.assertTrue(all(moment["end"] > moment["start"] for moment in expanded))

    def test_broll_intro_assets_are_assigned_to_some_expanded_variants(self):
        with TemporaryDirectory() as tmp_dir:
            Path(tmp_dir, "intro_a.mp4").touch()
            Path(tmp_dir, "intro_b.mov").touch()
            base_cfg = SimpleNamespace(
                HOOK_DURATION=1.5,
                ZOOM_SCALE=1.45,
                SUBTITLE_Y_POS=0.80,
                FONT_SUBTITLE="assets/fonts/Montserrat-ExtraBold.ttf",
                KARAOKE_ACTIVE_COLOR="#FFD600",
                KARAOKE_INACTIVE_OPACITY=1.0,
                BROLL_INTRO_ENABLED=True,
                BROLL_INTRO_DIR=tmp_dir,
                BROLL_INTRO_MIN_VARIANT_RATE=0.20,
                BROLL_INTRO_MAX_VARIANT_RATE=0.40,
                BROLL_INTRO_APPLY_TO_ORIGINAL=False,
                BROLL_INTRO_MAX_DURATION=2.5,
                BROLL_INTRO_REQUIRE_PRODUCT_MATCH=False,
            )
            moments = [{
                "clip_id": "clip_0001",
                "start": 0,
                "end": 30,
                "score": 9,
                "hook": "Serum best seller",
                "product": "Serum",
                "selected_text": "pakai serum proya ini",
            }]

            expanded = expand_moments_with_variants(moments, base_cfg, n_variants=6, seed=42)
            broll_variants = [
                moment["_variant"]
                for moment in expanded
                if moment["_variant"].broll_intro_path
            ]

            self.assertGreaterEqual(len(broll_variants), 1)
            self.assertLessEqual(len(broll_variants), 2)
            self.assertFalse(expanded[0]["_variant"].broll_intro_path)
            self.assertTrue(all("_broll" in variant.variant_id for variant in broll_variants))
            self.assertTrue(
                all(Path(variant.broll_intro_path).parent == Path(tmp_dir) for variant in broll_variants)
            )

    def test_broll_intro_slot_varies_by_base_clip(self):
        with TemporaryDirectory() as tmp_dir:
            Path(tmp_dir, "intro_a.mp4").touch()
            Path(tmp_dir, "intro_b.mp4").touch()
            base_cfg = SimpleNamespace(
                HOOK_DURATION=1.5,
                ZOOM_SCALE=1.45,
                SUBTITLE_Y_POS=0.80,
                FONT_SUBTITLE="assets/fonts/Montserrat-ExtraBold.ttf",
                KARAOKE_ACTIVE_COLOR="#FFD600",
                KARAOKE_INACTIVE_OPACITY=1.0,
                BROLL_INTRO_ENABLED=True,
                BROLL_INTRO_DIR=tmp_dir,
                BROLL_INTRO_MIN_VARIANT_RATE=0.20,
                BROLL_INTRO_MAX_VARIANT_RATE=0.20,
                BROLL_INTRO_APPLY_TO_ORIGINAL=False,
                BROLL_INTRO_MAX_DURATION=2.5,
                BROLL_INTRO_REQUIRE_PRODUCT_MATCH=False,
            )
            moments = [
                {
                    "clip_id": f"clip_{idx:04d}",
                    "start": idx * 40,
                    "end": idx * 40 + 30,
                    "score": 9,
                    "hook": "Serum best seller",
                    "product": "Serum",
                    "selected_text": "pakai serum proya ini",
                }
                for idx in range(1, 12)
            ]

            expanded = expand_moments_with_variants(moments, base_cfg, n_variants=6, seed=42)
            broll_slots = {
                moment["clip_id"].split("_v", 1)[0]: moment["_variant"].variant_index
                for moment in expanded
                if moment["_variant"].broll_intro_path
            }

            self.assertEqual(len(broll_slots), len(moments))
            self.assertGreater(len(set(broll_slots.values())), 1)
            self.assertNotIn(0, broll_slots.values())

    def test_product_broll_intro_uses_matching_product_folder(self):
        with TemporaryDirectory() as tmp_dir:
            serum_dir = Path(tmp_dir, "Serum")
            toner_dir = Path(tmp_dir, "Toner")
            serum_dir.mkdir()
            toner_dir.mkdir()
            Path(serum_dir, "serum_intro.mp4").touch()
            Path(toner_dir, "toner_intro.mp4").touch()
            base_cfg = SimpleNamespace(
                HOOK_DURATION=1.5,
                ZOOM_SCALE=1.45,
                SUBTITLE_Y_POS=0.80,
                FONT_SUBTITLE="assets/fonts/Montserrat-ExtraBold.ttf",
                KARAOKE_ACTIVE_COLOR="#FFD600",
                KARAOKE_INACTIVE_OPACITY=1.0,
                PRODUCT_CLASSES={3: "Serum", 5: "Toner"},
                BROLL_INTRO_ENABLED=True,
                BROLL_INTRO_DIR=tmp_dir,
                BROLL_INTRO_MIN_VARIANT_RATE=0.40,
                BROLL_INTRO_MAX_VARIANT_RATE=0.40,
                BROLL_INTRO_APPLY_TO_ORIGINAL=False,
                BROLL_INTRO_MAX_DURATION=2.5,
                BROLL_INTRO_REQUIRE_PRODUCT_MATCH=True,
                BROLL_INTRO_ALLOW_GENERIC_ROOT=False,
                BROLL_INTRO_PRODUCT_ALIASES={"Serum": ["serum"], "Toner": ["toner"]},
            )
            moments = [{
                "clip_id": "clip_0001",
                "start": 0,
                "end": 30,
                "score": 9,
                "hook": "Serum best seller",
                "product": "Serum",
                "selected_text": "pakai serum proya ini",
            }]

            expanded = expand_moments_with_variants(moments, base_cfg, n_variants=6, seed=42)
            broll_variants = [
                moment["_variant"]
                for moment in expanded
                if moment["_variant"].broll_intro_path
            ]

            self.assertGreaterEqual(len(broll_variants), 1)
            self.assertTrue(
                all(Path(variant.broll_intro_path).parent == serum_dir for variant in broll_variants)
            )
            self.assertTrue(all(variant.broll_intro_product == "serum" for variant in broll_variants))

    def test_active_variation_profile_drives_expansion(self):
        with TemporaryDirectory() as tmp_dir:
            cfg = SimpleNamespace(
                WORKING_DIR=str(Path(tmp_dir) / "working"),
                OUTPUT_DIR=str(Path(tmp_dir) / "output"),
                VARIANTS_PER_CLIP=6,
                HOOK_DURATION=1.5,
                ZOOM_SCALE=1.45,
                SUBTITLE_Y_POS=0.80,
                FONT_SUBTITLE="assets/fonts/Montserrat-ExtraBold.ttf",
                FONT_HOOK="assets/fonts/Montserrat-ExtraBold.ttf",
                FONT_HOOK_FALLBACKS=[],
                KARAOKE_ACTIVE_COLOR="#FFD600",
                KARAOKE_INACTIVE_OPACITY=1.0,
                BROLL_INTRO_ENABLED=False,
                BGM_DIR=str(Path(tmp_dir) / "bgm"),
            )
            profile = default_profile(cfg)
            profile["variant_count"] = 2
            profile["variants"][0]["name"] = "Clean Control"
            profile["variants"][0]["hook_type"] = "text_b_roll"
            profile["variants"][0]["subtitle_enabled"] = False
            profile["variants"][1]["name"] = "Bar Variant"
            profile["variants"][1]["letterbox_enabled"] = True
            profile["variants"][1]["letterbox_top_frac"] = 0.11
            profile["variants"][1]["letterbox_bottom_frac"] = 0.27
            profile["variants"][1]["subtitle_y_frac"] = 0.57
            profile["variants"][1]["zoom_intensity"] = "none"
            profile["variants"][1]["product_zoom_enabled"] = False
            saved = save_active_profile(cfg, profile, expected_revision=default_profile(cfg)["revision"])
            moments = [{
                "clip_id": "clip_0001",
                "start": 10.0,
                "end": 40.0,
                "score": 9,
                "hook": "Serum best seller",
                "product": "Serum",
                "selected_text": "pakai serum proya ini",
            }]

            expanded = expand_moments_with_variants(moments, cfg, n_variants=6, seed=42)

            self.assertEqual(len(expanded), 2)
            self.assertEqual(expanded[0]["_variant"].display_name, "Clean Control")
            self.assertEqual(expanded[0]["_variant"].hook_type, "text_b_roll")
            self.assertFalse(expanded[0]["_variant"].subtitle_enabled)
            self.assertEqual(expanded[0]["_variant"].profile_revision, saved["revision"])
            self.assertTrue(expanded[1]["_variant"].letterbox_enabled)
            self.assertEqual(expanded[1]["_variant"].letterbox_top_frac, 0.11)
            self.assertEqual(expanded[1]["_variant"].letterbox_bottom_frac, 0.27)
            self.assertEqual(expanded[1]["_variant"].subtitle_y_frac, 0.57)
            self.assertEqual(expanded[1]["_variant"].zoom_intensity, "none")
            self.assertFalse(expanded[1]["_variant"].product_zoom_enabled)

    def test_apply_variant_to_cfg_sets_profile_render_overrides(self):
        base_cfg = SimpleNamespace(
            BGM_ENABLED=True,
            SFX_ENABLED=True,
            HOOK_DURATION=1.5,
            HOOK_FONTSIZE=100,
            ZOOM_SCALE=1.45,
        )
        variant = VariantConfig(
            variant_id="v1_test",
            variant_index=1,
            font_subtitle="assets/fonts/Anton-Regular.ttf",
            subtitle_base_color="#EFEFEF",
            karaoke_active_color="#00D4FF",
            hook_color="#EFEFEF",
            highlight_color="#00D4FF",
            bgm_mode="selected",
            bgm_path="assets/bgm/focus.mp3",
            sfx_enabled=False,
            zoom_intensity="none",
            product_zoom_enabled=False,
            subtitle_enabled=False,
            letterbox_enabled=True,
            subtitle_y_frac=0.57,
            letterbox_top_frac=0.11,
            letterbox_bottom_frac=0.27,
            hook_type="text_before_after_image",
        )

        patched = apply_variant_to_cfg(base_cfg, variant)

        self.assertEqual(patched.FONT_SUBTITLE, "assets/fonts/Anton-Regular.ttf")
        self.assertEqual(patched.SUBTITLE_BASE_COLOR, "#EFEFEF")
        self.assertEqual(patched.KARAOKE_ACTIVE_COLOR, "#00D4FF")
        self.assertTrue(patched.BGM_ENABLED)
        self.assertEqual(patched._bgm_path, "assets/bgm/focus.mp3")
        self.assertFalse(patched.SFX_ENABLED)
        self.assertTrue(patched._zoom_disabled)
        self.assertFalse(patched._product_zoom_enabled)
        self.assertFalse(patched._subtitle_enabled)
        self.assertTrue(patched._letterbox_enabled)
        self.assertEqual(patched._variant_subtitle_y_frac, 0.57)
        self.assertEqual(patched._letterbox_top_frac, 0.11)
        self.assertEqual(patched._letterbox_bottom_frac, 0.27)
        self.assertEqual(patched._hook_format, "text_before_after_image")


if __name__ == "__main__":
    unittest.main()
