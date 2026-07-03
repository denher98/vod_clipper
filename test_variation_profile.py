import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from variation_profile import (
    VariationRevisionConflict,
    active_profile_revision,
    default_profile,
    generate_previews,
    load_active_profile,
    normalize_profile,
    preview_source_ref,
    save_active_profile,
)


class VariationProfileTests(unittest.TestCase):
    def _cfg(self, root: Path):
        return SimpleNamespace(
            WORKING_DIR=str(root / "working"),
            OUTPUT_DIR=str(root / "output"),
            VARIANTS_PER_CLIP=9,
            FONT_SUBTITLE="assets/fonts/Montserrat-ExtraBold.ttf",
            FONT_HOOK="assets/fonts/Montserrat-ExtraBold.ttf",
            FONT_HOOK_FALLBACKS=[],
            SUBTITLE_FONT_DIR="assets/fonts",
            BGM_DIR=str(root / "bgm"),
        )

    def test_default_profile_clamps_count_and_assigns_new_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = self._cfg(Path(temp_dir))

            profile = default_profile(cfg)

            self.assertEqual(profile["schema_version"], 3)
            self.assertEqual(profile["variant_count"], 6)
            self.assertEqual(
                [idx for idx, item in enumerate(profile["variants"]) if item["letterbox_enabled"]],
                [5],
            )
            self.assertEqual(profile["variants"][0]["hook_type"], "text")
            self.assertTrue(profile["variants"][0]["subtitle_enabled"])
            self.assertTrue(profile["variants"][0]["product_zoom_enabled"])
            self.assertEqual(profile["variants"][0]["subtitle_y_frac"], 0.84)
            self.assertEqual(profile["variants"][0]["letterbox_top_frac"], 0.0)
            self.assertEqual(profile["variants"][0]["letterbox_bottom_frac"], 0.0)
            self.assertEqual(profile["variants"][5]["letterbox_top_frac"], 0.20)
            self.assertEqual(profile["variants"][5]["letterbox_bottom_frac"], 0.20)

    def test_schema_v2_payload_migrates_and_clamps_preview_layout_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = self._cfg(Path(temp_dir))

            loaded = normalize_profile(
                {
                    "schema_version": 2,
                    "variant_count": 3,
                    "variants": [
                        {"name": "Top", "subtitle_position": "top", "letterbox_enabled": True},
                        {
                            "name": "Center",
                            "subtitle_position": "center",
                            "subtitle_y_frac": 2.0,
                            "letterbox_enabled": True,
                            "letterbox_top_frac": -1,
                            "letterbox_bottom_frac": 1,
                        },
                        {
                            "name": "Bottom",
                            "subtitle_position": "bottom",
                            "subtitle_y_frac": 0.01,
                            "letterbox_enabled": False,
                        },
                    ],
                },
                cfg,
            )

            self.assertEqual(loaded["schema_version"], 3)
            self.assertEqual(loaded["variants"][0]["subtitle_y_frac"], 0.34)
            self.assertEqual(loaded["variants"][0]["letterbox_top_frac"], 0.20)
            self.assertEqual(loaded["variants"][0]["letterbox_bottom_frac"], 0.20)
            self.assertEqual(loaded["variants"][1]["subtitle_y_frac"], 0.92)
            self.assertEqual(loaded["variants"][1]["letterbox_top_frac"], 0.0)
            self.assertEqual(loaded["variants"][1]["letterbox_bottom_frac"], 0.40)
            self.assertEqual(loaded["variants"][2]["subtitle_y_frac"], 0.08)
            self.assertEqual(loaded["variants"][2]["letterbox_top_frac"], 0.0)
            self.assertEqual(loaded["variants"][2]["letterbox_bottom_frac"], 0.0)

    def test_save_normalizes_and_revision_conflict_protects_updates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = self._cfg(Path(temp_dir))
            profile = default_profile(cfg)
            profile["variant_count"] = 3
            profile["variants"][0]["letterbox_enabled"] = True
            profile["variants"][1]["letterbox_enabled"] = True
            profile["variants"][2]["letterbox_enabled"] = False
            profile["variants"][0]["hook_type"] = "pain"
            profile["variants"][1]["subtitle_enabled"] = False
            profile["variants"][2]["product_zoom_enabled"] = False
            profile["variants"][0]["subtitle_y_frac"] = 0.5
            profile["variants"][1]["letterbox_top_frac"] = 0.12
            profile["variants"][1]["letterbox_bottom_frac"] = 0.28

            saved = save_active_profile(cfg, profile, expected_revision=default_profile(cfg)["revision"])
            loaded = load_active_profile(cfg)

            self.assertEqual(saved["revision"], loaded["revision"])
            self.assertEqual(
                [idx for idx, item in enumerate(loaded["variants"]) if item["letterbox_enabled"]],
                [0, 1],
            )
            self.assertEqual(loaded["variants"][0]["hook_type"], "text")
            self.assertFalse(loaded["variants"][1]["subtitle_enabled"])
            self.assertFalse(loaded["variants"][2]["product_zoom_enabled"])
            self.assertEqual(loaded["variants"][0]["subtitle_y_frac"], 0.5)
            self.assertEqual(loaded["variants"][1]["letterbox_top_frac"], 0.12)
            self.assertEqual(loaded["variants"][1]["letterbox_bottom_frac"], 0.28)
            self.assertEqual(active_profile_revision(cfg), loaded["revision"])

            no_bars = dict(loaded)
            no_bars["variants"] = [
                dict(item, letterbox_enabled=False, letterbox_top_frac=0.0, letterbox_bottom_frac=0.0)
                for item in loaded["variants"]
            ]
            saved_no_bars = save_active_profile(cfg, no_bars, expected_revision=loaded["revision"])
            self.assertEqual(
                [idx for idx, item in enumerate(saved_no_bars["variants"]) if item["letterbox_enabled"]],
                [],
            )
            self.assertNotEqual(saved_no_bars["revision"], loaded["revision"])

            stale = dict(profile)
            stale["variant_count"] = 2
            with self.assertRaises(VariationRevisionConflict):
                save_active_profile(cfg, stale, expected_revision="stale")

    def test_preview_source_is_fixed_asset_and_does_not_scan_latest_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = self._cfg(Path(temp_dir))
            run_dir = Path(cfg.OUTPUT_DIR) / "run_001"
            run_dir.mkdir(parents=True)
            barred = run_dir / "v0_barred.mp4"
            clean = run_dir / "v1_clean.mp4"
            barred.touch()
            clean.touch()
            (run_dir / "manifest.json").write_text(
                """
[
  {
    "clip_id": "clip_0001_v0",
    "output_file": "v0_barred.mp4",
    "status": "ok",
    "letterbox_enabled": true
  },
  {
    "clip_id": "clip_0001_v1",
    "output_file": "v1_clean.mp4",
    "status": "ok",
    "letterbox_enabled": false
  }
]
""".strip(),
                encoding="utf-8",
            )

            fixed_source = Path(temp_dir) / "assets" / "variation_preview" / "raw_cut_preview.mp4"
            with mock.patch("variation_profile.FIXED_PREVIEW_SOURCE", fixed_source):
                source_ref = preview_source_ref(cfg)
                result = generate_previews(cfg, default_profile(cfg))

            self.assertEqual(source_ref["path"], str(fixed_source.resolve()))
            self.assertFalse(source_ref["exists"])
            self.assertEqual(result["source_clip"], str(fixed_source.resolve()))
            self.assertEqual(result["previews"], [])
            self.assertIn("Fixed preview clip", result["message"])


if __name__ == "__main__":
    unittest.main()
