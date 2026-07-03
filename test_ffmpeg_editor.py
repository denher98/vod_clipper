import types
import unittest

from ffmpeg_editor import (
    _add_before_after_overlay_filters,
    _build_zoom_expressions,
    _cpu_encode_fallback_cmd,
    _hook_layout_settings,
    _letterbox_bar_heights,
    _subtitle_line_centers,
    _variant_hook_format,
)


class FfmpegEditorFallbackTests(unittest.TestCase):
    def test_cpu_encode_fallback_replaces_nvenc_options(self):
        cfg = types.SimpleNamespace(OUTPUT_CRF=24)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            "in.mp4",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-cq",
            "26",
            "-rc",
            "vbr",
            "-b:v",
            "0",
            "-c:a",
            "aac",
            "out.mp4",
        ]

        fallback = _cpu_encode_fallback_cmd(cmd, cfg)

        self.assertIn("libx264", fallback)
        self.assertNotIn("h264_nvenc", fallback)
        self.assertNotIn("-cq", fallback)
        self.assertNotIn("-rc", fallback)
        self.assertNotIn("-b:v", fallback)
        self.assertEqual(fallback[fallback.index("-preset") + 1], "fast")
        self.assertEqual(fallback[fallback.index("-crf") + 1], "24")
        self.assertEqual(fallback[-1], "out.mp4")

    def test_before_after_hook_format_uses_opening_window(self):
        cfg = types.SimpleNamespace(HOOK_DURATION=2.5, BEFORE_AFTER_DURATION=2.5, BEFORE_AFTER_OPACITY=1.0)
        fc = []

        result = _add_before_after_overlay_filters(
            fc=fc,
            vid="[v0]",
            extra_inputs=[{"path": "before.png", "type": "ba"}],
            clip_duration=20.0,
            W=1080,
            H=1920,
            cfg=cfg,
            hook_format="text_before_after_image",
        )

        self.assertEqual(result, "[vba]")
        self.assertTrue(any("between(t,0.00," in item for item in fc))

    def test_visual_hook_format_normalizes_legacy_values_to_text(self):
        cfg = types.SimpleNamespace(_hook_format="pain")

        self.assertEqual(_variant_hook_format(cfg), "text")
        self.assertEqual(_variant_hook_format(cfg, "text_b_roll"), "text_b_roll")

    def test_face_zoom_can_render_without_product_zoom_trigger(self):
        expressions = _build_zoom_expressions(
            prod_trigger=None,
            face_zooms=[{"start": 1.0, "end": 2.5, "cx": 0.5, "cy": 0.3, "scale": 1.25}],
            clip_duration=8.0,
            W=1080,
            H=1920,
            zoom_dur=3.0,
            zoom_scale=1.45,
            timeline_fps=30.0,
        )

        self.assertIsNotNone(expressions)

    def test_letterbox_subtitle_position_honors_explicit_y_fraction(self):
        cfg = types.SimpleNamespace(
            _letterbox_enabled=True,
            _variant_subtitle_position="bottom",
            _variant_subtitle_y_frac=0.50,
            LETTERBOX_BAR_HEIGHT_FRAC=0.20,
        )

        y_line1, y_line2, _line_gap = _subtitle_line_centers(1920, 102, 0.50, cfg)

        self.assertLess(y_line2, int(1920 * 0.65))
        self.assertGreater(y_line1, int(1920 * 0.35))

    def test_letterbox_hook_layout_uses_independent_bar_bands(self):
        cfg = types.SimpleNamespace(
            _letterbox_enabled=True,
            _hook_layout_mode="standard",
            _letterbox_top_frac=0.10,
            _letterbox_bottom_frac=0.30,
        )

        layout = _hook_layout_settings(cfg, 1080, 1920)

        self.assertLess(layout["top_y"], int(1920 * 0.10))
        self.assertLess(layout["mid_y"], int(1920 * 0.10))
        self.assertGreater(layout["bottom_y"], int(1920 * 0.70))

    def test_letterbox_bar_heights_allow_zero_and_clamp_each_side(self):
        cfg = types.SimpleNamespace(
            _letterbox_enabled=True,
            _letterbox_top_frac=0.0,
            _letterbox_bottom_frac=0.80,
        )

        self.assertEqual(_letterbox_bar_heights(1000, cfg), (0, 400))


if __name__ == "__main__":
    unittest.main()
