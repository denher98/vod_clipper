import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from main import _build_clip_job, _completed_resume_rows, _render_fingerprint
from variation_profile import default_profile, save_active_profile


class RenderResumeTests(unittest.TestCase):
    def test_completed_resume_rows_skip_failed_and_require_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            raw_dir = output_dir / "raw"
            raw_dir.mkdir()
            ok_output = output_dir / "v1" / "clip_0001.mp4"
            ok_output.parent.mkdir()
            ok_output.write_bytes(b"ok")

            moments = [
                {"clip_id": "clip_0001", "start": 0, "end": 10, "score": 9, "hook": "a"},
                {"clip_id": "clip_0002", "start": 10, "end": 20, "score": 8, "hook": "b"},
                {"clip_id": "clip_0003", "start": 20, "end": 30, "score": 7, "hook": "c"},
            ]
            jobs = [_build_clip_job(moment, index, str(output_dir), raw_dir) for index, moment in enumerate(moments)]
            manifest = [
                {"clip_id": "clip_0001", "status": "ok", "output_file": "v1/clip_0001.mp4"},
                {"clip_id": "clip_0002", "status": "failed", "output_file": "clip_0002.mp4"},
                {"clip_id": "clip_0003", "status": "compliance_blocked", "output_file": "clip_0003.mp4"},
            ]

            rows = _completed_resume_rows(jobs, manifest, output_dir)

            self.assertEqual([row["clip_id"] for row in rows], ["clip_0001", "clip_0003"])

    def test_render_fingerprint_changes_when_variation_profile_revision_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cfg = SimpleNamespace(
                WORKING_DIR=str(root / "working"),
                OUTPUT_DIR=str(root / "output"),
                VARIANTS_PER_CLIP=4,
                OUTPUT_CODEC="h264_nvenc",
                FONT_SUBTITLE="assets/fonts/Montserrat-ExtraBold.ttf",
                FONT_HOOK="assets/fonts/Montserrat-ExtraBold.ttf",
                FONT_HOOK_FALLBACKS=[],
                SUBTITLE_FONT_DIR="assets/fonts",
                BGM_DIR=str(root / "bgm"),
            )
            profile = default_profile(cfg)
            first = save_active_profile(cfg, profile, expected_revision=default_profile(cfg)["revision"])
            first_fp = _render_fingerprint("missing.mp4", cfg, max_clips=None, cut_only=False)

            profile["variants"][0]["highlight_color"] = "#00D4FF"
            second = save_active_profile(cfg, profile, expected_revision=first["revision"])
            second_fp = _render_fingerprint("missing.mp4", cfg, max_clips=None, cut_only=False)
            repeat_fp = _render_fingerprint("missing.mp4", cfg, max_clips=None, cut_only=False)

            self.assertNotEqual(first["revision"], second["revision"])
            self.assertNotEqual(first_fp, second_fp)
            self.assertEqual(second_fp, repeat_fp)
            self.assertEqual(
                second_fp["extra"]["variation_profile_revision"],
                second["revision"],
            )


if __name__ == "__main__":
    unittest.main()
