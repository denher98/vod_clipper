import tempfile
import unittest
from pathlib import Path

from video_queue import EDIT_STAGE, PRE_EDIT_STAGES, STAGES, StageJob, VideoQueueRunner


class VideoQueueSchedulingTests(unittest.TestCase):
    def test_ffmpeg_start_backfills_next_transcribe_before_edit_finishes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input"
            input_dir.mkdir()
            first_video = input_dir / "a.mp4"
            second_video = input_dir / "b.mp4"
            first_video.write_bytes(b"")
            second_video.write_bytes(b"")

            runner = VideoQueueRunner(
                input_dir=str(input_dir),
                state_path=str(temp_path / "state.json"),
                max_retries=0,
                max_inflight_videos=1,
                poll_interval=0.5,
            )
            runner._sync_videos([first_video, second_video])

            first_key = str(first_video.resolve())
            second_key = str(second_video.resolve())
            observed_second_transcribe_statuses = []

            def fake_execute(job):
                with runner.state_lock:
                    observed_second_transcribe_statuses.append(
                        runner.state["videos"][second_key]["stages"]["transcribe"]["status"]
                    )

            runner._execute_stage = fake_execute

            with runner.state_lock:
                first_entry = runner.state["videos"][first_key]
                first_entry["status"] = "queued"
                for stage in PRE_EDIT_STAGES:
                    first_entry["stages"][stage]["status"] = "done"
                first_entry["stages"][EDIT_STAGE]["status"] = "queued"
                first_entry["stages"][EDIT_STAGE]["queued"] = True

                second_entry = runner.state["videos"][second_key]
                second_entry["status"] = "queued"
                for stage in STAGES:
                    second_entry["stages"][stage]["status"] = "pending"
                    second_entry["stages"][stage]["queued"] = False

            runner._run_job("ffmpeg-worker", StageJob(video_path=first_key, stage=EDIT_STAGE))

            self.assertEqual(observed_second_transcribe_statuses, ["queued"])
            with runner.state_lock:
                self.assertEqual(
                    runner.state["videos"][second_key]["stages"]["transcribe"]["status"],
                    "queued",
                )


if __name__ == "__main__":
    unittest.main()
