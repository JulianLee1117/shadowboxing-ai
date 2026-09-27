import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ml.video_timestamps import (
    build_map,
    frame,
    main,
    map_inclusive_bounds,
    parse_ffmpeg,
    parse_ffprobe,
    probe_video,
    time_base,
)


class VideoTimestampTests(unittest.TestCase):
    def mapping(self, pts=(100, 140, 230), duration=None):
        return build_map(
            {"index": 2, "timeBase": time_base("1/1000")},
            [frame(i, p, duration) for i, p in enumerate(pts)],
        )

    def test_vfr_native_origin_and_inclusive_end_preserved(self):
        result = map_inclusive_bounds(self.mapping(), 0, 1)
        self.assertEqual(result["startSeconds"], 0.1)
        self.assertEqual(result["endInclusiveSeconds"], 0.14)
        self.assertEqual(result["endExclusiveSeconds"], 0.23)
        self.assertEqual(result["endExclusiveSource"], "next_decoded_frame_pts")
        self.assertIsNone(result["peakSeconds"])

    def test_last_frame_end_requires_actual_duration(self):
        self.assertIsNone(
            map_inclusive_bounds(self.mapping(), 2, 2)["endExclusiveSeconds"]
        )
        result = map_inclusive_bounds(self.mapping(duration=17), 2, 2)
        self.assertEqual(result["endExclusiveSeconds"], 0.247)
        self.assertEqual(
            result["endExclusiveSource"], "last_decoded_frame_declared_duration"
        )

    def test_invalid_bounds_not_clipped_or_coerced(self):
        for a, b in [(-1, 1), (1, 0), (0, 3), (True, 1), (0.1, 1)]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                map_inclusive_bounds(self.mapping(), a, b)

    def test_missing_or_nonmonotonic_pts_abstains(self):
        for pts in [(0, None, 60), (0, 20, 20), (0, 40, 30)]:
            m = self.mapping(pts)
            self.assertFalse(m["diagnostics"]["validForIntervalMapping"])
            with self.assertRaises(ValueError):
                map_inclusive_bounds(m, 0, 2)

    def test_negative_native_origin_is_not_zeroed(self):
        self.assertEqual(
            map_inclusive_bounds(self.mapping((-40, 0, 30)), 0, 1)["startSeconds"],
            -0.04,
        )

    def test_ffprobe_does_not_replace_missing_pts_with_best_effort(self):
        stream, frames = parse_ffprobe(
            {
                "streams": [
                    {"codec_type": "video", "index": 3, "time_base": "1/90000"}
                ],
                "frames": [
                    {"stream_index": 3, "best_effort_timestamp": 3000},
                    {"stream_index": 3, "pts": "6000", "duration": "3000"},
                ],
            }
        )
        self.assertIsNone(frames[0]["ptsTicks"])
        self.assertEqual(frames[1]["durationTicks"], 3000)
        self.assertEqual(stream["index"], 3)

    def test_ffprobe_wrong_stream_rejected(self):
        with self.assertRaises(ValueError):
            parse_ffprobe(
                {
                    "streams": [
                        {"codec_type": "video", "index": 2, "time_base": "1/1000"}
                    ],
                    "frames": [{"stream_index": 0, "pts": 1}],
                }
            )

    def test_ffmpeg_integer_pts_beat_rounded_text(self):
        log = "Stream #0:2 -> #0:0 (h264)\n[Parsed_showinfo_0] config in time_base: 1/15360, frame_rate: 30/1\n[Parsed_showinfo_0] n: 0 pts: 56315 pts_time:3.666341 duration: 512\n"
        stream, frames = parse_ffmpeg(log)
        self.assertEqual(stream["index"], 2)
        self.assertEqual(frames[0]["ptsTicks"], 56315)
        self.assertEqual(
            build_map(stream, frames)["frames"][0]["ptsSeconds"], 56315 / 15360
        )
        with self.assertRaises(ValueError):
            parse_ffmpeg(log.replace("n: 0", "n: 1"))
        with self.assertRaises(ValueError):
            parse_ffmpeg(log + "[Parsed_showinfo] config in time_base: 1/1000\n")

    def test_invalid_time_base_and_empty_video_rejected(self):
        for value in (None, "0/0", "-1/90000", "nan"):
            with self.assertRaises(ValueError):
                time_base(value)
        with self.assertRaises(ValueError):
            build_map({"timeBase": time_base("1/1000")}, [])

    def test_cli_existing_evidence_is_not_overwritten_or_probed(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "map.json"
            out.write_text("original")
            with patch("ml.video_timestamps.probe_video") as probe, self.assertRaises(
                ValueError
            ):
                main(["missing.mp4", "--ffprobe", "missing", "--output", str(out)])
            probe.assert_not_called()
            self.assertEqual(out.read_text(), "original")

    def fake_probe(self, directory, *, mutate=False, stderr=b"", status=0, limit=100):
        source, binary = Path(directory) / "input.mp4", Path(directory) / "ffprobe"
        source.write_bytes(b"source bytes")
        binary.write_bytes(b"executable bytes")
        payload = {
            "streams": [{"codec_type": "video", "index": 4, "time_base": "1/1000"}],
            "frames": [{"stream_index": 4, "pts": p} for p in [37, 80]],
        }

        def run(command, **kwargs):
            if command[-1] == "-version":
                return SimpleNamespace(stdout="ffprobe fixture\n", returncode=0)
            self.assertEqual(command[command.index("-select_streams") + 1], "v:1")
            kwargs["stdout"].write(json.dumps(payload).encode())
            kwargs["stderr"].write(stderr)
            if mutate:
                source.write_bytes(b"changed video")
            return SimpleNamespace(returncode=status)

        with patch(
            "ml.video_timestamps.resolve_executable", return_value=str(binary)
        ), patch("ml.video_timestamps.subprocess.run", side_effect=run):
            return probe_video(
                source, str(binary), "ffprobe", ordinal=1, max_frames=limit
            )

    def test_probe_provenance_stream_and_nonzero_pts(self):
        with tempfile.TemporaryDirectory() as d:
            result = self.fake_probe(d)
            self.assertEqual(result["stream"]["index"], 4)
            self.assertEqual(result["frames"][0]["ptsSeconds"], 0.037)
            self.assertEqual(result["probe"]["videoStreamOrdinal"], 1)
            self.assertEqual(len(result["source"]["sha256"]), 64)
            self.assertEqual(len(result["toolSourceSha256"]), 64)

    def test_changed_source_incomplete_or_error_probes_fail(self):
        for options in (
            {"mutate": True},
            {"limit": 1},
            {"stderr": b"decode error"},
            {"status": 1},
        ):
            with self.subTest(
                options=options
            ), tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
                self.fake_probe(d, **options)


if __name__ == "__main__":
    unittest.main()
