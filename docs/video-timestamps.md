# Native video timestamps

`ml.video_timestamps` binds zero-based decoded frame indices to the selected video's native presentation timestamps. It uses the Python standard library and an explicitly installed FFmpeg or ffprobe executable. It does not install anything, run a model, modify video, or overwrite existing evidence.

```sh
python3 -m ml.video_timestamps source.mp4 \
  --ffprobe /absolute/path/to/ffprobe --output data/frame-times.json

# Alternative when only FFmpeg is available:
python3 -m ml.video_timestamps source.mp4 \
  --ffmpeg /absolute/path/to/ffmpeg --output data/frame-times.json
```

The output parent directory must exist. `--video-stream 0` selects the first video stream, not necessarily global stream index zero. The manifest records the actual stream index/time base, integer PTS and available native frame durations, source video SHA256, tool-source SHA256, executable SHA256/version, exact command and output hashes. Integer ticks plus the rational time base are authoritative; floating-point seconds are a convenience.

FFmpeg uses full sequential decoding, `-copyts`, `showinfo` and passthrough output. ffprobe reads decoded frame `pts`; a `best_effort_timestamp` is deliberately not substituted. Neither path seeks, shifts the start to zero, interpolates missing times, or reconstructs timestamps from frame index/FPS. This matters for nonzero origins, variable frame rates and reordered compressed packets: indices refer to decoded presentation order, not compressed packet order. See the official [ffprobe](https://ffmpeg.org/ffprobe.html), [timestamp options](https://ffmpeg.org/ffmpeg.html), and [showinfo](https://ffmpeg.org/ffmpeg-filters.html#showinfo) documentation.

Missing/non-increasing PTS are retained as diagnostic failures. Such a map cannot be used by the interval helper. A default 100,000-frame limit, ten-minute decode timeout and 256 MiB probe-output parsing limit bound ordinary inspection; a failed decode or exceeded frame limit produces no successful map. Inspecting an untrusted source does not grant training or redistribution rights.

The FFmpeg path stops on decoding errors (`-xerror`); error-level ffprobe diagnostics also prevent export. Input and executable hashes are checked again after decoding. The ffprobe JSON parser is covered by fixtures; the actual local sample run used FFmpeg 7.1 because a standalone ffprobe was unavailable.

## Inclusive annotation bounds

```python
from ml.video_timestamps import map_inclusive_bounds

interval = map_inclusive_bounds(timestamp_map, start=263, end=278)
```

Bounds must be valid integer frame indices; they are never clipped. The helper returns the PTS of the first and last included frames. A separately named exclusive end uses the next decoded frame's PTS. At the video's last frame, it can use an explicitly decoded positive duration; without one it remains unknown. These are media presentation boundaries, not newly annotated movement boundaries. Peaks remain `None`.

Keep decoder-relative timestamps, browser source time and native PTS separate. A join requires identical source bytes, matching video stream and verified decoded frame indices. Do not apply a guessed constant offset to observations from another decode. Preserve original annotations and write any timestamp-bound version as a derived artifact with source hashes.

The BoxingWeb sample's full map and derived labels remain ignored local artifacts. It has 2,521 decoded frames, time base 1/15360 and first PTS 507 ticks (0.0330078125 seconds). All 18 earlier sampled integer PTS agree. The last PTS step is 100 ms, whereas the other 2,519 steps are 33.333… ms; the decoded final frame's declared end is approximately 84.133 seconds, beyond the container's rounded 84.03-second duration. This is retained as source timing evidence, not silently repaired. All 62 punch intervals map within the decoded coverage. Its source loader uses zero-based inclusive indices; this does not establish a convention for other datasets. No stance, actor identity, punch peak or training eligibility is inferred by this tool.

Run the dependency-free tests with `python3 -m unittest ml.tests.test_video_timestamps -v`.
