"""Read native decoded video PTS with an explicitly installed FFmpeg/ffprobe.

No dependencies beyond the standard library; no downloads, FPS arithmetic,
best-effort timestamp substitution, annotation changes or model inference.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

VERSION = "video-native-timestamps-1"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def integer(value, field, *, missing=False):
    if missing and value in (None, "N/A", "NOPTS"):
        return None
    if isinstance(value, bool) or not re.fullmatch(r"-?\d+", str(value)):
        raise ValueError(f"Invalid integer {field}: {value!r}")
    return int(value)


def time_base(value):
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError, TypeError) as exc:
        raise ValueError("Invalid stream time base") from exc
    if result <= 0:
        raise ValueError("Stream time base must be positive")
    return {"numerator": result.numerator, "denominator": result.denominator}


def frame(index, pts, duration=None):
    return {
        "frameIndex0Based": index,
        "ptsTicks": integer(pts, "pts", missing=True),
        "durationTicks": integer(duration, "duration", missing=True),
    }


def parse_ffprobe(payload):
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError("Probe must select exactly one video stream")
    stream = streams[0]
    if stream.get("codec_type") != "video":
        raise ValueError("Selected stream is not video")
    base = time_base(stream.get("time_base"))
    index = integer(stream.get("index"), "stream index")
    frames = []
    for item in payload.get("frames", []):
        if integer(item.get("stream_index"), "frame stream index") != index:
            raise ValueError("Frame belongs to a different stream")
        # best_effort_timestamp is deliberately not a substitute for missing PTS.
        frames.append(
            frame(
                len(frames),
                item.get("pts"),
                item.get("duration", item.get("pkt_duration")),
            )
        )
    return {
        "index": index,
        "timeBase": base,
        "metadata": stream,
    }, frames


def parse_ffmpeg(log):
    bases = re.findall(r"config in time_base:\s*(\d+/\d+)", log)
    if not bases or len(set(bases)) != 1:
        raise ValueError("Missing or changing decoded stream time base")
    mapping = re.findall(r"Stream #0:(\d+)(?:\([^)]*\))? -> #0:\d+", log)
    if len(mapping) != 1:
        raise ValueError("Expected one explicitly mapped video stream")
    frames = []
    for line in log.splitlines():
        if "showinfo" not in line or not re.search(r"\bn:\s*\d+\s+pts:", line):
            continue
        match = re.search(r"\bn:\s*(\d+)\s+pts:\s*(\S+)\s+pts_time:", line)
        if not match:
            raise ValueError("Malformed decoded frame timestamp")
        n = int(match[1])
        if n != len(frames):
            raise ValueError("Decoded frame indices are not contiguous from zero")
        duration = re.search(r"\bduration:\s*(\S+)", line)
        frames.append(frame(n, match[2], duration[1] if duration else None))
    return {"index": int(mapping[0]), "timeBase": time_base(bases[0])}, frames


def build_map(stream, frames):
    if not frames:
        raise ValueError("No decoded video frames")
    base = Fraction(stream["timeBase"]["numerator"], stream["timeBase"]["denominator"])
    missing, backwards, invalid_duration = [], [], []
    result = []
    for i, item in enumerate(frames):
        if item["frameIndex0Based"] != i:
            raise ValueError("Frame indices must be contiguous from zero")
        pts = integer(item["ptsTicks"], "pts", missing=True)
        duration = integer(item.get("durationTicks"), "duration", missing=True)
        if pts is None:
            missing.append(i)
        if (
            i
            and pts is not None
            and frames[i - 1]["ptsTicks"] is not None
            and pts <= frames[i - 1]["ptsTicks"]
        ):
            backwards.append([i - 1, i])
        if duration is not None and duration <= 0:
            invalid_duration.append(i)
        result.append(
            {**item, "ptsSeconds": None if pts is None else float(pts * base)}
        )
    return {
        "schemaVersion": VERSION,
        "stream": stream,
        "frameCount": len(result),
        "frames": result,
        "diagnostics": {
            "missingPtsFrameIndices": missing,
            "nonIncreasingPtsTransitions": backwards,
            "nonpositiveDurationFrameIndices": invalid_duration,
            "validForIntervalMapping": not missing and not backwards,
        },
    }


def map_inclusive_bounds(mapping, start, end):
    """Bind inclusive indices to observed PTS, never infer a punch peak."""
    start, end = integer(start, "start frame"), integer(end, "end frame")
    frames = mapping["frames"]
    if start < 0 or end < start or end >= len(frames):
        raise ValueError("Inclusive frame bounds are outside the decoded video")
    if not mapping["diagnostics"]["validForIntervalMapping"]:
        raise ValueError("Timestamp anomalies must be resolved before interval mapping")
    base = Fraction(
        mapping["stream"]["timeBase"]["numerator"],
        mapping["stream"]["timeBase"]["denominator"],
    )
    a, b = frames[start]["ptsTicks"], frames[end]["ptsTicks"]
    exclusive, method = None, None
    if end + 1 < len(frames):
        exclusive, method = frames[end + 1]["ptsTicks"], "next_decoded_frame_pts"
    elif (
        frames[end].get("durationTicks") is not None
        and frames[end]["durationTicks"] > 0
    ):
        exclusive = b + frames[end]["durationTicks"]
        method = "last_decoded_frame_declared_duration"
    return {
        "startFrameIndex0Based": start,
        "endFrameIndex0BasedInclusive": end,
        "startPtsTicks": a,
        "endInclusivePtsTicks": b,
        "startSeconds": float(a * base),
        "endInclusiveSeconds": float(b * base),
        "endExclusivePtsTicks": exclusive,
        "endExclusiveSeconds": None if exclusive is None else float(exclusive * base),
        "endExclusiveSource": method,
        "peakFrame": None,
        "peakSeconds": None,
    }


def resolve_executable(value):
    executable = shutil.which(value)
    if executable is None:
        raise ValueError(
            f"Executable not found: {value}. Supply an installed binary; nothing is downloaded."
        )
    return str(Path(executable).resolve())


def probe_video(video, executable, engine, ordinal=0, max_frames=100000, timeout=600):
    source = Path(video).resolve()
    if not source.is_file():
        raise ValueError("Input must be a local regular video file")
    if ordinal < 0 or not 1 <= max_frames <= 1000000 or not 1 <= timeout <= 3600:
        raise ValueError("Invalid stream, frame limit or timeout")
    source_hash = digest(source)
    tool_hash = digest(__file__)
    executable = resolve_executable(executable)
    binary_hash = digest(executable)
    version = subprocess.run(
        [executable, "-version"], check=True, capture_output=True, text=True, timeout=10
    ).stdout.splitlines()[0]
    if engine == "ffprobe":
        command = [
            executable,
            "-v",
            "error",
            "-select_streams",
            f"v:{ordinal}",
            "-show_streams",
            "-show_frames",
            "-show_entries",
            "stream=index,codec_type,codec_name,time_base,start_pts,duration_ts,nb_frames,width,height:frame=stream_index,pts,duration,pkt_duration",
            "-of",
            "json",
            str(source),
        ]
    elif engine == "ffmpeg":
        command = [
            executable,
            "-nostdin",
            "-hide_banner",
            "-nostats",
            "-xerror",
            "-copyts",
            "-i",
            str(source),
            "-map",
            f"0:v:{ordinal}",
            "-vf",
            "showinfo",
            "-fps_mode",
            "passthrough",
            "-frames:v",
            str(max_frames + 1),
            "-an",
            "-sn",
            "-dn",
            "-f",
            "null",
            "-",
        ]
    else:
        raise ValueError("Unsupported probe engine")
    with tempfile.TemporaryDirectory(prefix="corner-video-pts-") as directory:
        stdout, stderr = Path(directory) / "stdout", Path(directory) / "stderr"
        with stdout.open("wb") as out, stderr.open("wb") as err:
            completed = subprocess.run(
                command, stdout=out, stderr=err, timeout=timeout, check=False
            )
        if completed.returncode:
            raise ValueError(
                f"Video decode failed ({completed.returncode}): {stderr.read_text(errors='replace')[-2000:]}"
            )
        if max(stdout.stat().st_size, stderr.stat().st_size) > 256 * 1024 * 1024:
            raise ValueError("Probe output exceeds 256 MiB; no mapping written")
        if engine == "ffprobe" and stderr.stat().st_size:
            raise ValueError("ffprobe reported decoding errors; no mapping exported")
        stream, frames = (
            parse_ffprobe(json.loads(stdout.read_text()))
            if engine == "ffprobe"
            else parse_ffmpeg(stderr.read_text())
        )
        if len(frames) > max_frames:
            raise ValueError("Frame limit exceeded; incomplete mapping is not exported")
        result = build_map(stream, frames)
        result["probe"] = {
            "engine": engine,
            "executable": executable,
            "executableSha256": binary_hash,
            "version": version,
            "command": command,
            "stdoutSha256": digest(stdout),
            "stderrSha256": digest(stderr),
            "completed": True,
            "videoStreamOrdinal": ordinal,
        }
    if (
        digest(source) != source_hash
        or digest(executable) != binary_hash
        or digest(__file__) != tool_hash
    ):
        raise ValueError("Input, executable or tool source changed during decoding")
    result["source"] = {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": source_hash,
    }
    result["toolSourceSha256"] = tool_hash
    result["createdAt"] = datetime.now(timezone.utc).isoformat()
    result["timestampPolicy"] = (
        "Decoded native PTS; zero-based presentation-order frames. No seek, FPS reconstruction, start-at-zero shift or best-effort substitution. Decoder-relative timestamps are separate evidence."
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--output", required=True)
    engines = parser.add_mutually_exclusive_group(required=True)
    engines.add_argument("--ffprobe", help="Installed ffprobe executable")
    engines.add_argument(
        "--ffmpeg", help="Installed FFmpeg executable (copyts/showinfo)"
    )
    parser.add_argument(
        "--video-stream", type=int, default=0, help="Zero-based video stream ordinal"
    )
    parser.add_argument("--max-frames", type=int, default=100000)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise ValueError("Output exists; refusing to overwrite timestamp evidence")
    result = probe_video(
        args.video,
        args.ffprobe or args.ffmpeg,
        "ffprobe" if args.ffprobe else "ffmpeg",
        args.video_stream,
        args.max_frames,
        args.timeout_seconds,
    )
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "frameCount": result["frameCount"],
                "diagnostics": result["diagnostics"],
            }
        )
    )


if __name__ == "__main__":
    main()
