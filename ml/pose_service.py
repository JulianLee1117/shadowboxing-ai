"""Optional loopback-only RTMPose/RTMW service; images remain in memory.

Start explicitly: python -m ml.pose_service --manifest LOCAL.json
--minimum-score 0.5 --provider coreml. The cutoff is an experimental policy,
not calibrated visibility. No model downloads or camera access occur here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
import json
import math
from pathlib import Path
import secrets
import socket
import tempfile
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

from .extract import (
    configure_detector_score_policy,
    detector_postprocessing,
    load_manifest,
    native_keypoints,
    provider_configuration,
    session_providers,
    summarize_execution_profile,
    verify_coreml_execution,
)
from .landmarks import COCO_TO_MEDIAPIPE

PROTOCOL_VERSION = "local-pose-1"
LIMITS = {
    "maximumImageBytes": 3_000_000,
    "maximumPixels": 921_600,
    "maximumDimension": 2048,
    "maximumSessions": 4,
    "idleSessionSeconds": 60,
    "maximumSessionSeconds": 1800,
    "inferenceConcurrency": 1,
    "queueCapacity": 0,
}
DEFAULT_ORIGINS = ("http://127.0.0.1:5173", "http://localhost:5173")


def configure_service_coreml(model, manifest, ort, configuration, directory):
    """Keep YOLOX dynamic NMS on CPU; pose retains normal CoreML partitioning."""
    by_model = {"detector": [{"RequireStaticInputShapes": "1"}, {}], "pose": [{}, {}]}
    configuration["providerOptionsByModel"] = by_model
    configuration.pop("providerOptions", None)
    for key, attribute in (("detector", "det_model"), ("pose", "pose_model")):
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str((directory / key).resolve())
        session = ort.InferenceSession(
            manifest[key]["path"],
            sess_options=options,
            providers=configuration["providers"],
            provider_options=by_model[key],
        )
        session.disable_fallback()
        if session.get_providers() != configuration["providers"]:
            raise ValueError(
                f"{key} did not register the requested execution providers"
            )
        getattr(model, attribute).session = session


class RequestError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Read JPEG dimensions before allocation; do not decode an oversized image."""
    if not data.startswith(b"\xff\xd8"):
        raise RequestError(415, "Only JPEG image bytes are accepted")
    offset = 2
    while offset < len(data):
        if data[offset] != 0xFF:
            raise RequestError(400, "Malformed JPEG header")
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in (0xD9, 0xDA):
            break
        if marker in (0x01, *range(0xD0, 0xD8)):
            continue
        if offset + 2 > len(data):
            break
        size = int.from_bytes(data[offset : offset + 2], "big")
        if size < 2 or offset + size > len(data):
            break
        if marker in (
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        ):
            if size < 8:
                break
            height = int.from_bytes(data[offset + 3 : offset + 5], "big")
            width = int.from_bytes(data[offset + 5 : offset + 7], "big")
            if (
                width <= 0
                or height <= 0
                or max(width, height) > LIMITS["maximumDimension"]
                or width * height > LIMITS["maximumPixels"]
            ):
                raise RequestError(
                    413, "Image dimensions exceed the local service limit"
                )
            return width, height
        offset += size
    raise RequestError(400, "JPEG dimensions are missing or malformed")


class PoseBackend:
    """A persistent local model pair. No pixels, results or profiles accumulate."""

    def __init__(self, manifest_path: Path, provider: str, minimum_score: float):
        if not math.isfinite(minimum_score) or not 0 < minimum_score < 1:
            raise ValueError(
                "minimum-score must be finite and strictly between zero and one"
            )
        import cv2
        import numpy as np
        import onnxruntime as ort
        from rtmlib import RTMPose, YOLOX

        if version("rtmlib") != "0.0.16":
            raise ValueError("Install the pinned requirements-extract.txt environment")
        manifest = load_manifest(manifest_path)
        configuration = provider_configuration(provider, ort.get_available_providers())
        model = SimpleNamespace(
            det_model=YOLOX(
                manifest["detector"]["path"],
                model_input_size=tuple(manifest["detector"]["inputSize"]),
                backend="onnxruntime",
                device="cpu",
            ),
            pose_model=RTMPose(
                manifest["pose"]["path"],
                model_input_size=tuple(manifest["pose"]["inputSize"]),
                to_openpose=False,
                backend="onnxruntime",
                device="cpu",
            ),
        )
        configure_detector_score_policy(model.det_model)
        profiles = None
        if provider == "coreml":
            runtime = (
                Path(__file__).resolve().parents[1] / "data" / "local-pose-runtime"
            )
            runtime.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="warmup-", dir=runtime
            ) as directory:
                configure_service_coreml(
                    model, manifest, ort, configuration, Path(directory)
                )
                try:
                    blank = np.zeros((640, 640, 3), dtype=np.uint8)
                    model.det_model(blank)
                    # Exercise pose even when the blank image has no person.
                    # Warmup observations are never returned as real detections.
                    model.pose_model(
                        blank, bboxes=np.array([[0, 0, 640, 640]], dtype=np.float32)
                    )
                finally:
                    profiles = {
                        key: summarize_execution_profile(
                            Path(getattr(model, attribute).session.end_profiling())
                        )
                        for key, attribute in (
                            ("detector", "det_model"),
                            ("pose", "pose_model"),
                        )
                    }
                verify_coreml_execution(profiles)
        self.cv2, self.np, self.model = cv2, np, model
        self.family = manifest["family"]
        self.color_order = manifest["pose"].get("inputColorOrder", "BGR")
        model_id = "rtmpose-m" if self.family == "rtmpose-body" else "rtmw-l"
        # Restrict these public identities to the specific supported checkpoints.
        expected_name = "rtmpose-m_" if model_id == "rtmpose-m" else "rtmw-dw-x-l_"
        if expected_name not in manifest["pose"]["sourceUrl"]:
            raise ValueError(
                "This service supports the fingerprinted RTMPose-M or RTMW-L checkpoints only"
            )
        self.estimator = {
            "id": model_id,
            "scoreType": "simcc",
            "minimumScore": minimum_score,
        }
        self.model_info = {
            "id": model_id,
            "family": self.family,
            "backend": "onnxruntime",
            "delegate": "CoreML+CPU" if provider == "coreml" else "CPU",
            "estimator": self.estimator,
            "confidenceSemantics": "native_SimCC_score_not_MediaPipe_visibility_or_calibrated_correctness",
            "confidencePolicy": "explicit_experimental_unvalidated_threshold",
            "modelManifest": {
                key: {
                    field: manifest[key].get(field)
                    for field in ("sourceUrl", "sha256", "inputSize", "inputColorOrder")
                }
                for key in ("detector", "pose")
            },
            "sessionProviders": session_providers(model),
            "providerConfiguration": configuration,
            "warmupExecution": (
                {
                    key: {
                        field: profile[field]
                        for field in (
                            "sha256",
                            "executedNodeProviderEvents",
                            "interpretation",
                        )
                    }
                    for key, profile in profiles.items()
                }
                if profiles
                else None
            ),
            "profilingDuringFrames": False,
            "packages": {
                key: version(key)
                for key in ("rtmlib", "onnxruntime", "numpy", "opencv-python")
            },
            "detectorPostprocessing": detector_postprocessing(model),
            "mapping": "COCO body indices mapped to partial MediaPipe33; score only; no depth",
            "personPolicy": "exactly_one_detector_box_otherwise_abstain",
            "transport": "loopback_JPEG_quality_0.95_unmirrored_pixels",
        }
        self.recognizer_bundle = None

    def attach_recognizer(self, manifest_path: Path):
        """Explicit local opt-in; incompatible weights abort startup, never fallback."""
        from .recognizer import ModelBundle

        bundle = ModelBundle.load(manifest_path, self.model_info)
        self.recognizer_bundle = bundle
        self.model_info["recognizer"] = bundle.model_info

    def infer(self, data: bytes, timestamp: float, dimensions: tuple[int, int]) -> dict:
        started = time.perf_counter()
        pixels = self.cv2.imdecode(
            self.np.frombuffer(data, dtype=self.np.uint8), self.cv2.IMREAD_COLOR
        )
        if pixels is None:
            raise RequestError(400, "JPEG could not be decoded")
        height, width = pixels.shape[:2]
        if (width, height) != dimensions:
            raise RequestError(400, "Decoded dimensions do not match JPEG header")
        decoded = time.perf_counter()
        boxes = self.model.det_model(pixels)
        detected = time.perf_counter()
        people = len(boxes)
        points, scores = None, None
        if people == 1:
            pose_pixels = pixels[:, :, ::-1] if self.color_order == "RGB" else pixels
            all_points, all_scores = self.model.pose_model(pose_pixels, bboxes=boxes)
            if len(all_points) != 1 or len(all_scores) != 1:
                raise RuntimeError(
                    "Pose output must contain exactly one detected person"
                )
            points, scores = all_points[0], all_scores[0]
        posed = time.perf_counter()
        inference_ms = (posed - decoded) * 1000
        landmarks, native = [], []
        if people == 1:
            expected = 17 if self.family == "rtmpose-body" else 133
            if len(points) != expected:
                raise RuntimeError("Model output point order does not match manifest")
            native = native_keypoints(points, scores, width, height)
            landmarks = [{"x": 0.0, "y": 0.0, "score": 0.0} for _ in range(33)]
            for source, target in COCO_TO_MEDIAPIPE.items():
                point = native[source]
                if point is not None:
                    landmarks[target] = dict(point)
        return {
            "frame": {
                "t": timestamp,
                "width": width,
                "height": height,
                "landmarks": landmarks,
                "inferenceMs": inference_ms,
                "estimator": self.estimator,
            },
            "nativeKeypoints": native,
            "personCount": people,
            "detectorObservations": getattr(
                self.model.det_model, "last_detection_observations", []
            ),
            "timing": {
                "decodeMs": (decoded - started) * 1000,
                "detectorMs": (detected - decoded) * 1000,
                "poseMs": (posed - detected) * 1000,
                "serviceMs": (time.perf_counter() - started) * 1000,
            },
        }


@dataclass
class SessionState:
    token: str
    origin: str
    created: float
    touched: float
    last_timestamp: float = -1.0
    recognizer: object | None = None


class PoseService:
    def __init__(self, backend, allowed_origins=DEFAULT_ORIGINS):
        self.backend = backend
        self.origins = frozenset(allowed_origins)
        for origin in self.origins:
            parsed = urlsplit(origin)
            if (
                parsed.scheme != "http"
                or parsed.hostname not in ("127.0.0.1", "localhost")
                or parsed.path
                or parsed.query
                or parsed.fragment
                or parsed.username
            ):
                raise ValueError("Allowed origins must be exact HTTP loopback origins")
        self.sessions: dict[str, SessionState] = {}
        self.state_lock = threading.Lock()
        self.inference_lock = threading.Lock()

    def cleanup(self):
        now = time.monotonic()
        with self.state_lock:
            expired = [
                key
                for key, value in self.sessions.items()
                if now - value.touched > LIMITS["idleSessionSeconds"]
                or now - value.created > LIMITS["maximumSessionSeconds"]
            ]
            for key in expired:
                self._close_locked(key)

    def _close_locked(self, identifier):
        state = self.sessions.pop(identifier, None)
        if state is not None and state.recognizer is not None:
            state.recognizer.dispose()

    def close(self, identifier):
        with self.state_lock:
            self._close_locked(identifier)

    def close_all(self):
        with self.state_lock:
            for identifier in list(self.sessions):
                self._close_locked(identifier)

    def open(self, origin):
        self.cleanup()
        with self.state_lock:
            if len(self.sessions) >= LIMITS["maximumSessions"]:
                raise RequestError(429, "Local pose session limit reached")
            identifier, token = secrets.token_urlsafe(18), secrets.token_urlsafe(32)
            now = time.monotonic()
            bundle = getattr(self.backend, "recognizer_bundle", None)
            self.sessions[identifier] = SessionState(
                token,
                origin,
                now,
                now,
                recognizer=bundle.create_session() if bundle else None,
            )
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "sessionId": identifier,
            "token": token,
            "modelInfo": self.backend.model_info,
            "limits": LIMITS,
        }

    def session(self, identifier, authorization, origin):
        self.cleanup()
        with self.state_lock:
            value = self.sessions.get(identifier)
            if (
                value is None
                or value.origin != origin
                or not secrets.compare_digest(authorization, f"Bearer {value.token}")
            ):
                raise RequestError(
                    401, "Unknown, expired or unauthorized local pose session"
                )
            return value


class PoseHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 8

    def __init__(self, address, service):
        if address[0] != "127.0.0.1":
            raise ValueError("Pose service must bind to 127.0.0.1")
        self.service = service
        self.handlers = threading.BoundedSemaphore(8)
        super().__init__(address, PoseHandler)

    def process_request(self, request, client_address):
        if not self.handlers.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.handlers.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.handlers.release()

    def service_actions(self):
        self.service.cleanup()


class PoseHandler(BaseHTTPRequestHandler):
    server_version = "LocalPose/1"
    sys_version = ""
    protocol_version = "HTTP/1.0"

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, *_args):
        pass  # No image, token, timestamp or local-path access logs.

    def reply(self, status, value):
        data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
        self.send_response(status)
        origin = self.headers.get("Origin", "")
        if origin in self.server.service.origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(data)))
        if self.command == "OPTIONS":
            self.send_header(
                "Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"
            )
            self.send_header(
                "Access-Control-Allow-Headers",
                "Content-Type, Authorization, X-Frame-Time-Ms",
            )
        self.end_headers()
        self.wfile.write(data)

    def body(self, limit):
        if (
            self.headers.get("Transfer-Encoding")
            or len(self.headers.get_all("Content-Length", [])) != 1
        ):
            raise RequestError(411, "Exactly one Content-Length is required")
        try:
            length = int(self.headers["Content-Length"])
        except (ValueError, TypeError):
            raise RequestError(400, "Invalid Content-Length") from None
        if length <= 0 or length > limit:
            raise RequestError(413, "Request body exceeds the local service limit")
        data = self.rfile.read(length)
        if len(data) != length:
            raise RequestError(400, "Incomplete request body")
        return data

    def handle_api(self):
        service = self.server.service
        origin = self.headers.get("Origin", "")
        expected_hosts = {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        }
        if (
            self.headers.get("Host") not in expected_hosts
            or origin not in service.origins
        ):
            raise RequestError(
                403,
                "Only the explicitly allowed local application may use this service",
            )
        path = urlsplit(self.path)
        if path.query or path.fragment:
            raise RequestError(400, "Query parameters are unsupported")
        if self.command == "OPTIONS":
            self.reply(200, {"protocolVersion": PROTOCOL_VERSION})
            return
        if self.command == "GET" and path.path == "/v1/health":
            self.reply(
                200,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "ready": True,
                    "modelInfo": service.backend.model_info,
                    "limits": LIMITS,
                },
            )
            return
        if self.command == "POST" and path.path == "/v1/sessions":
            if self.headers.get_content_type() != "application/json":
                raise RequestError(
                    415, "Session initialization requires application/json"
                )
            if json.loads(self.body(1024)) != {}:
                raise RequestError(
                    400, "Session initialization expects an empty object"
                )
            self.reply(201, service.open(origin))
            return
        parts = path.path.strip("/").split("/")
        if len(parts) not in (3, 4) or parts[:2] != ["v1", "sessions"]:
            raise RequestError(404, "Unknown local pose endpoint")
        identifier = parts[2]
        state = service.session(
            identifier, self.headers.get("Authorization", ""), origin
        )
        if self.command == "DELETE" and len(parts) == 3:
            service.close(identifier)
            self.reply(200, {"closed": True})
            return
        if self.command != "POST" or len(parts) != 4 or parts[3] != "frame":
            raise RequestError(404, "Unknown local pose endpoint")
        if self.headers.get_content_type() != "image/jpeg":
            raise RequestError(415, "Frame input must be image/jpeg")
        try:
            timestamp = float(self.headers.get("X-Frame-Time-Ms", ""))
        except ValueError:
            raise RequestError(
                400, "A finite nonnegative source timestamp is required"
            ) from None
        if not math.isfinite(timestamp) or timestamp < 0:
            raise RequestError(400, "A finite nonnegative source timestamp is required")
        if not service.inference_lock.acquire(blocking=False):
            raise RequestError(
                429, "Local pose inference is busy; skip this frame instead of queueing"
            )
        try:
            if timestamp <= state.last_timestamp:
                raise RequestError(
                    409,
                    "Frame timestamps must increase; open a new session after a seek",
                )
            data = self.body(LIMITS["maximumImageBytes"])
            result = service.backend.infer(data, timestamp, jpeg_dimensions(data))
            with service.state_lock:
                if service.sessions.get(identifier) is not state:
                    raise RequestError(
                        409, "Session closed while inference was running"
                    )
                if state.recognizer is not None:
                    recognized_at = time.perf_counter()
                    result["frame"]["recognition"] = state.recognizer.update(
                        result["frame"]
                    )
                    recognition_ms = (time.perf_counter() - recognized_at) * 1000
                    timing = result.setdefault("timing", {})
                    timing["recognizerMs"] = recognition_ms
                    if "serviceMs" in timing:
                        timing["serviceMs"] += recognition_ms
                state.last_timestamp = timestamp
                state.touched = time.monotonic()
            self.reply(200, result)
        finally:
            service.inference_lock.release()

    def dispatch(self):
        try:
            self.handle_api()
        except RequestError as error:
            self.reply(error.status, {"error": str(error)})
        except (ValueError, UnicodeError):
            self.reply(400, {"error": "Malformed local pose request"})
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            pass
        except Exception:
            self.reply(
                500,
                {"error": "Local pose inference failed; original video is unaffected"},
            )

    do_GET = do_POST = do_DELETE = do_OPTIONS = dispatch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--minimum-score",
        type=float,
        required=True,
        help="Explicit experimental native SimCC cutoff; not calibrated visibility",
    )
    parser.add_argument("--provider", choices=("cpu", "coreml"), default="cpu")
    parser.add_argument(
        "--recognizer",
        type=Path,
        help="Optional frozen personal-recognizer manifest; local experimental weights only",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--origin", action="append", help="Exact allowed HTTP loopback origin"
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        backend = PoseBackend(args.manifest, args.provider, args.minimum_score)
        if args.recognizer is not None:
            backend.attach_recognizer(args.recognizer)
        service = PoseService(backend, args.origin or DEFAULT_ORIGINS)
        with PoseHTTPServer(("127.0.0.1", args.port), service) as server:
            print(
                json.dumps(
                    {
                        "status": "ready",
                        "url": f"http://127.0.0.1:{args.port}",
                        "modelInfo": backend.model_info,
                        "startedAt": datetime.now(timezone.utc).isoformat(),
                    }
                ),
                flush=True,
            )
            try:
                server.serve_forever(poll_interval=0.5)
            except KeyboardInterrupt:
                pass
            finally:
                service.close_all()
        return 0
    except Exception as error:
        print(f"Local pose service could not start: {error}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
