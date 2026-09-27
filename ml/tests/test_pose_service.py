import http.client
import json
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from ml.pose_service import (
    DEFAULT_ORIGINS,
    LIMITS,
    PoseBackend,
    PoseHTTPServer,
    PoseService,
    RequestError,
    configure_service_coreml,
    jpeg_dimensions,
)


def jpeg_header(width=32, height=24):
    # Enough encoded structure to test bounds before the fake decoder is called.
    return (
        b"\xff\xd8\xff\xc0\x00\x0b\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x01\x01\x11\x00\xff\xd9"
    )


class FakeBackend:
    model_info = {"id": "rtmpose-m", "delegate": "CPU"}

    def __init__(self):
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = False

    def infer(self, data, timestamp, dimensions):
        self.calls.append((timestamp, dimensions))
        self.entered.set()
        if self.block:
            self.release.wait(3)
        return {
            "frame": {
                "t": timestamp,
                "width": dimensions[0],
                "height": dimensions[1],
                "landmarks": [],
                "inferenceMs": 1,
            },
            "personCount": 0,
        }


class PoseServiceTests(unittest.TestCase):
    def test_optional_recognizer_history_is_isolated_and_disposed_with_sessions(self):
        created = []

        class Recognizer:
            def __init__(self):
                self.times = []
                self.disposed = False

            def update(self, frame):
                self.times.append(frame["t"])
                return {"events": [], "seen": len(self.times)}

            def dispose(self):
                self.disposed = True

        def create():
            state = Recognizer()
            created.append(state)
            return state

        self.backend.recognizer_bundle = SimpleNamespace(create_session=create)
        first, second = self.open(), self.open()
        self.assertEqual(self.frame(first, 0)[1]["frame"]["recognition"]["seen"], 1)
        self.assertEqual(self.frame(first, 33)[1]["frame"]["recognition"]["seen"], 2)
        self.assertEqual(self.frame(second, 0)[1]["frame"]["recognition"]["seen"], 1)
        self.service.close(first["sessionId"])
        self.assertTrue(created[0].disposed)
        self.assertFalse(created[1].disposed)
        self.service.sessions[second["sessionId"]].touched -= (
            LIMITS["idleSessionSeconds"] + 1
        )
        self.service.cleanup()
        self.assertTrue(created[1].disposed)
        self.assertFalse(self.service.sessions)

    def test_native_threshold_rejects_closed_interval_endpoints_before_model_loading(
        self,
    ):
        for threshold in (0, 1, -1, 1.1, float("nan")):
            with self.assertRaisesRegex(ValueError, "strictly between"):
                PoseBackend(Path("does-not-exist.json"), "cpu", threshold)

    def test_coreml_keeps_dynamic_detector_shapes_on_cpu_without_whole_session_retry(
        self,
    ):
        sessions = []

        class Session:
            def __init__(self, model, **kwargs):
                self.arguments = kwargs
                self.fallback_disabled = False
                sessions.append(self)

            def disable_fallback(self):
                self.fallback_disabled = True

            def get_providers(self):
                return self.arguments["providers"]

        ort = SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=Session)
        model = SimpleNamespace(
            det_model=SimpleNamespace(), pose_model=SimpleNamespace()
        )
        configuration = {
            "providers": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
            "providerOptions": [{}, {}],
        }
        configure_service_coreml(
            model,
            {"detector": {"path": "detector"}, "pose": {"path": "pose"}},
            ort,
            configuration,
            Path("/tmp/local-pose-test"),
        )
        self.assertEqual(
            sessions[0].arguments["provider_options"],
            [{"RequireStaticInputShapes": "1"}, {}],
        )
        self.assertEqual(sessions[1].arguments["provider_options"], [{}, {}])
        self.assertTrue(all(session.fallback_disabled for session in sessions))
        self.assertEqual(
            configuration["providerOptionsByModel"]["detector"],
            sessions[0].arguments["provider_options"],
        )
        self.assertNotIn("providerOptions", configuration)

    def setUp(self):
        self.backend = FakeBackend()
        self.service = PoseService(self.backend)
        self.server = PoseHTTPServer(("127.0.0.1", 0), self.service)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.02},
            daemon=True,
        )
        self.thread.start()

    def tearDown(self):
        self.backend.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def request(
        self, method, path, body=None, *, origin=DEFAULT_ORIGINS[0], headers=None
    ):
        actual = {"Origin": origin} if origin is not None else {}
        actual.update(headers or {})
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=4
        )
        try:
            connection.request(method, path, body=body, headers=actual)
            response = connection.getresponse()
            return (
                response.status,
                json.loads(response.read()),
                dict(response.getheaders()),
            )
        finally:
            connection.close()

    def open(self):
        status, body, _ = self.request(
            "POST", "/v1/sessions", b"{}", headers={"Content-Type": "application/json"}
        )
        self.assertEqual(status, 201)
        return body

    def frame(self, credentials, timestamp=0, data=None):
        return self.request(
            "POST",
            f"/v1/sessions/{credentials['sessionId']}/frame",
            data or jpeg_header(),
            headers={
                "Content-Type": "image/jpeg",
                "X-Frame-Time-Ms": str(timestamp),
                "Authorization": "Bearer " + credentials["token"],
            },
        )

    def test_session_requires_allowed_origin_and_loopback_host(self):
        for origin in (
            None,
            "null",
            "https://attacker.example",
            "http://127.0.0.1:9999",
        ):
            status, _, headers = self.request(
                "POST",
                "/v1/sessions",
                b"{}",
                origin=origin,
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(status, 403)
            self.assertNotIn("Access-Control-Allow-Origin", headers)
        status, _, _ = self.request(
            "GET", "/v1/health", headers={"Host": "attacker.example"}
        )
        self.assertEqual(status, 403)
        self.assertFalse(self.service.sessions)

    def test_known_origin_gets_exact_cors_and_no_cache(self):
        status, result, headers = self.request("GET", "/v1/health")
        self.assertEqual(status, 200)
        self.assertTrue(result["ready"])
        self.assertEqual(headers["Access-Control-Allow-Origin"], DEFAULT_ORIGINS[0])
        self.assertEqual(headers["Cache-Control"], "no-store")

    def test_timestamps_and_session_tokens_are_enforced(self):
        credentials = self.open()
        status, result, _ = self.frame(credentials, 123.45)
        self.assertEqual(status, 200)
        self.assertEqual(result["frame"]["t"], 123.45)
        self.assertEqual(self.frame(credentials, 123.45)[0], 409)
        self.assertEqual(self.frame(credentials, float("nan"))[0], 400)
        self.assertEqual(self.frame({**credentials, "token": "wrong"}, 200)[0], 401)
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(self.frame(credentials, 200)[0], 200)

    def test_size_limits_reject_before_inference(self):
        credentials = self.open()
        self.assertEqual(self.frame(credentials, 1, jpeg_header(3000, 3000))[0], 413)
        self.assertEqual(self.frame(credentials, 2, b"not a JPEG")[0], 415)
        status, _, _ = self.request(
            "POST",
            f"/v1/sessions/{credentials['sessionId']}/frame",
            b"x",
            headers={
                "Content-Type": "image/jpeg",
                "X-Frame-Time-Ms": "3",
                "Authorization": "Bearer " + credentials["token"],
                "Content-Length": str(LIMITS["maximumImageBytes"] + 1),
            },
        )
        self.assertEqual(status, 413)
        self.assertFalse(self.backend.calls)

    def test_backpressure_does_not_queue_and_close_invalidates_inflight_result(self):
        credentials = self.open()
        self.backend.block = True
        result = []
        pending = threading.Thread(
            target=lambda: result.append(self.frame(credentials, 10))
        )
        pending.start()
        self.assertTrue(self.backend.entered.wait(1))
        self.assertEqual(self.frame(credentials, 20)[0], 429)
        status, _, _ = self.request(
            "DELETE",
            f"/v1/sessions/{credentials['sessionId']}",
            headers={"Authorization": "Bearer " + credentials["token"]},
        )
        self.assertEqual(status, 200)
        self.backend.release.set()
        pending.join(2)
        self.assertEqual(result[0][0], 409)
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(self.frame(credentials, 30)[0], 401)

    def test_expired_and_excess_sessions_are_bounded(self):
        credentials = [self.open() for _ in range(LIMITS["maximumSessions"])]
        self.assertEqual(
            self.request(
                "POST",
                "/v1/sessions",
                b"{}",
                headers={"Content-Type": "application/json"},
            )[0],
            429,
        )
        with self.service.state_lock:
            self.service.sessions[credentials[0]["sessionId"]].touched = (
                time.monotonic() - LIMITS["idleSessionSeconds"] - 1
            )
        self.assertEqual(self.frame(credentials[0])[0], 401)
        self.open()
        self.assertEqual(len(self.service.sessions), LIMITS["maximumSessions"])

    def test_origins_and_listen_addresses_cannot_be_public(self):
        with self.assertRaises(ValueError):
            PoseService(self.backend, ["https://remote.example"])
        with self.assertRaises(ValueError):
            PoseHTTPServer(("0.0.0.0", 0), self.service)

    def test_jpeg_header_parser_never_accepts_truncated_or_excess_pixels(self):
        self.assertEqual(jpeg_dimensions(jpeg_header()), (32, 24))
        for data in (b"", b"\xff\xd8", jpeg_header()[:8], jpeg_header(1500, 1500)):
            with self.assertRaises(RequestError):
                jpeg_dimensions(data)


if __name__ == "__main__":
    unittest.main()
