import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from ml.extract import (
    configure_coreml_sessions,
    provider_configuration,
    summarize_execution_profile,
    verify_coreml_execution,
)


class ProviderTests(unittest.TestCase):
    def test_cpu_selection_stays_explicit_even_when_coreml_is_available(self):
        config = provider_configuration(
            "cpu", ["CoreMLExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(config["providers"], ["CPUExecutionProvider"])
        self.assertEqual(config["providerOptions"], [{}])
        self.assertEqual(config["hardwareExecution"], "not_established_by_provider_selection")

    def test_unavailable_coreml_never_aliases_to_cpu(self):
        with self.assertRaisesRegex(ValueError, "CoreMLExecutionProvider"):
            provider_configuration("coreml", ["CPUExecutionProvider"])
        with self.assertRaisesRegex(ValueError, "CPUExecutionProvider"):
            provider_configuration("coreml", ["CoreMLExecutionProvider"])
        with self.assertRaisesRegex(ValueError, "cpu or coreml"):
            provider_configuration("gpu", ["CPUExecutionProvider"])

    def make_runtime(self, registered):
        runtime = Mock()
        sessions = [Mock(), Mock()]
        for session in sessions:
            session.get_providers.return_value = registered
        runtime.InferenceSession.side_effect = sessions
        runtime.SessionOptions.side_effect = [SimpleNamespace(), SimpleNamespace()]
        model = SimpleNamespace(det_model=SimpleNamespace(session=Mock()),
                                pose_model=SimpleNamespace(session=Mock()))
        manifest = {"detector": {"path": "/fixture/detector.onnx"},
                    "pose": {"path": "/fixture/pose.onnx"}}
        return runtime, sessions, model, manifest

    def test_coreml_sessions_record_profiles_and_disable_whole_session_retry(self):
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        runtime, sessions, model, manifest = self.make_runtime(providers)
        config = provider_configuration("coreml", providers)
        with tempfile.TemporaryDirectory() as directory:
            timings = configure_coreml_sessions(model, manifest, runtime, config, Path(directory))
            for call, name in zip(runtime.InferenceSession.call_args_list, ("detector", "pose")):
                self.assertEqual(call.args[0], manifest[name]["path"])
                self.assertEqual(call.kwargs["providers"], providers)
                self.assertEqual(call.kwargs["provider_options"], [{}, {}])
                options = call.kwargs["sess_options"]
                self.assertTrue(options.enable_profiling)
                self.assertEqual(options.profile_file_prefix, str((Path(directory) / name).resolve()))
        self.assertIs(model.det_model.session, sessions[0])
        self.assertIs(model.pose_model.session, sessions[1])
        for session in sessions:
            session.disable_fallback.assert_called_once_with()
        self.assertEqual(set(timings), {"detector", "pose"})
        self.assertTrue(all(value >= 0 for value in timings.values()))

    def test_session_registration_fallback_to_cpu_is_rejected(self):
        runtime, _, model, manifest = self.make_runtime(["CPUExecutionProvider"])
        config = provider_configuration("coreml", ["CoreMLExecutionProvider", "CPUExecutionProvider"])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "did not register"):
                configure_coreml_sessions(model, manifest, runtime, config, Path(directory))
        self.assertEqual(runtime.InferenceSession.call_count, 1)

    def test_profile_uses_executed_node_events_not_registered_provider_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps([
                {"cat": "Session", "args": {"provider": "CoreMLExecutionProvider"}},
                {"cat": "Node", "args": {"provider": "CPUExecutionProvider", "op_name": "Gather"}},
                {"cat": "Node", "args": {"provider": "CoreMLExecutionProvider", "op_name": "CoreMLPartition"}},
                {"cat": "Node", "args": {}},
                None,
            ]))
            report = summarize_execution_profile(path)
            self.assertEqual(report["executedNodeProviderEvents"],
                             {"CPUExecutionProvider": 1, "CoreMLExecutionProvider": 1})
            self.assertEqual(len(report["sha256"]), 64)
            self.assertIn("not operation counts", report["interpretation"])
            verify_coreml_execution({"detector": report})

    def test_registered_but_entirely_cpu_execution_fails_verification(self):
        with self.assertRaisesRegex(ValueError, "no CoreMLExecutionProvider nodes"):
            verify_coreml_execution({
                "detector": {"executedNodeProviderEvents": {"CPUExecutionProvider": 20}},
                "pose": {"executedNodeProviderEvents": {}},
            })


if __name__ == "__main__":
    unittest.main()
