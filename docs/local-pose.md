# Optional local RTM tracker and personal recognizer

The app can send unmirrored frames to a persistent RTMPose-M process on the same Mac. This is an experimental alternative to the default MediaPipe Full worker. An optional, separately fingerprinted personal model recognizes straight punches, hooks and uppercuts. Neither option validates technique or establishes reliable recognition across people, camera angles and lighting.

Install the pinned `ml/requirements-extract.txt` dependencies in `.venv`, then obtain the two model archives linked in [`ml/models.example.json`](../ml/models.example.json). That manifest specifies archive and extracted ONNX fingerprints. Extract the detector as `models/yolox-m.onnx` and pose model as `models/rtmpose-m.onnx`; keep weights out of Git. The application never downloads weights at runtime. See [the extraction notes](benchmarking.md#optional-rtmpose--rtmw-extraction) for provenance and reuse limits.

Start both the app and local service in one foreground terminal:

```sh
npm run dev:mac
```

Open **http://127.0.0.1:5173**. Keep that terminal open; Ctrl+C stops both processes started by this launcher. Ports 5173 and 8765 must be free. If separately started development/service terminals already own them, stop those terminals first. The launcher does not replace or stop unrelated processes. Restart with the same command.

For existing local assets, save a configuration in `data/local-studio.json`, which is ignored by Git. Paths resolve from the repository root. For example:

```json
{
  "manifest": "ml/models.example.json",
  "provider": "coreml",
  "minimumScore": 0.55
}
```

Set `manifest` to the actual local manifest if weights are elsewhere. Explicit command arguments override configuration; `--config FILE` selects a different local configuration. Models and `.venv` must already exist. The launcher performs no installation, downloads or background startup registration.

Under **More options → Model**, choose **Precision · this Mac** before starting a camera or clip. This selection persists in this browser; starting the service alone does not select it. For an existing saved recording, use **Export & details → Review with Precision · this Mac**. Original video and capture results are preserved. Review records the actual estimator, runtime, weights and score policy. `npm run preview` uses the same local address after the development server has stopped.

The personal recognizer is an explicit opt-in and its weights are not included in Git. It requires a frozen local bundle and the additional pinned dependencies in `ml/requirements-recognizer.txt`:

```sh
.venv/bin/python -m pip install -r ml/requirements-recognizer.txt
npm run dev:mac -- --recognizer data/personal-recognizer/manifest.json
```

Replace that example path with the validated bundle. Alternatively, add `"recognizer": "data/personal-recognizer/manifest.json"` to the local configuration for subsequent restarts. Omit the field for pose-only tracking. A configured recognizer must match the pose weights, native score policy, feature and runtime source hashes, checkpoint hashes and protocol. Missing or incompatible output fails visibly; the app does not silently substitute geometric detections. Source-clock event times and physical hand are preserved; the browser maps straight punches to jab/cross using the selected stance. Personal-model outputs do not measure extension or guard quality.

The current personal model was fitted using one development recording. Other known recordings informed evaluation and model selection. That evidence does not establish accuracy on a new recording, another person, or a different setup. Keep a separate fresh-video check before broadening claims. The model uses an 800 ms arbitration delay after a candidate ends; in the actual live test, decisions arrived a median **1.07 seconds after the observed peak**, or **835 ms after the event end**. This is not instantaneous feedback. Leave roughly two seconds of visible follow-through after the final punch before stopping; unavailable future observations are not fabricated to complete a last-second detection.

The current M3 MacBook Pro measurement used an isolated Chromium session, a locally decoded 30-second recording, JPEG quality .95, RTMPose-M with verified Core ML partitions, and the frozen personal recognizer. The actual rendered app processed **19.6 fps**, approximately 20 fps, from a roughly 30 fps source. It retained the original video. This is measured throughput, not a 25/30 fps guarantee or an accuracy claim. No external network requests occurred in the test.

| Stage                                             | Saved-video pass, p50 / p95 | Live app, p50 / p95 |
| ------------------------------------------------- | --------------------------- | ------------------- |
| JPEG encoding                                     | 5.9 / 7.7 ms                | 8.8 / 19.1 ms       |
| Server JPEG decoding                              | 1.9 / 2.0 ms                | 2.0 / 2.2 ms        |
| Person detector                                   | 19.4 / 20.7 ms              | 19.3 / 20.7 ms      |
| Pose model                                        | 8.4 / 9.5 ms                | 8.5 / 9.4 ms        |
| Personal recognizer                               | 8.1 / 15.5 ms               | 8.3 / 14.5 ms       |
| Complete client call, including transport         | 47.0 / 55.0 ms              | 50.0 / 61.4 ms      |
| Capture-to-result age, including waiting snapshot | Sequential decoding         | 68.6 / 97.0 ms      |

The complete saved-video pass processed 899 source frames in 43.3 seconds; a second 747-frame recording took 36.3 seconds. Offline decoding waits for each inference and preserves source timestamps. Live capture permits one inference and at most one replaceable waiting snapshot, so it can drop tracking frames while retaining video. Never queue old live images to manufacture a higher processing rate. Timing varies with machine load and capture settings; inference-only timings do not establish live frame rate.

Native saved-video analysis has a **six-minute processing cap**. Its other limits remain 185 seconds of source video, 5,550 processed frames and 250 MB per recording. A capped job with usable results is marked partial; the original recording remains untouched. The browser-worker path retains its four-minute processing cap.

`.55` is an explicit exploratory native SimCC score floor, not a calibrated probability of a correct joint or a transplanted MediaPipe visibility threshold. Frames preserve native `score` values and their policy. A high score does not prove correct hand identity. The personal model preserves its distinct feature and temporal rules; pose-only tracking retains the geometric observer. Neither path grades form.

The service binds only to `127.0.0.1:8765`. The app proxy forwards requests with an exact origin allowlist and short-lived bearer-token sessions. One inference runs at a time; competing requests receive a busy response instead of accumulating. Live capture skips busy work; saved-video analysis retries short contention within a bounded, cancelable window. Every personal-recognition session owns its bounded history and is disposed on close or expiry. No pixels or inference history are written by the service. Evidence saved by the browser remains local.

Core ML warmup verifies actual execution with explicit CPU partitions, then ends profiling. The detector uses `RequireStaticInputShapes=1` so dynamic NMS outputs, including empty-person views, remain on CPU; pose keeps its ordinary Core ML partitioning. These provider profiles do not identify a particular Apple compute unit. JPEG transport uses quality .95, at most 1280 pixels on the long edge and 921,600 pixels. This differs from the lossless browser-worker path and offline PNG extraction. Empty or multiple detected-person views do not produce a new person's pose. Previously supported temporal decisions may arrive later. The adapter applies YOLOX's configured .7 box threshold because the pinned upstream embedded-NMS path otherwise hardcodes .3.

For separate terminals, use `npm run pose:serve -- --manifest ml/models.example.json --provider coreml --minimum-score .55` and `npm run dev`. Python's provider default is CPU; the combined Mac launcher explicitly selects Core ML. RTMW-L needs its own manifest, while the app's normal Precision option expects RTMPose-M. The personal bundle currently requires the exact RTMPose-M pipeline it was validated against.
