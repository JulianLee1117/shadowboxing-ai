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

Under **More options → Model**, choose **Precision · this Mac** before starting a camera or clip. This selection persists in this browser; starting the service alone does not select it. New rounds show their live results immediately, with no automatic second pass. For an optional fresh pass over a saved recording, open **Re-run analysis**. Original video and capture results are preserved. Review records the actual estimator, runtime, weights and score policy. `npm run preview` uses the same local address after the development server has stopped.

The personal recognizer is an explicit opt-in and its weights are not included in Git. It requires a frozen local bundle and the additional pinned dependencies in `ml/requirements-recognizer.txt`:

```sh
.venv/bin/python -m pip install -r ml/requirements-recognizer.txt
npm run dev:mac -- --recognizer data/personal-recognizer/manifest.json
```

Replace that example path with the validated bundle. Alternatively, add `"recognizer": "data/personal-recognizer/manifest.json"` to the local configuration for subsequent restarts. Omit the field for pose-only tracking. A configured recognizer must match the pose weights, native score policy, feature and runtime source hashes, checkpoint hashes and protocol. Missing or incompatible output fails visibly; the app does not silently substitute geometric detections. Source-clock event times and physical hand are preserved; the browser maps straight punches to jab/cross using the selected stance. Personal-model outputs do not measure extension or guard quality.

The current personal model was fitted using seven personal development recordings, including the latest mixed-punch round. Whole-recording leave-out checks still show poor transfer on excluded mixed-punch footage. Fitted improvements do not establish accuracy on a new round, another person, or another setup. Keep a separate fresh-video check before broadening claims.

The decoder now uses a 150 ms arbitration hold after an observed event ends, reduced from 300 ms. Classification and recovery can add delay, so this is not a promise of feedback 150 ms after a punch. The change can alter the proposal winner and event boundaries; the [latency review](research/detection-latency-2026-09.md) records that tradeoff and the unchanged weights. Leave a few seconds of visible follow-through after the final punch before stopping; unavailable observations are not fabricated to finish a last-second detection. [Personal recognition](personal-recognition.md) describes event splitting, background rejection, bounded missing-arm retention and the training protocol.

Live capture permits one inference and at most one replaceable waiting snapshot, so tracking frames can be dropped while the original video is retained. Saved-video decoding instead waits for each inference and preserves source timestamps. Never queue old live images to manufacture a higher processing rate.

Two 30-second recordings played through the actual app in real time on the development M3 MacBook Pro processed 23.2 and 23.9 pose frames per second. Browser frame callback to completed inference took 58–62 ms median and 76–81 ms p95; this excludes camera sensor age and subsequent rendering. JPEG encoding took 5.2–5.5 ms median; the complete local request, including encoding and recognition, took 41–43 ms median and 48 ms p95. Recognition still needs a temporal window: across emitted detections, final decisions followed their reported source-time peak by about 0.53 seconds median, even though the skeleton updated sooner. These are imported-file playback measurements with the final local bundle, not fresh webcam capture or a sustained 20-minute performance test.

Those controlled runs used an isolated Chromium test browser. A final optional re-analysis in the user's ordinary Chrome tab processed all 900 frames in 111.6 seconds, with identical landmarks and event objects to the corresponding controlled full-frame pass. Its median encoding and service times were 46.9 and 56.2 ms, substantially slower than the isolated runs. The cause of that environment difference has not been established. Source-video cadence is not processing throughput; verify live performance in the actual foreground webcam session before treating the controlled rates as representative.

Two subsequent actual webcam recordings, lasting 30 and 19 seconds, measured 19.8 pose observations per second. Callback-to-inference frame age was about 66 ms median and 78–80 ms p95; neither recording had an observation gap over 150 ms. These short captures establish that processing continued in the low-detection round, not that its classifications were correct or that a sustained-session target has passed.

Native saved-video analysis has a **six-minute processing cap**. Its other limits remain 185 seconds of source video, 5,550 processed frames and 250 MB per recording. A capped job with usable results is marked partial; the original recording remains untouched. The browser-worker path retains its four-minute processing cap.

`.55` is an explicit exploratory native SimCC score floor, not a calibrated probability of a correct joint or a transplanted MediaPipe visibility threshold. Frames preserve native `score` values and their policy. A high score does not prove correct hand identity. The personal model preserves its distinct feature and temporal rules; pose-only tracking retains the geometric observer. Neither path grades form.

The service binds only to `127.0.0.1:8765`. The app proxy forwards requests with an exact origin allowlist and short-lived bearer-token sessions. One inference runs at a time; competing requests receive a busy response instead of accumulating. Live capture skips busy work; saved-video analysis retries short contention within a bounded, cancelable window. Every personal-recognition session owns its bounded history and is disposed on close or expiry. No pixels or inference history are written by the service. Evidence saved by the browser remains local.

Core ML warmup verifies actual execution with explicit CPU partitions, then ends profiling. The detector uses `RequireStaticInputShapes=1` so dynamic NMS outputs, including empty-person views, remain on CPU; pose keeps its ordinary Core ML partitioning. These provider profiles do not identify a particular Apple compute unit. JPEG transport uses quality .95, at most 1280 pixels on the long edge and 921,600 pixels. This differs from the lossless browser-worker path and offline PNG extraction. Empty or multiple detected-person views do not produce a new person's pose. Previously supported temporal decisions may arrive later. The adapter applies YOLOX's configured .7 box threshold because the pinned upstream embedded-NMS path otherwise hardcodes .3.

JPEG encoding runs in a dedicated worker with one transferred snapshot and no encoding queue. Cancellation or disposal terminates that worker and rejects pending work. The pixel dimensions, quality and byte limit remain unchanged. A complete 899-frame saved-video comparison produced exactly identical timestamps, landmarks and punch events after this transport change; encoding measured 5.3 ms median and 5.8 ms p95 in that run. This stage measurement does not guarantee overall live frame rate. A separate detector-caching experiment was rejected because it missed additional punches despite reducing inference time; the service still detects the person on every processed frame.

For separate terminals, use `npm run pose:serve -- --manifest ml/models.example.json --provider coreml --minimum-score .55` and `npm run dev`. Python's provider default is CPU; the combined Mac launcher explicitly selects Core ML. RTMW-L needs its own manifest, while the app's normal Precision option expects RTMPose-M. The personal bundle currently requires the exact RTMPose-M pipeline it was validated against.
