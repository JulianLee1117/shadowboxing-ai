# Local pose assets

Run `npm run models:setup` after `npm install`. This downloads the official Full
Pose Landmarker float16 version-1 bundle and copies the pinned
`@mediapipe/tasks-vision@1.0.1` JavaScript/WASM loaders from `node_modules` to
`public/wasm`. Use `npm run models:setup -- --all` to add Lite and Heavy.

The exact downloaded byte sizes and SHA-256 values are recorded in
[`manifest.json`](manifest.json). On first download these hashes are established
from Google's HTTPS response, not a vendor signature; subsequent setup runs
verify the pinned values. Keep the manifest under version control, while `.task`
weights and `public/wasm/` are generated, ignored assets. Do not put training
footage in this directory: every file under `public` can be served to a browser.

Official bundle sizes checked by HTTPS headers on 2026-09-26:

| Variant | Bytes | Approximate size |
| --- | ---: | ---: |
| Lite | 5,777,746 | 5.51 MiB |
| Full | 9,398,198 | 8.96 MiB |
| Heavy | 30,664,242 | 29.24 MiB |

The setup also copies all six SDK WASM/loader files (about 33.8 MiB on disk);
the browser fetches the selected loader and binary, not all six.

Provenance:

- [Google model overview](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker)
- [BlazePose GHUM model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf)
- Exact Google Storage bundle URLs are in the manifest.
- Runtime package is pinned to 1.0.1 in the application dependency lockfile.

The model card and runtime identify Apache 2.0 licensing. Preserve applicable
third-party notices with a redistributed build. The runtime includes licensed
third-party code; the npm package and its source are the provenance for it.

Asset requests use only same-origin `/models/` and `/wasm/` paths. No external
CDN is used, and no camera image is sent to the asset hosts. SDK 1.0.1 contains
usage logging; the isolated worker rejects all cross-origin `fetch` requests,
including that telemetry endpoint. The worker first
tries the GPU delegate, then retries with a fresh CPU worker if initialization
fails. The selected delegate is available as `VisionRunner.delegate`.

`PoseFrame.inferenceMs` measures only the synchronous detector call. It excludes
frame capture, transfer and scheduling; the application separately measures its
callback-to-result interval. Source timestamps and unmirrored pixel dimensions
are preserved. Use a fresh runner for a new source, restart or seek so a previous
tracking state cannot contaminate a new sequence.
