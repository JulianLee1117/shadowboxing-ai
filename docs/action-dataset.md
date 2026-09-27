# Six-punch research dataset preparation

`ml/action_dataset.py` audits existing local session exports and prepares observed-time action targets. It does not download footage, guess an external annotation format, train a model, grade technique, or supply browser model weights.

The current browser detector (`projected-six-punch-v7-supported-rise`), review annotation interface and Python evaluator accept `jab`, `cross`, `hook`, and `uppercut`, with anatomical hand labels. Lead/rear role distinguishes the six displayed punch identities. Browser recognition uses experimental geometric rules, not a learned multiclass model, and class support does not imply reliable recognition. The experimental trainer in `ml/temporal.py` remains binary per arm: it treats hooks/uppercuts as negative examples and assigns jab/cross from stance. This dataset format is a separate, versioned foundation rather than an incompatible change to that trainer. Do not use that binary trainer as a six-punch baseline.

## Manifest and commands

Save a manifest under ignored `data/`. Paths are relative to the manifest; absolute local paths also work. Nothing is downloaded.

```json
{
  "schemaVersion": "action-dataset-1",
  "splitPolicy": "recording",
  "expectedDomains": ["shadowboxing"],
  "entries": [
    {
      "id": "round-a",
      "session": "round-a-labeled.json",
      "video": "round-a.webm",
      "split": "train",
      "participantId": "personal-1",
      "captureDay": "2026-09-26",
      "sourceGroup": "original-round-a",
      "domain": "shadowboxing",
      "view": "three-quarter",
      "rights": {
        "status": "owned",
        "reference": "Operator's declaration for this personal recording"
      },
      "labelScope": {
        "labels": ["jab", "cross"],
        "complete": true
      }
    }
  ]
}
```

```sh
python3 -m ml.action_dataset audit data/labels/manifest.json \
  --output data/experiments/action-audit.json
python3 -m ml.action_dataset prepare data/labels/manifest.json \
  --output data/experiments/action-dataset-001
python3 -m unittest ml.tests.test_action_dataset -v
```

Outputs must be new. `audit` writes a report and exits with status 2 when invalid. `prepare` requires a valid audit and creates `index.json`, `audit.json`, and one `session-ID.json` per entry. Original sessions, poses and recordings are never edited. An incomplete class/domain inventory is reported rather than silently filled; all-train preparation is allowed but has no held-out evaluation set.

Each input needs a valid existing schema 1.0 session, a known orthodox/southpaw stance, strictly increasing pose timestamps and actual observed frames. Synthetic demonstration sessions are excluded. `labelScope.complete: true` additionally requires `annotationsComplete: true` in that session. A valid audit establishes software consistency, not that the annotations are correct or the dataset is sufficient.

`video` is optional. When supplied, its bytes are hashed; an optional `videoSha256` must match. This records which local video the operator supplied. Old session exports lack a cryptographic recording association, so a matching supplied-file hash does **not** prove that its poses came from that video. The report preserves that limitation.

## Labels and unknown intervals

The six identities are `jab`, `cross`, `lead_hook`, `rear_hook`, `lead_uppercut`, and `rear_uppercut`. Every prepared event also retains physical `left`/`right` hand, stance-dependent `lead`/`rear` role and motion family. Southpaw swaps the physical hand associated with each role. A jab/cross annotation inconsistent with hand and stance is rejected; hooks and uppercuts receive their role without changing their hand.

Targets have four possible family indices per arm: `0=background`, `1=straight`, `2=hook`, `3=uppercut`. They are sampled only at the original frame timestamps. Unknown targets use `familyIndex: null`, `mask: false`, and no event ID; they must not enter a supervised loss as background. Intervals are half-open `[startMs,endMs)`, avoiding two labels at a shared boundary.

- Known positive annotations produce their family target and original event ID. Boundary times are retained separately; no peak or movement phase is fabricated.
- Unlabeled time becomes known background only when the source explicitly asserts complete annotation of **all four labels**. Complete jab/cross-only clips keep every unlabeled interval masked for multiclass training.
- Partial or positive-only annotation uses `complete: false`; unlabeled intervals remain masked even if all four class names are listed.
- An unknown-hand punch masks both arms during its interval. `unobservable` with a known hand masks that arm; `unknown` masks both. Unknown takes precedence over a positive interval.
- Two positive event intervals on the same physical arm cannot overlap: resolve their boundaries first. Events on opposite arms may overlap. Repeated same-family punches retain distinct event IDs for later boundary training.

The builder preserves pose coordinates, model provenance and raw confidence fields. It never copies a hand position, substitutes confident-looking joints, interpolates a missing observation, or treats model confidence as annotation truth. Native RTMPose scores still need a model-specific observation policy; this builder does not make the existing MediaPipe feature thresholds appropriate for RTMPose.

External clip-level labels require a separate documented adapter and review. A clip labeled “hook” does not establish that every frame is a hook, which anatomical arm performs it, whether the camera is mirrored, or where a continuous event starts and ends. Do not invent interval labels or background labels to satisfy this interface.

## Provenance and split checks

Required entry metadata includes participant identity, capture day, original-source group, domain, view and split. Domains are `shadowboxing`, `bag`, `padwork`, `sparring`, or `unknown`; view is an explicit descriptive string. Crops, re-encodes, extracted poses and alternate model passes of one recording belong to the same `sourceGroup`.

The tool fingerprints the manifest, original session, pose observations, annotations and optional video. Duplicate pose captures are rejected even if the session ID or inference telemetry changes. Source groups and identical video hashes cannot cross splits under any policy:

| Split policy      | Additional isolation                                                    |
| ----------------- | ----------------------------------------------------------------------- |
| `recording`       | Separate original recordings; same-person/same-day overlap is reported. |
| `participant-day` | Each participant/day group belongs to one split.                        |
| `participant`     | Each participant belongs to one split.                                  |

These identities are operator metadata, not independently verified identities. Re-encoded or otherwise altered duplicates still require truthful source grouping. The report lists per-split classes, domain/class gaps, participants, participant-days, observed views, unknown arm-frame targets and cross-split groups. It separately reports positive event counts and actual known arm-frame targets, since an annotation can have no usable sampled target.

`rights.status` is `owned`, `licensed`, or `unreviewed`, with a `reference` describing the declaration or source terms. Unreviewed status blocks preparation. Accepted declarations are **not legal verification**, and a dataset's availability online is not evidence of permission. Record and review actual source terms before using external data; this tool does not authorize training or redistribution.

## Next model increment

First obtain reviewed continuous examples of all six identities plus idle/guard movements and hard negatives. Keep natural rapid repeats, partial returns, different views and actual occlusions. Reserve complete recordings and later participant-days before fitting or tuning. Action labels and coach-adjudicated technique labels are separate evidence; a recognized uppercut cannot by itself establish correct form.

A minimal research extension is a new version of the existing causal TCN with per-arm four-family outputs, both-arm/torso context, unknown-label masking, and explicit onset/offset supervision. Classifying frames alone will still merge adjacent punches of the same family. Freeze cadence/context rules and an event decoder before evaluating on untouched recordings. Keep it offline until numerical/runtime parity is tested. No such multiclass model is trained by this tool.

The existing evaluator can assess these classes using `--labels jab,cross,hook,uppercut`; matching already requires exact anatomical hand and class. Its displayed per-class metrics currently combine lead/rear hooks and uppercuts, so a six-role breakdown and confusion analysis remain additional evaluation work.

ST-GCN and PoseC3D are reasonable later transfer baselines, but their available action checkpoints are not ready six-punch classifiers. For example, MMAction2's official [ST-GCN configuration](https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py) uses a 60-class head and 100-frame uniform sampling, while its [NTU60 label map](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/label_map_ntu60.txt) has a broad punching/slapping class. That configuration is not evidence of causal punch boundaries or Mac throughput. The official [PoseC3D implementation](https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/posec3d/README.md) offers another representation to compare after establishing the boxing data and evaluation protocol.
