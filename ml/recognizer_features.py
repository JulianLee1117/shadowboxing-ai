"""Numerical features for the explicitly enabled personal recognizer.

Coordinates remain observed image-plane positions. A cached scalar torso length
can support normalization for at most250ms; it never reconstructs a joint.
"""

import math
import numpy as np

J = [11, 12, 13, 14, 15, 16, 23, 24]


def temporal_features(frames):
    rows = []
    valid = []
    prev = None
    for f in frames:
        l = f["landmarks"]
        arms = []
        vs = []
        if len(l) < 25:
            rows.append(np.zeros((2, 40)))
            valid.append([False, False])
            prev = None
            continue
        xy = np.array([[p["x"] * f["width"] / f["height"], p["y"]] for p in l])
        scores = np.array([p.get("score", p.get("visibility", 0)) for p in l])
        scale = np.linalg.norm((xy[11] + xy[12] - xy[23] - xy[24]) / 2)
        if not np.isfinite(scale) or scale < 0.08:
            rows.append(np.zeros((2, 40)))
            valid.append([False, False])
            prev = None
            continue
        for h in [0, 1]:
            s, e, w, hip = 11 + h, 13 + h, 15 + h, 23 + h
            o = 1 - h
            # Current observed geometry; no inferred coordinates. Scores remain features.
            base = np.r_[
                (xy[[e, w, 11 + o, 13 + o, 15 + o, hip, 23 + o]] - xy[s]).reshape(-1)
                / scale,
                np.clip(scores[[s, e, w, hip, 11 + o]], 0, 1),
            ]
            base = np.clip(np.nan_to_num(base), -4, 4)
            velocity = (
                np.zeros(19)
                if prev is None
                else np.clip(
                    (base - prev[h]) / max(0.01, (f["t"] - prev_t) / 1000) / 15, -4, 4
                )
            )
            arms.append(
                np.r_[
                    base,
                    velocity,
                    f.get("_ageMs", 0) / 100,
                    float(f.get("_newSample", True)),
                ]
            )
            vs.append(
                bool(np.isfinite(xy[[s, e, w]]).all() and min(scores[[e, w]]) >= 0.55)
            )
        rows.append(arms)
        valid.append(vs)
        prev = np.array([a[:19] for a in arms])
        prev_t = f["t"]
    return np.array(rows, dtype=np.float32).transpose(1, 2, 0), np.array(valid).T


def grid(frames, source_fps=30):
    # Simulate source throughput without choosing a future source frame. A 30Hz
    # causal hold grid never fabricates/interpolates a joint; age is explicit.
    available = []
    idx = 0
    last_source = None
    for delivery in np.arange(frames[0]["t"], frames[-1]["t"] + 1, 1000 / source_fps):
        while idx + 1 < len(frames) and frames[idx + 1]["t"] <= delivery:
            idx += 1
        if frames[idx]["t"] != last_source:
            available.append(frames[idx])
            last_source = frames[idx]["t"]
    out = []
    i = 0
    previous_source = None
    for tick in np.arange(available[0]["t"], available[-1]["t"] + 1, 1000 / 30):
        while i + 1 < len(available) and available[i + 1]["t"] <= tick:
            i += 1
        source = available[i]
        age = tick - source["t"]
        out.append(
            {
                **source,
                "t": float(tick),
                "_sourceT": source["t"],
                "_ageMs": float(age),
                "_newSample": previous_source != source["t"],
                "landmarks": source["landmarks"] if age <= 100 else [],
            }
        )
        previous_source = source["t"]
    return out


def family_features(frames):
    coords = np.full((len(frames), 8, 2), np.nan)
    scores = np.zeros((len(frames), 8))
    features = np.zeros((len(frames), 16), np.float32)
    geom = np.full((len(frames), 2, 6), np.nan)
    cached_scale = None
    cached_t = -1e9
    previous_shoulders = None
    for i, f in enumerate(frames):
        l = f["landmarks"]
        if len(l) < 25:
            continue
        xy = np.array([[l[j]["x"] * f["width"] / f["height"], l[j]["y"]] for j in J])
        sc = np.array([l[j].get("score", l[j].get("visibility", 0)) for j in J])
        scores[i] = sc
        torso = np.linalg.norm((xy[0] + xy[1] - xy[6] - xy[7]) / 2)
        sw = np.linalg.norm(xy[0] - xy[1])
        if torso < 0.08:
            continue
        shoulders = xy[:2].copy()
        jump = (
            previous_shoulders is not None
            and np.max(np.linalg.norm(shoulders - previous_shoulders, axis=1))
            > max(torso, 0.08) * 0.35
        )
        previous_shoulders = shoulders
        if jump:
            cached_scale = None
        if sc[[0, 1, 6, 7]].min() >= 0.55:
            cached_scale = torso
            cached_t = f["t"]
        elif cached_scale is not None and f["t"] - cached_t <= 250:
            torso = cached_scale
        else:
            continue
        sw = max(sw, 0.05)
        coords[i] = (xy - (xy[0] + xy[1]) / 2) / torso
        feat = (xy - (xy[0] + xy[1]) / 2) / sw
        feat[sc <= 0.5] = 0
        features[i] = feat.flatten()
        for h in [0, 1]:
            e = (xy[2 + h] - xy[h]) / torso
            w = (xy[4 + h] - xy[h]) / torso
            den = np.linalg.norm(e) * np.linalg.norm(w - e)
            if den < 1e-6:
                continue
            a = math.degrees(math.acos(np.clip(np.dot(-e, w - e) / den, -1, 1)))
            geom[i, h] = [w[0], w[1], e[0], e[1], np.linalg.norm(w), a]
    return features, geom, scores


def classification_features(frames):
    coords = np.full((len(frames), 8, 2), np.nan)
    scores = np.zeros((len(frames), 8))
    features = np.zeros((len(frames), 16), np.float32)
    geom = np.full((len(frames), 2, 6), np.nan)
    for i, f in enumerate(frames):
        l = f["landmarks"]
        if len(l) < 25:
            continue
        xy = np.array([[l[j]["x"] * f["width"] / f["height"], l[j]["y"]] for j in J])
        sc = np.array([l[j].get("score", l[j].get("visibility", 0)) for j in J])
        scores[i] = sc
        torso = np.linalg.norm((xy[0] + xy[1] - xy[6] - xy[7]) / 2)
        sw = np.linalg.norm(xy[0] - xy[1])
        if torso < 0.08:
            continue
        sw = max(sw, 0.05)
        coords[i] = (xy - (xy[0] + xy[1]) / 2) / torso
        feat = (xy - (xy[0] + xy[1]) / 2) / sw
        feat[sc <= 0.5] = 0
        features[i] = feat.flatten()
        for h in [0, 1]:
            e = (xy[2 + h] - xy[h]) / torso
            w = (xy[4 + h] - xy[h]) / torso
            den = np.linalg.norm(e) * np.linalg.norm(w - e)
            if den < 1e-6:
                continue
            a = math.degrees(math.acos(np.clip(np.dot(-e, w - e) / den, -1, 1)))
            geom[i, h] = [w[0], w[1], e[0], e[1], np.linalg.norm(w), a]
    return features, geom, scores
