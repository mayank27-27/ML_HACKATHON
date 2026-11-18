"""PathSnap's API using only GPS tracks and road-network files for inference."""

from __future__ import annotations

import os
import asyncio
import io
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware


BACKEND_DIR = Path(__file__).resolve().parent
DATASET_ROOT = Path(
    os.environ.get("PATHSNAP_DATASET_ROOT", str(BACKEND_DIR.parent / "dataset"))
).expanduser().resolve()
MODEL_PATH = Path(os.environ.get("PATHSNAP_MODEL_PATH", "")).expanduser() if os.environ.get("PATHSNAP_MODEL_PATH") else None
CONFIG_PATH = Path(os.environ.get("PATHSNAP_CONFIG_PATH", str(BACKEND_DIR.parent / "pathsnap_config.joblib"))).expanduser()
STUDENT_TEST_DIR = DATASET_ROOT / "student_test"

FEATURES = [
    "point_seg_dist_m", "midpoint_dist_m", "segment_length_m", "heading_diff_deg",
    "gps_speed_mps", "time_delta_s", "candidate_rank", "dist_from_best_m",
]
EARTH_R = 6371000.0
MAX_REASONABLE_SPEED_MPS = 55.0
MAX_UPLOAD_FILE_BYTES = 16 * 1024 * 1024
MAX_UPLOAD_TOTAL_BYTES = 32 * 1024 * 1024
MAX_UPLOAD_OBSERVATIONS = 50000
MAX_UPLOAD_NODES = 250000
MAX_UPLOAD_ARCS = 500000
UPLOAD_INFERENCE_LIMIT = asyncio.Semaphore(1)

app = FastAPI(title="PathSnap API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_origin_regex=r"https?://.*:5173",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def discover_trip_ids() -> list[str]:
    """Return folder names containing a matching .track file, unchanged as strings."""
    if not STUDENT_TEST_DIR.is_dir():
        raise HTTPException(status_code=500, detail=f"student_test directory not found: {STUDENT_TEST_DIR}")
    return sorted(
        p.name for p in STUDENT_TEST_DIR.iterdir()
        if p.is_dir() and (p / f"{p.name}.track").is_file()
    )


def read_track(trip_id: str) -> list[dict[str, Any]]:
    if trip_id not in discover_trip_ids():
        raise HTTPException(status_code=404, detail="Unknown trip_id")
    track_path = STUDENT_TEST_DIR / trip_id / f"{trip_id}.track"
    points: list[dict[str, Any]] = []
    try:
        with track_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) < 3:
                    raise ValueError(f"expected longitude latitude timestamp at line {line_number}")
                longitude, latitude, timestamp = map(float, fields[:3])
                timestamp_value: int | float = int(timestamp) if timestamp.is_integer() else timestamp
                points.append({"latitude": latitude, "longitude": longitude, "timestamp": timestamp_value})
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not read track file: {exc}") from exc
    return points


def read_numeric_file(path: Path, columns: int, dtype=float):
    """Read the notebook's whitespace-delimited, headerless trip files."""
    import numpy as np
    import pandas as pd

    try:
        try:
            values = pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype, engine="c").to_numpy()
        except Exception:
            values = np.loadtxt(path, dtype=dtype, ndmin=2)
    except Exception as exc:
        raise RuntimeError(f"Could not parse {path.name}: {exc}") from exc
    if values.ndim < 2 or values.shape[1] < columns:
        raise RuntimeError(f"Unexpected column count in {path}")
    return values[:, :columns]


def parse_numeric_upload(content: bytes, label: str, columns: int):
    """Parse whitespace-separated upload bytes without writing uploaded files."""
    import numpy as np
    import pandas as pd

    try:
        try:
            values = pd.read_csv(io.BytesIO(content), sep=r"\s+", header=None, dtype=float, engine="c").to_numpy()
        except Exception:
            values = np.loadtxt(io.BytesIO(content), dtype=float, ndmin=2)
    except Exception as exc:
        raise ValueError(f"{label} must be a whitespace-separated numeric file: {exc}") from exc
    if values.ndim < 2 or values.shape[1] < columns:
        raise ValueError(f"{label} must contain at least {columns} columns")
    result = values[:, :columns]
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains non-finite values")
    return result


def parse_uploaded_trip(track_bytes: bytes, nodes_bytes: bytes, arcs_bytes: bytes):
    """Validate upload formats, geographic ranges and arc-to-node references."""
    import numpy as np

    track = parse_numeric_upload(track_bytes, ".track", 3)
    nodes = parse_numeric_upload(nodes_bytes, ".nodes", 2)
    arc_values = parse_numeric_upload(arcs_bytes, ".arcs", 2)
    if len(track) > MAX_UPLOAD_OBSERVATIONS:
        raise ValueError(f".track exceeds {MAX_UPLOAD_OBSERVATIONS} observations")
    if len(nodes) > MAX_UPLOAD_NODES:
        raise ValueError(f".nodes exceeds {MAX_UPLOAD_NODES} nodes")
    if len(arc_values) > MAX_UPLOAD_ARCS:
        raise ValueError(f".arcs exceeds {MAX_UPLOAD_ARCS} arcs")
    if len(track) < 2:
        raise ValueError(".track must contain at least two observations")
    if len(nodes) == 0:
        raise ValueError(".nodes must contain at least one node")
    if len(arc_values) == 0:
        raise ValueError(".arcs must contain at least one arc")
    if (np.abs(track[:, 0]) > 180).any() or (np.abs(track[:, 1]) > 90).any():
        raise ValueError(".track longitude/latitude must be within [-180,180] and [-90,90]")
    if (np.abs(nodes[:, 0]) > 180).any() or (np.abs(nodes[:, 1]) > 90).any():
        raise ValueError(".nodes longitude/latitude must be within [-180,180] and [-90,90]")
    if not np.equal(arc_values, np.floor(arc_values)).all():
        raise ValueError(".arcs endpoints must be integer node indices")
    arcs = arc_values.astype(np.int64)
    if ((arcs < 0) | (arcs >= len(nodes))).any():
        raise ValueError(".arcs contains node indices outside the .nodes file")
    return track, nodes, arcs


def clean_track_for_modelling(raw_track):
    """Same GPS cleaning steps and order as the training notebook."""
    import numpy as np

    raw = np.asarray(raw_track, dtype=float)
    keep = np.isfinite(raw[:, :3]).all(axis=1)
    work = raw[keep].copy()
    geo = (np.abs(work[:, 0]) <= 180) & (np.abs(work[:, 1]) <= 90)
    work = work[geo]
    if len(work):
        exact_dup = np.zeros(len(work), dtype=bool)
        exact_dup[1:] = np.all(work[1:] == work[:-1], axis=1)
        work = work[~exact_dup]
    if len(work):
        seen = set()
        timestamp_dup = np.zeros(len(work), dtype=bool)
        for i, timestamp in enumerate(work[:, 2]):
            key = float(timestamp)
            if key in seen:
                timestamp_dup[i] = True
            else:
                seen.add(key)
        work = work[~timestamp_dup]
    return work


def local_xy(lon, lat, lon0, lat0):
    import numpy as np
    x = np.radians(lon - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_R
    return np.column_stack([x, y])


def point_segment_distance(P, A, B):
    import numpy as np
    AB = B - A
    denom = np.maximum(np.sum(AB * AB, axis=1), 1e-9)
    AP = P - A
    u = np.clip(np.sum(AP * AB, axis=1) / denom, 0, 1)
    Q = A + u[:, None] * AB
    return np.sqrt(np.sum((P - Q) ** 2, axis=1))


def bearing_xy(v):
    import numpy as np
    return (np.degrees(np.arctan2(v[:, 0], v[:, 1])) + 360) % 360


def angle_diff(a, b):
    import numpy as np
    return np.abs((a - b + 180) % 360 - 180)


def make_inference_candidates(track, nodes, arcs, k: int):
    """Reproduce notebook candidate/features, omitting route-derived labels only."""
    import numpy as np
    import pandas as pd
    from scipy.spatial import cKDTree

    if len(track) < 2 or len(arcs) == 0:
        return pd.DataFrame(columns=["obs_idx", "arc_id", *FEATURES]), len(track)
    valid_arcs = ((arcs[:, 0] >= 0) & (arcs[:, 0] < len(nodes)) &
                  (arcs[:, 1] >= 0) & (arcs[:, 1] < len(nodes)))
    if not np.all(valid_arcs):
        raise RuntimeError(f"Invalid arc endpoints: {int((~valid_arcs).sum())}")
    arc_ids = np.flatnonzero(valid_arcs)
    if not len(arc_ids):
        return pd.DataFrame(columns=["obs_idx", "arc_id", *FEATURES]), len(track)

    lon0 = float(np.mean(track[:, 0])); lat0 = float(np.mean(track[:, 1]))
    P = local_xy(track[:, 0], track[:, 1], lon0, lat0)
    N = local_xy(nodes[:, 0], nodes[:, 1], lon0, lat0)
    starts, ends = arcs[arc_ids, 0], arcs[arc_ids, 1]
    A, B = N[starts], N[ends]
    mids = (A + B) / 2
    segvec = B - A
    seglen = np.sqrt((segvec ** 2).sum(axis=1))
    segbearing = bearing_xy(segvec)
    kk = min(k, len(arc_ids))
    _, local_candidates = cKDTree(mids).query(P, k=kk)
    if kk == 1:
        local_candidates = local_candidates[:, None]

    dxy = np.diff(P, axis=0, prepend=P[:1])
    dt = np.diff(track[:, 2], prepend=track[0, 2])
    positive_dt = dt[dt > 0]
    fallback_dt = float(np.median(positive_dt)) if len(positive_dt) else 1.0
    dt = np.where(dt > 0, dt, fallback_dt)
    speed = np.clip(np.sqrt((dxy ** 2).sum(axis=1)) / dt, 0, MAX_REASONABLE_SPEED_MPS)
    heading = bearing_xy(dxy)

    records = []
    for i in range(len(P)):
        local_ids = local_candidates[i]
        ci = arc_ids[local_ids]
        dist = point_segment_distance(np.repeat(P[i:i+1], kk, axis=0), A[local_ids], B[local_ids])
        to_mid = np.sqrt(((mids[local_ids] - P[i]) ** 2).sum(axis=1))
        bdiff = angle_diff(np.full(kk, heading[i]), segbearing[local_ids])
        best = float(np.min(dist))
        # The notebook rank(method="first") preserves the candidate order on ties.
        ranks = np.argsort(np.argsort(dist, kind="stable"), kind="stable") + 1
        for j, arc_id in enumerate(ci):
            records.append((i, int(arc_id), float(dist[j]), float(to_mid[j]),
                            float(seglen[local_ids[j]]), float(bdiff[j]), float(speed[i]),
                            float(dt[i]), float(ranks[j]), float(dist[j] - best)))
    return pd.DataFrame(records, columns=["obs_idx", "arc_id", *FEATURES]), 0


@lru_cache(maxsize=1)
def load_predictor():
    """Load and validate the saved artifacts; never fit or modify the estimator."""
    import joblib

    model_file = MODEL_PATH or (BACKEND_DIR.parent / "pathsnap_best_model.joblib")
    if not model_file.is_file():
        raise FileNotFoundError(f"Saved model not found: {model_file}")
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Saved model config not found: {CONFIG_PATH}")
    model = joblib.load(model_file)
    config = joblib.load(CONFIG_PATH)
    configured_features = config.get("features")
    if configured_features != FEATURES:
        raise RuntimeError(f"Model config feature order mismatch: {configured_features!r}")
    if hasattr(model, "n_features_in_") and int(model.n_features_in_) != len(FEATURES):
        raise RuntimeError(f"Estimator expects {model.n_features_in_} features; config has {len(FEATURES)}")
    booster = getattr(model, "booster_", None)
    if booster is not None and list(booster.feature_name()) != FEATURES:
        raise RuntimeError(f"Estimator feature order mismatch: {booster.feature_name()!r}")
    if not callable(getattr(model, "predict_proba", None)):
        raise RuntimeError("Saved estimator does not provide predict_proba")
    return model, config


def continuity_rate(path, arcs):
    """Port of the notebook's classifier route-continuity calculation."""
    import numpy as np

    if len(path) < 2:
        return np.nan
    valid = [(a, b) for a, b in zip(path[:-1], path[1:])
             if 0 <= a < len(arcs) and 0 <= b < len(arcs)]
    if not valid:
        return np.nan
    return float(np.mean([arcs[a, 1] == arcs[b, 0] for a, b in valid]))


def collect_optional_metrics(arcs, predicted_segments, inference_ms):
    """Return inference-only metrics; ground-truth metrics need separate evaluation."""
    metrics: dict[str, float | None] = {
        "top1_accuracy": None,
        "macro_f1": None,
        "route_continuity": None,
        "inference_ms": float(inference_ms),
    }
    try:
        path = [segment["arc_id"] for segment in predicted_segments]
        continuity = continuity_rate(path, arcs)
        import numpy as np
        metrics["route_continuity"] = float(continuity) if np.isfinite(continuity) else None
    except Exception:
        pass
    return metrics


def viterbi_decode_for_trip(candidate_rows, probabilities, arcs, jump_penalty=5.0):
    """Port of notebook viterbi_decode_for_trip without loading route labels."""
    import numpy as np

    tmp = candidate_rows[["obs_idx", "arc_id"]].copy()
    tmp["prob"] = np.asarray(probabilities)
    groups = [group.sort_values("prob", ascending=False).head(8)
              for _, group in tmp.groupby("obs_idx")]
    if not groups:
        return np.array([], dtype=int)
    states = [group.arc_id.astype(int).tolist() for group in groups]
    emissions = [np.log(np.maximum(group.prob.to_numpy(), 1e-9)) for group in groups]
    dp = [emissions[0].copy()]
    back = []
    for t in range(1, len(states)):
        prev, cur = states[t - 1], states[t]
        scores = np.full((len(prev), len(cur)), -jump_penalty, dtype=float)
        for i, arc_a in enumerate(prev):
            end_node = int(arcs[arc_a, 1])
            for j, arc_b in enumerate(cur):
                if end_node == int(arcs[arc_b, 0]):
                    scores[i, j] = 0.0
                elif arc_a == arc_b:
                    scores[i, j] = -0.5
        total = dp[-1][:, None] + scores
        ptr = np.argmax(total, axis=0)
        dp.append(emissions[t] + total[ptr, np.arange(len(cur))])
        back.append(ptr)
    path = [int(np.argmax(dp[-1]))]
    for ptr in reversed(back):
        path.append(int(ptr[path[-1]]))
    path.reverse()
    return np.array([states[t][path[t]] for t in range(len(path))], dtype=int)


def arc_segment(arc_id: int, observation_index: int, nodes, arcs) -> dict[str, Any]:
    from_node, to_node = map(int, arcs[arc_id])
    return {
        "observation_index": int(observation_index),
        "arc_id": int(arc_id),
        "start": {"latitude": float(nodes[from_node, 1]), "longitude": float(nodes[from_node, 0])},
        "end": {"latitude": float(nodes[to_node, 1]), "longitude": float(nodes[to_node, 0])},
    }


def predict_uploaded_trip(track_bytes: bytes, nodes_bytes: bytes, arcs_bytes: bytes, trip_id: str):
    """Run the saved-model plus HMM pipeline on uploaded files, with no route input."""
    import numpy as np

    model, config = load_predictor()
    # Warm the numerical imports before starting request inference timing.
    import pandas as pd  # noqa: F401
    from scipy.spatial import cKDTree  # noqa: F401

    inference_started = time.perf_counter()
    raw_track, nodes, arcs = parse_uploaded_trip(track_bytes, nodes_bytes, arcs_bytes)
    gps_observation_count = len(raw_track)
    track = clean_track_for_modelling(raw_track)
    candidates, initially_unmatched = make_inference_candidates(
        track, nodes, arcs, int(config["candidate_k"])
    )
    if candidates.empty:
        return {
            "trip_id": trip_id,
            "gps_observation_count": gps_observation_count,
            "predicted_segments": [],
            "reconstructed_segments": None,
            "unmatched_count": gps_observation_count,
            "inference_ms": (time.perf_counter() - inference_started) * 1000.0,
        }

    X = candidates[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    probabilities = model.predict_proba(X)[:, 1]
    scored = candidates.assign(score=probabilities)
    chosen = (scored.sort_values(["obs_idx", "score"], ascending=[True, False], kind="stable")
              .drop_duplicates("obs_idx"))
    predicted_segments = [arc_segment(int(row.arc_id), int(row.obs_idx), nodes, arcs)
                          for row in chosen.itertuples(index=False)]
    unmatched_count = int(
        gps_observation_count - len(track) + initially_unmatched + len(track) - len(chosen)
    )

    reconstructed_segments = None
    try:
        decoded = viterbi_decode_for_trip(candidates, probabilities, arcs)
        reconstructed_segments = [arc_segment(int(arc_id), obs_idx, nodes, arcs)
                                  for obs_idx, arc_id in enumerate(decoded)]
    except Exception:
        # The classifier output remains usable if continuity decoding is unavailable.
        reconstructed_segments = None

    inference_ms = (time.perf_counter() - inference_started) * 1000.0
    return {
        "trip_id": trip_id,
        "gps_observation_count": gps_observation_count,
        "predicted_segments": predicted_segments,
        "reconstructed_segments": reconstructed_segments,
        "unmatched_count": unmatched_count,
        "inference_ms": float(inference_ms),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/trips")
def trips() -> list[str]:
    return discover_trip_ids()


@app.get("/trips/{trip_id}/track")
def trip_track(trip_id: str) -> dict[str, Any]:
    points = read_track(trip_id)
    return {"trip_id": trip_id, "point_count": len(points), "points": points}


@app.post("/trips/{trip_id}/predict")
def predict_trip(trip_id: str) -> dict[str, Any]:
    # Prediction uses track, nodes and arcs only; the reference route is never read.
    read_track(trip_id)
    trip_dir = STUDENT_TEST_DIR / trip_id
    try:
        # Load the cached estimator and runtime modules before timing so this
        # field measures request inference work, not a cold Python/model import.
        model, config = load_predictor()
        import numpy as np
        import pandas as pd  # noqa: F401
        from scipy.spatial import cKDTree  # noqa: F401

        inference_started = time.perf_counter()
        raw_track = read_numeric_file(trip_dir / f"{trip_id}.track", 3, float)
        nodes = read_numeric_file(trip_dir / f"{trip_id}.nodes", 2, float)
        arcs = read_numeric_file(trip_dir / f"{trip_id}.arcs", 2, int)
        track = clean_track_for_modelling(raw_track)
        candidates, initially_unmatched = make_inference_candidates(track, nodes, arcs, int(config["candidate_k"]))
        if candidates.empty:
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            predicted_segments = []
            metrics = collect_optional_metrics(arcs, predicted_segments, inference_ms)
            return {"trip_id": trip_id, "predicted_segments": predicted_segments,
                    "unmatched_count": len(raw_track), "metrics": metrics}
        X = candidates[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        probabilities = model.predict_proba(X)[:, 1]
        candidates = candidates.assign(score=probabilities)
        chosen = candidates.sort_values(["obs_idx", "score"], ascending=[True, False], kind="stable").drop_duplicates("obs_idx")
        predicted = []
        for row in chosen.itertuples(index=False):
            arc_id = int(row.arc_id)
            from_node, to_node = map(int, arcs[arc_id])
            predicted.append({
                "observation_index": int(row.obs_idx), "arc_id": arc_id,
                "start": {"latitude": float(nodes[from_node, 1]), "longitude": float(nodes[from_node, 0])},
                "end": {"latitude": float(nodes[to_node, 1]), "longitude": float(nodes[to_node, 0])},
            })
        inference_ms = (time.perf_counter() - inference_started) * 1000.0
        unmatched_count = int(len(raw_track) - len(track) + initially_unmatched + len(track) - len(chosen))
        metrics = collect_optional_metrics(arcs, predicted, inference_ms)
        return {"trip_id": trip_id, "predicted_segments": predicted,
                "unmatched_count": unmatched_count, "metrics": metrics}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Prediction unavailable: {exc}") from exc


def uploaded_trip_id(upload, expected_suffix: str) -> str:
    filename = upload.filename or ""
    normalized = filename.replace("\\", "/")
    # Browser directory uploads may include the selected folder as a prefix.
    # Only the final uploaded filename identifies the trip; no path is opened.
    basename = normalized.rsplit("/", 1)[-1]
    if not basename.lower().endswith(expected_suffix):
        raise ValueError(f"Upload must be named <trip_id>{expected_suffix}")
    trip_id = basename[:-len(expected_suffix)]
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", trip_id):
        raise ValueError(f"Invalid trip ID in filename {filename!r}")
    return trip_id


async def read_limited_upload(upload, suffix: str) -> bytes:
    content = await upload.read(MAX_UPLOAD_FILE_BYTES + 1)
    if len(content) > MAX_UPLOAD_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"{suffix} exceeds 16 MiB")
    if not content:
        raise HTTPException(status_code=422, detail=f"{suffix} is empty")
    return content


@app.post("/predict-upload")
async def predict_upload(request: Request) -> dict[str, Any]:
    """Accept exactly one trip's track, nodes and arcs; never accepts route data."""
    if UPLOAD_INFERENCE_LIMIT.locked():
        raise HTTPException(status_code=429, detail="Another upload prediction is running; retry shortly")
    async with UPLOAD_INFERENCE_LIMIT:
        try:
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > MAX_UPLOAD_TOTAL_BYTES + 1024 * 1024:
                raise HTTPException(status_code=413, detail="Multipart request exceeds 33 MiB")
            form = await request.form(max_files=3, max_fields=3, max_part_size=MAX_UPLOAD_FILE_BYTES)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid multipart upload: {exc}") from exc

        try:
            expected_fields = {"track_file", "nodes_file", "arcs_file"}
            if set(form.keys()) != expected_fields:
                raise HTTPException(
                    status_code=422,
                    detail="Send exactly track_file, nodes_file and arcs_file; .route is not accepted",
                )
            values = {key: form.getlist(key) for key in expected_fields}
            if any(len(items) != 1 or not hasattr(items[0], "read") for items in values.values()):
                raise HTTPException(status_code=422, detail="Each multipart field must contain one file")
            track_upload = values["track_file"][0]
            nodes_upload = values["nodes_file"][0]
            arcs_upload = values["arcs_file"][0]
            try:
                ids = [
                    uploaded_trip_id(track_upload, ".track"),
                    uploaded_trip_id(nodes_upload, ".nodes"),
                    uploaded_trip_id(arcs_upload, ".arcs"),
                ]
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            if len(set(ids)) != 1:
                raise HTTPException(status_code=422, detail="Uploaded filenames must have matching trip IDs")

            track_bytes = await read_limited_upload(track_upload, ".track")
            nodes_bytes = await read_limited_upload(nodes_upload, ".nodes")
            arcs_bytes = await read_limited_upload(arcs_upload, ".arcs")
            if len(track_bytes) + len(nodes_bytes) + len(arcs_bytes) > MAX_UPLOAD_TOTAL_BYTES:
                raise HTTPException(status_code=413, detail="Combined uploaded files exceed 32 MiB")

            from starlette.concurrency import run_in_threadpool
            try:
                return await run_in_threadpool(
                    predict_uploaded_trip, track_bytes, nodes_bytes, arcs_bytes, ids[0]
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"Upload prediction unavailable: {exc}") from exc
        finally:
            await form.close()
