# PathSnap Laptop 1 backend

FastAPI inference server using the existing saved `LGBMClassifier` and notebook
feature pipeline. The service does not retrain the model. Neither prediction
endpoint reads `.route`; ground truth is used only by the browser for optional
visual comparison.

## Files and model contract

Expected project layout:

```text
PathSnap/
├── dataset/
├── pathsnap_best_model.joblib
├── pathsnap_config.joblib
├── laptop1_backend/
└── pathsnap_frontend/
```

Defaults point to the artifacts beside this backend directory. Override with
`PATHSNAP_DATASET_ROOT`, `PATHSNAP_MODEL_PATH`, or `PATHSNAP_CONFIG_PATH` if
your files are elsewhere.

The saved config and model use `candidate_k=6` and this exact ordered feature
list: `point_seg_dist_m`, `midpoint_dist_m`, `segment_length_m`,
`heading_diff_deg`, `gps_speed_mps`, `time_delta_s`, `candidate_rank`,
`dist_from_best_m`. Inference repeats the notebook's GPS cleaning, nearest
midpoint candidates, geometry and motion features, and saved estimator scoring.
`/predict-upload` then applies the notebook's Viterbi transition/emission
decoder to reconstruct a continuity-aware route.

## Windows PowerShell setup

```powershell
cd C:\path\to\PathSnap\laptop1_backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The saved sklearn object was produced with scikit-learn 1.6.1; requirements pin
that version. Run the tests:

```powershell
python -m unittest discover -s tests -v
```

Start the API locally on Laptop 1:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

For a second laptop, stop the server with `Ctrl+C`, run `ipconfig` to find the
active adapter IPv4 address, and restart it on the LAN:

```powershell
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Allow inbound TCP port 8000 on Windows Firewall's **Private** profile if asked.
From the frontend laptop, use `http://<LAPTOP1_IPV4>:8000` as
`VITE_API_BASE_URL` and check `GET /health`.

## macOS development on one laptop

From this directory, use the tested Python 3.12 environment if present:

```zsh
source .venv312/bin/activate
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

For a new Mac setup:

```zsh
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

## Endpoints

- `GET /health` → `{"status":"ok"}`.
- `GET /trips` → actual student-test trip IDs as strings.
- `GET /trips/{trip_id}/track` → raw points and point count; unknown IDs return
  404.
- `POST /trips/{trip_id}/predict` → the existing student-test prediction
  response. Accuracy and F1 are null because inference never reads references.
- `POST /predict-upload` → multipart prediction for an evaluation trip.

### Upload contract

Send exactly these three multipart fields. All filenames must share the same
eight-digit trip ID:

| Multipart name | File name example |
| --- | --- |
| `track_file` | `00000015.track` |
| `nodes_file` | `00000015.nodes` |
| `arcs_file` | `00000015.arcs` |

Do not send `.route` or any extra form field. `trip_id` is read from the common
filename stem. The endpoint validates numeric formats, finite values,
longitude/latitude ranges, integer arc endpoints, and endpoint indices. Limits:
16 MiB per file, 32 MiB total, 50,000 observations, 250,000 nodes, and 500,000
arcs. One upload inference runs at a time; concurrent uploads receive 429.

Successful response shape:

```json
{
  "trip_id": "00000015",
  "gps_observation_count": 0,
  "predicted_segments": [
    {"observation_index": 0, "arc_id": 0,
     "start": {"latitude": 0, "longitude": 0},
     "end": {"latitude": 0, "longitude": 0}}
  ],
  "reconstructed_segments": [],
  "unmatched_count": 0,
  "inference_ms": 0.0
}
```

The numbers above illustrate the schema, not a sample prediction. The live
response contains actual values. `reconstructed_segments` is `null` only if
Viterbi decoding is unavailable. Validation errors return 400/422; oversized
uploads return 413; model/runtime errors return 503. No accuracy or F1 is
returned by the upload endpoint.

### PowerShell upload example

Run from the evaluator trip directory:

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict-upload `
  -F "track_file=@00000015.track" `
  -F "nodes_file=@00000015.nodes" `
  -F "arcs_file=@00000015.arcs"
```

Quick local checks:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/trips
Invoke-RestMethod http://127.0.0.1:8000/trips/00000000/track
Invoke-RestMethod -Method Post http://127.0.0.1:8000/trips/00000000/predict
```

The frontend setup, evaluator-folder workflow, local ground-truth display, and
API schemas are in `../FRONTEND_README_LAPTOP2.md` and `../pathsnap_frontend/README.md`.
