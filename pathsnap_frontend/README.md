# PathSnap evaluation UI — Laptop 2

React, Vite, TypeScript and React Leaflet UI for loading the evaluator folder,
running saved-model inference, and comparing prediction with ground truth.
The `.route` file stays in browser memory for the ground-truth map layer. The
frontend's API helper explicitly sends only `.track`, `.nodes`, and `.arcs` to
`POST /predict-upload`.

## Run everything on one Mac

Open two Terminal windows from the project root.

**Terminal 1 — backend:**

```zsh
cd laptop1_backend
source .venv312/bin/activate
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — UI:**

```zsh
cd pathsnap_frontend
npm install
cp .env.example .env.local
npm run dev -- --host 127.0.0.1
```

Open `http://localhost:5173`. The UI shows API health in the header. Choose
`/Users/apple/Downloads/evaluator_test/` in the folder picker, select a trip,
run prediction, then enable **Ground truth · local** to compare the supplied
reference route.

If the local Python virtual environment is missing, create it in `laptop1_backend`:

```zsh
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install -r requirements.txt
```

## Connect two Windows laptops

On Laptop 1, activate the backend environment and run:

```powershell
cd C:\path\to\PathSnap\laptop1_backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Run `ipconfig` on Laptop 1 and find the active Wi-Fi/Ethernet **IPv4 Address**.
If prompted, allow Python through Windows Defender Firewall on Private
networks. If needed, in elevated PowerShell:

```powershell
New-NetFirewallRule -DisplayName "PathSnap FastAPI 8000" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 -Profile Private
```

On Laptop 2:

```powershell
cd C:\path\to\PathSnap\pathsnap_frontend
npm install
Copy-Item .env.example .env.local
notepad .env.local
```

Set `VITE_API_BASE_URL=http://<LAPTOP1_IPV4>:8000` in `.env.local`, save, then:

```powershell
npm run dev -- --host 0.0.0.0
```

Open the Vite URL printed by the command, usually `http://localhost:5173`. CORS
allows this origin. Verify Laptop 1 is reachable from Laptop 2:

```powershell
Invoke-RestMethod http://<LAPTOP1_IPV4>:8000/health
```

Expected: `status : ok`.

## Evaluator-folder workflow

The browser folder picker accepts a directory containing folders such as:

```text
evaluator_test/
├── 00000015/
│   ├── 00000015.track
│   ├── 00000015.nodes
│   ├── 00000015.arcs
│   └── 00000015.route
└── ...
```

The current reference folder contains these 10 IDs:
`00000015`, `00000017`, `00000023`, `00000030`, `00000049`, `00000059`,
`00000067`, `00000073`, `00000087`, `00000094`.

The UI validates matching IDs and parses GPS, nodes, arcs, and the optional
reference locally. It initially shows the GPS track/points. **Run prediction**
sends exactly these multipart fields:

| Form field | File |
| --- | --- |
| `track_file` | matching `<trip_id>.track` |
| `nodes_file` | matching `<trip_id>.nodes` |
| `arcs_file` | matching `<trip_id>.arcs` |

No `route_file` field is created. The `.route` file is never sent to the API.
After a successful response, the map can show classifier top-1 segments,
Viterbi reconstructed segments, and (when enabled) the reference overlay
computed locally from `.route` plus `.nodes`/`.arcs`. Every road segment is
drawn as its own two-point line; gaps are not joined. Ground truth is gated
until a prediction has completed, so the comparison is an explicit final step.

## API contract

Base URL: `VITE_API_BASE_URL`.

- `GET /health` → `{"status":"ok"}`.
- `GET /trips` → string trip IDs for the backend's student-test set.
- `GET /trips/{trip_id}/track` → `trip_id`, `point_count`, and GPS `points`.
- `POST /trips/{trip_id}/predict` → existing student-test model prediction.
- `POST /predict-upload` → evaluator upload request described below.

### Upload request example

From PowerShell in a trip folder:

```powershell
curl.exe -X POST "$env:VITE_API_BASE_URL/predict-upload" `
  -F "track_file=@00000015.track" `
  -F "nodes_file=@00000015.nodes" `
  -F "arcs_file=@00000015.arcs"
```

Expected multipart keys are exactly `track_file`, `nodes_file`, and `arcs_file`.
File basenames must share the same trip ID. Extra fields are rejected. The
endpoint validates whitespace-separated numeric formats, coordinate bounds,
integer endpoint IDs and arc-to-node references. A file may be at most 16 MiB,
the combined files at most 32 MiB. Oversized uploads return 413, malformed or
mismatched inputs return 422, simultaneous upload inference returns 429, and
model failures return 503.

### Upload response schema

```json
{
  "trip_id": "00000015",
  "gps_observation_count": 0,
  "predicted_segments": [
    {
      "observation_index": 0,
      "arc_id": 0,
      "start": {"latitude": 0, "longitude": 0},
      "end": {"latitude": 0, "longitude": 0}
    }
  ],
  "reconstructed_segments": [
    {
      "observation_index": 0,
      "arc_id": 0,
      "start": {"latitude": 0, "longitude": 0},
      "end": {"latitude": 0, "longitude": 0}
    }
  ],
  "unmatched_count": 0,
  "inference_ms": 0.0
}
```

This is a typed shape example, not real result data. Actual numbers and arcs
come from the selected files and saved model. `reconstructed_segments` may be
`null` if the decoder cannot produce a path. The upload response includes no
accuracy or F1; the API has no `.route` access. The UI displays unmatched count
and inference time only as returned, and does not invent evaluation metrics.

## Build, test and demo

Build the frontend:

```powershell
npm run build
```

For the demo: verify `/health`, select the evaluator folder, select a trip,
show GPS observations, run the model, show predicted and Viterbi layers, then
enable local ground truth. Keep the browser Network panel visible if you need to
demonstrate that the upload request contains only the three allowed file
fields. A request/response can also be shown in the Laptop 2 PowerShell window
using the `curl.exe` example above.

## Screenshots for the report

Capture a UI screenshot after one trip has loaded with the GPS and prediction
layers, and a second screenshot after enabling the local ground-truth layer.
For the API evidence, run the `curl.exe` multipart example from a PowerShell
window on the receiving laptop and capture the returned JSON. If testing on a
single computer, label those screenshots as local verification; do not present
them as proof of a two-device connection.
