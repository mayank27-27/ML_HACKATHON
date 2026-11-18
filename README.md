# PathSnap

PathSnap is a road map-matching project with a saved LightGBM model, a FastAPI inference service, and a React + Vite evaluation UI.

## Project contents

- `PathSnap_Model_Training_and_Evaluation_CLEANED_COMPLETE-2.ipynb` contains the cleaned-data workflow, candidate generation, feature engineering, model comparison and tuning, observation-level evaluation, HMM/Viterbi decoding, and Folium visualizations.
- `pathsnap_best_model.joblib` and `pathsnap_config.joblib` are the saved estimator and its inference configuration. The application loads these artifacts; it does not retrain.
- `laptop1_backend/` contains the FastAPI application, inference pipeline, dependencies, tests, and backend setup instructions.
- `pathsnap_frontend/` contains the React/TypeScript evaluator UI and its run instructions.
- `FRONTEND_README_LAPTOP2.md` is the teammate handoff for running the UI on a second laptop.

## Local data

Raw GPS trip files are not included in this public repository. Keep authorized data locally under `dataset/` using the directory structure described in `laptop1_backend/README.md`. The evaluation UI can also load an evaluator folder directly. Its `.route` reference file stays in the browser and is never sent to the prediction API.

## Quick start on macOS

From the project root, install and start the backend:

```bash
uv venv "$HOME/.venvs/pathsnap-backend" --python 3.12
uv pip install --python "$HOME/.venvs/pathsnap-backend/bin/python" -r laptop1_backend/requirements.txt
cd laptop1_backend
"$HOME/.venvs/pathsnap-backend/bin/python" -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info
```

In a second Terminal, start the UI:

```bash
cd pathsnap_frontend
npm ci
npm run dev -- --host 0.0.0.0
```

Open `http://localhost:5173/`. For a two-laptop setup, follow `FRONTEND_README_LAPTOP2.md` and set `VITE_API_BASE_URL` to Laptop 1's LAN IPv4 address.

## Evaluation behavior

The API receives only `.track`, `.nodes`, and `.arcs` for an uploaded trip. The browser reads `.route` locally and calculates selected-trip metrics using the notebook's nearest-reference-arc and candidate-hit evaluation logic. Macro F1 averages over the union of reference and predicted arc IDs among aligned observations.
