import os
import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

os.environ.setdefault(
    "PATHSNAP_DATASET_ROOT",
    str(Path(__file__).resolve().parents[2] / "dataset"),
)

import main  # noqa: E402

app = main.app


class PathSnapApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_trips_preserve_ids(self):
        response = self.client.get("/trips")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 10)
        self.assertIn("00000000", response.json())
        self.assertTrue(all(isinstance(trip_id, str) for trip_id in response.json()))

    def test_track_has_expected_actual_count(self):
        response = self.client.get("/trips/00000000/track")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["trip_id"], "00000000")
        self.assertEqual(body["point_count"], 2503)
        self.assertEqual(len(body["points"]), 2503)
        self.assertEqual(set(body["points"][0]), {"latitude", "longitude", "timestamp"})

    def test_unknown_trip_is_404(self):
        self.assertEqual(self.client.get("/trips/99999999/track").status_code, 404)

    def test_existing_prediction_keeps_route_metrics_unavailable(self):
        route_reads = []
        auditing = [True]

        def audit_file_opens(event, args):
            if not auditing[0] or event != "open" or not args:
                return
            try:
                suffix = Path(os.fsdecode(args[0])).suffix
            except (TypeError, ValueError):
                return
            if suffix == ".route":
                route_reads.append(os.fsdecode(args[0]))

        sys.addaudithook(audit_file_opens)
        response = self.client.post("/trips/00000000/predict")
        auditing[0] = False
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["trip_id"], "00000000")
        self.assertEqual(len(body["predicted_segments"]), 2503)
        self.assertEqual(body["unmatched_count"], 0)
        self.assertTrue(all("start" in segment and "end" in segment for segment in body["predicted_segments"]))
        self.assertEqual(route_reads, [])
        metrics = body["metrics"]
        self.assertEqual(set(metrics), {"top1_accuracy", "macro_f1", "route_continuity", "inference_ms"})
        self.assertIsNone(metrics["top1_accuracy"])
        self.assertIsNone(metrics["macro_f1"])
        for key in ("top1_accuracy", "macro_f1", "route_continuity"):
            if metrics[key] is not None:
                self.assertGreaterEqual(metrics[key], 0.0)
                self.assertLessEqual(metrics[key], 1.0)
        self.assertIsInstance(metrics["inference_ms"], (int, float))
        self.assertGreater(metrics["inference_ms"], 0.0)

    def test_upload_prediction_uses_three_files_and_viterbi(self):
        route_reads = []
        auditing = [True]
        def audit_file_opens(event, args):
            if not auditing[0] or event != "open" or not args:
                return
            try:
                suffix = Path(os.fsdecode(args[0])).suffix
            except (TypeError, ValueError):
                return
            if suffix == ".route":
                route_reads.append(os.fsdecode(args[0]))
        sys.addaudithook(audit_file_opens)
        trip_dir = Path(__file__).resolve().parents[2] / "dataset" / "student_test" / "00000000"
        files = {
            "track_file": ("00000000.track", (trip_dir / "00000000.track").read_bytes(), "text/plain"),
            "nodes_file": ("00000000.nodes", (trip_dir / "00000000.nodes").read_bytes(), "text/plain"),
            "arcs_file": ("00000000.arcs", (trip_dir / "00000000.arcs").read_bytes(), "text/plain"),
        }
        response = self.client.post("/predict-upload", files=files)
        auditing[0] = False
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(route_reads, [])
        body = response.json()
        self.assertEqual(body["trip_id"], "00000000")
        self.assertEqual(body["gps_observation_count"], 2503)
        self.assertEqual(len(body["predicted_segments"]), 2503)
        self.assertEqual(len(body["reconstructed_segments"]), 2503)
        self.assertEqual(body["unmatched_count"], 0)
        self.assertGreater(body["inference_ms"], 0)
        self.assertNotIn("top1_accuracy", body)
        self.assertNotIn("macro_f1", body)

    def test_upload_rejects_mismatched_trip_ids(self):
        files = {
            "track_file": ("00000000.track", b"1 2 3\n4 5 6\n", "text/plain"),
            "nodes_file": ("00000001.nodes", b"1 2\n4 5\n", "text/plain"),
            "arcs_file": ("00000000.arcs", b"0 1\n", "text/plain"),
        }
        response = self.client.post("/predict-upload", files=files)
        self.assertEqual(response.status_code, 422)

    def test_frontend_cors_preflight(self):
        response = self.client.options(
            "/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:5173")

    def test_frontend_cors_lan_preflight(self):
        response = self.client.options(
            "/health",
            headers={
                "Origin": "http://10.30.202.39:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://10.30.202.39:5173")


if __name__ == "__main__":
    unittest.main()
