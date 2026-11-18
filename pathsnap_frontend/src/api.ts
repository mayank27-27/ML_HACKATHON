export interface GeoPoint {
  latitude: number;
  longitude: number;
}

export interface Segment {
  observation_index: number;
  arc_id: number;
  start: GeoPoint;
  end: GeoPoint;
}

export interface UploadPredictionResponse {
  trip_id: string;
  gps_observation_count: number;
  predicted_segments: Segment[];
  reconstructed_segments: Segment[] | null;
  unmatched_count: number;
  inference_ms: number;
}

const baseUrl = () =>
  (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

export async function checkHealth(signal?: AbortSignal): Promise<void> {
  const response = await fetch(`${baseUrl()}/health`, { signal });
  if (!response.ok) throw new Error(`Health check failed (${response.status})`);
  const body = (await response.json()) as { status?: string };
  if (body.status !== 'ok') throw new Error('Backend returned an unexpected health response');
}

export async function predictUploadedTrip(files: {
  track: File;
  nodes: File;
  arcs: File;
}): Promise<UploadPredictionResponse> {
  const form = new FormData();
  // Intentionally append only the three prediction inputs. The local .route
  // file is never included in this form or sent to the backend.
  form.append('track_file', files.track);
  form.append('nodes_file', files.nodes);
  form.append('arcs_file', files.arcs);

  const response = await fetch(`${baseUrl()}/predict-upload`, {
    method: 'POST',
    body: form,
    signal: AbortSignal.timeout(120_000),
  });
  if (!response.ok) {
    const problem = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(problem?.detail || `Prediction failed (${response.status})`);
  }
  return (await response.json()) as UploadPredictionResponse;
}
