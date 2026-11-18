import { useEffect, useMemo, useRef, useState } from 'react';
import type { ChangeEvent, ReactNode } from 'react';
import {
  CircleMarker,
  MapContainer,
  Polyline,
  TileLayer,
  useMap,
} from 'react-leaflet';
import type { LatLngBoundsExpression, LatLngExpression } from 'leaflet';
import { checkHealth, predictUploadedTrip } from './api';
import type { GeoPoint, Segment, UploadPredictionResponse } from './api';
import './App.css';

type FileGroup = { track?: File; nodes?: File; arcs?: File; route?: File };
type Dataset = Record<string, Required<Pick<FileGroup, 'track' | 'nodes' | 'arcs'>> & Pick<FileGroup, 'route'>>;
type ParsedTrip = {
  points: GeoPoint[];
  arcs: Array<[number, number]>;
  alignedReferenceArcIds: Array<number | null>;
  referenceCandidateHits: boolean[];
  metricObservationCount: number;
  referenceSegments: Segment[];
  invalidReferenceCount: number;
};

type LocalMetrics = {
  top1Accuracy: number | null;
  macroF1: number | null;
  coverage: number | null;
  effectiveAccuracy: number | null;
  routeContinuity: number | null;
};

function tripFileInfo(file: File): { tripId: string; ext: keyof FileGroup } | null {
  const match = file.name.match(/^(\d{8})\.(track|nodes|arcs|route)$/i);
  if (!match) return null;
  return { tripId: match[1], ext: match[2].toLowerCase() as keyof FileGroup };
}

function parseRows(source: string, minColumns: number, fileLabel: string): number[][] {
  const rows: number[][] = [];
  for (const [lineIndex, line] of source.split(/\r?\n/).entries()) {
    const content = line.split('#', 1)[0].trim();
    if (!content) continue;
    const values = content.split(/\s+/).map(Number);
    if (values.length < minColumns || values.some((value) => !Number.isFinite(value))) {
      throw new Error(`${fileLabel} has invalid numeric data on line ${lineIndex + 1}`);
    }
    rows.push(values.slice(0, minColumns));
  }
  if (!rows.length) throw new Error(`${fileLabel} is empty`);
  return rows;
}

const EARTH_RADIUS_M = 6_371_000;
const TRAINING_CANDIDATE_K = 6;

function projectLocal(longitude: number, latitude: number, longitude0: number, latitude0: number): [number, number] {
  const x = ((longitude - longitude0) * Math.PI / 180) * EARTH_RADIUS_M * Math.cos(latitude0 * Math.PI / 180);
  const y = ((latitude - latitude0) * Math.PI / 180) * EARTH_RADIUS_M;
  return [x, y];
}

function pointSegmentDistance(point: [number, number], start: [number, number], end: [number, number]): number {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const denominator = Math.max(dx * dx + dy * dy, 1e-9);
  const fraction = Math.max(0, Math.min(1, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator));
  return Math.hypot(point[0] - (start[0] + fraction * dx), point[1] - (start[1] + fraction * dy));
}

async function parseTrip(files: Dataset[string]): Promise<ParsedTrip> {
  const [trackText, nodesText, arcsText, routeText] = await Promise.all([
    files.track.text(), files.nodes.text(), files.arcs.text(), files.route?.text(),
  ]);
  const trackRows = parseRows(trackText, 3, '.track');
  const nodeRows = parseRows(nodesText, 2, '.nodes');
  const arcRows = parseRows(arcsText, 2, '.arcs');
  const points = trackRows.map(([longitude, latitude]) => {
    if (Math.abs(longitude) > 180 || Math.abs(latitude) > 90) {
      throw new Error('.track contains coordinates outside longitude/latitude ranges');
    }
    return { latitude, longitude };
  });
  for (const [longitude, latitude] of nodeRows) {
    if (Math.abs(longitude) > 180 || Math.abs(latitude) > 90) {
      throw new Error('.nodes contains coordinates outside longitude/latitude ranges');
    }
  }
  const arcs = arcRows.map(([from, to], index) => {
    if (!Number.isInteger(from) || !Number.isInteger(to) || from < 0 || to < 0 || from >= nodeRows.length || to >= nodeRows.length) {
      throw new Error(`.arcs has an invalid node index on row ${index + 1}`);
    }
    return [from, to] as [number, number];
  });
  const routeIds = routeText ? parseRows(routeText, 1, '.route').map(([arcId]) => arcId) : [];
  const referenceSegments: Segment[] = [];
  let invalidReferenceCount = 0;
  routeIds.forEach((arcId, index) => {
    if (!Number.isInteger(arcId) || arcId < 0 || arcId >= arcs.length) {
      invalidReferenceCount += 1;
      return;
    }
    const [from, to] = arcs[arcId];
    const [fromLon, fromLat] = nodeRows[from];
    const [toLon, toLat] = nodeRows[to];
    referenceSegments.push({
      observation_index: index,
      arc_id: arcId,
      start: { latitude: fromLat, longitude: fromLon },
      end: { latitude: toLat, longitude: toLon },
    });
  });

  // Mirror the notebook's modelling cleaner before aligning labels. Prediction
  // observation_index values refer to this cleaned sequence, not route rows.
  const cleanedRows = trackRows.filter((row) => Math.abs(row[0]) <= 180 && Math.abs(row[1]) <= 90);
  const deduplicatedRows: number[][] = [];
  const seenTimestamps = new Set<number>();
  for (const row of cleanedRows) {
    const previous = deduplicatedRows[deduplicatedRows.length - 1];
    if (previous && row.every((value, index) => value === previous[index])) continue;
    if (seenTimestamps.has(row[2])) continue;
    seenTimestamps.add(row[2]);
    deduplicatedRows.push(row);
  }
  const validRouteIds = [...new Set(routeIds.filter((arcId) => Number.isInteger(arcId) && arcId >= 0 && arcId < arcs.length))]
    .sort((a, b) => a - b);
  const longitude0 = deduplicatedRows.reduce((sum, row) => sum + row[0], 0) / Math.max(1, deduplicatedRows.length);
  const latitude0 = deduplicatedRows.reduce((sum, row) => sum + row[1], 0) / Math.max(1, deduplicatedRows.length);
  const projectedNodes = nodeRows.map(([longitude, latitude]) => projectLocal(longitude, latitude, longitude0, latitude0));
  const projectedPoints = deduplicatedRows.map(([longitude, latitude]) => projectLocal(longitude, latitude, longitude0, latitude0));
  const arcMidpoints = arcs.map(([from, to]) => {
    const start = projectedNodes[from];
    const end = projectedNodes[to];
    return [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2] as [number, number];
  });
  const alignedReferenceArcIds: Array<number | null> = [];
  const referenceCandidateHits: boolean[] = [];
  for (const point of projectedPoints) {
    let nearestReference: number | null = null;
    let nearestReferenceDistance = Number.POSITIVE_INFINITY;
    for (const arcId of validRouteIds) {
      const [from, to] = arcs[arcId];
      const distance = pointSegmentDistance(point, projectedNodes[from], projectedNodes[to]);
      if (distance < nearestReferenceDistance) {
        nearestReferenceDistance = distance;
        nearestReference = arcId;
      }
    }
    alignedReferenceArcIds.push(nearestReference);

    // Match the notebook/model candidate builder: nearest segment midpoints
    // in local metres, with K taken from the saved training configuration.
    const nearestCandidates: Array<{ distance: number; arcId: number }> = [];
    for (let arcId = 0; arcId < arcMidpoints.length; arcId += 1) {
      const midpoint = arcMidpoints[arcId];
      const distance = Math.hypot(point[0] - midpoint[0], point[1] - midpoint[1]);
      let insertAt = nearestCandidates.findIndex((candidate) => distance < candidate.distance);
      if (insertAt < 0) insertAt = nearestCandidates.length;
      if (insertAt < TRAINING_CANDIDATE_K) {
        nearestCandidates.splice(insertAt, 0, { distance, arcId });
        if (nearestCandidates.length > TRAINING_CANDIDATE_K) nearestCandidates.pop();
      }
    }
    referenceCandidateHits.push(nearestReference !== null && nearestCandidates.some((candidate) => candidate.arcId === nearestReference));
  }
  return {
    points, arcs, alignedReferenceArcIds, referenceCandidateHits,
    metricObservationCount: deduplicatedRows.length, referenceSegments, invalidReferenceCount,
  };
}

function evaluateLocally(parsed: ParsedTrip, prediction: UploadPredictionResponse): LocalMetrics {
  const predictedByObservation = new Map(
    prediction.predicted_segments.map((segment) => [segment.observation_index, segment.arc_id]),
  );
  const aligned: Array<{ truth: number; predicted: number }> = [];
  parsed.alignedReferenceArcIds.forEach((truth, observationIndex) => {
    if (truth === null || !parsed.referenceCandidateHits[observationIndex]) return;
    const predicted = predictedByObservation.get(observationIndex);
    if (predicted !== undefined) aligned.push({ truth, predicted });
  });

  const correctCount = aligned.filter(({ truth, predicted }) => truth === predicted).length;
  const top1Accuracy = aligned.length ? correctCount / aligned.length : null;
  const coverage = parsed.metricObservationCount ? aligned.length / parsed.metricObservationCount : null;
  const effectiveAccuracy = parsed.metricObservationCount ? correctCount / parsed.metricObservationCount : null;
  let macroF1: number | null = null;
  if (aligned.length) {
    const labels = new Set(aligned.flatMap(({ truth, predicted }) => [truth, predicted]));
    const f1Scores = [...labels].map((label) => {
      let truePositive = 0;
      let falsePositive = 0;
      let falseNegative = 0;
      for (const row of aligned) {
        if (row.truth === label && row.predicted === label) truePositive += 1;
        else if (row.truth !== label && row.predicted === label) falsePositive += 1;
        else if (row.truth === label && row.predicted !== label) falseNegative += 1;
      }
      const denominator = 2 * truePositive + falsePositive + falseNegative;
      return denominator ? (2 * truePositive) / denominator : 0;
    });
    macroF1 = f1Scores.reduce((sum, score) => sum + score, 0) / f1Scores.length;
  }

  const predictedPath = [...prediction.predicted_segments]
    .sort((a, b) => a.observation_index - b.observation_index)
    .map((segment) => segment.arc_id);
  let routeContinuity: number | null = null;
  if (predictedPath.length > 1) {
    const links = predictedPath.slice(0, -1).map((fromArc, index) => {
      const toArc = predictedPath[index + 1];
      return parsed.arcs[fromArc]?.[1] === parsed.arcs[toArc]?.[0];
    });
    routeContinuity = links.length ? links.filter(Boolean).length / links.length : null;
  }
  return { top1Accuracy, macroF1, coverage, effectiveAccuracy, routeContinuity };
}

function formatPercent(value: number | null) {
  return value === null ? '—' : `${(value * 100).toFixed(1)}%`;
}

function BoundsUpdater({ points }: { points: GeoPoint[] }) {
  const map = useMap();
  useEffect(() => {
    if (points.length) {
      const bounds: LatLngBoundsExpression = points.map((p): [number, number] => [p.latitude, p.longitude]);
      map.fitBounds(bounds, { padding: [30, 30], maxZoom: 16 });
    }
  }, [map, points]);
  return null;
}

function MapLayer({
  title,
  count,
  color,
  checked,
  disabled,
  onChange,
}: {
  title: string;
  count?: number;
  color: string;
  checked: boolean;
  disabled?: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <label className={`layer-row ${disabled ? 'is-disabled' : ''}`}>
      <span className="layer-mark" style={{ backgroundColor: color }} />
      <span>{title}</span>
      {count !== undefined && <small>{count.toLocaleString()}</small>}
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(event) => onChange(event.target.checked)} />
    </label>
  );
}

function Metric({ label, value }: { label: string; value: ReactNode }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

function App() {
  const folderInput = useRef<HTMLInputElement>(null);
  const [dataset, setDataset] = useState<Dataset>({});
  const [selectedId, setSelectedId] = useState('');
  const [parsed, setParsed] = useState<ParsedTrip | null>(null);
  const [prediction, setPrediction] = useState<UploadPredictionResponse | null>(null);
  const [apiState, setApiState] = useState<'checking' | 'online' | 'offline'>('checking');
  const [isParsing, setIsParsing] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [error, setError] = useState('');
  const [showGps, setShowGps] = useState(true);
  const [showClassifier, setShowClassifier] = useState(false);
  const [showReconstructed, setShowReconstructed] = useState(true);
  const [showReference, setShowReference] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    const refreshHealth = () => {
      checkHealth(controller.signal)
        .then(() => active && setApiState('online'))
        .catch(() => active && setApiState('offline'));
    };
    refreshHealth();
    const timer = window.setInterval(refreshHealth, 5000);
    return () => {
      active = false;
      window.clearInterval(timer);
      controller.abort();
    };
  }, []);

  useEffect(() => {
    folderInput.current?.setAttribute('webkitdirectory', '');
    folderInput.current?.setAttribute('directory', '');
  }, []);

  const tripIds = useMemo(() => Object.keys(dataset).sort(), [dataset]);
  const tripFiles = selectedId ? dataset[selectedId] : undefined;
  const predictionSegments = prediction?.predicted_segments ?? [];
  const reconstructedSegments = prediction?.reconstructed_segments ?? [];
  const referenceSegments = parsed?.referenceSegments ?? [];
  const localMetrics = useMemo(
    () => parsed && prediction ? evaluateLocally(parsed, prediction) : null,
    [parsed, prediction],
  );

  const visibleCoords = useMemo(() => {
    const coords: GeoPoint[] = [];
    if (showGps && parsed) coords.push(...parsed.points);
    const addSegments = (segments: Segment[]) => {
      for (const segment of segments) coords.push(segment.start, segment.end);
    };
    if (showClassifier) addSegments(predictionSegments);
    if (showReconstructed) addSegments(reconstructedSegments);
    if (showReference) addSegments(referenceSegments);
    return coords;
  }, [showGps, parsed, showClassifier, predictionSegments, showReconstructed, reconstructedSegments, showReference, referenceSegments]);

  async function handleFolder(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    setError('');
    setPrediction(null);
    setParsed(null);
    setSelectedId('');
    setShowGps(true);
    setShowClassifier(false);
    setShowReconstructed(true);
    setShowReference(false);
    const groups: Record<string, FileGroup> = {};
    for (const file of files) {
      const details = tripFileInfo(file);
      if (!details) continue;
      groups[details.tripId] ??= {};
      groups[details.tripId][details.ext] = file;
    }
    const complete: Dataset = {};
    for (const [tripId, group] of Object.entries(groups)) {
      if (group.track && group.nodes && group.arcs) {
        complete[tripId] = { track: group.track, nodes: group.nodes, arcs: group.arcs, route: group.route };
      }
    }
    setDataset(complete);
    const first = Object.keys(complete).sort()[0] ?? '';
    setSelectedId(first);
    if (!first) {
      setError('No complete trip folders found. Select a folder containing matching .track, .nodes, and .arcs files.');
      return;
    }
    await loadLocalTrip(first, complete[first]);
  }

  async function loadLocalTrip(tripId: string, files: Dataset[string]) {
    setIsParsing(true);
    setError('');
    setPrediction(null);
    setShowGps(true);
    setShowClassifier(false);
    setShowReference(false);
    setShowReconstructed(true);
    try {
      setParsed(await parseTrip(files));
    } catch (cause) {
      setParsed(null);
      setError(cause instanceof Error ? cause.message : 'Could not read the selected trip files');
    } finally {
      setIsParsing(false);
    }
    setSelectedId(tripId);
  }

  async function handleTripChange(tripId: string) {
    setSelectedId(tripId);
    setPrediction(null);
    setShowClassifier(false);
    setShowReconstructed(true);
    setShowGps(true);
    setShowReference(false);
    const files = dataset[tripId];
    if (files) await loadLocalTrip(tripId, files);
  }

  async function handlePredict() {
    if (!tripFiles) return;
    setIsPredicting(true);
    setError('');
    setPrediction(null);
    setShowClassifier(false);
    setShowReconstructed(true);
    try {
      const result = await predictUploadedTrip(tripFiles);
      setPrediction(result);
      setShowGps(true);
      setShowClassifier(false);
      setShowReconstructed(Boolean(result.reconstructed_segments));
      setShowReference(Boolean(tripFiles.route));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Prediction request failed');
    } finally {
      setIsPredicting(false);
    }
  }

  const center: LatLngExpression = parsed?.points[0]
    ? [parsed.points[0].latitude, parsed.points[0].longitude]
    : [56.2, 54.8];
  const pointStride = parsed ? Math.max(1, Math.ceil(parsed.points.length / 650)) : 1;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-symbol"><span /><span /><span /></div>
          <div><strong>PathSnap</strong><small>ROAD TRACE EVALUATOR</small></div>
        </div>
        <div className="topbar-right">
          <div className={`connection ${apiState}`}><i />{apiState === 'online' ? 'API connected' : apiState === 'offline' ? 'API offline' : 'Checking API'}</div>
          <span className="topbar-divider" />
          <span className="workspace-label">Laptop 2 · Evaluation workspace</span>
        </div>
      </header>

      <section className="workspace">
        <aside className="control-panel">
          <div className="eyebrow">MODEL VALIDATION</div>
          <h1>Inspect a trip</h1>
          <p className="intro">Upload an evaluator folder, run the saved model, and compare its route to the local reference.</p>

          <section className="panel-section">
            <div className="section-heading"><span className="step-number">01</span><div><h2>Evaluation dataset</h2><p>Choose the folder containing the trip files</p></div></div>
            <input
              ref={folderInput}
              className="visually-hidden"
              type="file"
              multiple
              onChange={handleFolder}
            />
            <button className="folder-picker" type="button" onClick={() => folderInput.current?.click()}>
              <span className="folder-icon">↥</span>
              <span><strong>{tripIds.length ? 'Choose another folder' : 'Choose evaluator folder'}</strong><small>Select the evaluator_test directory</small></span>
              <span className="picker-chevron">›</span>
            </button>
            <div className="privacy-note"><span>LOCAL REFERENCE</span><p>The <code>.route</code> file stays in this browser. Only track, nodes, and arcs are sent for prediction.</p></div>
            {tripIds.length > 0 && (
              <label className="field-label" htmlFor="trip-select">TRIP · {tripIds.length} FOUND</label>
            )}
            {tripIds.length > 0 && (
              <select id="trip-select" value={selectedId} onChange={(event) => void handleTripChange(event.target.value)}>
                {tripIds.map((tripId) => <option key={tripId} value={tripId}>{tripId}{dataset[tripId].route ? '' : ' · no reference'}</option>)}
              </select>
            )}
            {tripFiles && <div className="file-checklist">
              {(['track', 'nodes', 'arcs'] as const).map((ext) => <div key={ext}><b>{ext}</b><span>{tripFiles[ext].name}</span><i>✓</i></div>)}
              <div className={tripFiles.route ? 'reference-file' : 'missing-file'}><b>route</b><span>{tripFiles.route?.name ?? 'Optional ground truth not found'}</span><i>{tripFiles.route ? 'LOCAL' : '—'}</i></div>
            </div>}
          </section>

          <section className="panel-section run-section">
            <div className="section-heading"><span className="step-number">02</span><div><h2>Model prediction</h2><p>Candidate scoring with Viterbi reconstruction</p></div></div>
            <button className="predict-button" type="button" disabled={!tripFiles || isPredicting || apiState !== 'online'} onClick={() => void handlePredict()}>
              {isPredicting ? <><span className="spinner" /> Running model…</> : <>Run prediction <span>→</span></>}
            </button>
            {apiState === 'offline' && <div className="inline-warning">Start the backend at the address in <code>VITE_API_BASE_URL</code>.</div>}
          </section>

          {error && <div className="error-box" role="alert"><strong>Something needs attention</strong><span>{error}</span></div>}

          <section className="panel-section layers-section">
            <div className="section-heading"><span className="step-number">03</span><div><h2>Map layers</h2><p>Compare GPS, prediction, and local reference</p></div></div>
            <MapLayer title="GPS observations" count={parsed?.points.length} color="#176bd1" checked={showGps} onChange={setShowGps} disabled={!parsed} />
            <MapLayer title="Classifier top-1" count={predictionSegments.length || undefined} color="#f59e0b" checked={showClassifier} onChange={setShowClassifier} disabled={!prediction} />
            <MapLayer title="Predicted path · Viterbi" count={prediction?.reconstructed_segments?.length} color="#078f7e" checked={showReconstructed && !!prediction?.reconstructed_segments} onChange={setShowReconstructed} disabled={!prediction?.reconstructed_segments} />
            <MapLayer title="Ground truth · local" count={referenceSegments.length || undefined} color="#dc4c64" checked={showReference} onChange={setShowReference} disabled={!prediction || !tripFiles?.route} />
          </section>

          <footer className="side-footer"><span className="shield">◇</span><span>Reference data stays on this device<br /><b>.route is never sent to the API</b></span></footer>
        </aside>

        <section className="map-workspace">
          <div className="map-toolbar">
            <div><span className="map-kicker">TRIP MAP</span><strong>{selectedId || 'No trip selected'}</strong></div>
            <div className="map-toolbar-right">
              {isParsing && <span className="subtle-status"><span className="spinner small" /> Reading local files</span>}
              {isPredicting && <span className="subtle-status"><span className="spinner small" /> Scoring candidates · decoding path</span>}
              {prediction && <span className="complete-chip"><i /> Prediction ready</span>}
            </div>
          </div>
          <div className="map-card">
            <MapContainer center={center} zoom={13} scrollWheelZoom className="leaflet-map">
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
              <BoundsUpdater points={visibleCoords} />
              {showGps && parsed && <>
                <Polyline positions={parsed.points.map((p): [number, number] => [p.latitude, p.longitude])} pathOptions={{ color: '#176bd1', weight: 3, opacity: 0.72 }} />
                {parsed.points.filter((_, index) => index % pointStride === 0).map((point, index) => (
                  <CircleMarker key={`gps-${index}`} center={[point.latitude, point.longitude]} radius={2.4} pathOptions={{ color: '#fff', weight: 1, fillColor: '#176bd1', fillOpacity: 0.95 }} />
                ))}
              </>}
              {showClassifier && predictionSegments.length > 0 && <Polyline
                positions={predictionSegments.map((segment) => [[segment.start.latitude, segment.start.longitude], [segment.end.latitude, segment.end.longitude]])}
                pathOptions={{ color: '#f59e0b', weight: 3, opacity: 0.58 }}
              />}
              {showReference && referenceSegments.length > 0 && <Polyline
                positions={referenceSegments.map((segment) => [[segment.start.latitude, segment.start.longitude], [segment.end.latitude, segment.end.longitude]])}
                pathOptions={{ color: '#dc4c64', weight: 7, opacity: 0.9, dashArray: '6 6' }}
              />}
              {showReconstructed && reconstructedSegments.length > 0 && <Polyline
                positions={reconstructedSegments.map((segment) => [[segment.start.latitude, segment.start.longitude], [segment.end.latitude, segment.end.longitude]])}
                pathOptions={{ color: '#078f7e', weight: 3.5, opacity: 0.98 }}
              />}
            </MapContainer>
            <div className="map-legend" aria-label="Map legend">
              <span><i className="legend-gps" />GPS track</span>
              <span><i className="legend-predicted" />Predicted path</span>
              <span><i className="legend-reference" />Ground truth</span>
            </div>
            {!parsed && !isParsing && <div className="map-empty"><div className="map-empty-icon">⌖</div><strong>Your trip map will appear here</strong><span>Choose the evaluator folder to begin.</span></div>}
            {isParsing && <div className="map-overlay"><span className="spinner" /> Loading track and local reference…</div>}
          </div>

          <div className="results-strip">
            <Metric label="GPS observations" value={parsed?.points.length.toLocaleString() ?? '—'} />
            <Metric label="Predicted segments" value={prediction?.predicted_segments.length.toLocaleString() ?? '—'} />
            <Metric label="Reconstructed" value={prediction?.reconstructed_segments?.length.toLocaleString() ?? '—'} />
            <Metric label="Unmatched" value={prediction?.unmatched_count.toLocaleString() ?? '—'} />
            <Metric label="Inference time" value={prediction ? `${prediction.inference_ms.toFixed(1)} ms` : '—'} />
          </div>
          <section className="evaluation-card">
            <div className="evaluation-heading"><strong>Trip evaluation</strong><span>{tripFiles?.route ? 'Calculated locally from .route' : 'Select a trip with a .route file'}</span></div>
            <div className="evaluation-grid">
              <Metric label="TOP-1 ACCURACY" value={formatPercent(localMetrics?.top1Accuracy ?? null)} />
              <Metric label="MACRO F1" value={formatPercent(localMetrics?.macroF1 ?? null)} />
              <Metric label="COVERAGE" value={formatPercent(localMetrics?.coverage ?? null)} />
              <Metric label="EFFECTIVE ACCURACY" value={formatPercent(localMetrics?.effectiveAccuracy ?? null)} />
              <Metric label="ROUTE CONTINUITY" value={formatPercent(localMetrics?.routeContinuity ?? null)} />
            </div>
            <div className="evaluation-footnote">
              Each cleaned GPS point is labeled by snapping it to the nearest route arc, matching the notebook evaluator. Top-1 and macro F1 use observations where that reference arc is in the model's six-candidate set; macro F1 averages the union of reference and predicted arc IDs. Coverage and effective accuracy include candidate misses and unmatched points. The .route file stays in this browser.
              {parsed?.invalidReferenceCount ? ` ${parsed.invalidReferenceCount} invalid reference arc IDs were skipped.` : ''}
            </div>
          </section>
        </section>
      </section>
    </main>
  );
}

export default App;
