import { StrictMode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import maplibregl, { type GeoJSONSource, type Map as MapLibreMap } from "maplibre-gl";
import type { FeatureCollection, Point } from "geojson";
import {
  Activity,
  ArrowDownRight,
  ArrowRight,
  Bike,
  Boxes,
  Check,
  ChevronDown,
  CircleAlert,
  CircleDot,
  Command,
  Crosshair,
  Database,
  Expand,
  FlaskConical,
  Gauge,
  Grid2X2,
  Layers3,
  ListFilter,
  LocateFixed,
  Map as MapIcon,
  Maximize2,
  Minimize2,
  MoonStar,
  Network,
  RefreshCw,
  Route,
  Search,
  Send,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  TerminalSquare,
  TimerReset,
  TriangleAlert,
  X,
  Zap,
} from "lucide-react";
import "maplibre-gl/dist/maplibre-gl.css";
import "./styles.css";

type View = "command" | "network" | "dispatch" | "model";
type RiskLevel = "high" | "medium" | "low";

type Station = {
  station_id: string;
  station_name: string;
  latitude: number;
  longitude: number;
  capacity: number;
  bikes_available: number;
  docks_available: number;
  predicted_departures: number;
  predicted_arrivals: number;
  departures_lower: number;
  departures_upper: number;
  arrivals_lower: number;
  arrivals_upper: number;
  projected_bikes: number;
  projected_bikes_lower: number;
  projected_bikes_upper: number;
  empty_risk_score: number;
  full_risk_score: number;
  risk_score: number;
  risk_type: "empty" | "full" | "balanced";
  risk_level: RiskLevel;
  alignment_method?: string;
  alignment_distance_km?: number;
};

type RiskResponse = {
  as_of: string;
  feed_age_seconds: number;
  max_feed_age_seconds: number;
  feed_is_stale: boolean;
  model_version: string;
  champion: string;
  release_status: string;
  data_mode: string;
  horizon_minutes: number;
  station_count: number;
  system_station_count: number;
  high_risk_stations: number;
  model_station_coverage: number;
  alignment_methods: Record<string, number>;
  stations: Station[];
};

type ReplayPolicy = {
  with_rebalancing?: {
    service_failures: number;
    relative_failure_reduction: number;
    bikes_moved: number;
    bike_km: number;
  };
};

type ModelSummary = {
  model_version: string;
  data_mode: string;
  target_definition: string;
  rows: number;
  stations: number;
  time_range: string[];
  promotion: {
    champion?: string;
    mean_relative_improvement?: number;
    candidate_wins?: number;
    required_wins?: number;
  };
  frozen_test: {
    combined_mae?: number;
    baseline_combined_mae?: number;
    candidate_relative_improvement?: number;
    top_10pct_imbalance_recall?: number;
    departures_interval_coverage?: number;
    arrivals_interval_coverage?: number;
  };
  decision_replay: {
    release_status?: string;
    candidate_dominates_baseline?: boolean;
    baseline?: ReplayPolicy;
    candidate?: ReplayPolicy;
  };
  final_windows: Record<string, string>;
};

type PlanMove = {
  from_station_id: string;
  from_station_name: string;
  to_station_id: string;
  to_station_name: string;
  bikes: number;
  distance_km: number;
  destination_empty_risk: number;
};

type Plan = {
  plan_id: string;
  created_at: string;
  snapshot_at: string;
  model_version: string;
  release_status: string;
  algorithm: string;
  total_bikes_moved: number;
  requested_bikes: number;
  unfilled_bikes: number;
  max_total_moves: number;
  moves: PlanMove[];
  limitations: string;
};

const views: Array<{ id: View; label: string; caption: string; icon: typeof Grid2X2 }> = [
  { id: "command", label: "Command", caption: "Live decision surface", icon: Grid2X2 },
  { id: "network", label: "Network", caption: "Station topology", icon: Network },
  { id: "dispatch", label: "Dispatch", caption: "Move orchestration", icon: Route },
  { id: "model", label: "Model", caption: "Release governance", icon: FlaskConical },
];

const riskColor = (level: RiskLevel) =>
  level === "high" ? "var(--signal)" : level === "medium" ? "var(--warning)" : "var(--positive)";
const formatPercent = (value = 0, digits = 0) => `${(value * 100).toFixed(digits)}%`;
const formatNumber = (value?: number, digits = 1) =>
  value == null || Number.isNaN(value) ? "—" : value.toLocaleString(undefined, { maximumFractionDigits: digits });
const formatAge = (seconds = 0) => (seconds < 60 ? `${Math.round(seconds)} sec` : `${Math.round(seconds / 60)} min`);
const formatTime = (value?: string) =>
  value
    ? new Intl.DateTimeFormat("en", { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(
        new Date(value),
      )
    : "—";

async function api<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
  return payload as T;
}

function BrandMark() {
  return (
    <div className="brand-mark" aria-hidden="true">
      <svg viewBox="0 0 42 42">
        <path d="M7 7h11v11H7zM24 7h11v11H24zM7 24h11v11H7z" />
        <path className="brand-live" d="M29.5 23 36 35H23z" />
      </svg>
    </div>
  );
}

function Rail({ view, onView, onPalette }: { view: View; onView: (view: View) => void; onPalette: () => void }) {
  return (
    <aside className="rail" aria-label="Primary navigation">
      <button className="brand-button" onClick={() => onView("command")} aria-label="VeloGuard command center">
        <BrandMark />
      </button>
      <nav className="rail-nav">
        {views.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              className={view === item.id ? "rail-item is-active" : "rail-item"}
              onClick={() => onView(item.id)}
              aria-label={item.label}
              aria-current={view === item.id ? "page" : undefined}
              data-tooltip={item.label}
            >
              <Icon size={19} strokeWidth={1.7} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>
      <div className="rail-spacer" />
      <button className="rail-item palette-trigger" onClick={onPalette} data-tooltip="Command menu" aria-label="Open command menu">
        <Command size={19} />
        <span>Commands</span>
      </button>
      <div className="operator-avatar" aria-label="Operator profile">HY</div>
    </aside>
  );
}

function TopBar({
  view,
  data,
  loading,
  onRefresh,
  onCollect,
  onPalette,
}: {
  view: View;
  data: RiskResponse | null;
  loading: string | null;
  onRefresh: () => void;
  onCollect: () => void;
  onPalette: () => void;
}) {
  const current = views.find((item) => item.id === view)!;
  return (
    <header className="topbar">
      <div className="topbar-context">
        <span className="topbar-kicker">VELO / NYC-01</span>
        <span className="topbar-slash">/</span>
        <strong>{current.label}</strong>
        <span className="topbar-caption">{current.caption}</span>
      </div>
      <button className="command-search" onClick={onPalette} aria-label="Search commands and stations">
        <Search size={15} />
        <span>Search stations or run a command</span>
        <kbd>⌘ K</kbd>
      </button>
      <div className="topbar-actions">
        <div className={data?.feed_is_stale ? "feed-chip is-stale" : "feed-chip"}>
          <span className="live-pulse" />
          <span>{data ? (data.feed_is_stale ? "STALE FEED" : "GBFS LIVE") : "NO FEED"}</span>
          <b>{data ? formatAge(data.feed_age_seconds) : "—"}</b>
        </div>
        <button className="icon-button" onClick={onRefresh} disabled={Boolean(loading)} aria-label="Refresh risk data">
          <RefreshCw size={16} className={loading === "refresh" ? "spin" : ""} />
        </button>
        <button className="button button-subtle collect-button" onClick={onCollect} disabled={Boolean(loading)}>
          <Database size={15} />
          {loading === "collect" ? "Collecting" : "Collect snapshot"}
        </button>
      </div>
    </header>
  );
}

function MetricSpark({ points, tone = "mint" }: { points: number[]; tone?: "mint" | "coral" | "amber" | "blue" }) {
  const max = Math.max(...points);
  const min = Math.min(...points);
  const path = points
    .map((point, index) => {
      const x = (index / (points.length - 1)) * 100;
      const y = 28 - ((point - min) / Math.max(max - min, 1)) * 23;
      return `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg className={`metric-spark spark-${tone}`} viewBox="0 0 100 30" preserveAspectRatio="none" aria-hidden="true">
      <path className="spark-area" d={`${path} L100,30 L0,30 Z`} />
      <path className="spark-line" d={path} />
    </svg>
  );
}

function MetricCard({
  label,
  value,
  meta,
  icon: Icon,
  points,
  tone = "mint",
}: {
  label: string;
  value: string;
  meta: string;
  icon: typeof Activity;
  points: number[];
  tone?: "mint" | "coral" | "amber" | "blue";
}) {
  return (
    <article className={`metric-card tone-${tone}`}>
      <div className="metric-heading">
        <span>{label}</span>
        <Icon size={16} />
      </div>
      <div className="metric-value">{value}</div>
      <div className="metric-meta">{meta}</div>
      <MetricSpark points={points} tone={tone} />
    </article>
  );
}

function StatusBanner({ data }: { data: RiskResponse }) {
  const shadow = data.release_status === "shadow_mode_only";
  const stale = data.feed_is_stale;
  return (
    <section className={`status-banner ${stale ? "status-stale" : shadow ? "status-shadow" : "status-approved"}`}>
      <div className="status-symbol">{stale ? <TriangleAlert /> : shadow ? <FlaskConical /> : <ShieldCheck />}</div>
      <div>
        <span className="micro-label">DECISION AUTHORITY</span>
        <strong>{stale ? "Feed freshness breach" : shadow ? "Shadow policy / review only" : "Policy approved"}</strong>
      </div>
      <p>
        {stale
          ? `Latest source is ${formatAge(data.feed_age_seconds)} old. Collect a fresh snapshot before generating moves.`
          : shadow
            ? "Forecast champion has not passed the operational dominance gate. Plans are simulations for analyst review."
            : "Model and decision gates are both clear for assisted operation."}
      </p>
      <div className="status-fingerprint">
        <span>MODEL</span>
        <code>{data.model_version}</code>
      </div>
    </section>
  );
}

function RiskDonut({ stations }: { stations: Station[] }) {
  const counts = useMemo(
    () => ({
      high: stations.filter((station) => station.risk_level === "high").length,
      medium: stations.filter((station) => station.risk_level === "medium").length,
      low: stations.filter((station) => station.risk_level === "low").length,
    }),
    [stations],
  );
  const total = Math.max(stations.length, 1);
  const high = (counts.high / total) * 100;
  const medium = high + (counts.medium / total) * 100;
  return (
    <div className="risk-donut-wrap">
      <div
        className="risk-donut"
        style={{
          background: `conic-gradient(var(--signal) 0 ${high}%, var(--warning) ${high}% ${medium}%, var(--positive) ${medium}% 100%)`,
        }}
        aria-label={`${counts.high} high risk, ${counts.medium} medium risk, ${counts.low} low risk`}
      >
        <div><strong>{counts.high}</strong><span>critical</span></div>
      </div>
      <div className="donut-legend">
        {(["high", "medium", "low"] as RiskLevel[]).map((level) => (
          <div key={level}><i style={{ background: riskColor(level) }} /><span>{level}</span><b>{counts[level]}</b></div>
        ))}
      </div>
    </div>
  );
}

function MapPanel({
  stations,
  selected,
  onSelect,
}: {
  stations: Station[];
  selected: Station | null;
  onSelect: (station: Station) => void;
}) {
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const stationsRef = useRef(stations);
  const [expanded, setExpanded] = useState(false);
  const [mapReady, setMapReady] = useState(false);
  stationsRef.current = stations;

  useEffect(() => {
    if (!container.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: container.current,
      style: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
      center: [-73.995, 40.735],
      zoom: 11.7,
      pitch: 48,
      bearing: -18,
      attributionControl: false,
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: true }), "bottom-right");
    map.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-left");
    map.on("load", () => {
      map.addSource("stations", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      map.addLayer({
        id: "risk-halo",
        type: "circle",
        source: "stations",
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["get", "risk"], 0, 13, 1, 36],
          "circle-color": ["match", ["get", "level"], "high", "#ff5d73", "medium", "#ffb45a", "#44e3b1"],
          "circle-opacity": 0.12,
          "circle-blur": 0.45,
        },
      });
      map.addLayer({
        id: "risk-core",
        type: "circle",
        source: "stations",
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["get", "risk"], 0, 4, 1, 11],
          "circle-color": ["match", ["get", "level"], "high", "#ff5d73", "medium", "#ffb45a", "#44e3b1"],
          "circle-stroke-width": ["case", ["boolean", ["feature-state", "selected"], false], 3, 1],
          "circle-stroke-color": ["case", ["boolean", ["feature-state", "selected"], false], "#ffffff", "#071012"],
          "circle-opacity": 0.94,
        },
      });
      map.on("mouseenter", "risk-core", () => { map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", "risk-core", () => { map.getCanvas().style.cursor = ""; });
      map.on("click", "risk-core", (event) => {
        const id = String(event.features?.[0]?.properties?.stationId || "");
        const station = stationsRef.current.find((item) => item.station_id === id);
        if (station) onSelect(station);
      });
      setMapReady(true);
    });
    mapRef.current = map;
    return () => { map.remove(); mapRef.current = null; };
  }, [onSelect]);

  useEffect(() => {
    const map = mapRef.current;
    if (!mapReady || !map) return;
    const source = map.getSource("stations") as GeoJSONSource;
    const collection: FeatureCollection<Point> = {
      type: "FeatureCollection",
      features: stations.map((station) => ({
        type: "Feature",
        id: station.station_id,
        geometry: { type: "Point", coordinates: [station.longitude, station.latitude] },
        properties: { stationId: station.station_id, risk: station.risk_score, level: station.risk_level },
      })),
    };
    source.setData(collection);
    if (stations.length) {
      const bounds = new maplibregl.LngLatBounds();
      stations.forEach((station) => bounds.extend([station.longitude, station.latitude]));
      map.fitBounds(bounds, { padding: 70, duration: 800, maxZoom: 13.1 });
    }
  }, [stations, mapReady]);

  useEffect(() => {
    const map = mapRef.current;
    if (!mapReady || !map) return;
    stations.forEach((station) => {
      map.setFeatureState({ source: "stations", id: station.station_id }, { selected: station.station_id === selected?.station_id });
    });
    if (selected) map.easeTo({ center: [selected.longitude, selected.latitude], zoom: 14.2, duration: 750 });
  }, [selected, stations, mapReady]);

  useEffect(() => {
    window.setTimeout(() => mapRef.current?.resize(), 150);
  }, [expanded]);

  return (
    <section className={expanded ? "map-panel panel is-expanded" : "map-panel panel"}>
      <div className="panel-header over-map">
        <div>
          <span className="micro-label">GEOSPATIAL RISK FIELD</span>
          <h2>Network pressure</h2>
        </div>
        <div className="map-toolbar">
          <div className="map-legend"><i className="legend-high" />Empty / full threat<i className="legend-low" />Balanced</div>
          <button className="icon-button glass" onClick={() => setExpanded((value) => !value)} aria-label={expanded ? "Exit expanded map" : "Expand map"}>
            {expanded ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
          </button>
        </div>
      </div>
      <div ref={container} className="map-canvas" aria-label="Interactive map of station risk" />
      <div className="map-grid" aria-hidden="true" />
      <div className="map-coordinate"><Crosshair size={13} /> 40.7350° N / 73.9950° W</div>
      {!mapReady && <div className="map-loading"><LocateFixed />INITIALIZING VECTOR FIELD</div>}
      <div className="map-risk-card">
        <span className="micro-label">RISK COMPOSITION</span>
        <RiskDonut stations={stations} />
      </div>
    </section>
  );
}

function CapacityGauge({ station }: { station: Station }) {
  const current = Math.max(0, Math.min(100, (station.bikes_available / Math.max(station.capacity, 1)) * 100));
  const projected = Math.max(0, Math.min(100, (station.projected_bikes / Math.max(station.capacity, 1)) * 100));
  const lower = Math.max(0, Math.min(100, (station.projected_bikes_lower / Math.max(station.capacity, 1)) * 100));
  const upper = Math.max(0, Math.min(100, (station.projected_bikes_upper / Math.max(station.capacity, 1)) * 100));
  return (
    <div className="capacity-gauge" aria-label={`Current capacity ${current.toFixed(0)} percent`}>
      <div className="gauge-track">
        <span className="gauge-safe" />
        <span className="gauge-range" style={{ left: `${lower}%`, width: `${Math.max(upper - lower, 1)}%` }} />
        <span className="gauge-current" style={{ left: `${current}%` }} />
        <span className="gauge-projected" style={{ left: `${projected}%` }} />
      </div>
      <div className="gauge-labels"><span>empty</span><span>operating envelope</span><span>full</span></div>
    </div>
  );
}

function StationInspector({ station, onClose }: { station: Station | null; onClose?: () => void }) {
  if (!station) {
    return (
      <aside className="inspector panel inspector-empty">
        <div className="empty-orbit"><CircleDot /><span /></div>
        <span className="micro-label">STATION INSPECTOR</span>
        <h2>No station locked</h2>
        <p>Select a map node or risk queue row to inspect forecast bounds, capacity, identity alignment, and decision pressure.</p>
      </aside>
    );
  }
  const net = station.predicted_arrivals - station.predicted_departures;
  return (
    <aside className="inspector panel">
      <div className="inspector-head">
        <div className={`risk-beacon beacon-${station.risk_level}`}><span /></div>
        <div>
          <span className="micro-label">STATION / {station.station_id.slice(-8).toUpperCase()}</span>
          <h2>{station.station_name}</h2>
        </div>
        {onClose && <button className="icon-button" onClick={onClose} aria-label="Close station inspector"><X size={16} /></button>}
      </div>
      <div className="inspector-risk-line">
        <div><span>{station.risk_type} risk</span><strong style={{ color: riskColor(station.risk_level) }}>{formatPercent(station.risk_score)}</strong></div>
        <span className={`risk-pill risk-${station.risk_level}`}>{station.risk_level}</span>
      </div>
      <div className="station-reading">
        <div><span>Bikes now</span><strong>{station.bikes_available}</strong><small>of {station.capacity}</small></div>
        <ArrowRight size={18} />
        <div><span>Projected</span><strong>{formatNumber(station.projected_bikes)}</strong><small>in 30 min</small></div>
      </div>
      <CapacityGauge station={station} />
      <div className="bound-card">
        <div className="bound-header"><span>UNCERTAINTY ENVELOPE</span><b>{formatNumber(station.projected_bikes_lower)} — {formatNumber(station.projected_bikes_upper)}</b></div>
        <div className="bound-axis"><i /><i /><i /><span /></div>
        <div className="bound-labels"><span>P10 conservative</span><span>P90 conservative</span></div>
      </div>
      <div className="flow-pair">
        <div><span><ArrowDownRight size={14} /> departures</span><strong>{formatNumber(station.predicted_departures)}</strong><small>{formatNumber(station.departures_lower)}—{formatNumber(station.departures_upper)}</small></div>
        <div><span><ArrowDownRight className="arrivals-icon" size={14} /> arrivals</span><strong>{formatNumber(station.predicted_arrivals)}</strong><small>{formatNumber(station.arrivals_lower)}—{formatNumber(station.arrivals_upper)}</small></div>
      </div>
      <div className="net-flow"><span>Net pressure</span><b className={net < 0 ? "negative" : "positive"}>{net > 0 ? "+" : ""}{formatNumber(net)} bikes</b></div>
      <div className="identity-card">
        <ShieldCheck size={15} />
        <div><span>IDENTITY RESOLUTION</span><strong>{station.alignment_method?.replace("_", " ") || "station id"}</strong></div>
        <code>{station.alignment_distance_km != null ? `${station.alignment_distance_km} km` : "exact"}</code>
      </div>
    </aside>
  );
}

function RiskQueue({ stations, selected, onSelect }: { stations: Station[]; selected: Station | null; onSelect: (station: Station) => void }) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<"all" | RiskLevel>("all");
  const [sort, setSort] = useState<"risk" | "name" | "bikes">("risk");
  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return stations
      .filter((station) => filter === "all" || station.risk_level === filter)
      .filter((station) => !needle || station.station_name.toLowerCase().includes(needle) || station.station_id.includes(needle))
      .sort((a, b) => sort === "name" ? a.station_name.localeCompare(b.station_name) : sort === "bikes" ? a.bikes_available - b.bikes_available : b.risk_score - a.risk_score);
  }, [stations, query, filter, sort]);
  return (
    <section className="risk-queue panel">
      <div className="panel-header queue-header">
        <div><span className="micro-label">PRIORITIZED EXCEPTIONS</span><h2>Risk queue <em>{visible.length}</em></h2></div>
        <div className="queue-tools">
          <label className="table-search"><Search size={14} /><span className="sr-only">Search stations</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Filter stations" /></label>
          <div className="segmented" aria-label="Risk filter">
            {(["all", "high", "medium", "low"] as const).map((level) => <button key={level} className={filter === level ? "is-active" : ""} onClick={() => setFilter(level)}>{level}</button>)}
          </div>
          <label className="sort-control"><ListFilter size={14} /><span className="sr-only">Sort stations</span><select value={sort} onChange={(event) => setSort(event.target.value as typeof sort)}><option value="risk">Risk</option><option value="name">Name</option><option value="bikes">Inventory</option></select><ChevronDown size={13} /></label>
        </div>
      </div>
      <div className="table-scroll">
        <table className="risk-table">
          <thead><tr><th>Station</th><th>Condition</th><th>Inventory</th><th>30m projection</th><th>Uncertainty</th><th aria-label="Open" /></tr></thead>
          <tbody>
            {visible.map((station, index) => {
              const fill = (station.bikes_available / Math.max(station.capacity, 1)) * 100;
              return (
                <tr key={station.station_id} className={selected?.station_id === station.station_id ? "is-selected" : ""} onClick={() => onSelect(station)} tabIndex={0} onKeyDown={(event) => { if (event.key === "Enter") onSelect(station); }}>
                  <td><span className="row-index">{String(index + 1).padStart(2, "0")}</span><div><strong>{station.station_name}</strong><small>{station.station_id.slice(-10)} · {station.alignment_method?.replace("_", " ") || "identity"}</small></div></td>
                  <td><span className={`risk-pill risk-${station.risk_level}`}><i />{station.risk_type} {formatPercent(station.risk_score)}</span></td>
                  <td><div className="inventory-cell"><span><b>{station.bikes_available}</b> / {station.capacity}</span><i><u style={{ width: `${fill}%` }} /></i></div></td>
                  <td><strong>{formatNumber(station.projected_bikes)}</strong><small className={station.predicted_arrivals - station.predicted_departures < 0 ? "negative" : "positive"}>{station.predicted_arrivals - station.predicted_departures > 0 ? "+" : ""}{formatNumber(station.predicted_arrivals - station.predicted_departures)}</small></td>
                  <td><div className="uncertainty-cell"><span>{formatNumber(station.projected_bikes_lower)}</span><i><u style={{ left: `${Math.max(0, station.projected_bikes_lower / station.capacity * 100)}%`, width: `${Math.max(4, (station.projected_bikes_upper - station.projected_bikes_lower) / station.capacity * 100)}%` }} /></i><span>{formatNumber(station.projected_bikes_upper)}</span></div></td>
                  <td><button className="row-open" aria-label={`Inspect ${station.station_name}`}><ArrowRight size={14} /></button></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function CommandView({ data, selected, onSelect }: { data: RiskResponse; selected: Station | null; onSelect: (station: Station) => void }) {
  const avgRisk = data.stations.reduce((sum, station) => sum + station.risk_score, 0) / Math.max(data.stations.length, 1);
  const atCapacity = data.stations.filter((station) => station.bikes_available <= 2 || station.docks_available <= 2).length;
  return (
    <>
      <div className="page-intro">
        <div><span className="eyebrow"><span /> LIVE DECISION SURFACE</span><h1>Network command</h1><p>Uncertainty-aware station pressure, model governance, and constrained dispatch in one operational field.</p></div>
        <div className="intro-meta"><span>AS OF</span><strong>{formatTime(data.as_of)}</strong><small>America / New York · 30 minute horizon</small></div>
      </div>
      <StatusBanner data={data} />
      <section className="metrics-grid" aria-label="Network metrics">
        <MetricCard label="Stations in scope" value={String(data.station_count)} meta={`${data.system_station_count.toLocaleString()} system-wide · ${formatPercent(data.model_station_coverage, 1)} coverage`} icon={Network} points={[25,28,31,29,34,36,39]} />
        <MetricCard label="Critical pressure" value={String(data.high_risk_stations)} meta={`${formatPercent(data.high_risk_stations / Math.max(data.station_count, 1))} of scored stations`} icon={Zap} points={[5,8,7,11,9,12,data.high_risk_stations]} tone="coral" />
        <MetricCard label="Mean risk load" value={formatPercent(avgRisk)} meta={`${atCapacity} stations at physical edge`} icon={Gauge} points={[.21,.34,.28,.4,.36,.43,avgRisk]} tone="amber" />
        <MetricCard label="Feed freshness" value={formatAge(data.feed_age_seconds)} meta={`SLO threshold ${formatAge(data.max_feed_age_seconds)}`} icon={TimerReset} points={[9,8,12,6,4,3,Math.max(1,data.feed_age_seconds/60)]} tone="blue" />
      </section>
      <div className="command-grid">
        <MapPanel stations={data.stations} selected={selected} onSelect={onSelect} />
        <StationInspector station={selected} />
      </div>
      <RiskQueue stations={data.stations} selected={selected} onSelect={onSelect} />
    </>
  );
}

function NetworkView({ data, onSelect }: { data: RiskResponse; onSelect: (station: Station) => void }) {
  const ordered = [...data.stations].sort((a, b) => b.risk_score - a.risk_score);
  return (
    <>
      <div className="page-intro compact"><div><span className="eyebrow"><span /> TOPOLOGY / CAPACITY / IDENTITY</span><h1>Network matrix</h1><p>Every modeled station resolved against the live system and normalized into one comparable operational lattice.</p></div></div>
      <section className="network-summary panel">
        <div className="network-orbit"><span className="orbit-a" /><span className="orbit-b" /><Network /><strong>{data.station_count}</strong><small>aligned nodes</small></div>
        <div className="network-facts">
          <div><span>Exact / name identity</span><strong>{data.alignment_methods.station_id || 0} / {data.alignment_methods.normalized_name || 0}</strong><small>deterministic resolution</small></div>
          <div><span>Coordinate fallback</span><strong>{data.alignment_methods.coordinate || 0}</strong><small>maximum 150 m radius</small></div>
          <div><span>Unmodeled network</span><strong>{data.system_station_count - data.station_count}</strong><small>excluded, never guessed</small></div>
          <div><span>Capacity in scope</span><strong>{ordered.reduce((sum, station) => sum + station.capacity, 0).toLocaleString()}</strong><small>physical bike slots</small></div>
        </div>
      </section>
      <section className="matrix-panel panel">
        <div className="panel-header"><div><span className="micro-label">RISK × INVENTORY MATRIX</span><h2>Station lattice</h2></div><div className="matrix-axis"><span>low pressure</span><ArrowRight size={14} /><span>high pressure</span></div></div>
        <div className="station-matrix">
          {ordered.map((station) => {
            const fill = station.bikes_available / Math.max(station.capacity, 1);
            return (
              <button key={station.station_id} className={`matrix-node node-${station.risk_level}`} onClick={() => onSelect(station)}>
                <div className="matrix-node-top"><span>{station.risk_type}</span><b>{formatPercent(station.risk_score)}</b></div>
                <strong>{station.station_name}</strong>
                <small>{station.bikes_available} bikes · {station.docks_available} docks</small>
                <div className="matrix-capacity"><i style={{ width: `${fill * 100}%` }} /><u style={{ left: `${Math.max(0, station.projected_bikes / station.capacity * 100)}%` }} /></div>
              </button>
            );
          })}
        </div>
      </section>
      <RiskQueue stations={data.stations} selected={null} onSelect={onSelect} />
    </>
  );
}

function DispatchView({ data, plan, loading, onGenerate }: { data: RiskResponse; plan: Plan | null; loading: string | null; onGenerate: () => void }) {
  return (
    <>
      <div className="page-intro compact">
        <div><span className="eyebrow"><span /> HUMAN-IN-THE-LOOP ORCHESTRATION</span><h1>Dispatch laboratory</h1><p>Convert conservative station bounds into deterministic one-hop moves. Shadow policies remain simulations.</p></div>
        <button className="button button-primary hero-action" onClick={onGenerate} disabled={Boolean(loading) || data.feed_is_stale}><Sparkles size={16} />{loading === "plan" ? "Solving safe moves" : "Generate simulation"}</button>
      </div>
      <StatusBanner data={data} />
      <div className="dispatch-grid">
        <section className="dispatch-board panel">
          <div className="panel-header"><div><span className="micro-label">MOVE SEQUENCE</span><h2>{plan ? `Plan ${plan.plan_id.slice(0, 8)}` : "No active plan"}</h2></div>{plan && <span className="algorithm-chip"><TerminalSquare size={13} />{plan.algorithm}</span>}</div>
          {!plan ? (
            <div className="dispatch-empty"><div className="route-glyph"><Route /><i /><i /><i /></div><h3>Ready to simulate</h3><p>A plan uses fresh inventory, forecast envelopes, safety docks, a four-kilometer route cap, and a global move budget.</p><button className="button button-primary" onClick={onGenerate} disabled={data.feed_is_stale}><Send size={15} />Generate plan</button></div>
          ) : (
            <div className="move-sequence">
              <div className="shadow-callout"><FlaskConical /><div><strong>Shadow output</strong><span>For analyst review. Not dispatch-authorized.</span></div><code>{plan.model_version}</code></div>
              {plan.moves.map((move, index) => (
                <article className="move-card" key={`${move.from_station_id}-${move.to_station_id}-${index}`}>
                  <div className="move-index">{String(index + 1).padStart(2, "0")}</div>
                  <div className="move-node source"><i /><span>Source</span><strong>{move.from_station_name}</strong><small>safe surplus</small></div>
                  <div className="route-line"><span>{move.bikes} bikes</span><i /><Route size={15} /><i /><small>{move.distance_km} km</small></div>
                  <div className="move-node destination"><i /><span>Destination</span><strong>{move.to_station_name}</strong><small>{formatPercent(move.destination_empty_risk)} empty risk</small></div>
                </article>
              ))}
            </div>
          )}
        </section>
        <aside className="plan-summary panel">
          <div className="panel-header"><div><span className="micro-label">PLAN ENVELOPE</span><h2>Constraint ledger</h2></div><SlidersHorizontal size={17} /></div>
          <div className="plan-score"><span>MOVE BUDGET</span><strong>{plan?.total_bikes_moved ?? 0}<small> / {plan?.max_total_moves ?? 40}</small></strong><div><i style={{ width: `${((plan?.total_bikes_moved ?? 0) / (plan?.max_total_moves ?? 40)) * 100}%` }} /></div></div>
          <div className="constraint-list">
            <div><span><Boxes />Requested bikes</span><b>{plan?.requested_bikes ?? "—"}</b></div>
            <div><span><CircleAlert />Unmatched demand</span><b className="warning">{plan?.unfilled_bikes ?? "—"}</b></div>
            <div><span><Route />Transfer legs</span><b>{plan?.moves.length ?? "—"}</b></div>
            <div><span><MapIcon />Total bike-km</span><b>{plan ? formatNumber(plan.moves.reduce((sum, move) => sum + move.bikes * move.distance_km, 0)) : "—"}</b></div>
          </div>
          <div className="safety-stack">
            <span className="micro-label">HARD INVARIANTS</span>
            {["Source safety floor", "Destination dock margin", "4 km route radius", "Global move ceiling"].map((item) => <div key={item}><Check size={13} /><span>{item}</span><b>enforced</b></div>)}
          </div>
          {plan && <p className="plan-limit"><TriangleAlert size={14} />{plan.limitations}</p>}
        </aside>
      </div>
    </>
  );
}

function ScoreBar({ label, value, target, invert = false }: { label: string; value: number; target: number; invert?: boolean }) {
  const passed = invert ? value <= target : value >= target;
  return (
    <div className="score-bar">
      <div><span>{label}</span><strong>{formatPercent(value, 2)}</strong><b className={passed ? "pass" : "fail"}>{passed ? "PASS" : "BELOW TARGET"}</b></div>
      <div className="score-track"><i style={{ width: `${Math.min(value * 100, 100)}%` }} /><u style={{ left: `${target * 100}%` }} /></div>
      <small>target {formatPercent(target)}</small>
    </div>
  );
}

function ModelView({ data, model }: { data: RiskResponse; model: ModelSummary | null }) {
  const frozen = model?.frozen_test || {};
  const replay = model?.decision_replay;
  const baseline = replay?.baseline?.with_rebalancing;
  const candidate = replay?.candidate?.with_rebalancing;
  const failureDelta = candidate && baseline ? candidate.service_failures - baseline.service_failures : 0;
  const moveDelta = candidate && baseline ? candidate.bikes_moved - baseline.bikes_moved : 0;
  return (
    <>
      <div className="page-intro compact"><div><span className="eyebrow"><span /> MODEL RISK MANAGEMENT</span><h1>Release governance</h1><p>Forecast quality and operational behavior are separate gates. The release status is evidence, not decoration.</p></div></div>
      <section className="release-hero panel">
        <div className="release-ring"><div><FlaskConical /><strong>SHADOW</strong><span>MODE ONLY</span></div></div>
        <div className="release-copy"><span className="micro-label">CURRENT RELEASE VERDICT</span><h2>Forecast champion.<br /><em>Decision challenger.</em></h2><p>The candidate cleared every chronological forecast fold, but did not Pareto-dominate the baseline decision policy. It stays in shadow mode until the tradeoff is resolved.</p><div className="model-tags"><code>{data.model_version}</code><span>{data.data_mode}</span><span>{data.horizon_minutes}m horizon</span></div></div>
        <div className="release-gates">
          <div className="gate passed"><span>01</span><div><strong>Chronological validation</strong><small>{model?.promotion.candidate_wins ?? 0}/3 fold wins</small></div><Check /></div>
          <div className="gate passed"><span>02</span><div><strong>Forecast promotion</strong><small>{formatPercent(model?.promotion.mean_relative_improvement, 2)} mean lift</small></div><Check /></div>
          <div className="gate blocked"><span>03</span><div><strong>Decision dominance</strong><small>operational tradeoff unresolved</small></div><X /></div>
        </div>
      </section>
      <div className="governance-grid">
        <section className="quality-panel panel">
          <div className="panel-header"><div><span className="micro-label">FROZEN TEST / UNTOUCHED</span><h2>Forecast evidence</h2></div><span className="window-chip">{formatNumber(model?.rows, 0)} rows</span></div>
          <div className="quality-lead"><div><span>Candidate MAE</span><strong>{formatNumber(frozen.combined_mae, 3)}</strong></div><div className="vs-line"><i /><span>vs</span><i /></div><div><span>Baseline MAE</span><strong>{formatNumber(frozen.baseline_combined_mae, 3)}</strong></div><div className="lift-badge"><ArrowDownRight />{formatPercent(frozen.candidate_relative_improvement, 2)} lower</div></div>
          <ScoreBar label="Departure interval coverage" value={frozen.departures_interval_coverage || 0} target={0.9} />
          <ScoreBar label="Arrival interval coverage" value={frozen.arrivals_interval_coverage || 0} target={0.9} />
          <ScoreBar label="Top-decile imbalance recall" value={frozen.top_10pct_imbalance_recall || 0} target={0.4} />
        </section>
        <section className="tradeoff-panel panel">
          <div className="panel-header"><div><span className="micro-label">DECISION REPLAY</span><h2>Pareto frontier</h2></div><span className="window-chip">frozen horizon</span></div>
          <div className="tradeoff-plot">
            <div className="plot-y">service failures <ArrowRight /></div><div className="plot-x">bikes moved <ArrowRight /></div>
            <span className="quadrant q1">dominant</span><span className="quadrant q2">cost tradeoff</span>
            <div className="plot-point baseline" style={{ left: "68%", top: "42%" }}><i /><span>Baseline</span><b>{baseline?.service_failures}</b></div>
            <div className="plot-point candidate" style={{ left: "45%", top: "64%" }}><i /><span>Candidate</span><b>{candidate?.service_failures}</b></div>
          </div>
          <div className="tradeoff-summary"><div><span>Failure delta</span><strong className="negative">+{failureDelta}</strong><small>candidate worse</small></div><div><span>Move delta</span><strong className="positive">{moveDelta}</strong><small>candidate lighter</small></div><div><span>Verdict</span><strong>NO DOMINANCE</strong><small>shadow retained</small></div></div>
        </section>
      </div>
      <section className="lineage-panel panel">
        <div className="panel-header"><div><span className="micro-label">MODEL LINEAGE</span><h2>Evidence chain</h2></div><Layers3 size={17} /></div>
        <div className="lineage-track">
          {[{n:"01",t:"Official trips",m:`${formatNumber(model?.rows,0)} supervised rows`,i:Database},{n:"02",t:"Rolling folds",m:"strict chronological windows",i:Activity},{n:"03",t:"Calibration",m:"finite-sample residual bounds",i:Gauge},{n:"04",t:"Frozen replay",m:"forecast + decision evidence",i:Route},{n:"05",t:"Shadow release",m:data.model_version,i:MoonStar}].map((item,index) => { const Icon=item.i; return <div className="lineage-step" key={item.n}><span>{item.n}</span><i><Icon /></i><div><strong>{item.t}</strong><small>{item.m}</small></div>{index<4&&<ArrowRight />}</div>; })}
        </div>
      </section>
    </>
  );
}

function CommandPalette({
  open,
  onClose,
  stations,
  onView,
  onStation,
  onCollect,
  onRefresh,
  onPlan,
}: {
  open: boolean;
  onClose: () => void;
  stations: Station[];
  onView: (view: View) => void;
  onStation: (station: Station) => void;
  onCollect: () => void;
  onRefresh: () => void;
  onPlan: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const [query, setQuery] = useState("");
  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) { dialog.showModal(); window.setTimeout(() => dialog.querySelector("input")?.focus(), 30); }
    if (!open && dialog.open) dialog.close();
  }, [open]);
  const execute = (action: () => void) => { action(); setQuery(""); onClose(); };
  const needle = query.toLowerCase();
  const matchingStations = stations.filter((station) => station.station_name.toLowerCase().includes(needle)).slice(0, 6);
  return (
    <dialog ref={ref} className="command-dialog" onClose={onClose} onClick={(event) => { if (event.target === ref.current) onClose(); }}>
      <div className="command-input"><Search /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search the network or run an operation…" aria-label="Command search" /><kbd>ESC</kbd></div>
      <div className="command-results">
        <span className="command-group-label">Navigate</span>
        {views.filter((item) => item.label.toLowerCase().includes(needle) || !needle).map((item) => { const Icon=item.icon; return <button key={item.id} onClick={() => execute(() => onView(item.id))}><i><Icon /></i><div><strong>{item.label}</strong><span>{item.caption}</span></div><kbd>↵</kbd></button>; })}
        <span className="command-group-label">Operations</span>
        {[{label:"Collect live GBFS snapshot",caption:"Refresh source inventory",icon:Database,action:onCollect},{label:"Refresh station risks",caption:"Recompute the decision surface",icon:RefreshCw,action:onRefresh},{label:"Generate shadow plan",caption:"Run constrained transfer solver",icon:Sparkles,action:onPlan}].filter((item)=>item.label.toLowerCase().includes(needle)||!needle).map((item)=>{const Icon=item.icon;return <button key={item.label} onClick={()=>execute(item.action)}><i><Icon /></i><div><strong>{item.label}</strong><span>{item.caption}</span></div><kbd>↵</kbd></button>;})}
        {needle && matchingStations.length > 0 && <><span className="command-group-label">Stations</span>{matchingStations.map((station)=><button key={station.station_id} onClick={()=>execute(()=>onStation(station))}><i className={`command-risk risk-${station.risk_level}`}><CircleDot /></i><div><strong>{station.station_name}</strong><span>{station.risk_type} risk · {formatPercent(station.risk_score)}</span></div><kbd>↵</kbd></button>)}</>}
        {needle && !matchingStations.length && !views.some((item)=>item.label.toLowerCase().includes(needle)) && <div className="command-empty">No matching command or station.</div>}
      </div>
      <footer><span><kbd>↑</kbd><kbd>↓</kbd> navigate</span><span><kbd>↵</kbd> execute</span><span>VELO COMMAND INDEX</span></footer>
    </dialog>
  );
}

function LoadingScreen() {
  return <div className="loading-screen"><BrandMark /><div className="loading-word"><span>VELO</span><strong>GUARD</strong></div><div className="loading-rail"><i /></div><p>Synchronizing decision surface</p></div>;
}

function EmptyState({ error, onDemo }: { error: string; onDemo: () => void }) {
  return <main className="fatal-state"><div className="fatal-grid" /><BrandMark /><span className="eyebrow"><span /> SYSTEM NOT READY</span><h1>Decision surface unavailable.</h1><p>{error}</p><div className="fatal-commands"><code>veloguard demo</code><ArrowRight /><code>veloguard serve</code></div><button className="button button-primary" onClick={onDemo}><RefreshCw size={15} />Retry connection</button></main>;
}

function App() {
  const [view, setView] = useState<View>("command");
  const [data, setData] = useState<RiskResponse | null>(null);
  const [model, setModel] = useState<ModelSummary | null>(null);
  const [selected, setSelected] = useState<Station | null>(null);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [loading, setLoading] = useState<string | null>("boot");
  const [error, setError] = useState("");
  const [palette, setPalette] = useState(false);
  const [toast, setToast] = useState("");

  const load = useCallback(async (allowStale = true) => {
    setLoading("refresh"); setError("");
    try {
      const [risks, summary] = await Promise.all([
        api<RiskResponse>(`/v1/stations/risks?limit=500&allow_stale=${allowStale}`),
        api<ModelSummary>("/v1/model/summary"),
      ]);
      setData(risks); setModel(summary); setSelected((current) => risks.stations.find((station) => station.station_id === current?.station_id) || risks.stations[0] || null);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Unable to load VeloGuard"); }
    finally { setLoading(null); }
  }, []);

  useEffect(() => { load(true); }, [load]);
  useEffect(() => {
    const listener = (event: KeyboardEvent) => {
      if (event.key.toLowerCase() === "k" && (event.ctrlKey || event.metaKey)) { event.preventDefault(); setPalette((value) => !value); }
    };
    document.addEventListener("keydown", listener); return () => document.removeEventListener("keydown", listener);
  }, []);
  useEffect(() => { if (!toast) return; const timer = window.setTimeout(() => setToast(""), 3800); return () => window.clearTimeout(timer); }, [toast]);

  const collect = async () => {
    setLoading("collect");
    try { await api("/v1/snapshots/collect", { method: "POST" }); await load(false); setToast("Fresh GBFS snapshot synchronized"); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Snapshot collection failed"); setLoading(null); }
  };
  const generate = async () => {
    if (data?.feed_is_stale) { setToast("Collect a fresh snapshot before planning"); return; }
    setLoading("plan");
    try {
      const result = await api<Plan>("/v1/rebalance-plans", { method: "POST", headers: { "Content-Type": "application/json", "Idempotency-Key": crypto.randomUUID() }, body: JSON.stringify({ max_total_moves: 40 }) });
      setPlan(result); setView("dispatch"); setToast(`Shadow plan ${result.plan_id.slice(0, 8)} generated`);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Plan generation failed"); }
    finally { setLoading(null); }
  };
  const selectStation = useCallback((station: Station) => { setSelected(station); setView("command"); }, []);

  if (loading === "boot" && !data) return <LoadingScreen />;
  if (!data) return <EmptyState error={error || "Run the offline demo to create a model and snapshot."} onDemo={() => load(true)} />;
  return (
    <div className="app-shell">
      <Rail view={view} onView={setView} onPalette={() => setPalette(true)} />
      <div className="app-frame">
        <TopBar view={view} data={data} loading={loading} onRefresh={() => load(true)} onCollect={collect} onPalette={() => setPalette(true)} />
        <main className="workspace">
          {view === "command" && <CommandView data={data} selected={selected} onSelect={setSelected} />}
          {view === "network" && <NetworkView data={data} onSelect={selectStation} />}
          {view === "dispatch" && <DispatchView data={data} plan={plan} loading={loading} onGenerate={generate} />}
          {view === "model" && <ModelView data={data} model={model} />}
        </main>
      </div>
      <CommandPalette open={palette} onClose={() => setPalette(false)} stations={data.stations} onView={setView} onStation={selectStation} onCollect={collect} onRefresh={() => load(true)} onPlan={generate} />
      {toast && <div className="toast" role="status"><Check size={15} /><span>{toast}</span></div>}
      {error && <div className="error-toast" role="alert"><CircleAlert size={16} /><span>{error}</span><button onClick={() => setError("")} aria-label="Dismiss error"><X size={14} /></button></div>}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);
