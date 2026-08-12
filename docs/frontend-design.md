# VeloGuard Frontend Design

## Product intent

The console is a decision surface for three jobs: locate near-term station pressure, inspect why a station is risky, and simulate a capacity-safe response without implying automated dispatch authority. Complexity is earned by the domain: geospatial state, uncertainty, freshness, constrained moves, and model release evidence must remain visible at the same time.

The design deliberately avoids a generic admin-dashboard template. Its identity comes from domain-specific information hierarchy and interactions rather than decorative animation.

## GitHub architecture research

| Reference | Useful idea | VeloGuard decision |
|---|---|---|
| [Grafana dashboards](https://github.com/grafana/grafana/blob/main/docs/sources/fundamentals/dashboards-overview/index.md) | Data source, transformation, and panel are separate concerns | Preserve API response -> view model -> panel boundaries |
| [Grafana Scenes](https://github.com/grafana/scenes) | Composable data-driven application surfaces | Compose four task-oriented views; do not import the framework |
| [Kibana](https://github.com/elastic/kibana) | Dense operational navigation and inspectable evidence | Persistent rail, freshness header, exception-first tables |
| [MapLibre GL JS](https://github.com/maplibre/maplibre-gl-js) | WebGL map layers, feature state, pitch, and interaction | Use MapLibre for the risk field and selected-station emphasis |
| [cmdk](https://github.com/dip/cmdk) | Keyboard-first navigation and action discovery | Implement the pattern with native `dialog` to avoid another UI dependency |
| [TanStack Table](https://github.com/TanStack/table) | Headless sorting/filtering for large tables | Not adopted: the scoped queue is small enough for native React state and semantic HTML |

## Information architecture

```text
Global rail + freshness header
├── Command: network KPIs, WebGL risk field, station evidence, priority queue
├── Network: identity coverage, capacity topology, full station inventory
├── Dispatch: simulated move sequence, constraint ledger, shadow authority
└── Model: promotion folds, frozen metrics, interval coverage, Pareto verdict
```

The command palette is a cross-cutting index over views, operational actions, and stations. `Ctrl/Cmd + K` opens it without taking authority away from the API.

## Visual system

- Near-black graphite field with a restrained grid communicates an operational instrument, not a marketing page.
- Acid lime marks current navigation and system identity; cyan marks freshness and balanced state; amber marks governance; magenta marks risk.
- Display type, compact mono metadata, thin rules, asymmetric panels, and map overlays establish hierarchy without a component library.
- Motion is limited to state transitions and live affordances and is disabled by `prefers-reduced-motion`.
- At tablet widths, the rail becomes a bottom command bar. At 390 px, cards become two columns, dense secondary labels collapse, tables remain horizontally scrollable within their own region, and the document itself has no horizontal overflow.

## State and failure behavior

Server state is fetched on load and after collection or refresh. The UI owns only ephemeral interaction state. A stale snapshot remains inspectable but plan controls are disabled; API errors produce a visible toast and retain the last usable decision surface. Release authority is repeated in Command, Dispatch, and Model views so a shadow model cannot be mistaken for production authorization.

## Build and verification

```bash
# Linux / WSL
cd frontend
pnpm install
pnpm build
```

The Vite build emits production assets to `web/`, which FastAPI mounts at `/assets` and serves at `/`. Verification covers TypeScript compilation, Python API tests, desktop and 390 px browser interaction, horizontal overflow, command navigation, shadow plan generation, and browser console errors.

## Intentional limits

- No client state framework: the state graph does not justify one.
- No chart library: current charts are small, accessible SVG/CSS views of already computed data.
- No virtualized table: the portfolio is intentionally scoped to 40 stations.
- No automatic dispatch controls: every plan is explicitly a simulation until the decision gate passes.
