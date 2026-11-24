# IDAMS Interface Design System (DESIGN.md)

**Document Reference:** `frontend/DESIGN.md`  
**Version:** 3.0.0 (Automotive Monitoring Instrument Specification)  
**System:** Intelligent Driver Activity Monitoring & Safety Agent (IDAMS)  
**Target Environment:** React 18, Vite 5, Tailwind CSS, Three.js  
**Last Updated:** November 2026  

---

## 1. Design Intent

The **Intelligent Driver Activity Monitoring & Safety Agent (IDAMS)** interface is an **automotive monitoring instrument and supervisory cockpit**, not a generic commercial AI dashboard, marketing landing page, or consumer tech portal.

### 1.1 Core Principles
1. **Technical Precision:** Interface components deliver dense, legible, and actionable telemetry without decorative frivolity.
2. **Minimal & Calm:** The visual atmosphere recedes to highlight the vehicle's physical state, risk assessment, and explainability. No neon glows, purple gradients, glassmorphism, or holograms.
3. **Automotive Focus:** Inspired by authentic automotive instrumentation, avionics control rooms, and engineering testbeds.
4. **Information Density & Legibility:** Information is organized into structured, neutral panels with high typographic contrast and strict grid discipline.
5. **Explainability First:** The primary question answered by any screen is: *What is the current safety state, and exactly what evidence triggered it?*

---

## 2. Source of Truth for States and Numbers

The frontend is strictly a **supervisory presentation and interactive client**. It never calculates, infers, interpolates, or overrides safety states or risk scores.

### 2.1 Backend Authority
- All risk scores ($[0.0, 100.0]$), safety states, model confidences, alert severities, triggered rules, and rule explanations originate directly from the backend Safety Agent (`ml/agent/`).
- The frontend preserves the exact numerical precision and state boundaries emitted by the backend:
  - `NORMAL`: $[0.0, 25.0)$
  - `CAUTION`: $[25.0, 50.0)$
  - `WARNING`: $[50.0, 75.0)$
  - `HIGH_RISK`: $[75.0, 100.0]$
  - `INSUFFICIENT_DATA`: Score set to $0.0$ or `--` during transport or sensor pipeline outages.

### 2.2 Four Distinct Operational Telemetry Paths
The frontend must transparently display the active data-source mode via a dedicated **SourceBadge**:
1. **Historical Benchmark Replay:** Telemetry streamed from public research datasets (Dataset B MPU6050 and Dataset C mobile IMU).
2. **Simulated / Demo Presets:** Pre-packaged scenario presets (`/api/simulated/scenarios`) for functional demonstration and client bench testing.
3. **Interactive Assessment APIs:** User-adjusted or scripted feature vectors submitted to `/api/telemetry/assess` or `/api/telemetry/assess/raw`.
4. **Physical Sensor Research Ingestion:** High-rate 20 Hz smartphone telemetry received via `/api/sensor/stream` and validated against 6 acceptance criteria by `data_quality_service.py`.
   - **Critical Invariant:** Physical smartphone telemetry remains strictly designated as **`NOT_VERIFIED`** and gated from production ML inference (`proven_compatibility = False`). The UI never presents physical sensor logging as validated production ML inference.

---

## 3. Color System

The color system uses semantic CSS custom properties with strict separation between neutral surfaces, interaction accents, and reserved safety state colors.

### 3.1 Dark Theme (Default)
| Token Name | CSS Variable | Hex Value | Purpose |
| :--- | :--- | :--- | :--- |
| **Background** | `--background` | `#111312` | Deep neutral obsidian floor |
| **Surface 1** | `--surface-1` | `#181B1A` | Sidebar, header, bottom nav, dialogs |
| **Surface 2** | `--surface-2` | `#202423` | Cards, content containers, table headers |
| **Surface 3** | `--surface-3` | `#2A2F2D` | Active nav items, hover surfaces, inputs |
| **Border** | `--border` | `#363C3A` | 1px structural boundaries |
| **Primary Text** | `--text-primary` | `#ECEFED` | High-contrast headings and active metrics |
| **Secondary Text** | `--text-secondary` | `#B4BCB8` | Subtitles, table body, field labels |
| **Muted Text** | `--text-muted` | `#8A938F` | Timestamps, units, inactive indicators |
| **Accent** | `--accent` | `#8FB3D9` | Interactive focus, selected tabs, slider thumbs |

### 3.2 Light Theme
| Token Name | CSS Variable | Hex Value | Purpose |
| :--- | :--- | :--- | :--- |
| **Background** | `--background` | `#F3F4F2` | Clean neutral parchment canvas |
| **Surface 1** | `--surface-1` | `#FFFFFF` | Sidebar, header, modals |
| **Surface 2** | `--surface-2` | `#EBEDEB` | Card containers, panels, table headers |
| **Surface 3** | `--surface-3` | `#DEE1DF` | Active nav items, hover states, inputs |
| **Border** | `--border` | `#CDD2CF` | 1px neutral dividing boundaries |
| **Primary Text** | `--text-primary` | `#161918` | High-contrast headings and readouts |
| **Secondary Text** | `--text-secondary` | `#454C49` | Subtitles, table body text |
| **Muted Text** | `--text-muted` | `#646C68` | Secondary labels, units, metadata |
| **Accent** | `--accent` | `#2B5F8F` | Interactive focus, selected tabs, active indicators |

### 3.3 Accent Color Discipline
- The accent color (`#8FB3D9` dark / `#2B5F8F` light) is reserved exclusively for **interaction chrome** (navigation indicators, active tabs, buttons, checkboxes, focus rings).
- The accent color **never conveys safety meaning**.

### 3.4 Reserved Safety State Colors
Safety colors are applied only when representing verified safety states or severe vehicle faults:

| Safety State | Dark Theme Hex | Light Theme Hex | Meaning |
| :--- | :--- | :--- | :--- |
| **`NORMAL`** | `#4CAF50` | `#2E7D32` | Kinematics nominal, safe cruising |
| **`CAUTION`** | `#F59E0B` | `#B45309` | Elevated driving style, mild hazard |
| **`WARNING`** | `#F97316` | `#C2410C` | Severe maneuver detected, limit exceeded |
| **`HIGH_RISK` / `CRITICAL`** | `#EF4444` | `#B91C1C` | Compound emergency hazard |
| **`INSUFFICIENT_DATA`** | `#8A938F` | `#646C68` | Telemetry dropout or dual pipeline outage |

---

## 4. Typography

### 4.1 Font Families
- **Interface Text:** **IBM Plex Sans** (`"IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`)
- **Telemetry & Numbers:** **IBM Plex Mono** (`"IBM Plex Mono", "SF Mono", Consolas, monospace`)

### 4.2 Typographic Hierarchy & Scale
| Role | Font Family | Size / Line Height | Weight | Tracking | Usage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Display / Score** | IBM Plex Mono | `32px / 40px` | Bold (`700`) | `-0.5px` | Risk Gauge Score, primary KPIs |
| **Title / H1** | IBM Plex Sans | `24px / 32px` | SemiBold (`600`) | `-0.25px` | Page Titles, modal headers |
| **Section / H2** | IBM Plex Sans | `18px / 24px` | SemiBold (`600`) | `0px` | Card headers, section dividers |
| **Subtitle / H3** | IBM Plex Sans | `14px / 20px` | Medium (`500`) | `0px` | Sub-panel labels, widget titles |
| **Body** | IBM Plex Sans | `14px / 20px` | Regular (`400`) | `0px` | Descriptive paragraphs, explanations |
| **Small / Label** | IBM Plex Sans | `12px / 16px` | Medium (`500`) | `0.2px` | Badges, form labels, nav items |
| **Mono Metric** | IBM Plex Mono | `12px / 16px` | Medium (`500`) | `0px` | Telemetry readouts, table metrics |
| **Micro / Caption** | IBM Plex Mono | `10px / 14px` | Regular (`400`) | `0.4px` | Timestamps, units, source hashes |

---

## 5. Spacing, Shape, and Elevation

### 5.1 Spacing Scale (4px Base Grid)
- `4px` (`0.25rem`): Micro gaps, inline badge padding
- `8px` (`0.50rem`): Icon-to-text spacing, compact button padding
- `12px` (`0.75rem`): Form field padding, list item gaps
- `16px` (`1.00rem`): Standard card interior padding, grid gap
- `20px` (`1.25rem`): Section spacing
- `24px` (`1.50rem`): Page padding, major component margins
- `32px` (`2.00rem`): Dashboard layout gutter
- `48px` (`3.00rem`): Viewport separation

### 5.2 Shapes & Radii
- **Base Components:** `4px` (`rounded-sm`) for buttons, inputs, chips, badges, and table rows.
- **Containers & Panels:** `8px` (`rounded-md`) for cards, modals, and the 3D viewport.
- **Forbidden Shapes:** No pill buttons (`rounded-full`), circular action buttons, or arbitrary radii.

### 5.3 Elevation & Surfaces
- Flat, matte visual planes.
- Borders: Crisp `1px solid var(--border)` defining every card and viewport boundary.
- Shadows: Minimal subtle drop shadow on modals only (`shadow-md`). Zero blur glows, bloom, or colored drop shadows.

---

## 6. Layout & Application Shell

### 6.1 Desktop Layout (>= 1024px)
- **Sidebar:** Width fixed at **240px**. Surface: `var(--surface-1)`. Border-right: `1px solid var(--border)`.
  - 7 primary navigation sections:
    1. Dashboard (`dashboard`)
    2. Live Monitoring (`live_monitoring`)
    3. Driving Behaviour (`driving_behaviour`)
    4. Safety Alerts (`safety_alerts`)
    5. Trip History (`trip_history`)
    6. Driving Insights (`driving_insights`)
    7. System Settings (`settings`)
  - Selected item: Background `var(--surface-3)`, left edge has a **2px solid var(--accent)** indicator.
- **Top Status Strip:** Height **44px**. Full-width sticky bar displaying:
  - Current Safety State Badge (`NORMAL`, `CAUTION`, `WARNING`, `HIGH_RISK`, `INSUFFICIENT_DATA`)
  - Unified Risk Score (`XX.X / 100.0 pts`)
  - Operational Source Badge (`Historical Replay`, `Simulation Preset`, `Physical Ingestion`, `Disconnected`)
  - Backend Connection & Model Status (`Models [4/4] Verified`)
  - Theme Toggle Switch
- **Content Viewport:** 12-column responsive grid with `16px` or `24px` gutters.

### 6.2 Tablet Layout (768px - 1023px)
- Collapsible sidebar (collapses to 64px icon-rail or hidden drawer).
- Status strip persists at top.
- Grid reflows to 6-column or single-column stack.

### 6.3 Mobile Layout (< 768px)
- Sticky top header with minimal status indicators and drawer toggle.
- Bottom navigation bar housing 4 primary items + "More" drawer item.
- Single-column vertical layout with touch targets $\ge 44\text{px}$.
- Zero horizontal scrolling.

---

## 7. Reusable Component System

### 7.1 StatusStrip (`components/layout/StatusStrip.jsx`)
- Displays overall system status in a compact, persistent horizontal strip.
- Elements: Safety State Chip, Mono Score, SourceBadge, Server Health, Theme Toggle.

### 7.2 SourceBadge (`components/common/SourceBadge.jsx`)
- Clearly labels the active telemetry ingestion mode:
  - `BENCHMARK_REPLAY`: Neutral gray badge
  - `SIMULATION_PRESET`: Neutral subtle blue badge
  - `PHYSICAL_RESEARCH`: Warning amber outline (`RESEARCH_ONLY - GATED`)
  - `DISCONNECTED`: Red outline

### 7.3 RiskGauge (`components/common/RiskGauge.jsx`)
- Semi-circular 270-degree SVG dial.
- Range: $0.0$ to $100.0$.
- Fixed threshold markers at $25.0$ (`CAUTION`), $50.0$ (`WARNING`), and $75.0$ (`HIGH_RISK`).
- Displays large mono score readout (`font-mono text-2xl`).
- Secondary residual-risk arc rendered only if backend explicitly supplies `residual_risk_points`.

### 7.4 AlertRow & EvidencePanel (`components/safety/AlertRow.jsx`)
- Compact tabular row:
  - Timestamp (`font-mono text-xs`)
  - Event Identifier (`font-sans font-medium`)
  - Severity Chip (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`)
  - Safety State (`NORMAL`, `CAUTION`, `WARNING`, `HIGH_RISK`)
  - Model Confidence (`XX.X%`)
  - Evidence Action Button (`View Evidence`)
- Evidence Panel modal / drawer: Displays triggered rules, raw sensor measurements, threshold comparisons, and causal explanations.

### 7.5 LiveReadout (`components/common/LiveReadout.jsx`)
- Standardized telemetry tile: Label, Large Mono Value, Unit, Nominal Range Indicator.
- Shows `N/A` with neutral styling if measurement is unavailable.

---

## 8. 3D Vehicle Simulation Scene (`components/three/CinematicVehicleScene.jsx`)

### 8.1 Visual Staging
- **Background:** Deep neutral graphite (`#111312` in dark mode, `#EBEDEB` in light mode).
- **Floor:** Flat ground plane with subtle 1.0m grid lines.
- **Lighting:**
  - Key Directional Light: Neutral white (`#FFFFFF`, intensity 2.2), soft shadows.
  - Fill Light: Soft neutral fill (`#ECEFED`, intensity 1.2).
  - Ambient / Hemisphere: Low-contrast sky-ground fill.
  - **No neon rim lighting, colored spotlights, or glowing laser effects.**
- **Vehicle Materials:**
  - Default finish: **Titanium Slate** (`#272D3B`, roughness 0.25, metalness 0.80).
  - Cabin Canopy: Dark neutral tinted glass (`opacity: 0.85`, roughness 0.10).
  - Wheels & Trim: Satin alloy rims with realistic dark rubber tires.
- **Ground State Ring:** Flat circular ring under the vehicle tinted to the active safety state color with minimal opacity (0.25).

### 8.2 Kinematic Motion Mapping
Animations reflect actual telemetry data:

| Driving Condition | Telemetry Trigger | Visual Kinematic Reaction |
| :--- | :--- | :--- |
| **Normal Cruising** | Maneuver = `0: NONE` | Quiescent suspension oscillation ($4\text{ Hz}$, $0.008$ amplitude) |
| **Aggressive Driving** | Style = `1: AGGRESSIVE` | Elevated chassis vibration ($8\text{ Hz}$, $0.020$ amplitude) |
| **Sudden Acceleration** | Maneuver = `1: SA` | Chassis pitch-up (nose up $+0.05\text{ rad}$), rear squat |
| **Sudden Braking** | Maneuver = `2: SD` | Chassis pitch-dive (nose down $-0.07\text{ rad}$), red taillights illuminated |
| **Sharp Right Turn** | Maneuver = `3: SRT` | Front wheels steer $+25^\circ$, body rolls left $+0.06\text{ rad}$, right blinker active |
| **Sharp Left Turn** | Maneuver = `4: SLT` | Front wheels steer $-25^\circ$, body rolls right $-0.06\text{ rad}$, left blinker active |

### 8.3 Camera Perspectives
Five selectable camera positions toggled via a segmented control below the canvas:
1. **Orbit:** Free 360-degree rotation via mouse drag and zoom via wheel.
2. **Chase:** Isometric trailing third-person view following the vehicle.
3. **Hood:** Forward-facing driver viewpoint over the vehicle bonnet.
4. **Top:** Orthographic overhead plan view showing vehicle trajectory.
5. **Three-Quarter:** Isometric elevated angle showcasing steering and roll.

### 8.4 Simulation Disclosure
The scene displays a permanent textual caption:  
*“Illustrative 3D Digital Twin — Motion driven by Safety Agent telemetry evaluation. Not real-time physical vehicle telemetry.”*

---

## 9. Icons

- **Icon Set:** `lucide-react`.
- **Stroke Width:** `1.75px` consistent across all icons.
- **Size Standards:** `16px` for inline labels and table rows; `20px` for section headers and primary navigation.
- **Icon Boxes:** No decorative colored squares or gradient backgrounds behind icons.

---

## 10. Motion & Transitions

- **Duration:** `150ms` for buttons and hover feedback; `200ms` for drawer and tab transitions.
- **Timing Function:** `cubic-bezier(0.16, 1, 0.3, 1)` (smooth ease-out). No bouncy spring oscillations.
- **Accessibility:** Full support for `prefers-reduced-motion`. When active, all 3D vehicle oscillations and camera lerping are disabled.

---

## 11. Copy & Plain-Language Standards

1. Use plain, driver-accessible English across primary views.
2. Never use em dashes (`—` or `--`); use colons, commas, or structured labels.
3. Never use emojis in UI chrome.
4. No AI marketing buzzwords ("revolutionary", "intelligent magic", "hyper-fast").
5. Display accurate SI units for all metrics: `m/s²`, `rad/s`, `km/h`, `s`, `ms`.

---

## 12. Accessibility (WCAG 2.2 AA)

1. **Color Contrast:** Minimum 4.5:1 text-to-background contrast in both dark and light themes.
2. **Focus Indicators:** Explicit `2px solid var(--accent)` outline on all interactive elements when keyboard-focused (`focus-visible`).
3. **Multi-Modal Indication:** Safety states are never communicated by color alone; every state is accompanied by a text label and icon.
4. **Touch Targets:** Minimum $40\times 40\text{px}$ on desktop, $44\times 44\text{px}$ on mobile.

---

## 13. Tokens Implementation

### 13.1 CSS Custom Properties (`src/index.css`)
```css
:root {
  /* Light Theme */
  --background: #F3F4F2;
  --surface-1: #FFFFFF;
  --surface-2: #EBEDEB;
  --surface-3: #DEE1DF;
  --border: #CDD2CF;
  --text-primary: #161918;
  --text-secondary: #454C49;
  --text-muted: #646C68;
  --accent: #2B5F8F;

  --state-normal: #2E7D32;
  --state-caution: #B45309;
  --state-warning: #C2410C;
  --state-high-risk: #B91C1C;
  --state-insufficient: #646C68;

  --font-sans: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --font-mono: "IBM Plex Mono", "SF Mono", Consolas, monospace;
}

html.dark {
  /* Dark Theme (Default) */
  --background: #111312;
  --surface-1: #181B1A;
  --surface-2: #202423;
  --surface-3: #2A2F2D;
  --border: #363C3A;
  --text-primary: #ECEFED;
  --text-secondary: #B4BCB8;
  --text-muted: #8A938F;
  --accent: #8FB3D9;

  --state-normal: #4CAF50;
  --state-caution: #F59E0B;
  --state-warning: #F97316;
  --state-high-risk: #EF4444;
  --state-insufficient: #8A938F;
}
```

### 13.2 Tailwind Configuration Extension
```javascript
module.exports = {
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: 'var(--background)',
        'surface-1': 'var(--surface-1)',
        'surface-2': 'var(--surface-2)',
        'surface-3': 'var(--surface-3)',
        border: 'var(--border)',
        'text-primary': 'var(--text-primary)',
        'text-secondary': 'var(--text-secondary)',
        'text-muted': 'var(--text-muted)',
        accent: 'var(--accent)',
        'state-normal': 'var(--state-normal)',
        'state-caution': 'var(--state-caution)',
        'state-warning': 'var(--state-warning)',
        'state-high-risk': 'var(--state-high-risk)',
        'state-insufficient': 'var(--state-insufficient)',
      },
      fontFamily: {
        sans: ['var(--font-sans)'],
        mono: ['var(--font-mono)'],
      },
      borderRadius: {
        sm: '4px',
        md: '8px',
      },
    },
  },
};
```

---

## 14. Anti-Pattern Checklist

- [x] No purple-to-blue or rainbow gradients.
- [x] No neon cyan or green glowing rim lights in the 3D scene.
- [x] No glassmorphism, blur effects, or translucent frosted overlays.
- [x] No pill-shaped buttons (`rounded-full`).
- [x] No gradient text.
- [x] No second accent color.
- [x] No calculated risk scores in JavaScript.
- [x] No fabricated telemetry values (missing values shown as `N/A`).
- [x] No claiming physical phone sensor data is validated production ML inference.
- [x] No em dashes in text.

---

## 15. Implementation & Rollout Order

1. **Phase 1: Design Tokens & Typography Foundation:** Implement CSS custom properties, Google Fonts IBM Plex, and Tailwind theme bindings.
2. **Phase 2: App Shell & Navigation:** Deploy 240px Sidebar, StatusStrip, ThemeToggle, and responsive drawers.
3. **Phase 3: Dashboard Redesign:** Align RiskGauge, 3D Vehicle Viewport, Explainability Panel, and Telemetry Tiles.
4. **Phase 4: Six Subsystem Pages:** Synchronize Live Monitoring, Driving Behaviour, Safety Alerts, Trip History, Driving Insights, and System Settings.
5. **Phase 5: 3D Scene Refinement:** Eliminate neon artifacts, add 1m grid floor, Titanium Slate finish, and camera perspective bar.
6. **Phase 6: Verification & Quality Assurance:** Test dark and light themes, responsive widths, model checksum verification, and production build.

---

## 16. Open Items & Engineering Roadmap

1. **On-Vehicle Physical Calibration:** Live monitoring displays physical streams for data-quality inspection only (`proven_compatibility = False`) until static standstill calibration and on-road smoke tests pass.
2. **OBD-II / CAN Telemetry Expansion:** Future integration of steering wheel angle, throttle position, and brake pressure.
3. **Multi-Camera Digital Twin:** Direct synchronization with multi-camera driver monitoring hardware when available.
