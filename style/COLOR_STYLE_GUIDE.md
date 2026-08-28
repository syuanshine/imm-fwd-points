# Color style guide — IMM forward-points dashboards

This is the palette used in the matplotlib charts under `output/` (see
[charts.py](../imm_fwd/charts.py)), specified precisely enough to rebuild as a
Streamlit theme. It follows a design-system-agnostic method: hues are assigned
by the **job the color does** (identity vs magnitude vs polarity), not chosen
for looks, and every categorical pairing here has been validated for
colorblind-safe separation — see "Why these hexes" below before changing any
of them.

## 1. Currency → color (categorical, identity)

Fixed assignment. **Never re-map** — if the dashboard lets a user filter down
to 3 currencies, THB keeps its blue and TWD keeps its magenta; colors must
never shift to fill gaps left by hidden series, or a color stops meaning one
currency across views/sessions.

| Currency | Light mode | Dark mode | Hue |
|---|---|---|---|
| THB | `#2a78d6` | `#3987e5` | blue |
| IDR | `#eb6834` | `#d95926` | orange |
| INR | `#1baf7a` | `#199e70` | aqua |
| PHP | `#eda100` | `#c98500` | yellow |
| TWD | `#e87ba4` | `#d55181` | magenta |
| KRW | `#4a3aa7` | `#9085e9` | violet |

KRW was originally plain green (`#008300`); it's violet here because green sat
too close to INR's aqua for colorblind readers on a 6-up chart (validated worst-case
CVD ∆E rose from 3.2 → 6.1 with the swap — still see the caveat in §4). Never
put KRW back to a green — that regression is exactly what was fixed.

```css
:root {
  --ccy-thb: #2a78d6;
  --ccy-idr: #eb6834;
  --ccy-inr: #1baf7a;
  --ccy-php: #eda100;
  --ccy-twd: #e87ba4;
  --ccy-krw: #4a3aa7;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ccy-thb: #3987e5;
    --ccy-idr: #d95926;
    --ccy-inr: #199e70;
    --ccy-php: #c98500;
    --ccy-twd: #d55181;
    --ccy-krw: #9085e9;
  }
}
:root[data-theme="dark"] {
  --ccy-thb: #3987e5;
  --ccy-idr: #d95926;
  --ccy-inr: #199e70;
  --ccy-php: #c98500;
  --ccy-twd: #d55181;
  --ccy-krw: #9085e9;
}
```

```python
CCY_COLORS_LIGHT = {
    "THB": "#2a78d6", "IDR": "#eb6834", "INR": "#1baf7a",
    "PHP": "#eda100", "TWD": "#e87ba4", "KRW": "#4a3aa7",
}
CCY_COLORS_DARK = {
    "THB": "#3987e5", "IDR": "#d95926", "INR": "#199e70",
    "PHP": "#c98500", "TWD": "#d55181", "KRW": "#9085e9",
}
```

## 2. Surfaces & text (theme-aware)

| Role | Light | Dark |
|---|---|---|
| Chart / page surface | `#fcfcfb` | `#1a1a19` |
| Primary text | `#0b0b0b` | `#ffffff` |
| Secondary text (axis labels, captions) | `#52514e` | `#c3c2b7` |
| Gridlines / recessive strokes | `#e4e3df` | `#2f2e2b` (≈ same step-down from dark surface) |

Streamlit `.streamlit/config.toml` (light):
```toml
[theme]
base = "light"
backgroundColor = "#fcfcfb"
secondaryBackgroundColor = "#f0efec"
textColor = "#0b0b0b"
primaryColor = "#2a78d6"   # THB blue as the interactive accent (buttons, sliders)
```
Streamlit only ships ONE static theme per config — it can't auto-switch on
`prefers-color-scheme`. For real dark-mode support, either maintain a second
`config.toml` the user swaps to, or build charts as HTML/JS (Plotly, custom
components) using the CSS variables above so they follow `st.theme` /
browser preference. Don't try to fake a second Streamlit theme by hand-picking
per-widget colors — it drifts from this palette immediately.

## 3. Non-categorical colors — assign by job, not vibe

- **Sequential (magnitude — heatmap cell, single ramp)**: one hue, light→dark.
  Default = blue: `#cde2fb → #86b6ef → #3987e5 → #2a78d6 → #1c5cab → #0d366b`.
  Never build a sequential ramp by lightening a categorical color on the fly —
  use this fixed ramp.
- **Diverging (polarity — correlation matrix, the fair-value gap chart, P&L
  above/below zero)**: blue ↔ red through a **gray** midpoint, matching
  `charts.py::plot_corr_heatmap`: `#104281 → #3987e5 → #f0efec → #e66767 → #8f1f1e`.
  Midpoint is gray, never a hue — a colored zero-point reads as "some value,"
  not "nothing."
- **Status (good/warning/serious/critical — e.g. a z-score severity badge, a
  stale-data flag)**: reserve dedicated colors, distinct from the 6 currency
  hues, and never reuse a currency's color for a status meaning. Suggested,
  consistent with this set's warm/cool logic: good `#1baf7a` (aqua-green),
  warning `#eda100` (yellow), serious `#eb6834` (orange), critical `#e34948`
  (red) — pair every status color with an icon/label, not color alone.

## 4. Rules that keep this from silently breaking

1. **Fixed hue order, never cycled.** If a 7th series is ever needed (unlikely
   for this fixed 6-currency universe, but if extended), do not auto-generate
   a new hue — fold the extra series into "Other," split into small multiples,
   or explicitly pick and validate a 7th slot (candidates from the same
   design system: `#e34948` red or `#008300` green — validate before use, see
   below).
2. **All 6 pass the *adjacent*-pairs colorblind check** (the relevant one for
   bar charts, line charts, and single-color small-multiple panels — this
   dashboard's chart types) in both light and dark mode. They do **not**
   cleanly clear the stricter *all-pairs* check past 3 series (a documented
   property of any 6+ hue categorical set, not specific to this palette) — so
   **never rely on color alone to identify a currency**: always keep a text
   label on-screen (subplot title, legend, axis tick, tooltip) alongside the
   color, exactly as every chart in `charts.py` already does.
3. **Color follows the entity, not its rank or position.** A filter that
   drops IDR must not cause TWD to inherit IDR's old chart position/color.
4. **One axis, ever.** Never plot two differently-scaled series (e.g. raw
   points and annualized %) on a dual-axis chart — use small multiples or
   index both to a common base instead. This applies regardless of color.
5. **If you touch any hex above**, re-validate before shipping — don't
   eyeball it. Port or reuse the palette validator (`dataviz` skill,
   `scripts/validate_palette.js`; this project used a Python port since no
   Node runtime was available) and check the *adjacent*-pairs case in both
   light and dark at minimum.

## 5. Quick reference — Streamlit component mapping

| Dashboard element | Color source |
|---|---|
| Currency line/bar in a chart | `CCY_COLORS_*[ccy]` (§1) — never derived |
| Selected-currency highlight / active filter chip | same currency's color, full opacity; unselected currencies drop to ~35% alpha, not gray (keeps hue identity even when de-emphasized) |
| KPI tile background / accent | `--surface-1` / `st.metric` default — do not tint tiles with currency colors, that colors text unnecessarily |
| Correlation matrix | diverging blue↔red (§3), never categorical hues |
| Z-score / rich-cheap gauge | diverging blue↔red if signed, or sequential blue if magnitude-only — not a currency color, since the metric itself isn't currency-identity, it's a level |
| "Data stale" / "provider error" banner | status red `#e34948` + icon + text, never a currency color |
