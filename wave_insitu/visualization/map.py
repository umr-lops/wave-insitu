"""
insitu_map.py
=============
Interactive Folium map for in-situ ocean observation data.

Author: Edouard Gauvrit (edouard.gauvrit@ifremer.fr)

Supported platforms (all optional):
  - Saildrones  : produced by saildrone_loader.load_saildrone_catalog()
  - LDL DWSD    : produced by ldl_loader.load_ldl_catalog()

Features:
  - Trajectory decimation to keep the HTML lightweight
  - Unique color per track name, consistent across all layer groups
  - Saildrones grouped by provider  (shown by default)
  - LDL drifters grouped by cyclone (shown by default)
  - Individual track layers available in LayerControl (hidden by default)
  - Click on a trajectory: highlight + multi-tab Plotly popup
    (one tab per available variable — only variables with data are shown)
  - Tooltip: platform-specific metadata + available variables
  - Double-handle time slider: show only tracks active within the selected window
  - Longitude wrapping (periodic map)
  - Floating color legend (commented out by default)
"""

import colorsys
import json

import folium
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Colormap utilities for TC tracks
# ---------------------------------------------------------------------------

def load_cpt_colormap(filepath: str) -> dict:
    """
    Load a GMT .cpt colormap file and return a matplotlib-compatible colorDict.

    Supports both RGB (0-255) and HSV color models.

    Parameters
    ----------
    filepath : path to the .cpt file

    Returns
    -------
    dict with keys 'red', 'green', 'blue' suitable for
    matplotlib.colors.LinearSegmentedColormap.from_list()
    """
    x, r, g, b = [], [], [], []
    color_model = "RGB"

    with open(filepath) as f:
        lines = f.readlines()

    x_temp = r_temp = g_temp = b_temp = 0.0

    for line in lines:
        ls = line.split()
        if not ls:
            continue
        if line[0] == "#":
            if ls[-1] == "HSV":
                color_model = "HSV"
            continue
        if ls[0] in ("B", "F", "N"):
            continue

        x.append(float(ls[0]))
        r.append(float(ls[1]))
        g.append(float(ls[2]))
        b.append(float(ls[3]))
        x_temp = float(ls[4])
        r_temp = float(ls[5])
        g_temp = float(ls[6])
        b_temp = float(ls[7])

    x.append(x_temp)
    r.append(r_temp)
    g.append(g_temp)
    b.append(b_temp)

    x = np.array(x)
    r = np.array(r)
    g = np.array(g)
    b = np.array(b)

    if color_model == "HSV":
        for i in range(len(r)):
            r[i], g[i], b[i] = colorsys.hsv_to_rgb(r[i] / 360.0, g[i], b[i])
    else:  # RGB 0-255
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0

    x_norm = (x - x[0]) / (x[-1] - x[0])

    color_dict = {
        "red":   [[x_norm[i], r[i], r[i]] for i in range(len(x))],
        "green": [[x_norm[i], g[i], g[i]] for i in range(len(x))],
        "blue":  [[x_norm[i], b[i], b[i]] for i in range(len(x))],
        # Store raw bounds for normalisation in the map renderer
        "_vmin": float(x[0]),
        "_vmax": float(x[-1]),
    }
    return color_dict


def colordict_to_hex(
    color_dict: dict,
    value: float,
    vmin: float | None = None,
    vmax: float | None = None,
) -> str:
    """
    Map a scalar value to a HEX color string using a loaded .cpt colorDict.

    The .cpt file stores normalised x-coordinates (0-1 after load_cpt_colormap).
    vmin / vmax define the physical data range to map onto [0, 1].
    If not provided, the raw _vmin/_vmax stored in color_dict are used
    (which equals 0-1 for most .cpt files — always pass explicit bounds).

    Parameters
    ----------
    color_dict : output of load_cpt_colormap()
    value      : physical scalar value to map (e.g. wind speed in m/s)
    vmin       : lower bound of the physical range (maps to colormap start)
    vmax       : upper bound of the physical range (maps to colormap end)

    Returns
    -------
    HEX color string, e.g. '#ff4400'
    """
    lo = vmin if vmin is not None else color_dict["_vmin"]
    hi = vmax if vmax is not None else color_dict["_vmax"]
    t  = float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))

    def _interp_channel(channel_key: str) -> float:
        pts = color_dict[channel_key]
        xs  = [p[0] for p in pts]
        ys  = [p[1] for p in pts]
        return float(np.interp(t, xs, ys))

    rv = int(_interp_channel("red")   * 255)
    gv = int(_interp_channel("green") * 255)
    bv = int(_interp_channel("blue")  * 255)
    return "#{:02x}{:02x}{:02x}".format(
        np.clip(rv, 0, 255),
        np.clip(gv, 0, 255),
        np.clip(bv, 0, 255),
    )


def build_colorbar_html(
    color_dict: dict,
    vmin: float,
    vmax: float,
    label: str = "Wind speed (knots)",
    n_steps: int = 256,
    width: int = 220,
    height: int = 16,
) -> str:
    """
    Generate a standalone HTML colorbar string for embedding in a Folium map.

    Renders as a CSS linear-gradient bar with min/max labels, suitable for
    injection via folium.Element into the map root HTML.

    Parameters
    ----------
    color_dict : output of load_cpt_colormap()
    vmin, vmax : physical data range
    label      : caption displayed above the bar
    n_steps    : number of gradient stops (higher = smoother)
    width      : bar width in pixels
    height     : bar height in pixels

    Returns
    -------
    HTML string for a floating colorbar panel.
    """
    stops = []
    for i in range(n_steps + 1):
        t   = i / n_steps
        val = vmin + t * (vmax - vmin)
        hex_color = colordict_to_hex(color_dict, val, vmin, vmax)
        pct = round(t * 100, 1)
        stops.append(f"{hex_color} {pct}%")

    gradient = ", ".join(stops)

    return f"""
<div id="tc-colorbar" style="
    position: fixed;
    top: 12px; right: 12px;
    z-index: 1000;
    background: rgba(25,25,25,0.88);
    border: 1px solid #555;
    border-radius: 6px;
    padding: 8px 12px 6px;
    font-family: sans-serif;
    backdrop-filter: blur(4px);
    min-width: {width + 24}px;
">
  <div style="font-size:10px;color:#ccc;margin-bottom:4px;
              letter-spacing:.04em">{label}</div>
  <div style="width:{width}px;height:{height}px;
              background:linear-gradient(to right,{gradient});
              border-radius:3px;"></div>
  <div style="display:flex;justify-content:space-between;
              font-size:10px;color:#aaa;margin-top:3px;">
    <span>{vmin:.0f}</span>
    <span>{vmin + (vmax-vmin)*0.25:.0f}</span>
    <span>{vmin + (vmax-vmin)*0.5:.0f}</span>
    <span>{vmin + (vmax-vmin)*0.75:.0f}</span>
    <span>{vmax:.0f}</span>
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# Variable registry
# ---------------------------------------------------------------------------

# Canonical variable name -> (display label, short label for tooltip, y-axis label)
VARIABLES: dict[str, tuple[str, str, str]] = {
    "significant_wave_height": ("Significant Wave Height", "Hs",    "Hs (m)"),
    "dominant_wave_period":    ("Dominant Wave Period",    "Tp",    "Tp (s)"),
    "average_period":          ("Average Period",          "Tavg",  "Tavg (s)"),
    "dominant_wave_direction": ("Wave Direction",          "Dir",   "Dir (deg)"),
    "wind_speed":              ("Wind Speed",              "WSpd",  "WSpd (m/s)"),
    "wind_direction":          ("Wind Direction",          "WDir",  "WDir (deg)"),
    "sea_surface_temperature": ("Sea Surface Temperature", "SST",   "SST (°C)"),
    "sea_level_pressure":      ("Sea Level Pressure",      "SLP",   "SLP (hPa)"),
}


def _short(var: str) -> str:
    return VARIABLES[var][1]

def _yaxis(var: str) -> str:
    return VARIABLES[var][2]


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

def _generate_colors(n: int) -> list[str]:
    """Generate n well-spaced colors using HLS -> HEX conversion."""
    colors = []
    for i in range(n):
        hue = (i / n + 0.05) % 1.0
        r, g, b = colorsys.hls_to_rgb(hue, 0.48, 0.82)
        colors.append("#{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(b * 255)
        ))
    return colors


def build_color_map(*dfs: pd.DataFrame) -> dict[str, str]:
    """
    Assign a unique HEX color to every unique 'name' value across all
    provided DataFrames. Colors are stable regardless of which DataFrames
    are passed together.
    """
    all_names: list[str] = []
    for df in dfs:
        if df is not None and not df.empty:
            all_names.extend(df["name"].unique().tolist())
    names  = sorted(set(all_names))
    colors = _generate_colors(len(names))
    return dict(zip(names, colors))


# ---------------------------------------------------------------------------
# Decimation
# ---------------------------------------------------------------------------

def decimate_trajectory(
    lats: np.ndarray,
    lons: np.ndarray,
    max_points: int = 800,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a trajectory to at most max_points by regular sub-sampling."""
    n = len(lats)
    if n <= max_points:
        return lats, lons
    indices = np.linspace(0, n - 1, max_points, dtype=int)
    indices = np.unique(np.concatenate([[0], indices, [n - 1]]))
    return lats[indices], lons[indices]


# ---------------------------------------------------------------------------
# Variable availability helpers
# ---------------------------------------------------------------------------

def get_available_variables(sub: pd.DataFrame) -> list[str]:
    """Return canonical variable names with at least one non-NaN value."""
    return [
        var for var in VARIABLES
        if var in sub.columns and sub[var].notna().any()
    ]


def format_variables_for_tooltip(available: list[str]) -> str:
    """Format available variable short labels as a comma-separated string."""
    if not available:
        return ""
    return ", ".join(_short(v) for v in available)


# ---------------------------------------------------------------------------
# Plotly multi-tab figure (embedded as JSON)
# ---------------------------------------------------------------------------

def _decimate_series(
    times: np.ndarray,
    values: np.ndarray,
    max_points: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Decimate a 1D series to max_points, keeping only non-NaN values."""
    valid_idx = np.where(~np.isnan(values.astype(float)))[0]
    if len(valid_idx) == 0:
        return np.array([]), np.array([])
    if len(valid_idx) > max_points:
        keep = np.linspace(0, len(valid_idx) - 1, max_points, dtype=int)
        valid_idx = valid_idx[keep]
    return times[valid_idx], values[valid_idx].astype(float)


def build_plotly_tabs_json(
    sub: pd.DataFrame,
    name: str,
    color: str,
    available: list[str],
    max_points: int = 500,
) -> str | None:
    """
    Build a multi-tab Plotly figure JSON for all available variables.
    Each variable gets its own tab via Plotly updatemenus buttons.
    Returns None if no plottable data exists.
    """
    if not available:
        return None

    times_str = sub["time"].astype(str).values

    traces  = []
    buttons = []

    for i, var in enumerate(available):
        t_dec, v_dec = _decimate_series(times_str, sub[var].values, max_points)
        if len(t_dec) == 0:
            continue

        traces.append({
            "type": "scatter",
            "mode": "lines",
            "x": t_dec.tolist(),
            "y": v_dec.tolist(),
            "line":  {"color": color, "width": 1.5},
            "name":  _short(var),
            "visible": i == 0,
            "hovertemplate": f"%{{x}}<br>{_short(var)} = %{{y:.2f}}<extra></extra>",
        })

        visibility = [False] * len(available)
        visibility[i] = True
        buttons.append({
            "label":  _short(var),
            "method": "update",
            "args": [
                {"visible": visibility},
                {"yaxis": {"title": _yaxis(var),
                           "showgrid": True, "gridcolor": "#444"}},
            ],
        })

    if not traces:
        return None

    fig_dict = {
        "data": traces,
        "layout": {
            "margin": {"l": 55, "r": 15, "t": 90, "b": 50},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(30,30,30,0.6)",
            "font": {"color": "#ddd", "size": 11},
            "title": {
                "text": name,
                "font": {"size": 13},
                "x": 0.5,
                "xanchor": "center",
                "y": 0.98,
                "yanchor": "top",
            },
            "xaxis": {"showgrid": True, "gridcolor": "#444"},
            "yaxis": {
                "title":    _yaxis(available[0]),
                "showgrid": True,
                "gridcolor": "#444",
            },
            "height": 480,
            "updatemenus": [{
                "type":      "buttons",
                "direction": "right",
                "x":         0.0,
                "xanchor":   "left",
                "y":         1.02,
                "yanchor":   "bottom",
                "showactive": True,
                "bgcolor": "rgba(50,50,50,0.85)",
                "bordercolor": "#666",
                "font": {"size": 11, "color": "#ddd"},
                "pad": {"t": 2, "b": 2, "l": 4, "r": 4},
                "buttons": buttons,
            }],
        },
    }
    return json.dumps(fig_dict)


def build_tc_windspeed_json(
    times: np.ndarray,
    wind_speeds: np.ndarray,
    name_sid: str,
    color: str,
    max_points: int = 500,
) -> str | None:
    """
    Build a Plotly figure JSON showing wind speed time series for a TC track.
    Returns None if no wind speed data exists or all values are NaN.
    """
    if wind_speeds is None or len(wind_speeds) == 0:
        return None
    
    if np.all(np.isnan(wind_speeds)):
        return None
    
    times_str = pd.to_datetime(times).astype(str).values
    
    # Decimate the data if needed
    t_dec, v_dec = _decimate_series(times_str, wind_speeds, max_points)
    if len(t_dec) == 0:
        return None

    fig_dict = {
        "data": [{
            "type": "scatter",
            "mode": "lines",
            "x": t_dec.tolist(),
            "y": v_dec.tolist(),
            "line": {"color": color, "width": 2},
            "name": "Wind Speed",
            "hovertemplate": "%{x}<br>Wind Speed = %{y:.1f} (m/s)<extra></extra>",
        }],
        "layout": {
            "margin": {"l": 55, "r": 15, "t": 90, "b": 50},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(30,30,30,0.6)",
            "font": {"color": "#ddd", "size": 11},
            "title": {
                "text": f"{name_sid}",
                "font": {"size": 13},
                "x": 0.5,
                "xanchor": "center",
                "y": 0.98,
                "yanchor": "top",
            },
            "xaxis": {"showgrid": True, "gridcolor": "#444"},
            "yaxis": {
                "title": "Wind Speed (m/s)",
                "showgrid": True,
                "gridcolor": "#444",
            },
            "height": 480,
        },
    }
    return json.dumps(fig_dict)


# ---------------------------------------------------------------------------
# JavaScript / HTML injected into the map
# ---------------------------------------------------------------------------

_PLOTLY_CDN = (
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/'
    'plotly.min.js"></script>'
)

_NOUISLIDER_CSS = (
    '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/'
    'noUiSlider/15.7.1/nouislider.min.css"/>'
)

_NOUISLIDER_JS = (
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/noUiSlider/15.7.1/'
    'nouislider.min.js"></script>'
)

_JS_HIGHLIGHT = """
<script>
(function() {
  // ---- Global track registry for time filtering ----
  // Each entry: { layer, tStart, tEnd, parentGroup, defaultVisible }
  window._insituRegistry = [];

  window._insituRegister = function(layer, tStart, tEnd, parentGroup) {
    window._insituRegistry.push({
      layer:      layer,
      tStart:     tStart,
      tEnd:       tEnd,
      parentGroup: parentGroup
    });
  };

  // ---- Time filter (called by slider) ----
  // Completely removes layers outside the time window from the map
  // (remove() kills tooltips and click events, unlike opacity:0).
  // Re-adds them when they come back in range, but only if their
  // parent FeatureGroup is currently active on the map.
  window._insituApplyTimeFilter = function(filterStart, filterEnd) {
    window._insituRegistry.forEach(function(entry) {
      var overlaps = entry.tStart <= filterEnd && entry.tEnd >= filterStart;

      // Parent FeatureGroup must be on the map (LayerControl check)
      var parentMap = entry.parentGroup && entry.parentGroup._map;
      var groupActive = !!parentMap;

      if (overlaps && groupActive) {
        // Add back to the FeatureGroup if not already present
        if (!entry.layer._map) {
          entry.parentGroup.addLayer(entry.layer);
        }
      } else {
        // Fully remove from map — no tooltip, no click, no hover
        if (entry.layer._map) {
          entry.parentGroup.removeLayer(entry.layer);
        }
      }
    });
  };

  document.addEventListener('DOMContentLoaded', function() {
    var _active    = null;
    var _origStyle = null;
    var popupEl    = document.getElementById('insitu-ts-popup');

    // ---- Highlight on click ----
    window._insituSelect = function(layer, figJson, trackName) {
      if (_active && _active !== layer) {
        _active.setStyle(_origStyle);
      }
      if (_active === layer) {
        layer.setStyle(_origStyle);
        _active = null;
        _origStyle = null;
        if (popupEl) popupEl.style.display = 'none';
        return;
      }
      _origStyle = {
        color:   layer.options.color,
        weight:  layer.options.weight,
        opacity: layer.options.opacity
      };
      layer.setStyle({weight: 6, opacity: 1.0});
      layer.bringToFront();
      _active = layer;

      if (!popupEl || !figJson) {
        if (popupEl) popupEl.style.display = 'none';
        return;
      }
      popupEl.style.display = 'block';
      var fig = JSON.parse(figJson);
      if (window.Plotly) {
        Plotly.react(
          'insitu-ts-chart', fig.data, fig.layout,
          {responsive: true, displayModeBar: false}
        );
      }
    };

    // Click on map background -> deselect
    var container = document.querySelector('.leaflet-container');
    var isDragging = false;
    
    if (container) {
      // Detect drag start/end to prevent popup closing during drag
      container.addEventListener('mousedown', function() {
        isDragging = false;
      });
      
      container.addEventListener('mousemove', function() {
        isDragging = true;
      });
      
      container.addEventListener('mouseup', function() {
        setTimeout(function() { isDragging = false; }, 10);
      });
      
      container.addEventListener('click', function(e) {
        if (e.target.classList.contains('leaflet-container') && !isDragging) {
          if (_active) {
            _active.setStyle(_origStyle);
            _active = null;
            _origStyle = null;
          }
          if (popupEl) popupEl.style.display = 'none';
        }
      });
    }

    // Re-apply time filter whenever a LayerControl group is toggled on.
    // Without this, toggling a group back on would show all its layers
    // regardless of the current slider window.
    if (window._map) {
      window._map.on('overlayadd', function() {
        var slider = document.getElementById('insitu-time-slider');
        if (slider && slider.noUiSlider) {
          var vals = slider.noUiSlider.get();
          window._insituApplyTimeFilter(parseFloat(vals[0]), parseFloat(vals[1]));
        }
      });
    }

    // ---- noUiSlider initialisation ----
    var sliderEl = document.getElementById('insitu-time-slider');
    if (!sliderEl || typeof noUiSlider === 'undefined') return;

    var cfg      = JSON.parse(document.getElementById('insitu-slider-config').textContent);
    var tMin     = cfg.tMin;
    var tMax     = cfg.tMax;
    var labelMin = document.getElementById('insitu-slider-label-min');
    var labelMax = document.getElementById('insitu-slider-label-max');

    noUiSlider.create(sliderEl, {
      start:   [tMin, tMax],
      connect: true,
      range:   {min: tMin, max: tMax},
      step:    86400000,       // 1 day in ms
      tooltips: false,
      behaviour: 'drag'
    });

    // Style the connect bar
    sliderEl.querySelector('.noUi-connect').style.background = '#4a9eff';

    function msToDateStr(ms) {
      return new Date(ms).toISOString().slice(0, 10);
    }

    var _debounceTimer = null;
    sliderEl.noUiSlider.on('update', function(values) {
      var v0 = parseFloat(values[0]);
      var v1 = parseFloat(values[1]);
      if (labelMin) labelMin.textContent = msToDateStr(v0);
      if (labelMax) labelMax.textContent = msToDateStr(v1);
      clearTimeout(_debounceTimer);
      _debounceTimer = setTimeout(function() {
        window._insituApplyTimeFilter(v0, v1);
      }, 100);
    });

    // Initial filter pass: wait for all Leaflet layers to be added to the map
    // (they are created asynchronously via MacroElement scripts)
    setTimeout(function() {
      var vals = sliderEl.noUiSlider.get();
      window._insituApplyTimeFilter(parseFloat(vals[0]), parseFloat(vals[1]));
    }, 800);
  });
})();
</script>
"""

# ---- Popup panel ----
# Adjust 'width' to change the panel width (default: 800px).
# Adjust 'height' in the div AND 'height' in build_plotly_tabs_json to resize the chart.
_TS_POPUP_HTML = """
<div id="insitu-ts-popup" style="
    display: none;
    position: fixed;
    bottom: 30px; left: 50%;
    transform: translateX(-50%);
    z-index: 2000;
    background: rgba(25,25,25,0.93);
    border: 1px solid #555;
    border-radius: 8px;
    padding: 10px 14px 6px;
    width: 1000px;
    max-width: 92vw;
    font-family: sans-serif;
    backdrop-filter: blur(6px);
">
  <div style="display:flex;justify-content:flex-end;margin-bottom:2px">
    <span onclick="document.getElementById('insitu-ts-popup').style.display='none'"
          style="cursor:pointer;color:#aaa;font-size:15px;line-height:1">&#x2715;</span>
  </div>
  <div id="insitu-ts-chart" style="width:100%;height:500px"></div>
</div>
"""

# ---- Time slider panel ----
# The slider bounds and initial values are injected by build_insitu_map()
# via a <script id="insitu-slider-config"> JSON block.
_TIME_SLIDER_HTML = """
<div id="insitu-slider-panel" style="
    position: fixed;
    bottom: 22px; left: 50%;
    transform: translateX(-50%);
    z-index: 1500;
    background: rgba(25,25,25,0.90);
    border: 1px solid #555;
    border-radius: 8px;
    padding: 10px 20px 12px;
    width: 1024px;
    max-width: 1024px;
    font-family: sans-serif;
    backdrop-filter: blur(6px);
">
  <div style="display:flex;justify-content:space-between;
              font-size:11px;color:#aaa;margin-bottom:6px;">
    <span>Time filter</span>
    <span style="font-size:10px;color:#666">drag handles to filter tracks</span>
  </div>
  <div id="insitu-time-slider" style="margin:0 6px 8px;"></div>
  <div style="display:flex;justify-content:space-between;font-size:11px;color:#ccc;">
    <span id="insitu-slider-label-min"></span>
    <span id="insitu-slider-label-max"></span>
  </div>
</div>
"""

_GRID_HTML = """
<script>
document.addEventListener('DOMContentLoaded', function() {
  if (!window._map || typeof L === 'undefined') return;

  var style = {color: '#ffffff', weight: 1.2, opacity: 0.52, interactive: false};
  var graticule = L.featureGroup();

  for (var lat = -80; lat <= 80; lat += 10) {
    graticule.addLayer(L.polyline([[lat, -180], [lat, 180]], style));
  }
  for (var lon = -180; lon <= 180; lon += 10) {
    graticule.addLayer(L.polyline([[90, lon], [-90, lon]], style));
  }

  graticule.addTo(window._map);
});
</script>
"""


# ---------------------------------------------------------------------------
# Polyline / marker factories
# ---------------------------------------------------------------------------

def _make_polyline_with_click(
    coords: list,
    color: str,
    tooltip_html: str,
    fig_json: str | None,
    track_name: str,
    t_start_ms: int,
    t_end_ms: int,
    weight: float = 2.5,
    opacity: float = 0.85,
) -> "folium.MacroElement":
    """Create a polyline with highlight-on-click, Plotly popup, and time registration."""
    from folium.elements import MacroElement
    from jinja2 import Template

    safe_name = track_name.replace("'", "\\'").replace('"', '\\"')

    if fig_json:
        safe_json = fig_json.replace("\\", "\\\\").replace("`", "\\`")
        click_call = f"window._insituSelect(this, `{safe_json}`, '{safe_name}');"
    else:
        click_call = f"window._insituSelect(this, null, '{safe_name}');"

    class ClickPolyline(MacroElement):
        _template = Template("""
            {% macro script(this, kwargs) %}
            (function() {
              var line = L.polyline({{ this.coords }}, {
                color:   {{ this.color }},
                weight:  {{ this.weight }},
                opacity: {{ this.opacity }}
              });
              line.bindTooltip({{ this.tooltip }}, {sticky: true});
              line.on('click', function(e) {
                L.DomEvent.stopPropagation(e);
                {{ this.click_call }}
              });
              var parentFG = {{ this._parent.get_name() }};
              line.addTo(parentFG);
              if (window._insituRegister) {
                window._insituRegister(line, {{ this.t_start }}, {{ this.t_end }}, parentFG);
              }
            })();
            {% endmacro %}
        """)

        def __init__(self, coords, color, tooltip, click_call, weight, t_start, t_end, opacity):
            super().__init__()
            self._name      = "ClickPolyline"
            self.coords     = json.dumps(coords)
            self.color      = json.dumps(color)
            self.tooltip    = json.dumps(tooltip)
            self.click_call = click_call
            self.weight     = weight
            self.opacity    = opacity
            self.t_start    = t_start   # Unix timestamp ms
            self.t_end      = t_end     # Unix timestamp ms

    return ClickPolyline(coords, color, tooltip_html, click_call, weight, t_start_ms, t_end_ms, opacity)


def _make_start_marker(
    coord: list,
    color: str,
    label: str,
    t_start_ms: int,
    t_end_ms: int,
    radius: int = 4,
) -> "folium.MacroElement":
    """Circle marker for trajectory start points (drifters, buoys), registered for time filtering."""
    from folium.elements import MacroElement
    from jinja2 import Template

    class RegisteredMarker(MacroElement):
        _template = Template("""
            {% macro script(this, kwargs) %}
            (function() {
              var marker = L.circleMarker({{ this.coord }}, {
                radius:      {{ this.radius }},
                color:       {{ this.color }},
                fillColor:   {{ this.color }},
                fillOpacity: 1.0,
                opacity:     0.85,
                weight:      2
              });
              marker.bindTooltip({{ this.tooltip }});
              var parentFG = {{ this._parent.get_name() }};
              marker.addTo(parentFG);
              if (window._insituRegister) {
                window._insituRegister(marker, {{ this.t_start }}, {{ this.t_end }}, parentFG);
              }
            })();
            {% endmacro %}
        """)

        def __init__(self, coord, color, tooltip, t_start, t_end, radius):
            super().__init__()
            self._name   = "RegisteredMarker"
            self.coord   = json.dumps(coord)
            self.color   = json.dumps(color)
            self.tooltip = json.dumps(tooltip)
            self.t_start = t_start
            self.t_end   = t_end
            self.radius  = radius

    return RegisteredMarker(coord, color, label, t_start_ms, t_end_ms, radius)


def _make_triangle_marker(
    coord: list,
    color: str,
    label: str,
    t_start_ms: int,
    t_end_ms: int,
    size: int = 10,
) -> "folium.MacroElement":
    """Triangle marker for saildrone trajectory start points, registered for time filtering."""
    from folium.elements import MacroElement
    from jinja2 import Template

    class TriangleMarker(MacroElement):
        _template = Template("""
            {% macro script(this, kwargs) %}
            (function() {
              var coords = {{ this.coord }};
              var size = {{ this.size }};
              
              // Create SVG triangle pointing up
              var svgString = '<svg height="' + size + '" width="' + size + '" viewBox="0 0 10 10" xmlns="http://www.w3.org/2000/svg">' +
                '<polygon points="5,0 10,10 0,10" fill="{{ this.color_raw }}" stroke="{{ this.color_raw }}" stroke-width="0.5" opacity="0.85" />' +
                '</svg>';
              
              var marker = L.marker(coords, {
                icon: L.divIcon({
                  html: svgString,
                  iconSize: [size, size],
                  className: 'sd-triangle-marker'
                })
              });
              marker.bindTooltip({{ this.tooltip }});
              var parentFG = {{ this._parent.get_name() }};
              marker.addTo(parentFG);
              if (window._insituRegister) {
                window._insituRegister(marker, {{ this.t_start }}, {{ this.t_end }}, parentFG);
              }
            })();
            {% endmacro %}
        """)

        def __init__(self, coord, color, tooltip, t_start, t_end, size):
            super().__init__()
            self._name   = "TriangleMarker"
            self.coord   = json.dumps(coord)
            self.color_raw = color
            self.tooltip = json.dumps(tooltip)
            self.t_start = t_start
            self.t_end   = t_end
            self.size    = size

    return TriangleMarker(coord, color, label, t_start_ms, t_end_ms, size)

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
def _get_coords(lat, lon):
    """
    Convert lat/lon coordinates to list format, splitting at meridian crossings.
    
    Detects and splits trajectories at large jumps in longitude (> 180°),
    preventing visual artifacts when trajectories cross the -180/+180 meridian.
    
    Args:
        lat: latitude values
        lon: longitude values
    
    Returns:
        List of coordinate segments. Each segment is a list of [lat, lon] pairs.
        If no meridian crossing detected, returns a single segment.
    """
    coords = [[float(la), float(lo)] for la, lo in zip(lat, lon)]
    
    if len(coords) < 2:
        return [coords]
    
    tmp = np.asarray(coords)
    indexes = np.where(abs(np.diff(tmp[:, 1])) > 180.0)[0]
    
    if indexes.size > 0:
        segments = []
        start_idx = 0
        for index in indexes:
            segments.append(coords[start_idx:index + 1])
            start_idx = index + 1
        segments.append(coords[start_idx:])
        return segments
    
    return [coords]


# ---------------------------------------------------------------------------
# Per-platform trajectory rendering
# ---------------------------------------------------------------------------

def _render_saildrone_tracks(
    df: pd.DataFrame,
    color_map: dict[str, str],
    provider_groups: dict[str, folium.FeatureGroup],
    max_points: int,
) -> None:
    """Render saildrone trajectories into provider FeatureGroups."""

    platform_id_map: dict[str, str] = {}
    if "platform_id" in df.columns:
        platform_id_map = df.groupby("name")["platform_id"].first().to_dict()

    meta = (
        df.groupby(["name", "provider"])
        .agg(date_start=("time", "min"), date_end=("time", "max"),
             n_points=("time", "count"),
             platform_type=("platform_type", "first"))
        .reset_index()
    )
    meta["date_start"] = meta["date_start"].dt.strftime("%Y-%m-%d")
    meta["date_end"]   = meta["date_end"].dt.strftime("%Y-%m-%d")

    for _, row in meta.iterrows():
        name     = row["name"]
        provider = row["provider"]
        color    = color_map[name]
        platform_id = platform_id_map.get(name, "")

        sub = df.loc[
            (df["name"] == name) & (df["provider"] == provider)
        ].sort_values("time")

        lats_d, lons_d = decimate_trajectory(
            sub["latitude"].values.astype(float),
            sub["longitude"].values.astype(float),
            max_points,
        )

        available = get_available_variables(sub)
        vars_str  = format_variables_for_tooltip(available)

        platform_id_line = f"Platform ID: {platform_id}<br>" if platform_id else ""
        provider_line = f"Provider: {row['provider']}<br>"
        platform_type_line = f"Type: {row['platform_type']}<br>"
        vars_line  = f"Variables: {vars_str}<br>" if vars_str else ""
        tooltip_html = (
            f"<b>{name}</b><br>"
            f"{platform_id_line}"
            f"{provider_line}"
            f"{platform_type_line}"
            f"From {row['date_start']} to {row['date_end']}<br>"
            f"{vars_line}"
            f"({row['n_points']:,} points)"
        )
        start_label = f"Start · {name}<br>{row['date_start']}"
        fig_json    = build_plotly_tabs_json(sub, name, color, available, max_points)

        t_start_ms = int(sub["time"].min().timestamp() * 1000)
        t_end_ms   = int(sub["time"].max().timestamp() * 1000)

        Coords = _get_coords(lats_d, lons_d)

        for coords in Coords:
            if len(coords) < 2:
                continue

            provider_groups[provider].add_child(_make_polyline_with_click(
                coords, color, tooltip_html, fig_json, name,
                t_start_ms, t_end_ms, weight=2.5
            ))
            provider_groups[provider].add_child(
                _make_triangle_marker(coords[0], color, start_label, t_start_ms, t_end_ms, size=12)
            )


def _render_ldl_tracks(
    df: pd.DataFrame,
    color_map: dict[str, str],
    ldl_group: folium.FeatureGroup,
    max_points: int,
) -> None:
    """Render LDL DWSD drifter trajectories into the single LDL FeatureGroup."""

    meta = (
        df.groupby(["name", "tc_name", "platform_id"])
        .agg(date_start=("time", "min"), date_end=("time", "max"),
             n_points=("time", "count"),
             provider=("provider", "first"),
             platform_type=("platform_type", "first"))
        .reset_index()
    )
    meta["date_start"] = meta["date_start"].dt.strftime("%Y-%m-%d")
    meta["date_end"]   = meta["date_end"].dt.strftime("%Y-%m-%d")

    for _, row in meta.iterrows():
        name     = row["name"]
        tc_name  = row["tc_name"]
        platform_id = row["platform_id"]
        color    = color_map[name]

        sub = df.loc[df["name"] == name].sort_values("time")

        lats_d, lons_d = decimate_trajectory(
            sub["latitude"].values.astype(float),
            sub["longitude"].values.astype(float),
            max_points,
        )

        available = get_available_variables(sub)
        vars_str  = format_variables_for_tooltip(available)

        provider_line = f"Provider: {row['provider']}<br>"
        platform_type_line = f"Type: {row['platform_type']}<br>"
        vars_line = f"Variables: {vars_str}<br>" if vars_str else ""
        tooltip_html = (
            f"<b>{name}</b><br>"
            f"Platform ID: {platform_id}<br>"
            f"{provider_line}"
            f"{platform_type_line}"
            f"TC: {tc_name}<br>"
            f"From {row['date_start']} to {row['date_end']}<br>"
            f"{vars_line}"
            f"({row['n_points']:,} points)"
        )
        start_label = f"Start · {name}<br>{row['date_start']}"
        fig_json    = build_plotly_tabs_json(sub, name, color, available, max_points)

        t_start_ms = int(sub["time"].min().timestamp() * 1000)
        t_end_ms   = int(sub["time"].max().timestamp() * 1000)

        Coords = _get_coords(lats_d, lons_d)
        for coords in Coords:
            if len(coords) < 2:
                continue
            # LDL drifters now use same line width as saildrones
            ldl_group.add_child(_make_polyline_with_click(
                coords, color, tooltip_html, fig_json, name,
                t_start_ms, t_end_ms, weight=2.5
            ))
            ldl_group.add_child(_make_start_marker(coords[0], color, start_label, t_start_ms, t_end_ms, radius=4))


def _render_kub_tracks(
    df: pd.DataFrame,
    color_map: dict[str, str],
    kub_group: folium.FeatureGroup,
    max_points: int,
) -> None:
    """Render KU buoy trajectories into the KUB FeatureGroup."""

    meta = (
        df.groupby(["name", "platform_id"])
        .agg(
            date_start=("time", "min"),
            date_end=("time", "max"),
            n_points=("time", "count"),
            tc=("tc_name", lambda x: x.unique().tolist()),
            provider=("provider", "first"),
            platform_type=("platform_type", "first")
        ).reset_index()
    )
    meta["date_start"] = meta["date_start"].dt.strftime("%Y-%m-%d")
    meta["date_end"]   = meta["date_end"].dt.strftime("%Y-%m-%d")

    for _, row in meta.iterrows():
        name     = row["name"]
        platform_id = row["platform_id"]
        tcs      = row["tc"]
        color    = color_map[name]

        sub = df.loc[df["name"] == name].sort_values("time")

        lats_d, lons_d = decimate_trajectory(
            sub["latitude"].values.astype(float),
            sub["longitude"].values.astype(float),
            max_points,
        )

        available = get_available_variables(sub)
        vars_str  = format_variables_for_tooltip(available)

        vars_line = f"Variables: {vars_str}<br>" if vars_str else ""
        provider_line = f"Provider: {row['provider']}<br>"
        platform_type_line = f"Type: {row['platform_type']}<br>"
        tooltip_html = (
            f"<b>{name}</b><br>"
            f"Platform ID: {platform_id}<br>"
            f"{provider_line}"
            f"{platform_type_line}"
            f"Potential TCs: {', '.join(tcs)}<br>"
            f"From {row['date_start']} to {row['date_end']}<br>"
            f"{vars_line}"
            f"({row['n_points']:,} points)"
        )
        start_label = f"Start · {name}<br>{row['date_start']}"
        fig_json    = build_plotly_tabs_json(sub, name, color, available, max_points)

        t_start_ms = int(sub["time"].min().timestamp() * 1000)
        t_end_ms   = int(sub["time"].max().timestamp() * 1000)

        Coords = _get_coords(lats_d, lons_d)
        for coords in Coords:
            if len(coords) < 2:
                continue
            kub_group.add_child(_make_polyline_with_click(
                coords, color, tooltip_html, fig_json, name,
                t_start_ms, t_end_ms, weight=2.5
            ))
            kub_group.add_child(_make_start_marker(coords[0], color, start_label, t_start_ms, t_end_ms, radius=4))


def _render_tc_tracks(
    df: pd.DataFrame,
    tc_groups: dict[int, folium.FeatureGroup],
    color_dict: dict | None,
    max_points: int,
    ws_vmin: float = 0.0,
    ws_vmax: float = 136.0,
) -> None:
    """
    Render TC trajectories into the TC FeatureGroup.

    Each storm gets:
      - An invisible full-trajectory polyline carrying the storm-level tooltip
        (name, sid, date range). This is what the user hovers over.
      - N-1 short 2-point colored segments encoding wind_speed via color_dict.
        These segments have no tooltip to avoid visual clutter.
      - Without color_dict: a single grey polyline per storm (no segments).

    All elements are registered in _insituRegistry for the time slider.
    """
    from folium.elements import MacroElement
    from jinja2 import Template

    # ---- Invisible full-track polyline (tooltip only) ----
    class TCTooltipLine(MacroElement):
        _template = Template("""
            {% macro script(this, kwargs) %}
            (function() {
              var line = L.polyline({{ this.coords }}, {
                color:   "rgba(0,0,0,0)",
                weight:  8,
                opacity: 0
              });
              line.bindTooltip({{ this.tooltip }}, {sticky: true});
              var parentFG = {{ this._parent.get_name() }};
              line.addTo(parentFG);
              if (window._insituRegister) {
                window._insituRegister(line, {{ this.t_start }}, {{ this.t_end }}, parentFG);
              }
            })();
            {% endmacro %}
        """)
        def __init__(self, coords, tooltip, t_start, t_end):
            super().__init__()
            self._name   = "TCTooltipLine"
            self.coords  = json.dumps(coords)
            self.tooltip = json.dumps(tooltip)
            self.t_start = t_start
            self.t_end   = t_end

    # ---- Short colored segment (no tooltip) ----
    class TCSegment(MacroElement):
        _template = Template("""
            {% macro script(this, kwargs) %}
            (function() {
              var line = L.polyline({{ this.coords }}, {
                color:       {{ this.color }},
                weight:      {{ this.weight }},
                opacity:     0.85,
                interactive: false
              });
              var parentFG = {{ this._parent.get_name() }};
              line.addTo(parentFG);
              if (window._insituRegister) {
                window._insituRegister(line, {{ this.t_start }}, {{ this.t_end }}, parentFG);
              }
            })();
            {% endmacro %}
        """)
        def __init__(self, coords, color, t_start, t_end, weight=2.5):
            super().__init__()
            self._name   = "TCSegment"
            self.coords  = json.dumps(coords)
            self.color   = json.dumps(color)
            self.t_start = t_start
            self.t_end   = t_end
            self.weight  = weight

    # ---- Grey shade for fallback (no colormap) ----
    def _grey_for_rank(rank: int, total: int) -> str:
        t   = rank / max(total - 1, 1)
        val = int(153 + t * 68)
        return "#{v:02x}{v:02x}{v:02x}".format(v=val)

    meta = (
        df.groupby("sid")
        .agg(
            name_sid  =("name_sid", "first"),
            date_start=("time",     "min"),
            date_end  =("time",     "max"),
        )
        .reset_index()
        .sort_values("date_start")
        .reset_index(drop=True)
    )
    meta["date_start"] = meta["date_start"].dt.strftime("%Y-%m-%d")
    meta["date_end"]   = meta["date_end"].dt.strftime("%Y-%m-%d")
    n_storms = len(meta)

    for rank, row in meta.iterrows():
        sid      = row["sid"]
        name_sid = row["name_sid"]
        sub      = df[df["sid"] == sid].sort_values("time").reset_index(drop=True)

        lats    = sub["latitude"].values.astype(float)
        lons    = sub["longitude"].values.astype(float)
        ws_vals = (
            sub["wind_speed"].values.astype(float)
            if "wind_speed" in sub.columns
            else np.full(len(sub), np.nan)
        )
        ws_knots = ws_vals * 1.94384  # m/s to knots
        times = sub["time"].values

        n = len(lats)
        if n < 2:
            continue

        if n > max_points:
            idx     = np.unique(np.concatenate([
                [0],
                np.linspace(0, n - 1, max_points, dtype=int),
                [n - 1],
            ]))
            lats    = lats[idx]
            lons    = lons[idx]
            ws_vals = ws_vals[idx]
            times   = times[idx]

        t_start_ms = int(pd.Timestamp(times[0]).timestamp()  * 1000)
        t_end_ms   = int(pd.Timestamp(times[-1]).timestamp() * 1000)

        # Determine which group this storm belongs to (max category)
        max_cat = int(sub["max_category"].max()) if "max_category" in sub.columns else 0
        max_cat = max(0, min(5, max_cat))   # clamp to valid range
        fg = tc_groups[max_cat]

        source = sub["source"].iloc[0] if "source" in sub.columns else "Unknown"

        # Storm-level tooltip on the invisible full-track line
        tooltip_full = (
            f"<b>{name_sid}</b><br>"
            f"ID: {sid}<br>"
            f"Category: {max_cat if max_cat > 0 else 'TS/TD'}<br>"
            f"From {row['date_start']} to {row['date_end']}"
            f"{'<br>Source: ' + source if source else ''}"
        )

        Coords = _get_coords(lats, lons)
        
        # Build wind speed visualization
        fig_json = build_tc_windspeed_json(times, ws_vals, name_sid, "#4a9eff", max_points)
        
        for coords in Coords:
            
            # Use clickable polyline instead of invisible tooltip line
            fg.add_child(_make_polyline_with_click(
                coords,
                color="#4a9eff",
                tooltip_html=tooltip_full,
                fig_json=fig_json,
                track_name=name_sid,
                t_start_ms=t_start_ms,
                t_end_ms=t_end_ms,
                weight=2.5,
                opacity=0.0
            ))

            if color_dict is not None:
                # Colored 2-point segments by wind speed (no tooltip)
                for i in range(len(coords) - 1):
                    ws_mean = float(np.nanmean([ws_knots[i], ws_knots[i + 1]]))
                    seg_color = (
                        colordict_to_hex(color_dict, ws_mean, ws_vmin, ws_vmax)
                        if not np.isnan(ws_mean)
                        else "#888888"
                    )
                    t0 = int(pd.Timestamp(times[i]).timestamp()     * 1000)
                    t1 = int(pd.Timestamp(times[i + 1]).timestamp() * 1000)
                    fg.add_child(TCSegment(
                        [coords[i], coords[i + 1]],
                        seg_color, t0, t1, weight=2.5,
                    ))
            else:
                # Single grey polyline per storm
                grey = _grey_for_rank(rank, n_storms)
                fg.add_child(TCSegment(
                    coords, grey, t_start_ms, t_end_ms, weight=2.5
                ))


# ---------------------------------------------------------------------------
# Main map builder
# ---------------------------------------------------------------------------

def build_insitu_map(
    saildrone_df: pd.DataFrame | None = None,
    ldl_df: pd.DataFrame | None = None,
    tc_df: pd.DataFrame | None = None,
    kub_df: pd.DataFrame | None = None,
    cpt_path: str | None = None,
    ws_vmin: float = 0.0,
    ws_vmax: float = 136.0,
    max_points_per_track: int = 800,
    tile: str = "CartoDB dark_matter",
    show_grid: bool = True,
    output_path: str | None = None,
) -> folium.Map:
    """
    Build an interactive Folium map combining in-situ ocean observations.

    Parameters
    ----------
    saildrone_df : DataFrame from saildrone_loader.load_saildrone_catalog()
                   Required columns: time, latitude, longitude, name, provider
    ldl_df       : DataFrame from ldl_loader.load_ldl_catalog()
                   Required columns: time, latitude, longitude, name,
                                     platform_id, tc_name
    tc_df        : DataFrame from tc_loader.load_tc_tracks()
                   Required columns: time, latitude, longitude, sid,
                                     name, name_sid, wind_speed
    kub_df       : DataFrame from kub_loader.load_kub_catalog()
                   Required columns: time, latitude, longitude, name, platform_id
    cpt_path     : path to a GMT .cpt colormap file for TC wind speed coloring
    ws_vmin      : wind speed lower bound for colormap normalisation (default: 0 m/s)
    ws_vmax      : wind speed upper bound for colormap normalisation (default: 136 m/s)
    max_points_per_track : max rendered points per trajectory (decimation)
    tile         : Folium tile layer name or TMS URL
    show_grid    : if True, adds a subtle latitude/longitude grid overlay
    output_path  : if provided, saves the map as HTML at this path

    Returns
    -------
    folium.Map
    """
    if saildrone_df is None and ldl_df is None and tc_df is None and kub_df is None:
        raise ValueError("At least one of saildrone_df, ldl_df, tc_df, or kub_df must be provided.")

    # --- Clean inputs ---
    def _clean(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None or df.empty:
            return None
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])
        before = len(df)
        df = df.dropna(subset=["latitude", "longitude"])
        dropped = before - len(df)
        if dropped > 0:
            print(f"Info: dropped {dropped:,} rows with NaN coordinates.")
        return df if not df.empty else None

    saildrone_df = _clean(saildrone_df)
    ldl_df       = _clean(ldl_df)
    kub_df       = _clean(kub_df)

    # --- Unified color map across all platforms ---
    color_map = build_color_map(
        *[d for d in (saildrone_df, ldl_df, kub_df) if d is not None]
    )

    # --- Map center: median of all available coordinates ---
    all_lats, all_lons = [], []
    for df in (saildrone_df, ldl_df, kub_df):
        if df is not None:
            all_lats.extend(df["latitude"].dropna().tolist())
            all_lons.extend(df["longitude"].dropna().tolist())

    center_lat = float(np.median(all_lats))
    center_lon = float(np.median(all_lons))

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=3,
        tiles=tile,
        control_scale=True,
        world_copy_jump=True,
    )

    # Set window._map for custom scripts
    m.get_root().html.add_child(folium.Element(f'<script>window._map = {m.get_name()};</script>'))

    m.get_root().header.add_child(folium.Element(_PLOTLY_CDN))
    m.get_root().header.add_child(folium.Element(_NOUISLIDER_CSS))
    m.get_root().header.add_child(folium.Element(_NOUISLIDER_JS))
    m.get_root().html.add_child(folium.Element(_TS_POPUP_HTML))
    m.get_root().html.add_child(folium.Element(_TIME_SLIDER_HTML))
    m.get_root().html.add_child(folium.Element(_JS_HIGHLIGHT))

    if show_grid:
        m.get_root().html.add_child(folium.Element(_GRID_HTML))

    # Compute global time bounds across all platforms for slider initialisation
    all_times: list[pd.Timestamp] = []
    for df in (saildrone_df, ldl_df, tc_df, kub_df):
        if df is not None:
            all_times.extend([df["time"].min(), df["time"].max()])
    t_min_ms = int(min(all_times).timestamp() * 1000)
    t_max_ms = int(max(all_times).timestamp() * 1000)

    slider_config = json.dumps({"tMin": t_min_ms, "tMax": t_max_ms})
    m.get_root().html.add_child(folium.Element(
        f'<script id="insitu-slider-config" type="application/json">'
        f'{slider_config}</script>'
    ))

    # --- Saildrone layers ---
    provider_groups: dict[str, folium.FeatureGroup] = {}

    if saildrone_df is not None:
        for provider in sorted(saildrone_df["provider"].unique()):
            provider_groups[provider] = folium.FeatureGroup(
                name=f"&#x26F5; SD ({provider})", show=True, overlay=True
            )
        _render_saildrone_tracks(
            saildrone_df, color_map,
            provider_groups,
            max_points_per_track,
        )

    # --- LDL DWSD layers ---
    ldl_group: folium.FeatureGroup | None = None

    if ldl_df is not None:
        ldl_group = folium.FeatureGroup(
            name="&#x1F6DF; DWSD (LDL)", show=True, overlay=True
        )
        _render_ldl_tracks(
            ldl_df, color_map,
            ldl_group,
            max_points_per_track,
        )

    # --- KU Buoy layers ---
    kub_group: folium.FeatureGroup | None = None

    if kub_df is not None:
        kub_group = folium.FeatureGroup(
            name="&#x1F6DF; Spotter (KUB)", show=True, overlay=True
        )
        _render_kub_tracks(
            kub_df, color_map,
            kub_group,
            max_points_per_track,
        )

    # --- TC layers ---
    tc_groups: dict[int, folium.FeatureGroup] = {}

    if tc_df is not None:
        color_dict = load_cpt_colormap(cpt_path) if cpt_path else None

        # One FeatureGroup per Saffir-Simpson category
        # cat 0 = TD/TS, cat 1-5 = hurricane categories
        _TC_CAT_LABELS = {
            0: ("&#x1F300; TD/TS",  False),
            1: ("&#x1F300; TC (cat. 1)", False),
            2: ("&#x1F300; TC (cat. 2)", False),
            3: ("&#x1F300; TC (cat. 3)", False),
            4: ("&#x1F300; TC (cat. 4)", False),
            5: ("&#x1F300; TC (cat. 5)", True),   # cat 5 shown by default
        }
        tc_groups: dict[int, folium.FeatureGroup] = {
            cat: folium.FeatureGroup(name=label, show=shown, overlay=True)
            for cat, (label, shown) in _TC_CAT_LABELS.items()
        }
        _render_tc_tracks(
            tc_df, tc_groups, color_dict,
            max_points_per_track, ws_vmin, ws_vmax
        )
        if color_dict is not None:
            colorbar_html = build_colorbar_html(
                color_dict, ws_vmin, ws_vmax,
            )
            m.get_root().html.add_child(folium.Element(colorbar_html))

    # --- Add all groups to map ---
    # TC groups first so they render below in-situ data
    if tc_df is not None:
        for fg in tc_groups.values():
            m.add_child(fg)
    # Provider groups (saildrones) shown by default
    for fg in provider_groups.values():
        m.add_child(fg)
    # Single LDL group shown by default
    if ldl_group is not None:
        m.add_child(ldl_group)
    # Single KU buoy group shown by default
    if kub_group is not None:
        m.add_child(kub_group)

    folium.LayerControl(position="topleft", collapsed=True).add_to(m)

    if output_path:
        m.save(output_path)
        print(f"Map saved: {output_path}")

    #return m


