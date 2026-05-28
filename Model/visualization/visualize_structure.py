import json
import webbrowser
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs
from plotly.utils import PlotlyJSONEncoder


SELECTED_SYMBOL = "RELIANCE"
DEFAULT_CANDLE_LIMIT = 300
CANDLE_COUNT_OPTIONS = [100, 200, 300, 500, 1000]


REQUIRED_COLUMNS = [
    "Datetime",
    "Open",
    "High",
    "Low",
    "Close",
    "symbol",
    "swing_high",
    "swing_low",
    "anchored_support",
    "anchored_resistance",
    "anchored_breakout",
    "anchored_breakdown",
    "support_anchor_1_position",
    "support_anchor_1_price",
    "support_anchor_2_position",
    "support_anchor_2_price",
    "resistance_anchor_1_position",
    "resistance_anchor_1_price",
    "resistance_anchor_2_position",
    "resistance_anchor_2_price",
]


def load_dataset():
    dataset_path = (
        Path(__file__).resolve().parent.parent
        / "features_data"
        / "master_feature_dataset.csv"
    )

    df = pd.read_csv(dataset_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="mixed")

    return df


def validate_columns(df):
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing_columns:
        raise ValueError(
            "Missing required columns for market structure visualization: "
            + ", ".join(missing_columns)
        )


def normalize_symbol(symbol):
    return str(symbol).upper().split(".")[0]


def default_symbol(symbols):
    selected_key = SELECTED_SYMBOL.upper()

    for symbol in symbols:
        if str(symbol).upper() == selected_key or normalize_symbol(symbol) == selected_key:
            return symbol

    return symbols[0]


def marker_offset(plot_df):
    price_range = plot_df["High"].max() - plot_df["Low"].min()

    if pd.isna(price_range) or price_range == 0:
        return 0

    return price_range * 0.012


def build_trendline_ray(plot_df, full_symbol_df, prefix):
    active_rows = plot_df.dropna(
        subset=[
            f"{prefix}_anchor_1_position",
            f"{prefix}_anchor_1_price",
            f"{prefix}_anchor_2_position",
            f"{prefix}_anchor_2_price",
        ]
    )

    if active_rows.empty:
        return None

    last_active_row = active_rows.iloc[-1]
    anchor_1_position = int(last_active_row[f"{prefix}_anchor_1_position"])
    anchor_2_position = int(last_active_row[f"{prefix}_anchor_2_position"])
    anchor_1_price = float(last_active_row[f"{prefix}_anchor_1_price"])
    anchor_2_price = float(last_active_row[f"{prefix}_anchor_2_price"])

    if anchor_1_position == anchor_2_position:
        return None

    slope = (anchor_2_price - anchor_1_price) / (anchor_2_position - anchor_1_position)
    ray_start = plot_df.index.min()
    ray_end = plot_df.index.max()
    ray_positions = list(range(ray_start, ray_end + 1))
    ray_df = full_symbol_df.loc[ray_positions]
    ray_prices = [
        anchor_1_price + slope * (position - anchor_1_position)
        for position in ray_positions
    ]

    return {
        "x": ray_df["Datetime"],
        "y": ray_prices,
        "anchorX": [
            full_symbol_df.loc[anchor_1_position, "Datetime"],
            full_symbol_df.loc[anchor_2_position, "Datetime"],
        ],
        "anchorY": [anchor_1_price, anchor_2_price],
        "slope": slope,
    }


def prepare_symbol_payload(symbol, symbol_df):
    full_symbol_df = (
        symbol_df
        .sort_values("Datetime")
        .reset_index(drop=True)
        .copy()
    )

    payload_rows = []

    for candle_count in CANDLE_COUNT_OPTIONS:
        plot_df = full_symbol_df.tail(candle_count).copy()
        offset = marker_offset(plot_df)

        y_min = plot_df["Low"].min()
        y_max = plot_df["High"].max()
        y_padding = max((y_max - y_min) * 0.08, y_max * 0.001)

        candle_payload = {
            "x": plot_df["Datetime"],
            "open": plot_df["Open"],
            "high": plot_df["High"],
            "low": plot_df["Low"],
            "close": plot_df["Close"],
            "yRange": [y_min - y_padding, y_max + y_padding],
            "xRange": [
                plot_df["Datetime"].iloc[0],
                plot_df["Datetime"].iloc[-1],
            ],
            "supportRay": build_trendline_ray(
                plot_df,
                full_symbol_df,
                "support",
            ),
            "resistanceRay": build_trendline_ray(
                plot_df,
                full_symbol_df,
                "resistance",
            ),
            "swingHighs": {
                "x": plot_df.loc[plot_df["swing_high"] == 1, "Datetime"],
                "y": plot_df.loc[plot_df["swing_high"] == 1, "High"],
            },
            "swingLows": {
                "x": plot_df.loc[plot_df["swing_low"] == 1, "Datetime"],
                "y": plot_df.loc[plot_df["swing_low"] == 1, "Low"],
            },
            "breakouts": {
                "x": plot_df.loc[plot_df["anchored_breakout"] == 1, "Datetime"],
                "y": plot_df.loc[plot_df["anchored_breakout"] == 1, "High"] + offset,
                "close": plot_df.loc[plot_df["anchored_breakout"] == 1, "Close"],
            },
            "breakdowns": {
                "x": plot_df.loc[plot_df["anchored_breakdown"] == 1, "Datetime"],
                "y": plot_df.loc[plot_df["anchored_breakdown"] == 1, "Low"] - offset,
                "close": plot_df.loc[plot_df["anchored_breakdown"] == 1, "Close"],
            },
        }

        payload_rows.append((str(candle_count), candle_payload))

    return dict(payload_rows)


def build_chart_payload(df):
    validate_columns(df)

    symbols = sorted(df["symbol"].dropna().astype(str).unique())

    if not symbols:
        raise ValueError("No symbols found in the feature dataset.")

    payload = {
        "symbols": symbols,
        "defaultSymbol": default_symbol(symbols),
        "defaultCandleCount": str(DEFAULT_CANDLE_LIMIT),
        "candleCounts": [str(option) for option in CANDLE_COUNT_OPTIONS],
        "data": {},
    }

    for symbol in symbols:
        payload["data"][symbol] = prepare_symbol_payload(
            symbol,
            df[df["symbol"].astype(str) == symbol],
        )

    return payload


def build_html(payload):
    payload_json = json.dumps(payload, cls=PlotlyJSONEncoder)
    plotly_js = get_plotlyjs()

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Market Structure Inspector</title>
  <script>{plotly_js}</script>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b0f14;
      --panel: #131722;
      --grid: #2a2e39;
      --text: #d1d4dc;
      --muted: #8a8f98;
      --green: #00c853;
      --red: #ff1744;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, Segoe UI, Arial, sans-serif;
      overflow: hidden;
    }}

    .toolbar {{
      height: 48px;
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 8px 12px;
      background: #0f131a;
      border-bottom: 1px solid #1f2430;
    }}

    .brand {{
      font-size: 13px;
      font-weight: 650;
      color: #f4f4f5;
      margin-right: 6px;
      white-space: nowrap;
    }}

    label {{
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
    }}

    select {{
      height: 30px;
      min-width: 112px;
      border: 1px solid #303747;
      border-radius: 4px;
      background: var(--panel);
      color: var(--text);
      padding: 0 8px;
      font-size: 13px;
    }}

    input[type="checkbox"] {{
      accent-color: #2962ff;
    }}

    #chart {{
      width: 100vw;
      height: calc(100vh - 48px);
    }}

    @media (max-width: 720px) {{
      body {{
        overflow: auto;
      }}

      .toolbar {{
        height: auto;
        flex-wrap: wrap;
      }}

      #chart {{
        height: calc(100vh - 86px);
        min-height: 560px;
      }}
    }}
  </style>
</head>
<body>
  <div class="toolbar">
    <div class="brand">Market Structure</div>
    <label>Symbol <select id="symbolSelect"></select></label>
    <label>Candles <select id="candleSelect"></select></label>
    <label><input id="supportToggle" type="checkbox" checked> Support</label>
    <label><input id="resistanceToggle" type="checkbox" checked> Resistance</label>
  </div>
  <div id="chart"></div>

  <script>
    const payload = {payload_json};

    const symbolSelect = document.getElementById("symbolSelect");
    const candleSelect = document.getElementById("candleSelect");
    const supportToggle = document.getElementById("supportToggle");
    const resistanceToggle = document.getElementById("resistanceToggle");

    function addOptions(select, values, selectedValue) {{
      select.innerHTML = "";
      values.forEach((value) => {{
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        option.selected = value === selectedValue;
        select.appendChild(option);
      }});
    }}

    addOptions(symbolSelect, payload.symbols, payload.defaultSymbol);
    addOptions(candleSelect, payload.candleCounts, payload.defaultCandleCount);

    function rayTrace(ray, name, color, visible) {{
      return {{
        type: "scatter",
        mode: "lines",
        x: ray.x,
        y: ray.y,
        name,
        visible,
        hovertemplate: `${{name}}<br>%{{x}}<br>%{{y:.2f}}<extra></extra>`,
        line: {{
          color,
          width: 2.6,
          dash: "solid",
        }},
      }};
    }}

    function buildTraces(chartData) {{
      const showSupport = supportToggle.checked;
      const showResistance = resistanceToggle.checked;
      const traces = [
        {{
          type: "candlestick",
          x: chartData.x,
          open: chartData.open,
          high: chartData.high,
          low: chartData.low,
          close: chartData.close,
          name: "Candles",
          increasing: {{
            line: {{ color: "#26a69a", width: 1.45 }},
            fillcolor: "#26a69a",
          }},
          decreasing: {{
            line: {{ color: "#ef5350", width: 1.45 }},
            fillcolor: "#ef5350",
          }},
        }},
      ];

      traces.push(
        {{
          type: "scatter",
          mode: "markers",
          x: chartData.swingLows.x,
          y: chartData.swingLows.y,
          name: "swing_low",
          marker: {{
            symbol: "triangle-up",
            color: "#00c853",
            size: 9,
            line: {{ color: "#071f12", width: 1 }},
          }},
          hovertemplate: "swing_low<br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
        }},
        {{
          type: "scatter",
          mode: "markers",
          x: chartData.swingHighs.x,
          y: chartData.swingHighs.y,
          name: "swing_high",
          marker: {{
            symbol: "triangle-down",
            color: "#ff1744",
            size: 9,
            line: {{ color: "#2b0008", width: 1 }},
          }},
          hovertemplate: "swing_high<br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
        }},
      );

      if (chartData.supportRay) {{
        traces.push(rayTrace(chartData.supportRay, "anchored_support_ray", "#00c853", showSupport));
      }}

      if (chartData.resistanceRay) {{
        traces.push(rayTrace(chartData.resistanceRay, "anchored_resistance_ray", "#ff1744", showResistance));
      }}

      traces.push(
        {{
          type: "scatter",
          mode: "markers",
          x: chartData.breakouts.x,
          y: chartData.breakouts.y,
          text: chartData.breakouts.close,
          name: "anchored_breakout",
          marker: {{
            symbol: "triangle-up",
            color: "#00e676",
            size: 13,
            line: {{ color: "#082b18", width: 1 }},
          }},
          hovertemplate: "anchored_breakout<br>%{{x}}<br>Close: %{{text:.2f}}<extra></extra>",
        }},
        {{
          type: "scatter",
          mode: "markers",
          x: chartData.breakdowns.x,
          y: chartData.breakdowns.y,
          text: chartData.breakdowns.close,
          name: "anchored_breakdown",
          marker: {{
            symbol: "triangle-down",
            color: "#ff1744",
            size: 13,
            line: {{ color: "#2b0008", width: 1 }},
          }},
          hovertemplate: "anchored_breakdown<br>%{{x}}<br>Close: %{{text:.2f}}<extra></extra>",
        }},
      );

      return traces;
    }}

    function buildLayout(symbol, candleCount, chartData) {{
      return {{
        title: {{
          text: `${{symbol}} - Anchored Market Structure (${{candleCount}} candles)`,
          x: 0.012,
          xanchor: "left",
          font: {{ size: 15, color: "#f4f4f5" }},
        }},
        template: "plotly_dark",
        paper_bgcolor: "#131722",
        plot_bgcolor: "#131722",
        font: {{ color: "#d1d4dc", size: 12 }},
        margin: {{ l: 58, r: 28, t: 38, b: 32 }},
        hovermode: "x unified",
        dragmode: "pan",
        showlegend: true,
        legend: {{
          orientation: "h",
          yanchor: "bottom",
          y: 1.01,
          xanchor: "right",
          x: 1,
          bgcolor: "rgba(19, 23, 34, 0.72)",
          font: {{ size: 11 }},
        }},
        xaxis: {{
          type: "date",
          range: chartData.xRange,
          rangeslider: {{ visible: false }},
          showgrid: true,
          gridcolor: "#2a2e39",
          zeroline: false,
          showspikes: true,
          spikemode: "across",
          spikecolor: "#8a8f98",
          spikethickness: 1,
        }},
        yaxis: {{
          range: chartData.yRange,
          fixedrange: false,
          autorange: false,
          side: "right",
          showgrid: true,
          gridcolor: "#2a2e39",
          zeroline: false,
          tickformat: ".2f",
        }},
      }};
    }}

    const config = {{
      responsive: true,
      scrollZoom: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
    }};

    function renderChart() {{
      const symbol = symbolSelect.value;
      const candleCount = candleSelect.value;
      const chartData = payload.data[symbol][candleCount];
      const traces = buildTraces(chartData);
      const layout = buildLayout(symbol, candleCount, chartData);

      Plotly.react("chart", traces, layout, config);
    }}

    symbolSelect.addEventListener("change", renderChart);
    candleSelect.addEventListener("change", renderChart);
    supportToggle.addEventListener("change", renderChart);
    resistanceToggle.addEventListener("change", renderChart);

    renderChart();
  </script>
</body>
</html>"""


def open_market_structure_chart(df):
    payload = build_chart_payload(df)
    html = build_html(payload)
    output_path = Path(__file__).resolve().parent / "market_structure_inspector.html"

    output_path.write_text(html, encoding="utf-8")
    webbrowser.open(output_path.as_uri())

    return output_path


def main():
    df = load_dataset()
    chart_path = open_market_structure_chart(df)
    print(f"Opened market structure chart: {chart_path}")


if __name__ == "__main__":
    main()
