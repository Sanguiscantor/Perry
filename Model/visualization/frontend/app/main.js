(function () {
  const payload = window.PERRY_MARKET_DATA;
  const symbolSelect = document.getElementById("symbolSelect");
  const candleRangeSelect = document.getElementById("candleRangeSelect");
  const supportToggle = document.getElementById("supportToggle");
  const resistanceToggle = document.getElementById("resistanceToggle");
  const pivotsToggle = document.getElementById("pivotsToggle");
  const signalsToggle = document.getElementById("signalsToggle");
  const chartContainer = document.getElementById("chartContainer");

  const chartBundle = window.PerryCharts.createPerryChart(chartContainer);
  const overlays = window.PerryOverlays.createOverlayManager(
    chartBundle.chart,
    chartBundle.candleSeries
  );

  function addOptions(select, values, selectedValue) {
    select.innerHTML = "";

    values.forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      option.selected = value === selectedValue;
      select.appendChild(option);
    });
  }

  function candlesFromRows(rows) {
    return rows.map((row) => ({
      time: row.time,
      open: row.open,
      high: row.high,
      low: row.low,
      close: row.close,
    }));
  }

  function selectedRows() {
    const symbol = symbolSelect.value;
    const candleCount = Number(candleRangeSelect.value);
    const allRows = payload.series[symbol] || [];

    return {
      allRows,
      visibleRows: allRows.slice(Math.max(allRows.length - candleCount, 0)),
    };
  }

  function overlayOptions() {
    return {
      showSupport: supportToggle.checked,
      showResistance: resistanceToggle.checked,
      showPivots: pivotsToggle.checked,
      showSignals: signalsToggle.checked,
    };
  }

  function render() {
    const { allRows, visibleRows } = selectedRows();

    chartBundle.setCandles(candlesFromRows(visibleRows));
    overlays.update(allRows, visibleRows, overlayOptions());
  }

  addOptions(symbolSelect, payload.symbols, payload.defaultSymbol);
  addOptions(
    candleRangeSelect,
    payload.candleCountOptions.map(String),
    String(payload.defaultCandleCount)
  );

  symbolSelect.addEventListener("change", render);
  candleRangeSelect.addEventListener("change", render);
  supportToggle.addEventListener("change", render);
  resistanceToggle.addEventListener("change", render);
  pivotsToggle.addEventListener("change", render);
  signalsToggle.addEventListener("change", render);

  render();
})();
