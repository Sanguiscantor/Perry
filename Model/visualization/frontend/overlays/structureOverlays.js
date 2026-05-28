(function () {
  const supportColor = "#00c853";
  const resistanceColor = "#ff1744";

  function hasAnchor(row, side) {
    return (
      row[`${side}Anchor1Position`] !== null &&
      row[`${side}Anchor1Price`] !== null &&
      row[`${side}Anchor2Position`] !== null &&
      row[`${side}Anchor2Price`] !== null
    );
  }

  function latestAnchor(visibleRows, side) {
    for (let index = visibleRows.length - 1; index >= 0; index -= 1) {
      const row = visibleRows[index];

      if (hasAnchor(row, side)) {
        return {
          anchor1Position: row[`${side}Anchor1Position`],
          anchor1Price: row[`${side}Anchor1Price`],
          anchor2Position: row[`${side}Anchor2Position`],
          anchor2Price: row[`${side}Anchor2Price`],
        };
      }
    }

    return null;
  }

  function buildRayData(allRows, visibleRows, side) {
    const anchor = latestAnchor(visibleRows, side);

    if (!anchor || anchor.anchor1Position === anchor.anchor2Position) {
      return [];
    }

    const slope = (
      (anchor.anchor2Price - anchor.anchor1Price) /
      (anchor.anchor2Position - anchor.anchor1Position)
    );
    const startPosition = visibleRows[0].position;
    const endPosition = visibleRows[visibleRows.length - 1].position;
    const data = [];

    for (let position = startPosition; position <= endPosition; position += 1) {
      const row = allRows[position];

      if (!row) {
        continue;
      }

      data.push({
        time: row.time,
        value: anchor.anchor1Price + slope * (position - anchor.anchor1Position),
      });
    }

    return data;
  }

  function createLineSeries(chartApi, color) {
    return chartApi.addSeries(
      LightweightCharts.LineSeries,
      {
        color,
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
        autoscaleInfoProvider: () => null,
      }
    );
  }

  function createOverlayManager(chartApi, candleSeries) {
    const supportSeries = createLineSeries(chartApi, supportColor);
    const resistanceSeries = createLineSeries(chartApi, resistanceColor);
    const markerApi = LightweightCharts.createSeriesMarkers(candleSeries, []);

    function applyMarkers(visibleRows, options) {
      const markers = [];

      if (options.showPivots) {
        visibleRows.forEach((row) => {
          if (row.swingHigh) {
            markers.push({
              time: row.time,
              position: "aboveBar",
              color: resistanceColor,
              shape: "arrowDown",
              text: "SH",
            });
          }

          if (row.swingLow) {
            markers.push({
              time: row.time,
              position: "belowBar",
              color: supportColor,
              shape: "arrowUp",
              text: "SL",
            });
          }
        });
      }

      if (options.showSignals) {
        visibleRows.forEach((row) => {
          if (row.anchoredBreakout) {
            markers.push({
              time: row.time,
              position: "aboveBar",
              color: "#00e676",
              shape: "arrowUp",
              text: "BO",
            });
          }

          if (row.anchoredBreakdown) {
            markers.push({
              time: row.time,
              position: "belowBar",
              color: "#ff5252",
              shape: "arrowDown",
              text: "BD",
            });
          }
        });
      }

      markerApi.setMarkers(markers);
    }

    function update(allRows, visibleRows, options) {
      supportSeries.setData(
        options.showSupport ? buildRayData(allRows, visibleRows, "support") : []
      );
      resistanceSeries.setData(
        options.showResistance ? buildRayData(allRows, visibleRows, "resistance") : []
      );
      applyMarkers(visibleRows, options);
    }

    return {
      update,
    };
  }

  window.PerryOverlays = {
    createOverlayManager,
  };
})();
