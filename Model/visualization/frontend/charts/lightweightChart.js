(function () {
  const colors = {
    background: "#131722",
    text: "#d1d4dc",
    grid: "#2a2e39",
    up: "#26a69a",
    down: "#ef5350",
  };

  function createPerryChart(container) {
    const chart = LightweightCharts.createChart(container, {
      autoSize: true,
      layout: {
        background: { type: "solid", color: colors.background },
        textColor: colors.text,
      },
      grid: {
        vertLines: { color: colors.grid },
        horzLines: { color: colors.grid },
      },
      rightPriceScale: {
        borderColor: "#3a3f4b",
        scaleMargins: {
          top: 0.08,
          bottom: 0.08,
        },
      },
      timeScale: {
        borderColor: "#3a3f4b",
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 8,
        barSpacing: 7,
        minBarSpacing: 3,
        fixLeftEdge: false,
        fixRightEdge: false,
      },
      crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: false,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
      },
    });

    const candleSeries = chart.addSeries(
      LightweightCharts.CandlestickSeries,
      {
        upColor: colors.up,
        downColor: colors.down,
        borderVisible: false,
        wickUpColor: colors.up,
        wickDownColor: colors.down,
        priceLineVisible: false,
      }
    );

    return {
      chart,
      candleSeries,
      setCandles(candles) {
        candleSeries.setData(candles);
        chart.timeScale().fitContent();
      },
      destroy() {
        chart.remove();
      },
    };
  }

  window.PerryCharts = {
    createPerryChart,
  };
})();
