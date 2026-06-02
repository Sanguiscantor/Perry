# Cross-Asset Information Flow Report

**Generated:** 2026-06-02T16:35:50+00:00

## Mechanism

Crypto majors share liquidity shocks. BTC volatility and move events should propagate to alts with short lag if information flows cross-sectionally.

## Lead–lag correlation (BTC return vs alt return)

- **ETHUSDT:** peak |corr| at lag **0** = **0.8134**
- **SOLUSDT:** peak |corr| at lag **0** = **0.7561**
- **BNBUSDT:** peak |corr| at lag **0** = **0.6994**
- **XRPUSDT:** peak |corr| at lag **0** = **0.6372**

## Predictive: BTC move → alt move

- **ETHUSDT:** best lag **0** bars, corr **0.4615**
- **SOLUSDT:** best lag **0** bars, corr **0.3511**
- **BNBUSDT:** best lag **0** bars, corr **0.3646**
- **XRPUSDT:** best lag **0** bars, corr **0.3404**

## Volatility propagation (|BTC ret| lag → |alt ret|)

- **ETHUSDT:** lag1=0.268, lag2=0.228, lag4=0.195, lag8=0.157, lag12=0.135
- **SOLUSDT:** lag1=0.270, lag2=0.244, lag4=0.205, lag8=0.168, lag12=0.147
- **BNBUSDT:** lag1=0.256, lag2=0.222, lag4=0.185, lag8=0.145, lag12=0.125
- **XRPUSDT:** lag1=0.238, lag2=0.202, lag4=0.173, lag8=0.142, lag12=0.120

## ETH → alts (1-bar return lead)

- **SOLUSDT:** corr **-0.0169**
- **BNBUSDT:** corr **0.0011**
- **XRPUSDT:** corr **-0.0123**

## OOS ablation (BTC move, frozen CatBoost spec)

| Config | Balanced acc | Macro F1 |
| --- | ---: | ---: |
| Baseline OHLCV | 0.6069 | 0.5992 |
| + Cross-asset | 0.6029 | 0.5951 |
| **Lift** | **-0.0039** | |

### Top new features (MI vs move)

- btc_range_pct_lag_1: 0.0342
- alt_dispersion_48: 0.0266
- btc_range_pct_lag_4: 0.0217
- btc_ret_lag_1: 0.0185
- btc_ret_lag_4: 0.0134
- bnb_vol_ratio_48: 0.0131
- btc_abs_ret_lag_2: 0.0128
- btc_abs_ret_lag_1: 0.0126
- sol_ret_lag_1: 0.0116
- btc_ret_lag_2: 0.0103
