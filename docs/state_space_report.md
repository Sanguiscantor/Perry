# State-Space Discovery Report

## Objective

Focus the Perry research program on missing market state variables, latent structure, regimes, and microstructure — not on model tuning.

This report captures the current evidence, existing architectural strengths, and a prioritized research plan for discovering the latent state that drives directional moves.

## Existing evidence and architectural foundation

- The repository already contains strong structure-based feature engineering in `Model/features_data/market_structure.py`.
- Prior research artifacts show the project has tracked:
  - price action and directional edge performance
  - basis and IV features
  - regime/state transition ideas
  - microstructure and order-flow awareness
- The highest-leverage gap is not better classifiers but better state variables.

## What has been learned so far

1. **Directional edge is weak relative to move detection**
   - Existing models and signals can detect volatility expansion or momentum, but direction remains noisy.
   - This is consistent with the hypothesis that the market state is only partially observed by standard OHLCV features.

2. **Discrete regimes alone are insufficient**
   - Previous work on regimes and `market_state_report.md` suggests that purely categorical regimes are unstable and context-dependent.
   - Market regimes should be treated as soft, continuous, and conditioned on microstructure plus structural geometry.

3. **Market structure signals are already present**
   - `market_structure.py` encodes swing points, trendline quality, touches, rejections, and persistence.
   - These are valuable candidate state features for breakout, compression, and directional pressure.

4. **Options/derivatives data remain an underused state source**
   - Deribit IV and skew are important latent state proxies for risk appetite and directional conviction.
   - Historical backfill is limited, so present snapshots and short-window term structure should be prioritized.

## Hypotheses for missing state variables

### Hypothesis 1: directional state is a hybrid of structure + microstructure

- Structural context: support/resistance, compression, trendline activity, breakout pressure.
- Microstructure context: order flow imbalance, liquidity sweeps, volume profile shifts, limit order book pressure.
- The dominant state variable may be a continuous pressure vector rather than a binary regime.

### Hypothesis 2: delay embeddings and latent manifolds reveal hidden price dynamics

- Standard lagged OHLCV is a shallow observation of the state.
- Delay-embedding the price path can recover attractors or slow manifold components.
- PCA or latent projections on returns can identify the dominant low-dimensional dynamics.

### Hypothesis 3: soft state probabilities are more useful than hard labels

- Use Gaussian mixture / continuous cluster probabilities rather than fixed regime labels.
- A soft state representation can capture transitions, regime blending, and temporary uncertainty.

### Hypothesis 4: derivatives term structure is a leading state indicator

- Changes in IV slope, skew, and basis may anticipate directional conviction.
- These should be merged with structure and microstructure state variables rather than modeled separately.

## New scaffolding added

- `Model/features_data/state_space.py`
  - Delay embedding helper for lagged state reconstruction
  - PCA latent state extraction
  - local price geometry / curvature features
  - Gaussian mixture soft state features
  - Combined `build_state_space_features()` helper

- `Model/state_space_research.py`
  - Exploratory driver to load raw market data
  - Build latent state and geometry samples
  - Save `artifacts/data/latent_state/state_space_features.csv`

## Recommended research path

1. **Validate microstructure state signals first**
   - Extract order-flow imbalance, big-tick sweep counts, volume spikes, and trade direction proxies.
   - Evaluate whether these signals improve directional edge when combined with structure features.

2. **Enhance structural feature coverage**
   - Add support/resistance strength, compression score, breakout impulse, and active trendline pressure.
   - Convert existing structure detections into a continuous state score.

3. **Build continuous latent state features**
   - Use delay embeddings over `Close` and/or `returns`.
   - Apply PCA/autoencoder-style projection to detect dominant market modes.
   - Compute soft state probabilities from mixture models rather than hard labels.

4. **Test derivatives and basis as state inputs**
   - Use IV snapshots, skew ratios, basis spreads, and term-structure momentum.
   - Combine them with structure and microstructure state features for cross-asset context.

5. **Focus on state-aware directional rules, not end-to-end model tuning**
   - Prioritize features that explain why price moves in one direction instead of the other.
   - Keep the research engine as the evaluation driver, but treat this work as state discovery.

## Priority experiments

- Experiment A: `support/resistance + breakout pressure + local curvature`
- Experiment B: `microstructure imbalance + order-flow pulse`
- Experiment C: `latent PCA/GMM state on returns + geometry features`
- Experiment D: `IV skew + basis term structure as leading state variables`

## Summary recommendation

The next stage of Perry research should be:
- preserve the current hybrid architecture,
- shift the active focus from model fitting to latent state discovery,
- use the newly added `state_space.py` scaffold to prototype projections,
- and prioritize microstructure + structure as the missing state variables for directional edge.
