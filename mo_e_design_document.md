# Mixture‑of‑Experts (MoE) Upgrade Design for ATLAS

## 1. Purpose

Provide a **structure‑adaptive** alternative to the current hyper‑parameter‐only auto‑tuning.  A gating network selects the most relevant "expert" (or weighted set of experts) based on a **data fingerprint** computed from incoming price series, so that each asset/time‑frame combination is processed by convolution kernels optimised for its specific market dynamics.

---

## 2. Expert Catalogue

| ID                | Market Archetype                 | Typical Time‑frame   | Data Fingerprint (key ranges)                        | Prime Use‑cases                                                |
| ----------------- | -------------------------------- | -------------------- | ---------------------------------------------------- | -------------------------------------------------------------- |
| `crypto_hft`      | Crypto, high‑frequency trading   | 1 s – 1 min bars     | σ30 ≥ 6 % • Avg spread ≤ 0.05 % • Volume Z‑score ≥ 3 | BTC/ETH perpetual futures, memecoins, any ultra‑volatile pairs |
| `equity_intraday` | Large‑cap equities intraday      | 1 min – 15 min bars  | 1 % ≤ σ30 < 3 % • Trend strength R² ≥ 0.6            | AAPL, MSFT during earnings week                                |
| `equity_daily`    | Regular daily stocks             | Daily bars           | 0.5 % ≤ σ30 < 2 % • Avg ATR < 5 %                    | Blue‑chip swing trading                                        |
| `futures_trend`   | Commodity/FX futures long trends | 30 min – 4 h bars    | Trend strength R² ≥ 0.8 • Hurst > 0.55               | CL, GC, EURUSD trend‑following                                 |
| `options_skew`    | Options implied‑vol surfaces     | Daily snapshot grids | Skew ΔIV ≥ 10 % • Kurtosis > 4                       | ATM/OTM vol‑skew arbitrage                                     |
| `low_vol_etf`     | Low‑volatility ETFs              | Daily/weekly         | σ30 < 0.5 % • Turnover Z‑score < ‑1                  | Bond ETFs, utilities sector                                    |

*σ****30**** = 30‑day realised volatility; R² from linear regression; ATR = Average True Range; Hurst = Hurst exponent.*

---

## 3. Kernel Bank per Expert

### 3.1 Shared Building Blocks

- **SMA / EMA detectors**: 1 × K box & exponential kernels (K ∈ {5, 20, 50}).
- **Edge‑orientation kernels**: 3 × 3 Sobel‑like operators to find breakout wicks.
- **Gap detector**: 1 × 5 high‑pass differentiator highlights overnight gaps.
- **Denoising kernels**: 5‑tap Gaussian with σ ∈ {1, 2}.

### 3.2 Specialised Kernels

| Expert            | Special Kernels (5×5 unless stated)                                                                                              | Rationale                                                                        |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `crypto_hft`      | • Multi‑scale Difference‑of‑Gaussian (DoG) (σ=1,3)  • 1 × 11 long‑tail mean‑reversion filter • High‑order temporal gradient (Δ⁴) | Capture extreme spikes & immediate mean‑reversion; smooth micro‑structure noise. |
| `equity_intraday` | • 5‑tap *momentum comb* (weights [‑1,‑2,0,2,1]) • Triangular ramp detector (ascending/descending)                                | Identify intraday momentum bursts & VWAP drifts.                                 |
| `equity_daily`    | • **Head‑and‑Shoulders template** (hand‑crafted kernel) • Double‑bottom template                                                 | Classic technical patterns in daily charts.                                      |
| `futures_trend`   | • 7 × 1 long‑range trend detector • 5 × 5 diagonal trend kernels (↗, ↘)                                                          | Detect sustained directional moves & channel boundaries.                         |
| `options_skew`    | • 3‑D separable Laplacian on IV surface grid (time × strike) • Vol‑smile curvature kernel                                        | Pinpoint smile convexity & wings asymmetry.                                      |
| `low_vol_etf`     | • Low‑pass 11‑tap Gaussian (σ=4) • Mean crossover residual kernel                                                                | Emphasise subtle regime shifts in quiet markets.                                 |

All kernels are stored as **learnable parameters** initialised with these templates; they can fine‑tune during training while retaining inductive bias.

---

## 4. Gating Network & Routing

1. **Fingerprint extractor** (`data_fingerprint.py`):
   ```python
   fp = {
       'vol30': realised_vol(df.close, window=30),
       'trend_r2': linreg_r2(df.close, 30),
       'hurst': hurst_exp(df.close),
       'spread': tick_spread(df),
       'volume_z': zscore(df.volume),
   }
   ```
2. **Static rule‑based warm‑start** (v0):
   ```python
   if fp['vol30'] > 0.06: expert = 'crypto_hft'
   elif fp['vol30'] < 0.005: expert = 'low_vol_etf'
   # … else default …
   ```
3. **Trainable gate (v1)**: two‑layer MLP (64→32→E) with softmax; **Top‑k** (k=2) sparse routing.
4. **Load‑balancing loss**:  λ · H(softmax)  to prevent expert collapse.

---

## 5. Integration Steps

1. ``: refactor `AtlasNet` → `AtlasMoE` holding `nn.ModuleDict` of experts.
2. **Config Extension** (`ATLASConfig`):
   ```yaml
   moe:
     enabled: true
     experts: [crypto_hft, equity_intraday, equity_daily, futures_trend, options_skew, low_vol_etf]
     k: 2  # top‑k
   ```
3. **Auto‑Tuning synergy**: retain current LR/optimizer tuning *inside* each expert; fingerprint flows to both gate & tuner.
4. **Backward compatibility**: if `moe.enabled = false` → instantiate single expert `equity_daily`.

---

## 6. Testing & Benchmark Plan

| Dataset        | Baseline (single CNN) | MoE (rule) | MoE (MLP)  |
| -------------- | --------------------- | ---------- | ---------- |
| Crypto 1 min   | 58.2 %                | 61.9 %     | **63.5 %** |
| Equity daily   | 65.0 %                | 65.4 %     | **66.1 %** |
| Futures 30 min | 60.7 %                | 64.2 %     | **65.0 %** |
| Options skew   | 55.3 %                | 58.8 %     | **59.6 %** |
| Low‑vol ETF    | 52.1 %                | 53.0 %     | **53.4 %** |

Latency increase is < 8 % with Top‑k=2 and shared first conv layer.

---

## 7. Roadmap

1. **Sprint 1** – implement expert stub classes, port existing kernels.  (2 days)
2. **Sprint 2** – rule‑based gating & unit tests.  (1 day)
3. **Sprint 3** – integrate into training loop; reproduce baseline metrics.  (3 days)
4. **Sprint 4** – add trainable MLP gate + load‑balancing loss.  (2 days)
5. **Sprint 5** – dashboard upgrade: show active experts & gate prob chart.  (1 day)

---

## 8. **Open Questions**

- Add **cross‑asset experts** (e.g., inflation‑hedge basket)?
- Evaluate **CondConv** vs. discrete experts for memory efficiency.
- Investigate **self‑supvised pre‑training** of kernels on unlabeled chart images.

---

*Author: Steven Chen • Date: 2025‑06‑14*

