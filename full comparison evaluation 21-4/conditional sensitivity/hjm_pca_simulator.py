"""
HJM-PCA Yield Curve Simulator — 11 Maturities (No Interpolation)
Based on Mueller (2022), adapted for FRED US Treasury data.

Pipeline:
  1. Load FRED yields (11 maturities)
  2. Bootstrap forward rates per observation
  3. PCA on end-of-month forward rate changes → V matrix (11×3)
  4. Compute HJM no-arbitrage drift A from V
  5. Simulate 200 scenarios × 60 monthly steps per starting curve
  6. Convert forward rates directly to yields (no discount factor step)
  7. Save scenarios in long format with Start_Curve_ID
"""

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────
MATURITIES  = np.array([1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])  # years
MAT_LABELS  = ['1M','3M','6M','1Y','2Y','3Y','5Y','7Y','10Y','20Y','30Y']
DTAU        = np.diff(np.concatenate([[0], MATURITIES]))  # interval widths ∆τ_k

YIELD_COLS  = ['Yield_1M','Yield_3M','Yield_6M','Yield_1Y','Yield_2Y',
               'Yield_3Y','Yield_5Y','Yield_7Y','Yield_10Y','Yield_20Y','Yield_30Y']

OUT_COLS = ['Y_1M','Y_3M','Y_6M','Y_1Y','Y_2Y','Y_3Y','Y_5Y','Y_7Y','Y_10Y','Y_20Y','Y_30Y']

N_PC        = 3            # principal components (level, slope, curvature)
T_HORIZON   = 5.0          # years
dt          = 1 / 12       # monthly time step
n_steps     = int(T_HORIZON / dt)  # 60 steps

CALIB_END   = '2020-12-31' # calibration cutoff
CSV_PATH    = 'raw_data.csv'
CSV_SAMPLING = 'initial_curves.csv'
N_SIM_PER_CURVE = 200   # ← scenarios per starting curve (FIX 1: was 20)
SEED        = 42


# ── Step 0: Load data ──────────────────────────────────────────────────────
df_starts = pd.read_csv(CSV_SAMPLING)
df = pd.read_csv(CSV_PATH, parse_dates=['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)
df_valid = df.dropna(subset=YIELD_COLS).copy()

# Calibration sample
df_calib = df_valid[df_valid['DATE'] <= CALIB_END].copy()
yields_calib = df_calib[YIELD_COLS].values / 100.0  # convert % → decimal

print(f"Calibration: {df_calib['DATE'].iloc[0].date()} → {df_calib['DATE'].iloc[-1].date()} "
      f"({len(yields_calib):,} obs)")
print(f"Starting curves: {len(df_starts)}")


# ── Step 1: Bootstrap forward rates from zero yields ──────────────────────
#
# Piecewise-constant forward rate for interval [T_{k-1}, T_k]:
#   f_k = (T_k * R_k - T_{k-1} * R_{k-1}) / (T_k - T_{k-1})
#
# For k=0: f_0 = R_0  (first yield = first forward rate directly)
#
def yields_to_forwards(y_vec):
    """Convert zero yields (length 11) to forward rates (length 11)."""
    f = np.empty(len(MATURITIES))
    f[0] = y_vec[0]
    for k in range(1, len(MATURITIES)):
        f[k] = (MATURITIES[k] * y_vec[k] - MATURITIES[k-1] * y_vec[k-1]) / DTAU[k]
    return f


# ── Step 6 (forward declaration): Convert forward rates → yields ───────────
#
# Direct formula (weighted average, no discount factor step needed):
#   R(T_k) = (1 / T_k) * sum_{j=1}^{k} f_j * ∆τ_j
#
def forwards_to_yields(f_vec):
    """Convert forward rates (length 11) to zero yields (length 11)."""
    return np.cumsum(f_vec * DTAU) / MATURITIES


# Build calibration forward curve matrix — shape (N_calib, 11)
print("Bootstrapping forward curves...", end=" ")
F_calib = np.array([yields_to_forwards(yields_calib[i]) for i in range(len(yields_calib))])
print(f"done. Shape: {F_calib.shape}")


# ── Step 2: End-of-month changes & annualised covariance ─────────────────
#
# Use end-of-month observations → monthly changes → annualise × 12
# Consistent with monthly simulation step dt = 1/12
#
df_calib_eom = df_calib.set_index('DATE').resample('ME').last().dropna(subset=YIELD_COLS)
yields_eom   = df_calib_eom[YIELD_COLS].values / 100.0
F_monthly    = np.array([yields_to_forwards(yields_eom[i]) for i in range(len(yields_eom))])
dF_monthly   = np.diff(F_monthly, axis=0)       # shape (N_months-1, 11)
Sigma_hat    = np.cov(dF_monthly.T) * 12        # annualised, shape (11, 11)
print(f"End-of-month obs: {len(F_monthly):,} | Monthly changes: {dF_monthly.shape} | Covariance: {Sigma_hat.shape} (annualised ×12)")


# ── Step 3: PCA — eigenvectors of Sigma_hat ───────────────────────────────
eigenvalues_raw, Q = np.linalg.eigh(Sigma_hat)
idx         = np.argsort(eigenvalues_raw)[::-1]   # sort descending
eigenvalues = eigenvalues_raw[idx]
Q           = Q[:, idx]

# Scale by sqrt(eigenvalue) — correct HJM volatility convention
V = Q[:, :N_PC] * np.sqrt(eigenvalues[:N_PC])    # shape (11, 3)

total_var = eigenvalues.sum()
print("\nPCA variance explained:")
for k in range(N_PC):
    cum = eigenvalues[:k+1].sum() / total_var * 100
    print(f"  PC{k+1}: {eigenvalues[k]/total_var*100:.1f}%  (cumulative: {cum:.1f}%)")


# ── Step 4: HJM no-arbitrage drift ────────────────────────────────────────
#
# Thesis Eq. (15): A_k = σ_k * ∫_0^{T_k} σ(s)^T ds
#
# Discretised with uneven spacing:
#   A_k = sum_j  V_{k,j} * (sum_{i<=k} V_{i,j} * ∆τ_i)
#
def compute_hjm_drift(V, dtau):
    A = np.zeros(V.shape[0])
    for j in range(V.shape[1]):
        integral = np.cumsum(V[:, j] * dtau)   # ∫_0^{T_k} σ_j(s) ds
        A += V[:, j] * integral
    return A

A = compute_hjm_drift(V, DTAU)
print(f"\nDrift A range: [{A.min()*1e4:.2f}, {A.max()*1e4:.2f}] bps/year")


# ── Step 5: Simulate forward curve paths ──────────────────────────────────
#
# Thesis Eq. (20):
#   F_{t+dt} = F_t + (A + ∂F_t/∂τ) * dt + V @ dW
#
# Three terms:
#   1. HJM drift:    A * dt
#   2. Aging/roll:   (∂F/∂τ) * dt — forward difference with uneven spacing
#   3. Stochastic:   V @ (Z * sqrt(dt)),  Z ~ N(0, I_3)
#
# ∂F_k/∂τ ≈ (F_{k+1} - F_k) / (T_{k+1} - T_k)   for k < 11
#          = ∂F_{10}/∂τ                              for k = 11 (repeat last)
#
np.random.seed(SEED)
N = len(MATURITIES)   # = 11
dtau_diff = np.diff(MATURITIES)   # T_{k+1} - T_k, length 10

sqrt_dt = np.sqrt(dt)
all_frames = []

for curve_idx, row in df_starts.iterrows():
    # Handle both Yield_XY and bare XY column names in initial_curves.csv
    try:
        start_vals = row[YIELD_COLS].values / 100.0
    except KeyError:
        bare_cols = [c.replace('Yield_', '') for c in YIELD_COLS]
        start_vals = row[bare_cols].values / 100.0

    F0 = yields_to_forwards(start_vals)
    Ft = np.tile(F0, (N_SIM_PER_CURVE, 1))          # (N_SIM_PER_CURVE, 11)
    Y_paths = np.zeros((N_SIM_PER_CURVE, n_steps, N))

    for t in range(n_steps):
        dFdtau         = np.zeros_like(Ft)
        dFdtau[:, :-1] = (Ft[:, 1:] - Ft[:, :-1]) / dtau_diff
        dFdtau[:, -1]  = dFdtau[:, -2]

        Z     = np.random.randn(N_SIM_PER_CURVE, N_PC)
        shock = (Z * sqrt_dt) @ V.T

        Ft = Ft + (A + dFdtau) * dt + shock
        Y_paths[:, t, :] = np.cumsum(Ft * DTAU, axis=1) / MATURITIES

    Y_paths *= 100   # decimal → percent

    # Build long-format slice for this starting curve
    scenario_ids = np.repeat(np.arange(1, N_SIM_PER_CURVE + 1), n_steps)
    months       = np.tile(np.arange(1, n_steps + 1), N_SIM_PER_CURVE)
    yields_flat  = Y_paths.reshape(-1, N)

    frame = pd.DataFrame(yields_flat, columns=OUT_COLS)
    frame.insert(0, 'Start_Curve_ID', curve_idx + 1)   # 1-indexed regime ID
    frame.insert(1, 'Scenario_ID',    scenario_ids)     # 1…N_SIM_PER_CURVE per curve
    frame.insert(2, 'Month',          months)
    all_frames.append(frame)

    print(f"  Simulated curve {curve_idx + 1}/{len(df_starts)}: "
          f"{REGIMES_LABELS.get(curve_idx+1, f'Curve {curve_idx+1}')} "
          f"({N_SIM_PER_CURVE} scenarios)")

# ── FIX 2: keep Start_Curve_ID intact, keep Scenario_ID as 1…200 per curve ──
# (old code dropped Start_ID and overwrote Scenario_ID globally — now removed)
df_out = pd.concat(all_frames, ignore_index=True)

df_out.to_csv('hjm_pca_3 scenarios 6-5.csv', index=False)

# ── Summary ───────────────────────────────────────────────────────────────
Y_terminal = df_out[df_out['Month'] == n_steps][OUT_COLS].values
print(f"\nTerminal yield statistics at month 60 (%) — all curves combined:")
print(f"{'Maturity':>8}  {'Mean':>8}  {'Std':>7}  {'P10':>7}  {'P90':>7}")
print("-" * 45)
for i, lbl in enumerate(OUT_COLS):
    v = Y_terminal[:, i]
    print(f"{lbl:>8}  {v.mean():>8.3f}  {v.std():>7.3f}  "
          f"{np.percentile(v,10):>7.3f}  {np.percentile(v,90):>7.3f}")

n_curves = df_out['Start_Curve_ID'].nunique()
print(f"\nSaved: hjm_pca_3 scenarios 6-5.csv")
print(f"  Rows: {len(df_out):,}  "
      f"({n_curves} curves × {N_SIM_PER_CURVE} scenarios × {n_steps} months)")
print(f"  Columns: {list(df_out.columns)}")
print(df_out.head(3).to_string(index=False))

# Per-curve terminal stats
print("\nPer-curve terminal mean at month 60 (10Y yield):")
col_10y = OUT_COLS[OUT_COLS.index('Y_10Y')]
for rid in sorted(df_out['Start_Curve_ID'].unique()):
    sub = df_out[(df_out['Start_Curve_ID']==rid) & (df_out['Month']==n_steps)]
    print(f"  Curve {rid}: mean 10Y = {sub[col_10y].mean():.3f}%  "
          f"std = {sub[col_10y].std():.3f}%")


# ── Regime label lookup (for progress printing above) ─────────────────────
# Note: defined here so it can be used in the loop above on re-runs;
# on first run it will silently fall back to "Curve N" labels.
REGIMES_LABELS = {
    1: "High-rate (2002)",
    2: "Medium-rate (2015)",
    3: "Low-rate (2018)",
}
