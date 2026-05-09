# ==========================================
# MVP2
# CHANGES FROM MVP1:
#
# [FIX-1] Zero-mean dy normalization
#         mu_dy is now always 0. We only divide changes by std_dy.
#         This removes the historical drift artifact (~-1% / 60m for 10Y)
#         that was baked into every generated path regardless of conditioning.
#
# [FIX-2] Train/val split adjusted for sufficient validation windows
#         With HORIZON=60 you need >61 months of val data for even 1 window.
#         Split moved to pre-2016 / 2016+ so val gets ~35+ windows
#         (assuming data through ~2023-2024).
#
# [FIX-3] Bug in generate_scenarios: paths_dy was undefined (denormalization
#         line was commented out but the variable was still used below).
#
# [NEW-1] TenorAttention: self-attention across the 11 tenor dimensions,
#         applied after residual blocks. Forces the model to explicitly
#         learn cross-tenor structure (level / slope / curvature modes)
#         rather than relying solely on the spatial convolutions.
#
# [NEW-2] Correlation auxiliary loss: during training, estimate x0 from
#         v_pred, compute the batch-averaged cross-tenor correlation matrix,
#         and penalise deviation from the empirical historical matrix.
#         Weight controlled by CORR_WEIGHT (default 0.05).
#
# [NEW-3] Correlated noise initialisation in sampling: instead of i.i.d.
#         Gaussian noise, start the reverse chain from noise with the
#         historical tenor correlation (via Cholesky factor). This anchors
#         the large-scale structure from the very first reverse step.
#
# [NEW-4] Cosine annealing LR scheduler.
#
# [NEW-5] Gradient clipping (max_norm=1.0).
# ==========================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ==========================================
# 1. DATASET
# ==========================================
class YieldPathDataset(torch.utils.data.Dataset):
    def __init__(self, df, horizon=60, yield_cols=None,
                 mu_y=None, std_y=None, std_dy=None):
        """
        Parameters
        ----------
        mu_dy is intentionally removed.  We normalise changes by dividing
        by std only (zero-mean assumption) so that historical trend is NOT
        baked into every generated scenario.
        """
        self.horizon = horizon
        self.yield_cols = yield_cols if yield_cols else [
            c for c in df.columns if 'Y_DGS' in c
        ]

        y  = df[self.yield_cols].values.astype(np.float32)
        dy = y[1:] - y[:-1]          # monthly first differences

        # --- Level normalisation (conditioning input) ---
        if mu_y is None or std_y is None:
            self.mu_y  = y.mean(axis=0)
            self.std_y = y.std(axis=0) + 1e-8
        else:
            self.mu_y  = mu_y
            self.std_y = std_y

        # --- [FIX-1] Change normalisation: scale only, no mean subtraction ---
        if std_dy is None:
            self.std_dy = dy.std(axis=0) + 1e-8
        else:
            self.std_dy = std_dy
        self.mu_dy = np.zeros_like(self.std_dy)   # always zero, kept for compat

        # Empirical correlation matrix of normalised changes (used by engine)
        dy_scaled        = dy / self.std_dy
        self.corr_matrix = np.corrcoef(dy_scaled.T).astype(np.float32)  # (N, N)

        y_norm  = (y  - self.mu_y)  / self.std_y
        dy_norm = dy / self.std_dy           # no mean subtraction

        self.paths    = []
        self.y_starts = []

        for i in range(len(y) - horizon - 1):
            self.paths.append(dy_norm[i : i + horizon])
            self.y_starts.append(y_norm[i])

        self.paths    = torch.tensor(
            np.array(self.paths), dtype=torch.float32
        ).unsqueeze(1)          # (B, 1, T, N_tenors)
        self.y_starts = torch.tensor(
            np.array(self.y_starts), dtype=torch.float32
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.paths[idx], self.y_starts[idx]


# ==========================================
# 2. SINUSOIDAL TIME EMBEDDING
# ==========================================
def sinusoidal_embedding(timesteps, dim):
    device   = timesteps.device
    half_dim = dim // 2
    factor   = np.log(10000) / (half_dim - 1)
    emb      = torch.exp(torch.arange(half_dim, device=device) * -factor)
    emb      = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb      = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# ==========================================
# 3. [NEW-1] TENOR SELF-ATTENTION
# ==========================================
class TenorAttention(nn.Module):
    """
    Self-attention across the maturity dimension at each time step.

    The 11 tenors are treated as a sequence of 'tokens'; multi-head
    attention lets each tenor attend to all others, explicitly learning
    co-movement patterns such as 2Y–10Y spread dynamics and the
    level / slope / curvature modes that dominate yield-curve variation.
    """

    def __init__(self, channels, n_heads=4):
        super().__init__()
        assert channels % n_heads == 0, "channels must be divisible by n_heads"
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (B, C, T, N_tenors)
        B, C, T, N = x.shape
        # Flatten batch and time so each (batch, time) slice attends over N tenors
        x_in      = x.permute(0, 2, 3, 1).reshape(B * T, N, C)
        attn_out, _ = self.attn(x_in, x_in, x_in)
        attn_out  = self.norm(attn_out)
        attn_out  = attn_out.reshape(B, T, N, C).permute(0, 3, 1, 2)
        return x + attn_out          # residual


# ==========================================
# 4. MODEL (RESIDUAL CNN + TENOR ATTENTION)
# ==========================================
class PathResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class PathDenoiser(nn.Module):
    def __init__(self, tenors=11, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(tenors, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.input_conv = nn.Conv2d(1, hidden_dim, 3, padding=1)
        self.res_layers = nn.ModuleList(
            [PathResBlock(hidden_dim) for _ in range(4)]
        )

        # [NEW-1] Tenor attention after spatial residual blocks
        self.tenor_attn = TenorAttention(hidden_dim, n_heads=4)

        self.output_conv = nn.Conv2d(hidden_dim, 1, 3, padding=1)

    def forward(self, x, t, y_start):
        t_emb = sinusoidal_embedding(t, self.hidden_dim)
        t_emb = self.time_mlp(t_emb).view(-1, self.hidden_dim, 1, 1)

        c_emb = self.cond_mlp(y_start).view(-1, self.hidden_dim, 1, 1)

        h = self.input_conv(x) + t_emb + c_emb

        for layer in self.res_layers:
            h = layer(h)

        # [NEW-1] Cross-tenor attention
        h = self.tenor_attn(h)

        return self.output_conv(h)


# ==========================================
# 5. NOISE SCHEDULE
# ==========================================
def make_beta_schedule(T, schedule="cosine", device="cpu"):
    if schedule == "linear":
        return torch.linspace(1e-4, 0.02, T, device=device)
    elif schedule == "cosine":
        steps = T + 1
        x = torch.linspace(0, T, steps, device=device)
        alphas_cumprod = (
            torch.cos(((x / T) + 0.008) / 1.008 * torch.pi / 2) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        beta = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(beta, 1e-4, 0.999)
    else:
        raise ValueError("schedule must be 'linear' or 'cosine'")


# ==========================================
# 6. DIFFUSION ENGINE
# ==========================================
class PathDiffusionEngine:
    def __init__(self, model, horizon=60, tenors=11, T=400,
                 device='cpu', corr_matrix=None):
        self.model   = model
        self.horizon = horizon
        self.tenors  = tenors
        self.T       = T
        self.device  = device

        self.beta      = make_beta_schedule(T, "cosine", device)
        self.alpha     = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # [NEW-3] Cholesky factor of historical tenor correlation
        if corr_matrix is not None:
            corr_reg    = corr_matrix + 1e-6 * np.eye(tenors)
            L           = np.linalg.cholesky(corr_reg).astype(np.float32)
            self.L_corr = torch.tensor(L, device=device)
        else:
            self.L_corr = None

    # ----------------------------------------------------------
    def _correlated_noise(self, shape):
        """Sample noise with historical cross-tenor correlations."""
        B, C, T, N = shape
        z = torch.randn(B * T, N, device=self.device)
        if self.L_corr is not None:
            z = z @ self.L_corr.T        # apply Cholesky factor
        return z.reshape(B, C, T, N)

    # ----------------------------------------------------------
    def train_loss(self, path_0, y_start,
                   corr_target=None, corr_weight=0.05):
        """
        V-prediction MSE loss + optional cross-tenor correlation loss.

        corr_target : (N, N) tensor — empirical correlation matrix from training data
        corr_weight : scalar weight for correlation auxiliary loss
        """
        batch = path_0.shape[0]
        t     = torch.randint(0, self.T, (batch,), device=self.device)

        noise  = torch.randn_like(path_0)
        a_hat  = self.alpha_hat[t].view(-1, 1, 1, 1)
        path_t = torch.sqrt(a_hat) * path_0 + torch.sqrt(1 - a_hat) * noise

        v_target = torch.sqrt(a_hat) * noise - torch.sqrt(1 - a_hat) * path_0
        v_pred   = self.model(path_t, t, y_start)

        mse_loss = F.mse_loss(v_pred, v_target)

        # [NEW-2] Correlation auxiliary loss
        if corr_target is not None and corr_weight > 0:
            # Reconstruct estimated x0 from v_pred
            x0_pred = torch.sqrt(a_hat) * path_t - torch.sqrt(1 - a_hat) * v_pred
            x0_flat = x0_pred.squeeze(1)                   # (B, T, N)

            # Zero-mean and unit-std per path
            x0_flat = x0_flat - x0_flat.mean(dim=1, keepdim=True)
            x0_flat = x0_flat / (x0_flat.std(dim=1, keepdim=True) + 1e-8)

            # Batch-averaged sample correlation matrix
            # (B, N, T) @ (B, T, N) / T -> (B, N, N)
            C_pred = torch.bmm(
                x0_flat.permute(0, 2, 1), x0_flat
            ) / (x0_flat.shape[1] - 1)
            C_mean = C_pred.mean(dim=0)                    # (N, N)

            corr_loss = F.mse_loss(C_mean, corr_target)
            total_loss = mse_loss + corr_weight * corr_loss
        else:
            total_loss = mse_loss

        return total_loss

    # ----------------------------------------------------------
    @torch.no_grad()
    def sample_path(self, y_start, use_correlated_noise=True):
        """Reverse diffusion with optional correlated noise initialisation."""
        n = y_start.shape[0]

        # [NEW-3] Start from historically correlated noise
        if use_correlated_noise and self.L_corr is not None:
            x = self._correlated_noise((n, 1, self.horizon, self.tenors))
        else:
            x = torch.randn(
                (n, 1, self.horizon, self.tenors), device=self.device
            )

        for i in reversed(range(self.T)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            v = self.model(x, t, y_start)

            a_hat = self.alpha_hat[i]
            x0    = torch.sqrt(a_hat) * x - torch.sqrt(1 - a_hat) * v

            if i > 0:
                a_prev    = self.alpha_hat[i - 1]
                direction = torch.sqrt(1 - a_prev) * (
                    (x - torch.sqrt(a_hat) * x0) / torch.sqrt(1 - a_hat)
                )
                x = torch.sqrt(a_prev) * x0 + direction
            else:
                x = x0

        return x.squeeze(1)          # (N, T, N_tenors)


# ==========================================
# 7. VALIDATION LOSS
# ==========================================
@torch.no_grad()
def evaluate_loss(engine, model, loader, device):
    model.eval()
    losses = []
    for p, ys in loader:
        p, ys = p.to(device), ys.to(device)
        # No correlation auxiliary loss during validation (just MSE)
        loss = engine.train_loss(p, ys, corr_target=None, corr_weight=0.0)
        losses.append(loss.item())
    return float(np.mean(losses))


# ==========================================
# 8. SCENARIO GENERATION
# ==========================================
def generate_scenarios(model, engine, start_curve,
                       mu_y, std_y, std_dy, yield_cols,
                       num_scenarios=200, batch_size=50,
                       filename="scenarios_monthly.csv"):
    """
    Generate num_scenarios of 60-month yield paths from a given starting curve.

    Note: mu_dy parameter from MVP1 is gone — changes are zero-mean normalised.
    """
    model.eval()
    device = engine.device

    start_curve  = np.array(start_curve, dtype=np.float32)
    start_norm   = (start_curve - mu_y) / std_y
    start_tensor = torch.tensor(start_norm).unsqueeze(0).to(device)

    all_paths = []
    n_batches = int(np.ceil(num_scenarios / batch_size))

    for b in range(n_batches):
        size    = min(batch_size, num_scenarios - b * batch_size)
        y_batch = start_tensor.repeat(size, 1)

        # (size, T, N) — normalised changes
        paths_dy_norm = engine.sample_path(y_batch).cpu().numpy()

        # [FIX-3] Denormalise changes (zero-mean, so multiply by std only)
        paths_dy     = paths_dy_norm * std_dy.reshape(1, 1, -1)

        # Reconstruct levels via cumulative sum
        paths_levels = (
            start_curve.reshape(1, 1, -1) + np.cumsum(paths_dy, axis=1)
        )
        all_paths.append(paths_levels)

    all_paths = np.vstack(all_paths)    # (num_scenarios, T, N_tenors)

    rows = []
    for s in range(num_scenarios):
        for t in range(engine.horizon):
            row = {"Scenario_ID": s + 1, "Month": t + 1}
            for i, col in enumerate(yield_cols):
                row[col] = all_paths[s, t, i]
            rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(filename, index=False)
    print(f"Saved {num_scenarios} scenarios to '{filename}'")
    return df_out


# ==========================================
# 9. MAIN TRAINING PIPELINE
# ==========================================
if __name__ == "__main__":

    HORIZON     = 60
    TENORS      = 11
    EPOCHS      = 50
    CORR_WEIGHT = 0.05   # weight for cross-tenor correlation auxiliary loss

    # --- Load and resample ---
    df = pd.read_csv("raw_data.csv", index_col=0, parse_dates=True)
    df = df.resample("ME").last().dropna()
    print(f"Total data: {len(df)} months  "
          f"({df.index[0].date()} → {df.index[-1].date()})")

    # --- [FIX-2] Train / val split with enough validation windows ---
    # With HORIZON=60 you need >61 months of val data.
    # Splitting at 2015-end gives 2016+ for validation:
    #   ~96 months (to 2023) → 96 - 60 - 1 = 35 validation windows.
    # Adjust the split year if your data ends earlier.
    SPLIT_YEAR = '2015'
    df_train = df.loc[:SPLIT_YEAR]
    df_val   = df.loc[str(int(SPLIT_YEAR) + 1):]

    n_train_windows = len(df_train) - HORIZON - 1
    n_val_windows   = len(df_val)   - HORIZON - 1

    print(f"Train months: {len(df_train)}  |  windows: {n_train_windows}")
    print(f"Val   months: {len(df_val)}    |  windows: {n_val_windows}")

    if n_val_windows <= 5:
        raise ValueError(
            f"Only {n_val_windows} validation windows — too few for meaningful "
            f"early stopping. Move SPLIT_YEAR earlier or reduce HORIZON."
        )

    # --- Datasets ---
    train_ds = YieldPathDataset(df_train, HORIZON)
    val_ds   = YieldPathDataset(
        df_val, HORIZON,
        yield_cols=train_ds.yield_cols,
        mu_y=train_ds.mu_y,
        std_y=train_ds.std_y,
        std_dy=train_ds.std_dy,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=32, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- [NEW-2] Correlation target tensor ---
    corr_target = torch.tensor(
        train_ds.corr_matrix, dtype=torch.float32
    ).to(device)

    # --- Model & engine ---
    model  = PathDenoiser(TENORS).to(device)
    engine = PathDiffusionEngine(
        model, HORIZON, TENORS, device=device,
        corr_matrix=train_ds.corr_matrix,   # [NEW-3] for correlated noise
    )

    # --- Optimiser + [NEW-4] LR scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-4, weight_decay=0.1
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    train_losses, val_losses = [], []
    best_val = float("inf")

    print("Training …")

    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []

        for p, ys in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            p, ys = p.to(device), ys.to(device)

            loss = engine.train_loss(
                p, ys,
                corr_target=corr_target,
                corr_weight=CORR_WEIGHT,
            )

            optimizer.zero_grad()
            loss.backward()
            # [NEW-5] Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_losses.append(loss.item())

        # [NEW-4] Step LR scheduler
        scheduler.step()

        train_loss = float(np.mean(batch_losses))
        val_loss   = evaluate_loss(engine, model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch+1:3d} | "
            f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "mu_y":        train_ds.mu_y,
                "std_y":       train_ds.std_y,
                "std_dy":      train_ds.std_dy,
                "mu_dy":       train_ds.mu_dy,   # zeros, kept for compat
                "yield_cols":  train_ds.yield_cols,
                "corr_matrix": train_ds.corr_matrix,
                "horizon":     HORIZON,
                "tenors":      TENORS,
                "T":           engine.T,
            }, "best_model.pt")
            print(f"  ✓ Saved best model (val: {best_val:.5f})")

    # --- Loss plot ---
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses,   label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss curves — MVP2")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curves_mvp2.png", dpi=150)
    plt.show()

    # --- Generate scenarios from the latest observed curve ---
    sample_curve = df.iloc[-1][train_ds.yield_cols].values

    generate_scenarios(
        model, engine, sample_curve,
        train_ds.mu_y, train_ds.std_y, train_ds.std_dy,
        train_ds.yield_cols,
        num_scenarios=200,
        filename="scenarios_mvp2.csv",
    )
