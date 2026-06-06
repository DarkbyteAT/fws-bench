"""Phase 2 integration: real SIREN renderer via ondes + loom, multi-seed run.

Replaces phase 1's ``tanh(G @ z)`` placeholder with a SIREN INR (from
``ondes``) modulated by FiLM derived from ``z``, composed through
``loom.render``. The mainnet remains a linear regressor ``y = W @ x`` with
``W ∈ R^(4×8)`` so the integration surface stays small; the renderer is the
component getting real for phase 2.

The G group carries two pytrees: the SIREN body, and a linear FiLM projector
that maps ``z ∈ R^{dim_z}`` to the per-layer ``(gamma, beta)`` schedule the
basis body consumes. ``z`` is the pure modulation latent.

Run:
    uv run python examples/phase2_real_siren.py
"""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import loom
import ondes
import optax
from jaxtyping import Array, Float
from landscape_archaeology import singular_spectrum

import fws_bench


RESEARCH_DIR = Path("/Users/ammar/Documents/Research/fractal-weight-spaces/research")
RESEARCH_FILE = RESEARCH_DIR / "2026-06-06-phase2-real-siren.md"

# --- Renderer hyperparameters ------------------------------------------------
DIM_Z = 32
HIDDEN_DIM = 32
NUM_HIDDEN_LAYERS = 3
COORD_DIM = 1
N_OUT = 4
N_IN = 8
N_W_ENTRIES = N_OUT * N_IN  # 32
NUM_OUTER_STEPS = 500
K_SEED = 10  # seeds for the multi-seed sweep


# --- Synthetic linear-regression task ----------------------------------------
def make_synthetic_regression(
    *,
    n_in: int = N_IN,
    n_out: int = N_OUT,
    n_train: int = 100,
    n_test: int = 50,
    seed: int = 0,
):
    """Same noise-free linear regression task as phase 1."""
    key = jax.random.key(seed)
    key_w, key_x_train, key_x_test = jax.random.split(key, 3)
    W_target = jax.random.normal(key_w, (n_out, n_in))
    x_train = jax.random.normal(key_x_train, (n_train, n_in))
    y_train = x_train @ W_target.T
    x_test = jax.random.normal(key_x_test, (n_test, n_in))
    y_test = x_test @ W_target.T

    def task_loss_fn(W: Float[Array, "n_out n_in"], batch: dict) -> Float[Array, ""]:
        preds = batch["x"] @ W.T
        return jnp.mean((preds - batch["y"]) ** 2)

    return task_loss_fn, {"x": x_train, "y": y_train}, {"x": x_test, "y": y_test}, W_target


# --- Generator: SIREN body + FiLM projector ----------------------------------
class Generator(eqx.Module):
    """Conditioning network ``G`` for the FWS reparameterisation.

    Owns the SIREN body and a linear FiLM projector that turns ``z`` into the
    per-layer ``(gamma, beta)`` schedule expected by ``ondes.SIREN.trunk``.
    Both pytrees are trained as the G group under the ``optax.multi_transform``
    partition.
    """

    siren: ondes.SIREN
    film_W: Float[Array, "n_layers two_hidden dim_z"]
    film_b: Float[Array, "n_layers two_hidden"]

    def __init__(
        self,
        *,
        dim_z: int,
        hidden_dim: int,
        num_hidden_layers: int,
        coord_dim: int,
        key: jax.Array,
    ) -> None:
        k_siren, k_w, k_b = jax.random.split(key, 3)
        self.siren = ondes.SIREN(
            in_dim=coord_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            key=k_siren,
        )
        # Small init so FiLM starts near (gamma=0, beta=0) — the body sees the
        # base SIREN forward at z_init and the optimiser learns modulation.
        bound = 1.0 / dim_z
        self.film_W = jax.random.uniform(k_w, (num_hidden_layers, 2 * hidden_dim, dim_z), minval=-bound, maxval=bound)
        self.film_b = jax.random.uniform(k_b, (num_hidden_layers, 2 * hidden_dim), minval=-bound, maxval=bound)

    def film_from_z(self, z: Float[Array, " dim_z"]) -> Float[Array, "n_layers two_hidden"]:
        """Linear projection ``z -> film`` shaped to ondes' contract."""
        return jnp.einsum("lhd,d->lh", self.film_W, z) + self.film_b


def make_coords(n_points: int) -> Float[Array, "n_points 1"]:
    """1-D coordinate grid on ``[-1, 1]`` (Sitzmann+ 2020 convention)."""
    return jnp.linspace(-1.0, 1.0, n_points).reshape(n_points, 1)


COORDS_32 = make_coords(N_W_ENTRIES)


def siren_render_fn(G: Generator, z: Float[Array, " dim_z"]) -> Float[Array, "n_out n_in"]:
    """Materialise W via ``loom.render`` with an ondes SIREN body and FiLM(z).

    The ``P`` template is a one-leaf pytree shaped ``(4, 8)``; ``loom.render``
    walks it and calls our renderer once. Inside the renderer we vmap the
    SIREN body over the 32 coords with the FiLM schedule produced from ``z``,
    then reshape the 32 scalar outputs into ``(4, 8)``.
    """
    P = {"W": jnp.zeros((N_OUT, N_IN))}

    def f(path, shape, dtype, params):
        G_in, z_in = params
        film = G_in.film_from_z(z_in)
        values = jax.vmap(lambda c: G_in.siren(c, film=film))(COORDS_32)
        return values.reshape(shape).astype(dtype)

    return loom.render(P, f, (G, z))["W"]


# --- Init helpers ------------------------------------------------------------
def make_inits(seed: int):
    """Initial pytrees for both arms at a given seed."""
    key = jax.random.key(seed)
    key_g, key_z, key_w = jax.random.split(key, 3)
    init_G = Generator(
        dim_z=DIM_Z,
        hidden_dim=HIDDEN_DIM,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        coord_dim=COORD_DIM,
        key=key_g,
    )
    init_z = jax.random.normal(key_z, (DIM_Z,)) * 0.1
    init_W = jax.random.normal(key_w, (N_OUT, N_IN)) * 0.1
    return init_G, init_z, init_W


# --- Spectrum probe ----------------------------------------------------------
def spectrum_at(
    G: Generator,
    z: Float[Array, " dim_z"],
    *,
    k: int = 10,
    num_iterations: int = 80,
    seed: int = 7,
) -> Float[Array, " k"]:
    """Top-k singular values of ``d render_fn / d z`` at the given ``(G, z)``."""
    operator = lambda z_var: siren_render_fn(G, z_var)  # noqa: E731
    return singular_spectrum(
        operator,
        z,
        k=k,
        num_iterations=num_iterations,
        key=jax.random.key(seed),
    )


# --- Per-seed run ------------------------------------------------------------
def run_seed(seed: int, task_loss_fn, train_batch, test_batch):
    """One paired_train run + spectrum probe at init and final."""
    init_G, init_z, init_W = make_inits(seed)
    result = fws_bench.paired_train(
        init_G_params=init_G,
        init_z_params=init_z,
        render_fn=siren_render_fn,
        init_W_params=init_W,
        task_loss_fn=task_loss_fn,
        task_batch=train_batch,
        regime=fws_bench.Regime.JOINT,
        G_optimiser=optax.adam(1e-3),
        z_optimiser=optax.adam(1e-3),
        W_optimiser=optax.adam(1e-3),
        num_outer_steps=NUM_OUTER_STEPS,
    )
    final_G = result.fws_arm.final_params["G"]
    final_z = result.fws_arm.final_params["z"]
    sigmas_init = spectrum_at(init_G, init_z)
    sigmas_final = spectrum_at(final_G, final_z)
    fws_final_loss = float(result.fws_arm.loss_trajectory[-1])
    w_final_loss = float(result.w_arm.loss_trajectory[-1])
    fws_test = float(task_loss_fn(siren_render_fn(final_G, final_z), test_batch))
    w_test = float(task_loss_fn(result.w_arm.final_params, test_batch))
    any_nan = bool(
        jnp.any(jnp.isnan(result.fws_arm.loss_trajectory)) or jnp.any(jnp.isnan(result.w_arm.loss_trajectory))
    )
    fws_nan = bool(jnp.any(jnp.isnan(result.fws_arm.loss_trajectory)))
    w_nan = bool(jnp.any(jnp.isnan(result.w_arm.loss_trajectory)))
    return {
        "seed": seed,
        "fws_final": fws_final_loss,
        "w_final": w_final_loss,
        "fws_test": fws_test,
        "w_test": w_test,
        "sigma_1_init": float(sigmas_init[0]),
        "sigma_1_final": float(sigmas_final[0]),
        "sigma_min_init": float(sigmas_init[-1]),
        "sigma_min_final": float(sigmas_final[-1]),
        "cond_init": float(sigmas_init[0] / jnp.maximum(sigmas_init[-1], jnp.finfo(sigmas_init.dtype).tiny)),
        "cond_final": float(sigmas_final[0] / jnp.maximum(sigmas_final[-1], jnp.finfo(sigmas_final.dtype).tiny)),
        "any_nan": any_nan,
        "fws_nan": fws_nan,
        "w_nan": w_nan,
    }


# --- Aggregation helpers -----------------------------------------------------
def median(xs: list[float]) -> float:
    return float(jnp.median(jnp.array(xs)))


def iqr(xs: list[float]) -> float:
    arr = jnp.array(xs)
    return float(jnp.quantile(arr, 0.75) - jnp.quantile(arr, 0.25))


def main() -> None:
    print("=" * 72)
    print("Phase 2 — real ondes SIREN + loom.render, multi-seed")
    print("=" * 72)
    print(f"dim_z={DIM_Z}, hidden_dim={HIDDEN_DIM}, num_hidden_layers={NUM_HIDDEN_LAYERS}")
    print(f"num_outer_steps={NUM_OUTER_STEPS}, K_seed={K_SEED}")
    print()

    task_loss_fn, train_batch, test_batch, W_target = make_synthetic_regression()

    per_seed: list[dict] = []
    for seed in range(K_SEED):
        row = run_seed(seed, task_loss_fn, train_batch, test_batch)
        per_seed.append(row)
        print(
            f"  seed {seed}: "
            f"fws_test={row['fws_test']:.6g}  w_test={row['w_test']:.6g}  "
            f"cond_init={row['cond_init']:.4g}  cond_final={row['cond_final']:.4g}  "
            f"nan_fws={row['fws_nan']}  nan_w={row['w_nan']}"
        )

    fws_tests = [r["fws_test"] for r in per_seed]
    w_tests = [r["w_test"] for r in per_seed]
    fws_finals = [r["fws_final"] for r in per_seed]
    w_finals = [r["w_final"] for r in per_seed]
    sigma_1_inits = [r["sigma_1_init"] for r in per_seed]
    sigma_1_finals = [r["sigma_1_final"] for r in per_seed]
    sigma_min_inits = [r["sigma_min_init"] for r in per_seed]
    sigma_min_finals = [r["sigma_min_final"] for r in per_seed]
    cond_inits = [r["cond_init"] for r in per_seed]
    cond_finals = [r["cond_final"] for r in per_seed]

    print()
    print("Distribution summary (verbatim, no statistical test):")
    print(
        f"  FWS test MSE: median={median(fws_tests):.6g}  IQR={iqr(fws_tests):.6g}  min={min(fws_tests):.6g}  max={max(fws_tests):.6g}"
    )
    print(
        f"  W   test MSE: median={median(w_tests):.6g}  IQR={iqr(w_tests):.6g}  min={min(w_tests):.6g}  max={max(w_tests):.6g}"
    )
    print(f"  cond(J) init  median: {median(cond_inits):.6g}")
    print(f"  cond(J) final median: {median(cond_finals):.6g}")

    # --- Research log ------------------------------------------------------
    ondes_commit = "fa4cd96"
    loom_commit = "a0fd5b5"

    per_seed_rows = "\n".join(
        f"| {r['seed']} | {r['fws_final']:.6g} | {r['w_final']:.6g} | {r['fws_test']:.6g} | {r['w_test']:.6g} | "
        f"{r['sigma_1_init']:.4g} | {r['sigma_1_final']:.4g} | {r['sigma_min_init']:.4g} | {r['sigma_min_final']:.4g} |"
        for r in per_seed
    )

    md = f"""# Phase 2 — real SIREN renderer, multi-seed — 2026-06-06

## Setup

- Renderer: ondes `SIREN` body (hidden_dim={HIDDEN_DIM}, num_hidden_layers={NUM_HIDDEN_LAYERS}, scalar readout) with FiLM modulation; the FiLM schedule is a learned linear projection from $z$ to $(\\gamma, \\beta)$ per hidden layer. Composed through `loom.render(P, f, (G, z))` where $P$ is a one-leaf pytree shaped $(4, 8)$. Coord grid: 32 points on $[-1, 1]$ (Sitzmann+ 2020 convention), output reshaped to $(4, 8)$. dim_z = {DIM_Z}.
- ondes commit: `{ondes_commit}` (main); loom commit: `{loom_commit}` (main).
- Task, mainnet, optimisers: unchanged from phase 1 — synthetic noise-free linear regression with $W_{{\\text{{target}}}} \\in \\mathbb{{R}}^{{4 \\times 8}}$, 100 train / 50 test points, full-batch MSE, Adam(1e-3) on every optax group.
- $K_{{\\text{{seed}}}} = {K_SEED}$; num_outer_steps = {NUM_OUTER_STEPS}.

## Results

### Paired-arm final test MSE distribution

|        | FWS arm | W arm |
|--------|---------|-------|
| median | {median(fws_tests):.6g} | {median(w_tests):.6g} |
| IQR    | {iqr(fws_tests):.6g} | {iqr(w_tests):.6g} |
| min    | {min(fws_tests):.6g} | {min(w_tests):.6g} |
| max    | {max(fws_tests):.6g} | {max(w_tests):.6g} |

Final training MSE distribution (last step of trajectory):

|        | FWS arm | W arm |
|--------|---------|-------|
| median | {median(fws_finals):.6g} | {median(w_finals):.6g} |
| IQR    | {iqr(fws_finals):.6g} | {iqr(w_finals):.6g} |

(Verbatim numbers; no statistical test, no causal claim.)

### Renderer-Jacobian spectrum shift

|                                   | $\\sigma_1$ | $\\sigma_{{\\min}}$ | $\\text{{cond}}(J)$ |
|-----------------------------------|-------------|---------------------|---------------------|
| at $z_{{\\text{{init}}}}$ (median) | {median(sigma_1_inits):.4g}  | {median(sigma_min_inits):.4g}  | {median(cond_inits):.4g} |
| at $z_{{\\text{{final}}}}$ (median) | {median(sigma_1_finals):.4g} | {median(sigma_min_finals):.4g} | {median(cond_finals):.4g} |

### Per-seed table (full data)

| seed | FWS final | W final | FWS test | W test | $\\sigma_1$ init | $\\sigma_1$ final | $\\sigma_{{\\min}}$ init | $\\sigma_{{\\min}}$ final |
|------|-----------|---------|----------|--------|------------------|--------------------|---------------------------|----------------------------|
{per_seed_rows}

## Honest caveats

- Linear-regression task, not classification — will not engage the per-leaf spectral fingerprint (HT-SR / radial-FFT $\\alpha$). That is phase 3+ work with a multi-leaf SiLU MLP.
- Identity $P$ everywhere — Householder reflections on conv kernels are also phase 3+; no conv leaves yet.
- {NUM_OUTER_STEPS} outer steps — convergence not asserted.
- $K_{{\\text{{seed}}}} = {K_SEED}$ gives distribution shape but is below the falsifier-convention floor (K=5 minimum for paired Wilcoxon, K=20 for confidence). No statistical comparison made.
- All numbers verbatim; no thresholds; no "FWS won/lost" claim. Just the matrix.

## What this demonstrates

End-to-end integration of the real ondes SIREN basis body plus loom's `render` primitive with fws-bench's `paired_train` harness and landscape-archaeology's `singular_spectrum` probe. The reparameterisation is now genuine — not a `tanh` stand-in. Whether it reshapes the loss landscape in a useful way is what phase 3+ measurement sweeps are for.
"""

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_FILE.write_text(md)
    print()
    print(f"Wrote research log to {RESEARCH_FILE}")


if __name__ == "__main__":
    main()
