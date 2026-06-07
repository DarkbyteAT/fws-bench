"""Phase 10 — Polar-projection ``P`` + per-leaf G_H normalisation.

Three folded changes from phase 9, all attacking phase 9's K=3 finding
that the FWS-hyper-recursive arm sat at chance (10.0%) on CIFAR-10
despite stage-0 clearing the kill rule by a wide margin (log10 spread
2.24 OoM > 1.0 threshold). Stage-0 detected geometry; phase 10 tests
whether the geometry was just *useless* — i.e. produced a $W$ tree that
didn't carry classification signal.

1. **G_H linear readout (drop output sine).** Phase 9's ``G_H`` ended
   with ``flat = sin(omega_0 · (W_out · h + b_out))`` — every layer
   sine-activated, including the output. Squashing the produced
   ``G_leaf`` parameter vector to ``[-1, 1]`` meant every leaf-renderer's
   first-layer SIREN weights were uniformly random in ``[-1, 1]``, so
   every leaf initialised to the same near-zero, low-frequency rendering
   regardless of leaf-id. Phase 10 removes the final sine: ``flat =
   W_out · h + b_out`` (linear). ``G_H``'s body is still SIREN; only the
   readout changes.

2. **Per-leaf normalisation of G_H's output slice (NEW)**. The
   companion phase 10 (linear-readout + Kaiming-per-leaf-on-rendered-W
   at fws-bench commit 42384fd) shipped K=3: FWS-parallel-no-G_H went
   from chance to 29.2%, a robust positive FWS signal, but FWS-hyper
   stayed at chance. Diagnosis: the Kaiming pre-factor corrected the
   *rendered W*, but the *SIREN parameters of each leaf-renderer*
   (G_leaf's W_in / b_in / W_h / b_h / W_out / b_out) still came
   straight from G_H's linear output with no per-leaf magnitude
   information — every leaf-renderer received parameters at the same
   global scale. Phase 10 (this file) z-scores each leaf's slice of
   G_H's flat output to mean 0 / std sqrt(2/slice_size) before
   unpacking into G_leaf params. The slice becomes Kaiming-statistic
   per leaf, decoupled from G_H's global output drift.

3. **Polar-decomposition pseudo-orthonormal projection ``P``.** The
   rendered tensor for each weight leaf is projected via Newton-Schulz
   iteration (``A ← 0.5 · A · (3·I − Aᵀ·A)``, 5 steps, Frobenius
   pre-normalised) to its nearest semi-orthogonal matrix: spectral
   norm = 1 by construction. Biases are normalised to a small constant
   std. Differentiable through standard autodiff (NS avoids the
   SVD-gradient collision pathology near init). Applied AFTER the
   per-leaf Kaiming scale.

Combined, the three fixes attack the FWS-hyper pathology along three
independent axes: free G_H's output range (1), give each leaf the right
parameter scale (2), and constrain the final rendered tensor to
well-conditioned magnitudes (3).

Stage-0 falsifier (unchanged kill rule): 3 ``G_H`` output-init scale
choices, ``sigma_min(d render / d z)`` at steps ``{0, 100, 1000, 5000}``,
log10 spread < 1.0 OoM at step 5000 → STOP.

With ``G_H``'s output layer now linear, the SIREN ``1 / (omega_0 ·
sqrt(in_fan))`` is no longer paper-faithful. The canonical init for a
linear projection is Xavier ``1 / sqrt(in_fan)``. We rename the middle
scale to ``linear_xavier`` for clarity.

Run::

    PHASE10_STAGE=falsifier uv run python examples/phase10_polar_projection.py
    PHASE10_STAGE=k3        uv run python examples/phase10_polar_projection.py
    PHASE10_STAGE=all       uv run python examples/phase10_polar_projection.py   # default
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from landscape_archaeology import singular_spectrum

from fws_bench import StageZeroVerdict, paired_train_4arm, stage0_falsifier


sys.path.insert(0, str(Path(__file__).parent))
from _common import arms, data, diagnostics, mainnet, reporting  # noqa: E402
from _common.arms import (  # noqa: E402
    DIM_Z,
    G_LEAF_HIDDEN_DIM,
    G_LEAF_PARAM_SIZE,
    MAX_G_LEAF_PARAM_SIZE,
    OMEGA_0,
    kaiming_leaf_scale,
    slice_g_leaf_flat,
)


RESEARCH_DIR = Path("/Users/ammar/Documents/Research/fractal-weight-spaces/research")
FIGURES_DIR = RESEARCH_DIR / "figures"
RESEARCH_FILE = RESEARCH_DIR / "2026-06-07-phase10-polar-projection.md"


# --- Phase-specific G_H: SIREN body + linear readout ------------------------
DIM_LEAF_EMB = 8
G_H_HIDDEN_DIM = 100
G_H_DEPTH = 3                    # sine layers + 1 linear readout


def g_h_w_out_scale(scale_kind: str) -> float:
    """Output-layer (linear readout's ``W_out``) init scale.

    With phase 10's linear readout, canonical Xavier ``1 / sqrt(in_fan)``
    is the paper-faithful choice; the three Stage-0 scales test whether
    deviating from canonical Xavier matters.

    - ``"existing"``: ``0.1 / sqrt(G_H_HIDDEN_DIM)``, heuristic carry-over.
    - ``"linear_xavier"``: ``1 / sqrt(G_H_HIDDEN_DIM)``, canonical Xavier.
    - ``"linear_xavier_10x"``: ``10 × linear_xavier``, over-scaled control.
    """
    sqrt_H = float(jnp.sqrt(jnp.array(G_H_HIDDEN_DIM, dtype=jnp.float32)))
    if scale_kind == "existing":
        return 0.1 / sqrt_H
    if scale_kind == "linear_xavier":
        return 1.0 / sqrt_H
    if scale_kind == "linear_xavier_10x":
        return 10.0 / sqrt_H
    raise ValueError(f"unknown scale_kind={scale_kind!r}")


SCALE_KINDS: tuple[str, ...] = ("existing", "linear_xavier", "linear_xavier_10x")


class HyperRenderer(eqx.Module):
    """Phase-10 ``G_H``: SIREN body + LINEAR readout.

    Forward at one ``(z, emb)``:

    .. math::
        h_0 &= \\sin(\\omega_0 \\cdot (W_{in} \\cdot [z;\\,emb] + b_{in})) \\\\
        h_k &= \\sin(\\omega_0 \\cdot (W_{h,k} \\cdot h_{k-1} + b_{h,k}))
                \\quad \\text{for } k = 1 \\ldots G\\_H\\_DEPTH - 2 \\\\
        \\text{flat} &= W_{out} \\cdot h_{G\\_H\\_DEPTH-2} + b_{out}

    Hidden layers SIREN-activated; output layer plain linear. The
    produced flat parameter vector is no longer squashed to ``[-1, 1]``,
    so each leaf-renderer's SIREN parameters can span O(1) freely.
    """

    leaf_embedding: Float[Array, "n_leaves dim_leaf_emb"]
    W_in: Float[Array, "hidden in_total"]
    b_in: Float[Array, " hidden"]
    W_h: tuple[Float[Array, "hidden hidden"], ...]
    b_h: tuple[Float[Array, " hidden"], ...]
    W_out: Float[Array, "g_leaf_param_size hidden"]
    b_out: Float[Array, " g_leaf_param_size"]

    def __init__(self, *, key: Array, out_scale_kind: str = "linear_xavier") -> None:
        keys = jax.random.split(key, 5 + 2 * (G_H_DEPTH - 2))
        k_emb, k_w_in, k_b_in, k_w_out, k_b_out, *k_hidden = keys
        in_total = DIM_Z + DIM_LEAF_EMB

        emb_bound = 1.0 / jnp.sqrt(jnp.array(DIM_LEAF_EMB, dtype=jnp.float32))
        self.leaf_embedding = jax.random.uniform(
            k_emb, (mainnet.N_LEAVES, DIM_LEAF_EMB), minval=-emb_bound, maxval=emb_bound,
        )

        # First layer: SIREN first-layer init (bound = 1 / in_total).
        bound_first = 1.0 / in_total
        self.W_in = jax.random.uniform(k_w_in, (G_H_HIDDEN_DIM, in_total),
                                       minval=-bound_first, maxval=bound_first)
        self.b_in = jax.random.uniform(k_b_in, (G_H_HIDDEN_DIM,),
                                       minval=-bound_first, maxval=bound_first)

        # Hidden sine layers: SIREN subsequent-layer init (bound = sqrt(6/H)/omega).
        bound_hidden = float(jnp.sqrt(jnp.array(6.0 / G_H_HIDDEN_DIM, dtype=jnp.float32))) / OMEGA_0
        n_hidden = G_H_DEPTH - 2
        W_h_list: list[Array] = []
        b_h_list: list[Array] = []
        for j in range(n_hidden):
            kw, kb = k_hidden[2 * j], k_hidden[2 * j + 1]
            W_h_list.append(jax.random.uniform(
                kw, (G_H_HIDDEN_DIM, G_H_HIDDEN_DIM), minval=-bound_hidden, maxval=bound_hidden))
            b_h_list.append(jax.random.uniform(
                kb, (G_H_HIDDEN_DIM,), minval=-bound_hidden, maxval=bound_hidden))
        self.W_h = tuple(W_h_list)
        self.b_h = tuple(b_h_list)

        # Output (LINEAR) projection: scale chosen by the Stage-0 falsifier.
        bound_out = g_h_w_out_scale(out_scale_kind)
        self.W_out = jax.random.uniform(k_w_out, (MAX_G_LEAF_PARAM_SIZE, G_H_HIDDEN_DIM),
                                        minval=-bound_out, maxval=bound_out)
        self.b_out = jax.random.uniform(k_b_out, (MAX_G_LEAF_PARAM_SIZE,),
                                        minval=-bound_out, maxval=bound_out)

    def produce_flat(self, z: Array, leaf_id: int) -> Array:
        emb = self.leaf_embedding[leaf_id]
        inp = jnp.concatenate([z, emb])
        h = jnp.sin(OMEGA_0 * (self.W_in @ inp + self.b_in))
        for W, b in zip(self.W_h, self.b_h, strict=True):
            h = jnp.sin(OMEGA_0 * (W @ h + b))
        return self.W_out @ h + self.b_out

    def produce(self, z: Array, leaf_id: int, rank: int) -> dict[str, Array]:
        # Phase 10 fix 3 (added 2026-06-07): per-leaf normalisation of G_H's
        # output slice BEFORE unpacking into G_leaf params. Without this,
        # G_H's linear readout produces a globally-scaled flat vector, so
        # each leaf-renderer's W_in / b_in / W_h / b_h / W_out / b_out get
        # the same magnitude regardless of leaf identity. Per-leaf
        # normalisation z-scores the slice to mean 0 / std sqrt(2/slice_size)
        # — Kaiming-like statistics on the produced parameter vector,
        # decoupled from G_H's global output drift.
        flat = self.produce_flat(z, leaf_id)
        slice_size = G_LEAF_PARAM_SIZE[rank]
        leaf_slice = flat[:slice_size]
        eps = jnp.finfo(leaf_slice.dtype).eps * slice_size
        mean = jnp.mean(leaf_slice)
        std = jnp.std(leaf_slice) + eps
        target_std = jnp.sqrt(jnp.array(2.0 / slice_size, dtype=leaf_slice.dtype))
        normalised = (leaf_slice - mean) / std * target_std
        return slice_g_leaf_flat(normalised, rank)


# --- Polar-decomposition pseudo-orthonormal projection ----------------------
# Newton-Schulz iteration W ← 0.5 · W · (3·I − Wᵀ·W) converges to the
# orthogonal polar factor when ‖W‖_2 < sqrt(3). We pre-normalise by the
# Frobenius norm (a safe upper bound on the spectral norm) and run a fixed
# 5 iterations. NS is fully differentiable through standard autodiff and
# avoids the SVD-gradient collision pathology at near-zero init.
NS_ITERATIONS: int = 5


def project_pseudo_orthonormal(W: Array, leaf_name: str) -> Array:
    """Polar projection via Newton-Schulz iteration: spectral norm = 1.

    For 2+D weight leaves: reshape to ``(out, prod(rest))``, pre-normalise
    by Frobenius norm, run ``NS_ITERATIONS`` Newton-Schulz steps, reshape
    back. The result has all singular values ≈ 1 (Frobenius
    ≈ ``sqrt(min(out, rest))``).

    For 1-D bias leaves: standardise to a small constant std.
    """
    del leaf_name  # accepted to fit projection_fn signature
    if W.ndim == 1:
        eps = jnp.finfo(W.dtype).eps * 16
        std = jnp.std(W) + eps
        return W * 0.01 / std

    original_shape = W.shape
    A = W.reshape(original_shape[0], -1)

    # Pre-normalise: Frobenius norm upper-bounds the spectral norm, so
    # A / ‖A‖_F has all singular values in (0, 1], inside the NS convergence
    # ball ‖A‖_2 < sqrt(3). Adding eps keeps the gradient finite when A=0.
    eps = jnp.finfo(A.dtype).eps * A.size
    A = A / (jnp.linalg.norm(A) + eps)

    # Newton-Schulz: A_{k+1} = 0.5 · A_k · (3·I − A_kᵀ·A_k)
    eye = jnp.eye(A.shape[1], dtype=A.dtype)
    for _ in range(NS_ITERATIONS):
        AtA = A.T @ A
        A = 0.5 * A @ (3.0 * eye - AtA)

    return A.reshape(original_shape)


# --- Phase-specific schedule ------------------------------------------------
EPOCHS = int(os.environ.get("PHASE10_EPOCHS", 5))
K_SEED = int(os.environ.get("PHASE10_K_SEED", 3))
BATCH_SIZE = 128
FALSIFIER_STEPS = int(os.environ.get("PHASE10_FALSIFIER_STEPS", 5000))
FALSIFIER_CHECKPOINTS: tuple[int, ...] = (0, 100, 1000, FALSIFIER_STEPS)

SIGMA_TOP_K = 5
SIGMA_POWER_ITERS = 10
HESSIAN_POWER_ITERS = 5
EVAL_HESS_BATCH = 1024


# --- CIFAR-mainnet eval helpers (passed to fws_bench.paired_train_4arm) ----
def _eval_test_loss_acc(
    params: dict[str, Array], test_x: np.ndarray, test_y: np.ndarray,
) -> tuple[float, float]:
    chunk = 1024
    n = test_x.shape[0]
    total_loss = 0.0
    total_correct = 0
    for i in range(0, n, chunk):
        bx = jnp.asarray(test_x[i:i + chunk])
        by = jnp.asarray(test_y[i:i + chunk])
        logits = jax.vmap(lambda x: mainnet.cnn_forward(params, x))(bx)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        one_hot = jax.nn.one_hot(by, mainnet.NUM_CLASSES)
        loss = -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))
        preds = jnp.argmax(logits, axis=-1)
        total_loss += float(loss) * bx.shape[0]
        total_correct += int(jnp.sum(preds == by))
    return total_loss / n, total_correct / n


def _all_test_preds(params: dict[str, Array], test_x: np.ndarray) -> np.ndarray:
    chunk = 1024
    n = test_x.shape[0]
    preds = np.empty(n, dtype=np.int32)
    for i in range(0, n, chunk):
        bx = jnp.asarray(test_x[i:i + chunk])
        logits = jax.vmap(lambda x: mainnet.cnn_forward(params, x))(bx)
        preds[i:i + chunk] = np.asarray(jnp.argmax(logits, axis=-1))
    return preds


def make_fws_hyper_arm(out_scale_kind: str = "linear_xavier") -> arms.Arm:
    return arms.make_fws_hyper(
        g_h_init=lambda key: HyperRenderer(key=key, out_scale_kind=out_scale_kind),
        leaf_scale_fn=kaiming_leaf_scale,
        projection_fn=project_pseudo_orthonormal,
        name="FWS-hyper-polar",
        short="fws_hyper",
    )


def make_arms_list() -> list[arms.Arm]:
    return [
        make_fws_hyper_arm("linear_xavier"),
        arms.make_fws_parallel(
            leaf_scale_fn=kaiming_leaf_scale,
            projection_fn=project_pseudo_orthonormal,
        ),
        arms.make_w_matched(),
        arms.make_w_overparam(),
    ]


# --- σ-probes --------------------------------------------------------------
_HYPER_RENDER = arms.make_fws_hyper(
    g_h_init=lambda _k: None,  # type: ignore[arg-type]
    leaf_scale_fn=kaiming_leaf_scale,
    projection_fn=project_pseudo_orthonormal,
).render_W
_PARALLEL_RENDER = arms.make_fws_parallel(
    leaf_scale_fn=kaiming_leaf_scale,
    projection_fn=project_pseudo_orthonormal,
).render_W


def sigma_at_z_hyper(state: dict) -> Array:
    op = lambda z_var: _HYPER_RENDER({"G": state["G"], "z": z_var})  # noqa: E731
    return singular_spectrum(op, state["z"],
                             k=SIGMA_TOP_K, num_iterations=SIGMA_POWER_ITERS,
                             key=jax.random.key(7))


def sigma_at_gh_hyper(state: dict) -> Array:
    op = lambda G_var: _HYPER_RENDER({"G": G_var, "z": state["z"]})  # noqa: E731
    return singular_spectrum(op, state["G"],
                             k=SIGMA_TOP_K, num_iterations=SIGMA_POWER_ITERS,
                             key=jax.random.key(13))


def sigma_at_z_parallel(state: dict) -> Array:
    op = lambda z_var: _PARALLEL_RENDER({"P": state["P"], "z": z_var})  # noqa: E731
    return singular_spectrum(op, state["z"],
                             k=SIGMA_TOP_K, num_iterations=SIGMA_POWER_ITERS,
                             key=jax.random.key(7))


_shared_train_x: np.ndarray
_shared_train_y: np.ndarray


def per_seed_diagnostics(seed: int, states: dict, final_Ws: dict[str, dict[str, Array]]) -> dict:
    """Compute σ-spectra, Hessian top-eigs, G_leaf cosines + params at convergence."""
    rng = np.random.default_rng(seed)
    fws_state = states["fws_hyper"]
    par_state = states["fws_parallel"]

    sigma_z_hyper = np.asarray(sigma_at_z_hyper(fws_state))
    sigma_gh_hyper = np.asarray(sigma_at_gh_hyper(fws_state))
    sigma_z_par = np.asarray(sigma_at_z_parallel(par_state))

    g_leaf_cosines = diagnostics.g_leaf_cosine_matrix(
        fws_state["G"].produce_flat, fws_state["z"],
    )
    g_leaf_params = {
        name: {
            k: np.asarray(v) for k, v in
            fws_state["G"].produce(fws_state["z"], mainnet.LEAF_ORDER.index(name), mainnet.LEAF_RANKS[name]).items()
        }
        for name in mainnet.LEAF_ORDER
    }

    hess: dict[str, float] = {}
    hess_idx = rng.choice(_shared_train_x.shape[0], size=EVAL_HESS_BATCH, replace=False)
    hess_batch = {"x": jnp.asarray(_shared_train_x[hess_idx]),
                  "y": jnp.asarray(_shared_train_y[hess_idx])}
    for short, W in final_Ws.items():
        grad_fn = jax.grad(lambda p, b=hess_batch: mainnet.cross_entropy_loss(p, b))
        hess[short] = float(np.asarray(singular_spectrum(
            grad_fn, W, k=SIGMA_TOP_K, num_iterations=HESSIAN_POWER_ITERS,
            key=jax.random.key(11)))[0])

    return {
        "sigma_z_hyper": sigma_z_hyper,
        "sigma_gh_hyper": sigma_gh_hyper,
        "sigma_z_par": sigma_z_par,
        "g_leaf_cosines": g_leaf_cosines,
        "g_leaf_params": g_leaf_params,
        **{f"hess_{k}": v for k, v in hess.items()},
    }


def main() -> None:
    global _shared_train_x, _shared_train_y
    stage = os.environ.get("PHASE10_STAGE", "all")

    print("=" * 72)
    print(f"Phase 10 — Polar-projection P  (stage={stage})")
    print("=" * 72)
    print("Loading CIFAR-10 ...")
    train_x, train_y, test_x, test_y = data.load_cifar10()
    raw_test_x = data.load_raw_test()
    print(f"  train: {train_x.shape} {train_y.shape} | test: {test_x.shape} {test_y.shape}")
    _shared_train_x, _shared_train_y = train_x, train_y

    arms_list = make_arms_list()
    counts: dict[str, int] = {}
    for a in arms_list:
        state = a.init(jax.random.key(0))
        counts[a.short] = diagnostics.count_params(state)
    print(f"\nG_leaf param size per rank: {G_LEAF_PARAM_SIZE}  (max={MAX_G_LEAF_PARAM_SIZE})")
    print(f"G_H hidden={G_H_HIDDEN_DIM}, depth={G_H_DEPTH} (SIREN+linear), z={DIM_Z}, "
          f"leaf_emb=({mainnet.N_LEAVES},{DIM_LEAF_EMB})")
    for a in arms_list:
        print(f"  {a.name:32s}: {counts[a.short]}")
    print(f"K_seed={K_SEED}, epochs={EPOCHS}, batch={BATCH_SIZE}, "
          f"falsifier_steps={FALSIFIER_STEPS}")
    print()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

    stage0_verdict: StageZeroVerdict | None = None
    if stage in ("falsifier", "all"):
        t0 = time.time()
        stage0_verdict = stage0_falsifier(
            arm_factory=lambda sk: make_fws_hyper_arm(sk),
            scale_kinds=SCALE_KINDS,
            train_x=train_x, train_y=train_y,
            sigma_at_z=sigma_at_z_hyper,
            num_steps=FALSIFIER_STEPS,
            checkpoints=FALSIFIER_CHECKPOINTS,
            threshold_oom=1.0,
            seed=0,
        )
        print(f"Stage 0 wall: {time.time() - t0:.1f}s")
        reporting.plot_stage0(
            stage0_verdict.records,
            FIGURES_DIR / "2026-06-07-phase10-polar-stage0-falsifier.png",
            title="Phase 10 — G_H W_out init-scale falsifier (linear readout + polar P)",
            scale_bounds={k: g_h_w_out_scale(k) for k in SCALE_KINDS},
        )

    per_seed: list[dict] = []
    nan_or_crash: list[str] = []
    k3_wall = 0.0
    if stage in ("k3", "all"):
        if stage0_verdict is not None and not stage0_verdict.proceed and stage == "all":
            print("\nSTOP — Stage 0 verdict says init recovery, not FWS prior. K=3 skipped.")
        else:
            per_seed, nan_or_crash, k3_wall = paired_train_4arm(
                arms=arms_list,
                train_x=train_x, train_y=train_y,
                eval_fn=lambda W: _eval_test_loss_acc(W, test_x, test_y),
                predict_fn=lambda W: _all_test_preds(W, test_x),
                num_epochs=EPOCHS, K_seed=K_SEED,
                batch_size=BATCH_SIZE,
                per_seed_diagnostics=per_seed_diagnostics,
            )

    summary_md = ""
    if per_seed:
        for a in arms_list:
            accs = np.array([r[f"final_{a.short}_acc"] for r in per_seed])
            m, iq, lo, hi = reporting.median_iqr_min_max(accs)
            print(f"  {a.name:32s}: median={m:.4f} IQR={iq:.4f} min={lo:.4f} max={hi:.4f}")
        summary_md = reporting.acc_summary_md_table(per_seed, arms_list, K_SEED, EPOCHS)

        reporting.plot_loss_trajectories(per_seed, arms_list,
                                         FIGURES_DIR / "2026-06-07-phase10-polar-loss-trajectories.png",
                                         title=f"Phase 10 — training-loss trajectories (K={K_SEED})")
        reporting.plot_acc_trajectories(per_seed, arms_list,
                                        FIGURES_DIR / "2026-06-07-phase10-polar-acc-trajectories.png",
                                        title=f"Phase 10 — test-accuracy trajectories (K={K_SEED})")
        reporting.plot_htsr_box(per_seed, arms_list,
                                FIGURES_DIR / "2026-06-07-phase10-polar-htsr-alpha-boxplot.png",
                                title=f"Phase 10 — HT-SR α on fc leaves (K={K_SEED})")
        reporting.plot_fft_box(per_seed, arms_list,
                               FIGURES_DIR / "2026-06-07-phase10-polar-radial-fft-alpha-boxplot.png",
                               title=f"Phase 10 — radial-FFT α on conv leaves (K={K_SEED}; k=5 under-recovery caveat)")
        reporting.plot_final_acc_box(per_seed, arms_list,
                                     FIGURES_DIR / "2026-06-07-phase10-polar-final-acc-boxplot.png",
                                     title=f"Phase 10 — final test accuracy (K={K_SEED}, {EPOCHS} epochs)")
        reporting.plot_jacobian_spectrum(
            per_seed,
            FIGURES_DIR / "2026-06-07-phase10-polar-jacobian-spectrum.png",
            title=f"Phase 10 — Jacobian σ-spectrum at final, top-{SIGMA_TOP_K} (K={K_SEED})",
            sigma_keys={
                "hyper: σ(∂render/∂z)": "sigma_z_hyper",
                "hyper: σ(∂render/∂G_H)": "sigma_gh_hyper",
                "parallel: σ(∂render/∂z)": "sigma_z_par",
            },
        )
        for a in arms_list:
            reporting.plot_conv_kernels(
                per_seed, a,
                FIGURES_DIR / f"2026-06-07-phase10-polar-conv-kernels-{a.short}.png",
                title=f"Phase 10 — {a.name} conv1 kernels (8 × 5×5 RGB, per-filter normalised)",
            )
            reporting.plot_arm_curation(
                per_seed, a, test_y, raw_test_x,
                FIGURES_DIR / f"2026-06-07-phase10-polar-curations-{a.short}.png",
                title=f"Phase 10 — {a.name} curation (best/median/worst by final test acc, K={K_SEED})",
                class_names=data.CIFAR_NAMES,
            )
        reporting.plot_fws_g_leaf_panel(
            per_seed,
            FIGURES_DIR / "2026-06-07-phase10-polar-fws-hyper-leaves.png",
            title="Phase 10 — per-leaf G_leaf params (FWS-hyper median seed {seed})",
            hidden_dim=G_LEAF_HIDDEN_DIM,
        )
        reporting.plot_g_leaf_cosine(
            per_seed,
            FIGURES_DIR / "2026-06-07-phase10-polar-g_leaf-cosine.png",
            title="FWS-hyper: pairwise G_leaf cosine (median seed {seed})",
        )

    stage0_md = ""
    if stage0_verdict is not None:
        stage0_md = reporting.stage0_md_table(
            stage0_verdict.records, stage0_verdict.checkpoints, stage0_verdict.text,
            FALSIFIER_STEPS,
            figure_link="figures/2026-06-07-phase10-polar-stage0-falsifier.png",
        )

    nan_md = "\n".join(f"- {line}" for line in nan_or_crash) if nan_or_crash else "- none."

    sections: list[tuple[str, str]] = [
        ("Background", _BACKGROUND_MD),
        ("Architecture (changes vs phase 9)", _ARCHITECTURE_MD),
    ]
    if stage0_md:
        sections.append(("Stage 0 — G_H W_out init-scale falsifier (BLOCKING)", stage0_md))
    if summary_md:
        figures_md = "\n\n".join([
            "![loss trajectories](figures/2026-06-07-phase10-polar-loss-trajectories.png)",
            "![test accuracy trajectories](figures/2026-06-07-phase10-polar-acc-trajectories.png)",
            "![HT-SR α (fc leaves)](figures/2026-06-07-phase10-polar-htsr-alpha-boxplot.png)",
            "![radial-FFT α (conv leaves)](figures/2026-06-07-phase10-polar-radial-fft-alpha-boxplot.png)",
            "![final test accuracy](figures/2026-06-07-phase10-polar-final-acc-boxplot.png)",
            "![Jacobian σ-spectrum at convergence](figures/2026-06-07-phase10-polar-jacobian-spectrum.png)",
            "### Curated outputs",
            *[f"![{a.name} conv1 kernels](figures/2026-06-07-phase10-polar-conv-kernels-{a.short}.png)"
              for a in arms_list],
            *[f"![{a.name} curation](figures/2026-06-07-phase10-polar-curations-{a.short}.png)"
              for a in arms_list],
            "![Per-leaf G_leaf params (FWS-hyper, median seed)](figures/2026-06-07-phase10-polar-fws-hyper-leaves.png)",
            "![FWS-hyper pairwise G_leaf cosine](figures/2026-06-07-phase10-polar-g_leaf-cosine.png)",
        ])
        sections.append((
            f"Stage 1 — K={K_SEED} fiducial cell",
            f"{summary_md}\n\n{figures_md}",
        ))
    sections.append(("NaN / crash report", nan_md))
    sections.append(("Honest caveats", _CAVEATS_MD.format(K_SEED=K_SEED, EPOCHS=EPOCHS,
                                                          SIGMA_TOP_K=SIGMA_TOP_K,
                                                          EVAL_HESS_BATCH=EVAL_HESS_BATCH,
                                                          k3_wall=k3_wall)))
    sections.append(("What's next", _NEXT_MD))

    reporting.render_research_log(reporting.ReportContext(
        title="Phase 10 — Polar-projection $P$ — 2026-06-07",
        out_path=RESEARCH_FILE,
        sections=sections,
        figures_dir=FIGURES_DIR,
    ))
    reporting.render_pdf(RESEARCH_FILE)


_BACKGROUND_MD = """
Phase 9 (recursive SIREN) cleared the Stage-0 falsifier (log10 spread
2.24 OoM ≫ 1.0 threshold) — yet the FWS-hyper-recursive arm landed at
chance (10.0%) on the CIFAR-10 classification task. The companion phase
10 build (linear-readout + Kaiming-per-leaf-on-rendered-W at fws-bench
commit 42384fd) shipped K=3 results: FWS-parallel-no-$G_H$ went from
17.91% to 29.20% (+11.3 pp), the first robust positive FWS signal
across ten phases, but FWS-hyper-rec stayed at chance (9.14%). Phase
10's phase 8 agent diagnosed why: the Kaiming pre-factor corrected the
rendered $W$, but each leaf-renderer's SIREN parameters (G_leaf's
$W_{\\text{in}}$, $b_{\\text{in}}$, $W_h$, $b_h$, $W_{\\text{out}}$,
$b_{\\text{out}}$) still came straight from G_H's linear output with no
per-leaf magnitude information — every leaf-renderer received
parameters at the same global scale.

Phase 10 (this phase) attacks the FWS-hyper-specific pathology along
three independent axes simultaneously: free G_H's output range
(linear readout), give each leaf the right SIREN parameter scale (G_H
output per-leaf normalisation), and constrain the final rendered tensor
to well-conditioned magnitudes (Newton-Schulz polar projection $P$).
"""

_ARCHITECTURE_MD = """
Three changes from phase 9, applied to the FWS-hyper arm; the
FWS-parallel arm gets changes (2) and (3) only (it has no $G_H$):

1. **$G_H$ linear readout.** Phase 9 ended ``G_H``'s forward with
   ``flat = sin(omega_0 · (W_out · h + b_out))``. Phase 10 drops the
   final sine: ``flat = W_out · h + b_out``. The hidden layers stay
   SIREN-activated. (FWS-hyper only.)

2. **Per-leaf normalisation of G_H's output slice.** After producing
   ``flat``, each leaf's ``G_LEAF_PARAM_SIZE[rank]``-length slice is
   z-scored to mean 0, std ``sqrt(2 / slice_size)`` — Kaiming statistics
   on the produced parameter vector, decoupled from G_H's global output
   drift. This addresses the diagnosis that G_H's linear readout feeds
   every leaf-renderer parameters at the same global scale. (FWS-hyper
   only; FWS-parallel's per-rank ``G_leaf`` instances are already
   SIREN-initialised.)

3. **Polar-decomposition pseudo-orthonormal projection ``P``.** The
   rendered tensor (after the per-leaf Kaiming pre-factor) is projected
   to its nearest semi-orthogonal matrix via Newton-Schulz iteration
   (``A ← 0.5 · A · (3·I − Aᵀ·A)``, 5 steps, Frobenius pre-normalised);
   the result has all singular values ≈ 1. Biases are normalised to a
   small constant std. Differentiable through standard autodiff (NS
   avoids the SVD-gradient collision pathology at near-zero init).
   (Applied to both FWS arms.)

The Stage-0 falsifier shape is unchanged: 3 ``G_H`` output-init scales,
``sigma_min(d render / d z)`` at ``{0, 100, 1000, 5000}``, log10 spread
< 1.0 OoM at step 5000 → STOP. The middle scale is renamed to
``linear_xavier`` (= ``1 / sqrt(G_H_HIDDEN_DIM)``) because the readout
is now linear, not sine-following.
"""

_CAVEATS_MD = """
- **K={K_SEED} smoke** — not K=10 confirmation.
- **{EPOCHS} epochs only.**
- **Polar projection is Newton-Schulz (5 iterations), not exact SVD.**
  Initial SVD-based polar produced NaN gradients at the σ-probe at init
  (near-zero G_leaf outputs → near-zero singular-value collisions in the
  SVD gradient). NS converges to all-σ-≈-1 in 5 steps for
  well-conditioned inputs (Frobenius pre-normalisation keeps inputs
  inside the convergence ball $\\|A\\|_2 < \\sqrt{{3}}$).
- **Per-leaf G_H output normalisation is parameter-free** — z-score the
  slice to mean 0, std sqrt(2/slice_size). Not LayerNorm with learnable
  scale (which would re-introduce a per-leaf parameter trained against
  the task loss). Future ablation: LayerNorm with learnable scale vs
  parameter-free z-score.
- **Bias projection is *not* orthonormal** — it's a std-renormalisation
  step. Treat bias projection as a numerical sanity, not a structural
  constraint.
- **σ probes are top-{SIGMA_TOP_K} power iteration with reduced iters.**
- **Hessian top eigenvalue** computed on a {EVAL_HESS_BATCH}-image subsample.
- **Wall-clock budget**: stage 1 K=3 ≈ {k3_wall:.1f}s.
"""

_NEXT_MD = """
Two possible Stage-0 outcomes:

- **STOP (init recovery)**: even polar-projection P + per-leaf Kaiming
  scale doesn't free $\\sigma_{\\min}$ from init-recovery dynamics. The
  per-leaf hyper-renderer family is then mechanistically equivalent to
  direct $W$ training regardless of how cleverly we constrain the
  rendering — pivot to the supervised-distillation framing.
- **PROCEED**: K=3 then K=10 confirmation. If FWS-hyper-polar matches or
  exceeds W matched at fixed budget, the polar projection is the
  load-bearing structural choice, and Newton-Schulz / lower-rank
  variants become the next ablation axis.
"""


if __name__ == "__main__":
    main()
