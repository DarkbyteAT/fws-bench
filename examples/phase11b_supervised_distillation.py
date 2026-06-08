"""Phase 11B — Supervised-distillation sanity check on FWS architectures.

Phases 6 / 8 / 9 / 10 left a representation-vs-dynamics ambiguity. The
FWS-hyper arm stayed at chance across every phase in which it ran
end-to-end, and FWS-parallel got off chance only at phase 10 (≈29% on
CIFAR-10). Two hypotheses are consistent with that record:

- **Representation.** The per-leaf hyper-renderer family — the rendered
  $W$ tree as a function of $(z, G_H)$ — cannot express the kind of
  weight tensor a trained CIFAR classifier needs. Even with a perfect
  outer optimiser the FWS arm would land far from any
  classifier-quality $W$.
- **Training dynamics.** The family can express trained weights, but the
  cross-entropy outer landscape (through softmax → CNN forward →
  rendering chain) is hostile to gradient descent at the scales we ran.

**Literature anchoring.** NeRN (Ashkenazi et al., arXiv:2212.13554,
2023) is the direct prior art: a coordinate→weight predictor trained
by L2 distillation against a fixed CIFAR/ImageNet classifier, with
explicit accuracy-preservation evaluation. Phase 11B largely
replicates NeRN's setup. The fws-bench novelty axis is **sine-only
SIREN backbone + polar projection $P$ on the latent** (NeRN uses an
MLP and imposes no geometric constraint on the latent code).

Both NeRN AND Schürholt et al. (Hyper-Representations,
arXiv:2209.14733, 2022) report a known failure mode of leaf-uniform
L2: low MSE, near-zero task accuracy, because weight distributions
vary by 1–2 OoM across layer types (fc ∼ 0.1, conv ∼ 0.01, bias ≈ 0).
Uniform L2 underweights small-std layers; tiny absolute perturbations
to those layers destroy classification accuracy. Schürholt's fix is
**layer-wise normalisation**: scale each leaf's contribution to the
loss by its target's standard deviation. Phase 11B runs both losses
as paired cells so the verdict is read from the 4-way grid below
rather than from a single number.

Phase 11B's design:

1. train a vanilla ``WideKernelCNN-SiLU`` on CIFAR-10 with Adam(1e-3)
   for 5 epochs — this hits ≈60% test accuracy and produces a single
   target weight tree $W^{\\star}$. No smoothness regulariser on
   $W^{\\star}$ (NeRN imposes one; we leave it off for the first-pass
   diagnostic).
2. for each FWS arm (FWS-hyper-polar with the phase 10 architecture;
   FWS-parallel-no-G_H with polar $P$), train the arm's parameters by
   Adam(1e-3) for 5000 outer steps under **each** of two losses:

   - **leaf-mean**:
     $\\sum_{\\ell} \\text{mean}((W_{\\ell}(\\text{state}) -
     W^{\\star}_{\\ell})^2)$ — leaf-uniform.
   - **layer-wise normalised** (Schürholt 2022): each leaf's mean
     squared error is divided by $\\text{std}(W^{\\star}_{\\ell})^2$,
     so each leaf contributes proportionally to its native magnitude.

3. evaluate the distilled rendered $W$ as a CNN on the CIFAR-10 test
   set — functional fidelity. Compare per-leaf HT-SR / radial-FFT
   $\\alpha$ between rendered and target.

Pre-registered 4-way interpretation grid (per arm, per loss):

1. **low L2 + high fidelity (≥ 0.40)** → architecture can both
   represent and preserve the trained net's function. Clean positive.
2. **low L2 + low fidelity** → NeRN/Schürholt brittleness; check
   whether the layer-wise cell rescues fidelity. If yes, the
   architecture *is* expressive and phase 6/8/9/10 failures were
   optimisation dynamics, not representation.
3. **high L2 + low fidelity** → architecture cannot fit the target;
   per-leaf hyper-renderer family is the wrong shape.
4. **high L2 + high fidelity** → implausible; instrumentation bug.

A robust pivot signal requires case 3 to fire under **both** losses
(the layer-wise cell rules out brittleness as the explanation for the
high-L2 failure).

Run::

    uv run python examples/phase11b_supervised_distillation.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree
from jaxtyping import Array


sys.path.insert(0, str(Path(__file__).parent))
from _common import arms, data, diagnostics, mainnet, reporting  # noqa: E402
from _common.arms import kaiming_leaf_scale  # noqa: E402
# Reuse phase 10's HyperRenderer + polar projection verbatim — distillation
# changes the LOSS, not the architecture under test.
from phase10_polar_projection import (  # noqa: E402
    HyperRenderer,
    project_pseudo_orthonormal,
)


RESEARCH_DIR = Path("/Users/ammar/Documents/Research/fractal-weight-spaces/research")
FIGURES_DIR = RESEARCH_DIR / "figures"
RESEARCH_FILE = RESEARCH_DIR / "2026-06-07-phase11b-supervised-distillation.md"


# --- Schedule --------------------------------------------------------------
TARGET_EPOCHS = int(os.environ.get("PHASE11B_TARGET_EPOCHS", 5))
TARGET_BATCH_SIZE = 128
DISTILL_STEPS = int(os.environ.get("PHASE11B_DISTILL_STEPS", 5000))
DISTILL_LR = 1e-3
K_SEED = int(os.environ.get("PHASE11B_K_SEED", 3))
LOG_EVERY = 100


# --- Target CNN training ---------------------------------------------------
def train_target_cnn(
    train_x: np.ndarray, train_y: np.ndarray,
    test_x: np.ndarray, test_y: np.ndarray,
    *, epochs: int, batch_size: int, seed: int = 0,
) -> tuple[dict[str, Array], float, list[tuple[int, float, float]]]:
    """Train ``WideKernelCNN-SiLU`` on CIFAR-10 via Adam(1e-3).

    Returns ``(final_params, final_test_acc, per_epoch_ckpts)``.
    """
    key = jax.random.key(seed)
    params = mainnet.init_cnn_params(key)
    optimiser = optax.adam(1e-3)
    opt_state = optimiser.init(params)

    @jax.jit
    def step(params: Any, opt_state: Any, batch: dict) -> tuple[Any, Any, Array]:
        loss, grads = jax.value_and_grad(mainnet.cross_entropy_loss)(params, batch)
        updates, new_opt = optimiser.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt, loss

    rng = np.random.default_rng(seed)
    n_train = train_x.shape[0]
    steps_per_epoch = n_train // batch_size
    ckpts: list[tuple[int, float, float]] = []
    global_step = 0
    for epoch in range(epochs):
        idx = np.arange(n_train)
        rng.shuffle(idx)
        for s in range(steps_per_epoch):
            bi = idx[s * batch_size:(s + 1) * batch_size]
            batch = {"x": jnp.asarray(train_x[bi]), "y": jnp.asarray(train_y[bi])}
            params, opt_state, loss = step(params, opt_state, batch)
            global_step += 1
            if global_step % 200 == 0:
                print(f"    target step {global_step}/{steps_per_epoch * epochs}: "
                      f"loss={float(loss):.4f}", flush=True)
        tl, ta = eval_test_loss_acc(params, test_x, test_y)
        ckpts.append((epoch + 1, tl, ta))
        print(f"  target epoch {epoch + 1}/{epochs}: test loss={tl:.4f} acc={ta:.4f}",
              flush=True)
    return params, ckpts[-1][2], ckpts


# --- Test-set evaluation ---------------------------------------------------
def eval_test_loss_acc(
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


# --- Distillation loss + loop ---------------------------------------------
# Two losses, run as paired cells per the Schürholt 2022 anchoring:
#
# - ``leaf_mean``         : sum over leaves of ``mean((W_rendered - W*)**2)``.
#                           Leaf-uniform; matches the NeRN default and the
#                           first-pass interpretation grid.
# - ``layerwise_normalised``: each leaf's mean-squared error is divided by
#                            ``var(W*_leaf) + eps``; layers with small native
#                            std contribute proportionally to large-std
#                            layers, so the optimiser cannot ignore fc /
#                            conv-bias rows.
LossKind = str  # "leaf_mean" | "layerwise_normalised"


def leaf_weights(target: dict[str, Array], kind: LossKind) -> dict[str, float]:
    """Compute per-leaf loss weights for the given loss kind.

    For ``leaf_mean`` every weight is 1. For ``layerwise_normalised``
    each weight is ``1 / (var(target_leaf) + eps)`` — Schürholt 2022's
    layer-wise normalisation. The eps protects bias leaves whose target
    std can be near zero from blowing up the gradient.
    """
    if kind == "leaf_mean":
        return {name: 1.0 for name in mainnet.LEAF_ORDER}
    if kind == "layerwise_normalised":
        weights: dict[str, float] = {}
        for name in mainnet.LEAF_ORDER:
            t = np.asarray(target[name])
            # eps tied to the dtype keeps a near-zero-std bias from producing
            # a gradient orders of magnitude larger than the weight leaves.
            eps = float(np.finfo(t.dtype).eps) * t.size
            var = float(t.var()) + eps
            weights[name] = 1.0 / var
        return weights
    raise ValueError(f"unknown loss kind {kind!r}")


def mse_to_target(
    rendered: dict[str, Array], target: dict[str, Array],
    *, weights: dict[str, float],
) -> Array:
    """Weighted per-leaf mean-squared error, summed across leaves.

    ``weights[name]`` is the per-leaf scalar multiplying that leaf's
    ``mean((W_rendered - W_target) ** 2)``. Uniform-1 weights give the
    NeRN-style leaf-uniform loss; ``1 / var(target_leaf)`` weights give
    Schürholt's layer-wise-normalised loss.
    """
    return sum(
        weights[name] * jnp.mean((rendered[name] - target[name]) ** 2)
        for name in mainnet.LEAF_ORDER
    )


def per_leaf_l2_sse(
    rendered: dict[str, Array], target: dict[str, Array],
) -> dict[str, float]:
    """Per-leaf $\\|W_{rendered} - W^{\\star}\\|^2$ (sum of squares, not mean).

    Loss-kind independent: used as the cell-commensurable raw L2 in the
    4-way grid verdict.
    """
    return {
        name: float(jnp.sum((rendered[name] - target[name]) ** 2))
        for name in mainnet.LEAF_ORDER
    }


def distill_arm(
    arm: arms.Arm, target_W: dict[str, Array],
    *, seed: int, num_steps: int, lr: float, loss_kind: LossKind,
) -> tuple[dict, list[tuple[int, float]], dict[str, Array]]:
    """L2-distill ``arm`` against ``target_W`` for ``num_steps`` Adam steps.

    Returns ``(final_state, [(step, total_loss), ...], rendered_W_final)``.
    """
    key = jax.random.key(seed)
    state = arm.init(key)

    # Override the arm's optimiser to a flat Adam(lr) on the full state pytree
    # so the brief's "Adam(1e-3)" spec applies uniformly across cells.
    optimiser = optax.adam(lr)
    opt_state = optimiser.init(state)

    weights = leaf_weights(target_W, loss_kind)

    def loss_fn(state: Any) -> Array:
        rendered = arm.render_W(state)
        return mse_to_target(rendered, target_W, weights=weights)

    @jax.jit
    def step(state: Any, opt_state: Any) -> tuple[Any, Any, Array]:
        loss, grads = jax.value_and_grad(loss_fn)(state)
        updates, new_opt = optimiser.update(grads, opt_state, state)
        return optax.apply_updates(state, updates), new_opt, loss

    trajectory: list[tuple[int, float]] = []
    for s in range(num_steps):
        state, opt_state, loss = step(state, opt_state)
        if s % LOG_EVERY == 0 or s == num_steps - 1:
            trajectory.append((s, float(loss)))
        if s % 1000 == 0:
            print(f"    distill step {s}/{num_steps} [{loss_kind}]: "
                  f"loss={float(loss):.6e}", flush=True)
    rendered = {k: np.asarray(v) for k, v in arm.render_W(state).items()}
    return state, trajectory, rendered


# --- Arm constructors (FWS arms only — distillation diagnostic) ------------
def make_fws_hyper_distill_arm() -> arms.Arm:
    """Phase-10 FWS-hyper-polar arm — full architecture, distillation loss."""
    return arms.make_fws_hyper(
        g_h_init=lambda key: HyperRenderer(key=key, out_scale_kind="linear_xavier"),
        leaf_scale_fn=kaiming_leaf_scale,
        projection_fn=project_pseudo_orthonormal,
        name="FWS-hyper-polar",
        short="fws_hyper",
    )


def make_fws_parallel_distill_arm() -> arms.Arm:
    """Phase-10 FWS-parallel-no-G_H arm with polar $P$."""
    return arms.make_fws_parallel(
        leaf_scale_fn=kaiming_leaf_scale,
        projection_fn=project_pseudo_orthonormal,
    )


# --- Plots -----------------------------------------------------------------
def plot_distill_trajectories(
    trajectories_per_cell: dict[str, dict[str, list[list[tuple[int, float]]]]],
    arms_order: list[arms.Arm],
    save_path: Path, *, title: str,
) -> None:
    """Per-cell × per-arm L2 distillation trajectories: faint per-seed + bold median."""
    import matplotlib.pyplot as plt
    cells = list(trajectories_per_cell.keys())
    fig, axes = plt.subplots(1, len(cells), figsize=(7.5 * len(cells), 4.8), sharey=False)
    if len(cells) == 1:
        axes = [axes]
    for ax, cell in zip(axes, cells, strict=True):
        per_arm = trajectories_per_cell[cell]
        for a in arms_order:
            rows = per_arm[a.short]
            for traj in rows:
                xs = [t[0] for t in traj]
                ys = [t[1] for t in traj]
                ax.plot(xs, ys, color=a.color, alpha=0.3, linewidth=1.0)
            stack = np.stack([[t[1] for t in traj] for traj in rows], axis=0)
            xs = [t[0] for t in rows[0]]
            ax.plot(xs, np.median(stack, axis=0), color=a.color, linewidth=2.0,
                    label=a.name)
        ax.set_yscale("log")
        ax.set(xlabel="distillation step", ylabel=f"total {cell} loss",
               title=f"{cell}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_functional_fidelity_box(
    accs_per_cell: dict[str, dict[str, list[float]]], target_acc: float,
    arms_order: list[arms.Arm],
    save_path: Path, *, title: str,
) -> None:
    """Per-cell × per-arm boxplot of test accuracy from the distilled W."""
    import matplotlib.pyplot as plt
    cells = list(accs_per_cell.keys())
    fig, axes = plt.subplots(1, len(cells), figsize=(7.0 * len(cells), 4.6),
                             sharey=True)
    if len(cells) == 1:
        axes = [axes]
    rng = np.random.default_rng(0)
    for ax, cell in zip(axes, cells, strict=True):
        accs = accs_per_cell[cell]
        data = [accs[a.short] for a in arms_order]
        cols = [a.color for a in arms_order]
        labels = [a.name.replace("FWS-parallel-no-G_H", "FWS-par") for a in arms_order]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                        tick_labels=labels)
        for patch, c in zip(bp["boxes"], cols, strict=True):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
            patch.set_edgecolor(c)
        for ml, c in zip(bp["medians"], cols, strict=True):
            ml.set_color(c)
            ml.set_linewidth(2.0)
        for i, (vals, c) in enumerate(zip(data, cols, strict=True), start=1):
            v = np.asarray(vals, dtype=float)
            jitter = rng.uniform(-0.10, 0.10, size=v.size)
            ax.scatter(np.full_like(v, i) + jitter, v, color=c, s=40, zorder=3,
                       edgecolor="white", linewidths=0.6)
        ax.axhline(target_acc, color="#000000", linestyle="--", linewidth=1.5,
                   label=f"target CNN ({target_acc:.3f})")
        ax.axhline(0.10, color="#888888", linestyle=":", linewidth=1.0, label="chance")
        ax.axhline(0.40, color="#888888", linestyle="-.", linewidth=1.0,
                   label="interp. threshold 0.40")
        ax.set_title(cell)
        ax.set_ylabel("test acc from distilled W")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_alpha_comparison(
    alphas_target: dict[str, tuple[float, float]],
    alphas_per_cell: dict[str, dict[str, list[dict[str, tuple[float, float]]]]],
    arms_order: list[arms.Arm],
    save_path: Path, *, title: str,
) -> None:
    """Per-cell × per-leaf $\\alpha$ comparison: target horizontal line + per-arm boxes."""
    import matplotlib.pyplot as plt
    leaves = ("conv1", "conv2", "fc1", "fc2")
    cells = list(alphas_per_cell.keys())
    fig, axes = plt.subplots(len(cells), 4, figsize=(14.0, 4.4 * len(cells)),
                             sharey=False)
    if len(cells) == 1:
        axes = np.array([axes])
    for row, cell in enumerate(cells):
        alphas_per_arm = alphas_per_cell[cell]
        for col, leaf in enumerate(leaves):
            ax = axes[row, col]
            data = [
                [seed_alpha[leaf][0] for seed_alpha in alphas_per_arm[a.short]]
                for a in arms_order
            ]
            cols_ = [a.color for a in arms_order]
            labels = [a.name.replace("FWS-parallel-no-G_H", "FWS-par") for a in arms_order]
            bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                            tick_labels=labels)
            for patch, c in zip(bp["boxes"], cols_, strict=True):
                patch.set_facecolor(c)
                patch.set_alpha(0.35)
                patch.set_edgecolor(c)
            for ml, c in zip(bp["medians"], cols_, strict=True):
                ml.set_color(c)
                ml.set_linewidth(2.0)
            alpha_t, r2_t = alphas_target[leaf]
            ax.axhline(alpha_t, color="#000000", linestyle="--", linewidth=1.5,
                       label=f"target α={alpha_t:.2f} (R²={r2_t:.2f})")
            ax.set_title(f"{cell} — {leaf}")
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# --- Reporting helpers -----------------------------------------------------
def per_arm_summary_md(
    arms_order: list[arms.Arm],
    final_loss: dict[str, list[float]],
    per_leaf_l2_per_arm: dict[str, list[dict[str, float]]],
    func_acc: dict[str, list[float]],
    target_acc: float,
    *, cell_name: str,
) -> str:
    """Per-arm verbatim loss + per-leaf L2 + functional-fidelity table."""
    rows: list[str] = []
    rows.append(
        f"| arm | total {cell_name} loss (median) | "
        f"total {cell_name} loss (min) | total {cell_name} loss (max) | "
        "func acc (median) | func acc (min) | func acc (max) |"
    )
    rows.append("|---|---|---|---|---|---|---|")
    for a in arms_order:
        m = np.asarray(final_loss[a.short])
        f = np.asarray(func_acc[a.short])
        rows.append(
            f"| {a.name} | {np.median(m):.4e} | {m.min():.4e} | {m.max():.4e} "
            f"| {np.median(f):.4f} | {f.min():.4f} | {f.max():.4f} |"
        )
    text = "\n".join(rows) + f"\n\nTarget CNN test accuracy: **{target_acc:.4f}**\n\n"

    text += (
        f"### Per-leaf raw L2 ($\\|W_{{\\text{{rendered}}}} - "
        f"W^{{\\star}}\\|^2$, sum of squares, "
        f"loss-kind-independent) — median across seeds\n\n"
    )
    leaf_hdr = "| arm | " + " | ".join(mainnet.LEAF_ORDER) + " |"
    leaf_sep = "|---|" + "|".join(["---"] * len(mainnet.LEAF_ORDER)) + "|"
    leaf_rows = [leaf_hdr, leaf_sep]
    for a in arms_order:
        median_l2 = {
            leaf: float(np.median([d[leaf] for d in per_leaf_l2_per_arm[a.short]]))
            for leaf in mainnet.LEAF_ORDER
        }
        cells = " | ".join(f"{median_l2[leaf]:.3e}" for leaf in mainnet.LEAF_ORDER)
        leaf_rows.append(f"| {a.name} | {cells} |")
    return text + "\n".join(leaf_rows)


def alpha_table_md(
    alphas_target: dict[str, tuple[float, float]],
    alphas_per_arm: dict[str, list[dict[str, tuple[float, float]]]],
    arms_order: list[arms.Arm],
) -> str:
    leaves = ("conv1", "conv2", "fc1", "fc2")
    rows = ["| leaf | target α (R²) | "
            + " | ".join(f"{a.name} α (R²) median" for a in arms_order) + " |"]
    rows.append("|---|---|" + "|".join(["---"] * len(arms_order)) + "|")
    for leaf in leaves:
        a_t, r2_t = alphas_target[leaf]
        cells: list[str] = [f"{a_t:.3f} ({r2_t:.3f})"]
        for a in arms_order:
            vals = alphas_per_arm[a.short]
            alpha_med = float(np.median([v[leaf][0] for v in vals]))
            r2_med = float(np.median([v[leaf][1] for v in vals]))
            cells.append(f"{alpha_med:.3f} ({r2_med:.3f})")
        rows.append(f"| {leaf} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _bucket(loss: float, fidelity: float, *,
            l2_clean_threshold: float, func_threshold: float) -> str:
    """Return one of {low_L2_high_F, low_L2_low_F, high_L2_high_F, high_L2_low_F}.

    L2 is computed as the **per-leaf raw sum-of-squares L2 distance**
    (loss-kind-independent), not the weighted training loss — so the
    "low L2" boundary is comparable across cells.
    """
    low_l2 = loss < l2_clean_threshold
    high_fidel = fidelity >= func_threshold
    return f"{'low' if low_l2 else 'high'}_L2_{'high' if high_fidel else 'low'}_F"


def decide_interpretation(
    raw_l2_median_per_cell: dict[str, dict[str, float]],
    func_acc_median_per_cell: dict[str, dict[str, float]],
    *, l2_clean_threshold: float, func_threshold: float,
) -> str:
    """Apply the 4-way pre-registered grid (per arm, per loss).

    The verdict shape is:

    - Per (arm, loss_kind) bucket out of {low_L2_high_F, low_L2_low_F,
      high_L2_high_F, high_L2_low_F}.
    - Cross-cell read for each arm: did the layer-wise cell rescue
      fidelity (case 2 → case 1 / 2 → 3 distinction)?
    - Cross-arm read: does the meta-renderer ($G_H$) specifically
      break representation, or do both arms hit the same wall?

    A robust pivot signal requires **both** arms to bucket as
    ``high_L2_low_F`` under **both** losses.
    """
    cells = list(raw_l2_median_per_cell.keys())
    arms_keys = list(raw_l2_median_per_cell[cells[0]].keys())

    bucket: dict[tuple[str, str], str] = {}
    for cell in cells:
        for arm in arms_keys:
            bucket[(cell, arm)] = _bucket(
                raw_l2_median_per_cell[cell][arm],
                func_acc_median_per_cell[cell][arm],
                l2_clean_threshold=l2_clean_threshold,
                func_threshold=func_threshold,
            )

    bucket_lines = [
        f"- {arm} × {cell}: **{bucket[(cell, arm)]}**"
        for arm in arms_keys for cell in cells
    ]

    all_pivot = all(bucket[(cell, arm)] == "high_L2_low_F"
                    for arm in arms_keys for cell in cells)
    layerwise_rescued = any(
        bucket[("leaf_mean", arm)] in {"low_L2_low_F", "high_L2_low_F"}
        and bucket[("layerwise_normalised", arm)] in {"low_L2_high_F", "high_L2_high_F"}
        for arm in arms_keys
    )
    parallel_only_fit = (
        bucket.get(("layerwise_normalised", "fws_parallel"))
        in {"low_L2_high_F", "high_L2_high_F"}
        and bucket.get(("layerwise_normalised", "fws_hyper"))
        in {"low_L2_low_F", "high_L2_low_F"}
    )

    if all_pivot:
        verdict = (
            "**Robust pivot signal: case 3 (high L2 + low fidelity) fires for "
            "every (arm, loss) pair.** The per-leaf hyper-renderer family "
            "cannot fit a trained CIFAR classifier's weight tree even when "
            "the loss is layer-wise-normalised. Layer-wise normalisation does "
            "not rescue fidelity, so NeRN/Schürholt brittleness is ruled out "
            "as the explanation for the high-L2 failure. The per-leaf "
            "hyper-renderer family is the wrong shape for representing real "
            "CIFAR networks; pivot programme architecture."
        )
    elif layerwise_rescued and parallel_only_fit:
        verdict = (
            "**Layer-wise normalisation rescues FWS-parallel's functional "
            "fidelity but not FWS-hyper's.** The per-rank parallel "
            "architecture *can* represent trained CIFAR weights given a "
            "Schürholt-balanced loss; the meta-renderer ($G_H$) breaks "
            "representation specifically. Phase 6/8/9/10 FWS-parallel "
            "failures were optimisation dynamics / loss-balancing, not "
            "representation. FWS-hyper still pivot-signal."
        )
    elif layerwise_rescued:
        verdict = (
            "**Layer-wise normalisation rescues functional fidelity for at "
            "least one arm.** NeRN/Schürholt brittleness was the dominant "
            "failure mode under leaf-uniform L2. The architecture(s) that "
            "fidelity-recover are expressive; phase 6/8/9/10 task-loss "
            "failures for those arms were optimisation dynamics / loss "
            "balancing, not representation."
        )
    else:
        verdict = (
            "**Mixed pattern.** Read the bucket table per (arm, loss). The "
            "verdict is not the pre-registered robust pivot signal nor a "
            "clean rescue; the result is hypothesis-generating, and the "
            "next phase should isolate which (arm, loss) bucket carries "
            "the signal."
        )
    return verdict + "\n\nPer-bucket verdict:\n" + "\n".join(bucket_lines)


LOSS_KINDS: tuple[LossKind, ...] = ("leaf_mean", "layerwise_normalised")


def main() -> None:
    print("=" * 72)
    print("Phase 11B — Supervised-distillation sanity check on FWS")
    print("=" * 72)
    print("Loading CIFAR-10 ...")
    train_x, train_y, test_x, test_y = data.load_cifar10()
    print(f"  train: {train_x.shape} | test: {test_x.shape}")
    print(f"  K_seed={K_SEED}, distill_steps={DISTILL_STEPS}, lr={DISTILL_LR}")
    print(f"  loss kinds: {LOSS_KINDS}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: train target W*. ------------------------------------------
    print("\n--- Step 1: train target WideKernelCNN-SiLU on CIFAR-10 ---")
    t0 = time.time()
    target_W, target_acc, target_ckpts = train_target_cnn(
        train_x, train_y, test_x, test_y,
        epochs=TARGET_EPOCHS, batch_size=TARGET_BATCH_SIZE, seed=0,
    )
    target_wall = time.time() - t0
    n_target_params = sum(int(np.asarray(v).size) for v in target_W.values())
    print(f"  target final test acc: {target_acc:.4f}  (wall {target_wall:.1f}s, "
          f"{n_target_params} params)")

    target_alphas = diagnostics.per_leaf_alphas(target_W)
    print("  target per-leaf α (R²):")
    for leaf, (a, r2) in target_alphas.items():
        print(f"    {leaf}: α={a:.3f} R²={r2:.3f}")

    # Per-leaf target std drives layerwise normalisation; print so the reader
    # sees the relative magnitudes Schürholt's normalisation rebalances.
    print("  target per-leaf std (drives layer-wise normalisation):")
    for name in mainnet.LEAF_ORDER:
        print(f"    {name}: std={float(np.asarray(target_W[name]).std()):.4e}")

    arms_order: list[arms.Arm] = [
        make_fws_hyper_distill_arm(),
        make_fws_parallel_distill_arm(),
    ]

    # Per-cell × per-arm storage. trajectories[cell][arm] -> list of seed
    # trajectories; same shape for final losses, per-leaf L2, fidelities, α.
    final_loss: dict[str, dict[str, list[float]]] = {
        cell: {a.short: [] for a in arms_order} for cell in LOSS_KINDS
    }
    per_leaf_l2: dict[str, dict[str, list[dict[str, float]]]] = {
        cell: {a.short: [] for a in arms_order} for cell in LOSS_KINDS
    }
    func_acc: dict[str, dict[str, list[float]]] = {
        cell: {a.short: [] for a in arms_order} for cell in LOSS_KINDS
    }
    alphas_per_cell: dict[str, dict[str, list[dict[str, tuple[float, float]]]]] = {
        cell: {a.short: [] for a in arms_order} for cell in LOSS_KINDS
    }
    trajectories: dict[str, dict[str, list[list[tuple[int, float]]]]] = {
        cell: {a.short: [] for a in arms_order} for cell in LOSS_KINDS
    }

    # --- Step 2: distill each (cell, arm) × K seeds. -----------------------
    distill_wall = 0.0
    for cell in LOSS_KINDS:
        for a in arms_order:
            print(f"\n--- Step 2: distill {a.name}  [loss={cell}]  (K={K_SEED}) ---")
            for seed in range(K_SEED):
                t_seed = time.time()
                _state, traj, rendered = distill_arm(
                    a, target_W,
                    seed=seed, num_steps=DISTILL_STEPS, lr=DISTILL_LR,
                    loss_kind=cell,
                )
                final = traj[-1][1]
                leaf_l2 = per_leaf_l2_sse(rendered, target_W)
                _, ta = eval_test_loss_acc(rendered, test_x, test_y)
                alphas = diagnostics.per_leaf_alphas(rendered)
                final_loss[cell][a.short].append(final)
                per_leaf_l2[cell][a.short].append(leaf_l2)
                func_acc[cell][a.short].append(ta)
                alphas_per_cell[cell][a.short].append(alphas)
                trajectories[cell][a.short].append(traj)
                seed_wall = time.time() - t_seed
                distill_wall += seed_wall
                print(f"  [{cell}] {a.short} seed {seed}: final_loss={final:.4e}  "
                      f"func_acc={ta:.4f}  ({seed_wall:.1f}s)")

    # --- Step 3: numbers, figures, log. ------------------------------------
    print("\n--- Step 3: aggregate and write log ---")

    # Plots are cell-aware: side-by-side panels per cell.
    plot_distill_trajectories(
        trajectories, arms_order,
        FIGURES_DIR / "2026-06-07-phase11b-distill-trajectories.png",
        title=f"Phase 11B — distillation loss trajectories (K={K_SEED})",
    )
    plot_functional_fidelity_box(
        func_acc, target_acc, arms_order,
        FIGURES_DIR / "2026-06-07-phase11b-functional-fidelity.png",
        title=f"Phase 11B — functional fidelity per loss kind (K={K_SEED})",
    )
    plot_alpha_comparison(
        target_alphas, alphas_per_cell, arms_order,
        FIGURES_DIR / "2026-06-07-phase11b-alpha-comparison.png",
        title=f"Phase 11B — per-leaf α distilled vs target (K={K_SEED})",
    )

    # Per-cell summary + α tables.
    cell_summaries: list[tuple[str, str]] = []
    for cell in LOSS_KINDS:
        s = per_arm_summary_md(
            arms_order, final_loss[cell], per_leaf_l2[cell], func_acc[cell],
            target_acc, cell_name=cell,
        )
        a = alpha_table_md(target_alphas, alphas_per_cell[cell], arms_order)
        cell_summaries.append((cell, s + "\n\n#### Per-leaf α — " + cell + "\n\n" + a))

    # Pre-registered grid: bucket on RAW L2 (sum of squares, loss-kind
    # independent), not the cell's training loss — so cells are commensurable.
    raw_l2_median_per_cell: dict[str, dict[str, float]] = {
        cell: {
            arm: float(np.median([
                sum(d.values()) for d in per_leaf_l2[cell][arm]
            ]))
            for arm in [a.short for a in arms_order]
        }
        for cell in LOSS_KINDS
    }
    func_acc_median_per_cell: dict[str, dict[str, float]] = {
        cell: {arm: float(np.median(v)) for arm, v in func_acc[cell].items()}
        for cell in LOSS_KINDS
    }
    verdict_text = decide_interpretation(
        raw_l2_median_per_cell, func_acc_median_per_cell,
        # Raw-L2 "clean" threshold: 1e-1 on the sum-of-squares across leaves.
        # The target's sum-of-squares ‖W*‖² is ~5.5e+02 dominated by fc1;
        # an L2 of < 1e-1 corresponds to <0.02% relative error.
        # Heuristic, surfaced for the reader.
        l2_clean_threshold=1e-1, func_threshold=0.40,
    )
    print("\nVerdict:\n" + verdict_text)

    # Compose log sections.
    step2_section_body = "\n\n".join(
        f"### Cell — {cell}\n\n{body}" for cell, body in cell_summaries
    )

    sections: list[tuple[str, str]] = [
        ("Background", _BACKGROUND_MD),
        ("Setup", _SETUP_MD.format(
            target_epochs=TARGET_EPOCHS, distill_steps=DISTILL_STEPS,
            distill_lr=DISTILL_LR, k_seed=K_SEED,
        )),
        ("Step 1 — target CNN", (
            f"Target ``WideKernelCNN-SiLU`` trained on CIFAR-10 with "
            f"Adam(1e-3) for {TARGET_EPOCHS} epochs reached "
            f"**test accuracy {target_acc:.4f}** ({n_target_params} params, "
            f"target wall {target_wall:.1f}s).\n\n"
            "Per-epoch checkpoints:\n\n| epoch | test loss | test acc |\n|---|---|---|\n"
            + "\n".join(f"| {e} | {tl:.4f} | {ta:.4f} |" for e, tl, ta in target_ckpts)
            + "\n\nPer-leaf target std (drives layer-wise normalisation):\n\n"
            "| leaf | std |\n|---|---|\n"
            + "\n".join(
                f"| {name} | {float(np.asarray(target_W[name]).std()):.4e} |"
                for name in mainnet.LEAF_ORDER
            )
        )),
        (f"Step 2 — L2 distillation, two-cell grid (K={K_SEED} per (arm, loss))",
         step2_section_body + "\n\n"
         "![distillation trajectories per cell]"
         "(figures/2026-06-07-phase11b-distill-trajectories.png)\n\n"
         "![functional fidelity per cell]"
         "(figures/2026-06-07-phase11b-functional-fidelity.png)\n\n"
         "![per-leaf α distilled vs target per cell]"
         "(figures/2026-06-07-phase11b-alpha-comparison.png)"),
        ("Interpretation (4-way grid, per arm × per loss)", verdict_text),
        ("Honest caveats", _CAVEATS_MD.format(
            k_seed=K_SEED, target_epochs=TARGET_EPOCHS,
            distill_steps=DISTILL_STEPS, distill_wall=distill_wall,
        )),
    ]
    reporting.render_research_log(reporting.ReportContext(
        title="Phase 11B — Supervised-distillation sanity check on FWS — 2026-06-07",
        out_path=RESEARCH_FILE,
        sections=sections,
        figures_dir=FIGURES_DIR,
    ))
    reporting.render_pdf(RESEARCH_FILE)


_BACKGROUND_MD = """
Phases 6 / 8 / 9 / 10 left a representation-vs-dynamics ambiguity. The
FWS-hyper arm stayed at chance across every phase in which it ran
end-to-end, and FWS-parallel only got off chance at phase 10
(≈29% on CIFAR-10). Two hypotheses are consistent with that record:

- **Representation.** The per-leaf hyper-renderer family — the rendered
  $W$ tree as a function of $(z, G_H)$ — cannot express the kind of
  weight tensor a trained CIFAR classifier needs. Even with a perfect
  outer optimiser the FWS arm would land far from any
  classifier-quality $W$.
- **Training dynamics.** The family *can* express trained weights, but
  the cross-entropy outer landscape (through softmax → CNN forward →
  rendering chain) is hostile to gradient descent at the scales we ran.

Phase 11B disambiguates by **direct L2 distillation against a fixed
target weight tree.** No task loss enters the FWS arm's optimisation.

**Literature anchoring.** NeRN (Ashkenazi et al., arXiv:2212.13554,
2023) is the direct prior art for this diagnostic: a coordinate→weight
predictor trained by L2 distillation against a fixed CIFAR/ImageNet
classifier, with explicit accuracy-preservation evaluation. Phase 11B
largely replicates NeRN's setup. The fws-bench novelty axis is
**sine-only SIREN backbone + polar projection $P$ on the latent** —
NeRN uses an MLP and imposes no geometric constraint on the latent.

Both NeRN AND Schürholt et al. (Hyper-Representations,
arXiv:2209.14733, 2022) report a known failure mode of leaf-uniform
L2: low MSE, near-zero task accuracy, because weight distributions
vary by 1–2 OoM across layer types (fc ∼ 0.1, conv ∼ 0.01, bias ≈ 0).
Uniform L2 underweights small-std layers; small absolute perturbations
to those layers destroy classification accuracy. Schürholt's fix is
**layer-wise normalisation**: scale each leaf's contribution to the
loss by its target's variance. Phase 11B runs both losses as paired
cells so the verdict reads off a 4-way grid, not from a single number.
"""

_SETUP_MD = """
1. Train a vanilla ``WideKernelCNN-SiLU`` on CIFAR-10 with Adam(1e-3)
   for {target_epochs} epochs. Save the trained weights as $W^{{\\star}}$.
   No smoothness regulariser on $W^{{\\star}}$ (NeRN imposes one; we
   leave it off for the first-pass diagnostic).
2. For each FWS arm in {{FWS-hyper-polar (full phase 10 architecture:
   $G_H$ SIREN body + linear readout + per-leaf $G_H$ output
   normalisation + Kaiming pre-factor + polar Newton-Schulz $P$),
   FWS-parallel-no-G_H + polar $P$}}, train the FWS state via
   Adam({distill_lr}) for {distill_steps} outer steps under each of two
   losses (paired cells; K={k_seed} per cell):

   - **leaf-mean**: $\\sum_{{\\ell}} \\text{{mean}}((W_{{\\ell}}(\\text{{state}}) -
     W^{{\\star}}_{{\\ell}})^2)$ — leaf-uniform; NeRN's default shape.
   - **layer-wise normalised** (Schürholt 2022): each leaf's mean
     squared error is divided by $\\text{{var}}(W^{{\\star}}_{{\\ell}})$,
     so each leaf contributes proportionally to its native magnitude.

3. Measure raw per-leaf $\\|W_{{\\text{{rendered}}}} - W^{{\\star}}\\|^2$
   (sum-of-squares, loss-kind independent — comparable across cells),
   functional fidelity (test accuracy of the distilled rendered $W$ on
   CIFAR-10), and per-leaf HT-SR / radial-FFT $\\alpha$ comparison.

**Pre-registered 4-way grid (per arm, per loss):**

1. *low L2 + high fidelity (≥ 0.40)* — architecture both represents
   the target and preserves its function. Clean positive.
2. *low L2 + low fidelity* — NeRN/Schürholt brittleness; check the
   layer-wise cell. If layer-wise rescues fidelity, the architecture
   is expressive and phase 6/8/9/10 failures were optimisation
   dynamics / loss balancing.
3. *high L2 + low fidelity* — architecture cannot fit the target.
   If both arms × both losses fire case 3, robust pivot signal.
4. *high L2 + high fidelity* — implausible; instrumentation bug.

L2-clean threshold for the bucket call: raw sum-of-squares
$\\sum_{{\\ell}} \\|W_{{\\ell}} - W^{{\\star}}_{{\\ell}}\\|^2 < 0.1$
(soft heuristic; verbatim numbers are the primary artefact).
"""

_CAVEATS_MD = """
- **K={k_seed} smoke** — not K=10 confirmation.
- **One target tree.** A single Adam(1e-3) × {target_epochs}-epoch run
  produced $W^{{\\star}}$. Different target weights (different init,
  different LR, longer training, or NeRN's smoothness regulariser)
  might be easier or harder to L2-fit; this phase reports the
  K={k_seed} numbers against one un-regularised $W^{{\\star}}$.
- **L2 "clean" threshold is a soft heuristic.** The bucket call uses
  a raw sum-of-squares < 0.1 cut for case-3 detection; the verbatim
  per-cell loss values and the raw per-leaf L2 numbers are the primary
  artefacts.
- **Distillation may exit at a local minimum of the rendering manifold.**
  If the FWS arm hits a flat region of the rendering function near
  $W^{{\\star}}$, the L2 stalls but the family could still in
  principle express $W^{{\\star}}$. {distill_steps} steps is the
  budget; the loss-trajectory plateau is a representation-side
  ceiling under the schedule, not a hard expressivity bound.
- **Bias-leaf variance can be near zero**, which would blow up the
  layer-wise weight $1 / \\text{{var}}$. We add
  ``eps = finfo(dtype).eps * leaf.size`` to var as a numerical
  guard; the resulting bias weights are bounded but large enough
  that bias L2 dominates the layer-wise loss early. The plateau is
  the diagnostic of interest, not the early-step shape.
- **No data augmentation, no LR schedule.** Plain Adam(1e-3).
- **Distillation wall (all cells, all seeds)**: {distill_wall:.1f}s.
"""


if __name__ == "__main__":
    main()
