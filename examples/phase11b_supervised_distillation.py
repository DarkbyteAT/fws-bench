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

Phase 11B disambiguates by **direct L2 distillation**. We

1. train a vanilla ``WideKernelCNN-SiLU`` on CIFAR-10 with Adam(1e-3)
   for 5 epochs — this hits ≈60% test accuracy and produces a single
   target weight tree $W^{\\star}$.
2. for each FWS arm (FWS-hyper-polar with the phase 10 architecture;
   FWS-parallel-no-G_H with polar $P$), train the arm's parameters by
   Adam(1e-3) to minimise the leaf-summed mean-squared error
   $\\sum_{\\ell} \\| W_{\\ell}(state) - W^{\\star}_{\\ell} \\|^2 / n_{\\ell}$
   for 5000 outer steps. **No task loss.** Pure L2.
3. evaluate the distilled rendered $W$ as a CNN on the CIFAR-10 test
   set — functional fidelity. Compare per-leaf HT-SR / radial-FFT
   $\\alpha$ between rendered and target.

Pre-registered interpretation rules:

- FWS-parallel L2-fits cleanly AND functional fidelity ≥ 40% →
  parallel architecture *can* represent trained CIFAR weights; phase 9
  / 10 task-loss failure was training dynamics, not representation.
- FWS-hyper L2-fits cleanly AND functional fidelity ≥ 40% → hyper
  architecture *can* represent trained CIFAR weights; its task-loss
  failure was about the chain rule through $G_H$, not expressive power.
- Neither L2-fits → per-leaf hyper-renderer family is the wrong shape;
  pivot.
- FWS-parallel fits but FWS-hyper doesn't → the meta-renderer layer
  specifically breaks representation, not just optimisation.

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
def mse_to_target(
    rendered: dict[str, Array], target: dict[str, Array],
) -> Array:
    """Per-leaf mean-squared error, summed across leaves.

    Each leaf contributes ``mean((W_rendered - W_target) ** 2)`` — the
    leaf-wise mean (rather than total sum of squares) keeps the scale
    of the loss insensitive to fc1's much-larger parameter count.
    """
    return sum(
        jnp.mean((rendered[name] - target[name]) ** 2)
        for name in mainnet.LEAF_ORDER
    )


def per_leaf_l2(
    rendered: dict[str, Array], target: dict[str, Array],
) -> dict[str, float]:
    """Per-leaf $\\|W_{rendered} - W^{\\star}\\|^2$ (sum of squares, not mean)."""
    return {
        name: float(jnp.sum((rendered[name] - target[name]) ** 2))
        for name in mainnet.LEAF_ORDER
    }


def distill_arm(
    arm: arms.Arm, target_W: dict[str, Array],
    *, seed: int, num_steps: int, lr: float,
) -> tuple[dict, list[tuple[int, float]], dict[str, Array]]:
    """L2-distill ``arm`` against ``target_W`` for ``num_steps`` Adam steps.

    Returns ``(final_state, [(step, total_mse), ...], rendered_W_final)``.
    """
    key = jax.random.key(seed)
    state = arm.init(key)

    # Override the arm's optimiser to a flat Adam(lr) on the full state pytree —
    # FWS-hyper's multi_transform partition is fine here too because we keep
    # using the arm's own optimiser through optax.apply_updates. But the arm
    # was constructed with its own LRs; for distillation we re-build a single
    # Adam(lr) over the state to match the brief's "Adam(1e-3)" spec.
    optimiser = optax.adam(lr)
    opt_state = optimiser.init(state)

    def loss_fn(state: Any) -> Array:
        rendered = arm.render_W(state)
        return mse_to_target(rendered, target_W)

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
            print(f"    distill step {s}/{num_steps}: total_mse={float(loss):.6e}",
                  flush=True)
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
    trajectories: dict[str, list[list[tuple[int, float]]]],
    arms_order: list[arms.Arm],
    save_path: Path, *, title: str,
) -> None:
    """Plot per-arm L2 distillation MSE trajectories: faint per-seed + bold median."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for a in arms_order:
        rows = trajectories[a.short]
        for traj in rows:
            xs = [t[0] for t in traj]
            ys = [t[1] for t in traj]
            ax.plot(xs, ys, color=a.color, alpha=0.3, linewidth=1.0)
        stack = np.stack([[t[1] for t in traj] for traj in rows], axis=0)
        xs = [t[0] for t in rows[0]]
        ax.plot(xs, np.median(stack, axis=0), color=a.color, linewidth=2.0, label=a.name)
    ax.set_yscale("log")
    ax.set(xlabel="distillation step", ylabel="total per-leaf MSE", title=title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_functional_fidelity_box(
    accs: dict[str, list[float]], target_acc: float,
    arms_order: list[arms.Arm],
    save_path: Path, *, title: str,
) -> None:
    """Boxplot of test accuracy from the distilled rendered W per arm + target line."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
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
    rng = np.random.default_rng(0)
    for i, (vals, c) in enumerate(zip(data, cols, strict=True), start=1):
        v = np.asarray(vals, dtype=float)
        jitter = rng.uniform(-0.10, 0.10, size=v.size)
        ax.scatter(np.full_like(v, i) + jitter, v, color=c, s=40, zorder=3,
                   edgecolor="white", linewidths=0.6)
    ax.axhline(target_acc, color="#000000", linestyle="--", linewidth=1.5,
               label=f"target CNN (acc={target_acc:.3f})")
    ax.axhline(0.10, color="#888888", linestyle=":", linewidth=1.0, label="chance (0.10)")
    ax.axhline(0.40, color="#888888", linestyle="-.", linewidth=1.0,
               label="interpretation threshold (0.40)")
    ax.set_ylabel("test accuracy from distilled rendered W")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_alpha_comparison(
    alphas_target: dict[str, tuple[float, float]],
    alphas_per_arm: dict[str, list[dict[str, tuple[float, float]]]],
    arms_order: list[arms.Arm],
    save_path: Path, *, title: str,
) -> None:
    """Per-leaf $\\alpha$ comparison: target horizontal line + per-arm boxes per leaf."""
    import matplotlib.pyplot as plt
    leaves = ("conv1", "conv2", "fc1", "fc2")
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 4.4), sharey=False)
    for ax, leaf in zip(axes, leaves, strict=True):
        data = [
            [seed_alpha[leaf][0] for seed_alpha in alphas_per_arm[a.short]]
            for a in arms_order
        ]
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
        alpha_t, r2_t = alphas_target[leaf]
        ax.axhline(alpha_t, color="#000000", linestyle="--", linewidth=1.5,
                   label=f"target α={alpha_t:.2f} (R²={r2_t:.2f})")
        ax.set_title(leaf)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# --- Reporting helpers -----------------------------------------------------
def per_arm_summary_md(
    arms_order: list[arms.Arm],
    final_mse: dict[str, list[float]],
    per_leaf_l2_per_arm: dict[str, list[dict[str, float]]],
    func_acc: dict[str, list[float]],
    target_acc: float,
) -> str:
    """Per-arm verbatim L2 + functional-fidelity table."""
    rows: list[str] = []
    rows.append(
        "| arm | total MSE (median) | total MSE (min) | total MSE (max) | "
        "func acc (median) | func acc (min) | func acc (max) |"
    )
    rows.append("|---|---|---|---|---|---|---|")
    for a in arms_order:
        m = np.asarray(final_mse[a.short])
        f = np.asarray(func_acc[a.short])
        rows.append(
            f"| {a.name} | {np.median(m):.4e} | {m.min():.4e} | {m.max():.4e} "
            f"| {np.median(f):.4f} | {f.min():.4f} | {f.max():.4f} |"
        )
    text = "\n".join(rows) + f"\n\nTarget CNN test accuracy: **{target_acc:.4f}**\n\n"

    text += "### Per-leaf L2 (sum of squares) — median across seeds\n\n"
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


def decide_interpretation(
    final_mse_median: dict[str, float], func_acc_median: dict[str, float],
    *, mse_clean_threshold: float, func_threshold: float,
) -> str:
    """Apply the pre-registered interpretation rules.

    The L2 "clean fit" threshold is heuristic — the brief says no
    pre-committed number on the L2 itself, the verdict comes from the
    *combination* of L2 trajectory + functional fidelity. Here we use
    a soft threshold to label each arm clean / not-clean for the
    pre-registered rule firing; the verbatim numbers are reported above.
    """
    par_clean = final_mse_median["fws_parallel"] < mse_clean_threshold
    par_fidel = func_acc_median["fws_parallel"] >= func_threshold
    hyp_clean = final_mse_median["fws_hyper"] < mse_clean_threshold
    hyp_fidel = func_acc_median["fws_hyper"] >= func_threshold

    if par_clean and par_fidel and hyp_clean and hyp_fidel:
        return (
            "**Both arms L2-fit cleanly AND clear functional fidelity ≥"
            f" {func_threshold:.2f}.** Both architectures can represent "
            "trained CIFAR weights — phase 6/8/9/10 task-loss failures "
            "were training dynamics, not representation."
        )
    if par_clean and par_fidel and not (hyp_clean and hyp_fidel):
        return (
            "**FWS-parallel L2-fits and clears functional fidelity, "
            "FWS-hyper does not.** The meta-renderer layer ($G_H$) "
            "specifically breaks representation, not just optimisation. "
            "The per-leaf parallel architecture *can* represent trained "
            "weights."
        )
    if hyp_clean and hyp_fidel and not (par_clean and par_fidel):
        return (
            "**FWS-hyper L2-fits and clears functional fidelity, "
            "FWS-parallel does not.** Surprising; would imply the "
            "hyper-renderer is *more* expressive than the per-rank "
            "parallel decomposition — inspect for a per-rank-G_leaf "
            "expressivity ceiling."
        )
    return (
        "**Neither arm clears the pre-registered combined "
        "L2-fit + functional-fidelity bar.** The per-leaf "
        "hyper-renderer family is the wrong shape for representing "
        "real CIFAR networks; pivot programme architecture."
    )


def main() -> None:
    print("=" * 72)
    print("Phase 11B — Supervised-distillation sanity check on FWS")
    print("=" * 72)
    print("Loading CIFAR-10 ...")
    train_x, train_y, test_x, test_y = data.load_cifar10()
    print(f"  train: {train_x.shape} | test: {test_x.shape}")
    print(f"  K_seed={K_SEED}, distill_steps={DISTILL_STEPS}, lr={DISTILL_LR}")

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

    # --- Step 2: distill each FWS arm. -------------------------------------
    arms_order: list[arms.Arm] = [
        make_fws_hyper_distill_arm(),
        make_fws_parallel_distill_arm(),
    ]
    final_mse: dict[str, list[float]] = {a.short: [] for a in arms_order}
    per_leaf_l2_per_arm: dict[str, list[dict[str, float]]] = {a.short: [] for a in arms_order}
    func_acc: dict[str, list[float]] = {a.short: [] for a in arms_order}
    alphas_per_arm: dict[str, list[dict[str, tuple[float, float]]]] = {
        a.short: [] for a in arms_order
    }
    trajectories: dict[str, list[list[tuple[int, float]]]] = {a.short: [] for a in arms_order}
    rendered_per_arm_first_seed: dict[str, dict[str, np.ndarray]] = {}

    distill_wall = 0.0
    for a in arms_order:
        print(f"\n--- Step 2: distill {a.name} (K={K_SEED}) ---")
        for seed in range(K_SEED):
            t_seed = time.time()
            _state, traj, rendered = distill_arm(
                a, target_W, seed=seed, num_steps=DISTILL_STEPS, lr=DISTILL_LR,
            )
            final = traj[-1][1]
            leaf_l2 = per_leaf_l2(rendered, target_W)
            tl, ta = eval_test_loss_acc(rendered, test_x, test_y)
            alphas = diagnostics.per_leaf_alphas(rendered)
            final_mse[a.short].append(final)
            per_leaf_l2_per_arm[a.short].append(leaf_l2)
            func_acc[a.short].append(ta)
            alphas_per_arm[a.short].append(alphas)
            trajectories[a.short].append(traj)
            if seed == 0:
                rendered_per_arm_first_seed[a.short] = rendered
            seed_wall = time.time() - t_seed
            distill_wall += seed_wall
            print(f"  {a.short} seed {seed}: final_mse={final:.4e}  "
                  f"func_acc={ta:.4f}  ({seed_wall:.1f}s)")

    # --- Step 3: numbers, figures, log. ------------------------------------
    print("\n--- Step 3: aggregate and write log ---")
    summary_md = per_arm_summary_md(
        arms_order, final_mse, per_leaf_l2_per_arm, func_acc, target_acc,
    )

    plot_distill_trajectories(
        trajectories, arms_order,
        FIGURES_DIR / "2026-06-07-phase11b-distill-trajectories.png",
        title=f"Phase 11B — distillation MSE trajectories (K={K_SEED})",
    )
    plot_functional_fidelity_box(
        func_acc, target_acc, arms_order,
        FIGURES_DIR / "2026-06-07-phase11b-functional-fidelity.png",
        title=f"Phase 11B — functional fidelity (CIFAR test acc on distilled W, K={K_SEED})",
    )
    plot_alpha_comparison(
        target_alphas, alphas_per_arm, arms_order,
        FIGURES_DIR / "2026-06-07-phase11b-alpha-comparison.png",
        title=f"Phase 11B — per-leaf α distilled vs target (K={K_SEED})",
    )

    final_mse_median = {k: float(np.median(v)) for k, v in final_mse.items()}
    func_acc_median = {k: float(np.median(v)) for k, v in func_acc.items()}
    verdict_text = decide_interpretation(
        final_mse_median, func_acc_median,
        # MSE-clean threshold is heuristic: the leaf-mean MSE has the same
        # order of magnitude as the leaf weights' variance under Kaiming init
        # (~ 1 / fan_in). Calling MSE "clean" at <1e-2 says the rendered W
        # is within an order of magnitude of init noise of the target W.
        # The pre-registered call is on the COMBINATION; we surface this
        # threshold explicitly so the reader sees the cut.
        mse_clean_threshold=1e-2, func_threshold=0.40,
    )
    print("\nVerdict:")
    print(f"  {verdict_text}")

    alpha_md = alpha_table_md(target_alphas, alphas_per_arm, arms_order)

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
        )),
        (f"Step 2 — L2 distillation (K={K_SEED} per arm)", (
            summary_md + "\n\n"
            "![distillation trajectories](figures/2026-06-07-phase11b-distill-trajectories.png)\n\n"
            "![functional fidelity (CIFAR test acc on distilled W)]"
            "(figures/2026-06-07-phase11b-functional-fidelity.png)"
        )),
        ("Step 3 — per-leaf α distilled vs target", (
            alpha_md + "\n\n"
            "![per-leaf α distilled vs target]"
            "(figures/2026-06-07-phase11b-alpha-comparison.png)"
        )),
        ("Interpretation (pre-registered rule firing)", verdict_text),
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
If the FWS arm can fit a known-good $W^{\\star}$ by L2, then the family
*can* express trained weights; the previous failures were
training-dynamics. If it cannot fit $W^{\\star}$ even by L2, the family
is the wrong shape.
"""

_SETUP_MD = """
1. Train a vanilla ``WideKernelCNN-SiLU`` on CIFAR-10 with Adam(1e-3)
   for {target_epochs} epochs. Save the trained weights as $W^{{\\star}}$.
2. For each FWS arm in {{FWS-hyper-polar (full phase 10 architecture:
   $G_H$ SIREN body + linear readout + per-leaf $G_H$ output
   normalisation + Kaiming pre-factor + polar Newton-Schulz $P$),
   FWS-parallel-no-G_H + polar $P$}}, train the FWS state via
   Adam({distill_lr}) to minimise
   $\\sum_{{\\ell}} \\text{{mean}}((W_{{\\ell}}(\\text{{state}}) -
   W^{{\\star}}_{{\\ell}})^2)$ for {distill_steps} outer steps. K={k_seed}.
3. Measure per-leaf $\\|W_{{\\text{{rendered}}}} - W^{{\\star}}\\|^2$,
   functional fidelity (test accuracy of the distilled rendered $W$ on
   CIFAR-10), and per-leaf HT-SR / radial-FFT $\\alpha$ comparison.

Pre-registered interpretation rules (combination on L2-clean fit AND
functional fidelity, not L2 alone):

- **FWS-parallel clean AND fidelity ≥ 0.40 AND FWS-hyper not** →
  the meta-renderer layer specifically breaks representation.
- **Both clean AND fidelity ≥ 0.40** → training dynamics, not
  representation, drove the failures.
- **Neither clean** → wrong family; pivot.
- **FWS-hyper clean AND parallel not** → surprising;
  per-rank-G_leaf expressivity ceiling.

L2-clean threshold for rule firing: total leaf-summed MSE < $10^{{-2}}$
(soft heuristic; the verdict reads numbers and trajectory in the
combination, per brief).
"""

_CAVEATS_MD = """
- **K={k_seed} smoke** — not K=10 confirmation.
- **One target tree.** A single Adam(1e-3) × {target_epochs}-epoch run
  produced $W^{{\\star}}$. Different target weights (different init,
  different LR, longer training) might be easier or harder to
  L2-fit; this phase reports the K={k_seed} numbers against one
  $W^{{\\star}}$.
- **L2 "clean" threshold is a soft heuristic.** The brief explicitly
  declines a pre-committed L2 number — the verdict reads the L2
  trajectory + functional fidelity in combination. We surface a
  threshold ($10^{{-2}}$) only for the boolean interpretation-rule
  firing; the verbatim L2 numbers are the primary artefact.
- **Distillation may exit at a local minimum of the rendering manifold.**
  If the FWS arm hits a flat region of the rendering function near
  $W^{{\\star}}$, the L2 stalls but the family could still in
  principle express $W^{{\\star}}$. We mitigate by running
  {distill_steps} steps; a STOP at the loss-trajectory plateau is a
  representation-side ceiling under the schedule, not a hard
  expressivity bound.
- **No data augmentation, no LR schedule.** Plain Adam(1e-3).
- **Distillation wall**: {distill_wall:.1f}s.
"""


if __name__ == "__main__":
    main()
