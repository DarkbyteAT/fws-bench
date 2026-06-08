"""Brick 1.5 — Noise probe: derive per-leaf reconstruction tolerance δ_leaf
from behaviour preservation on a trained ``WideKernelCNN-SiLU`` CIFAR-10
target.

Phases 7–12 hit case 3 of the 4-way grid (low L2 reconstruction error, low
functional fidelity) — a weight-rendering machine can sit close to a
trained target in L2 and still classify at chance. This run produces the
behaviour-anchored tolerance any future weight-rendering machine has to
clear before being declared "fit": the largest per-leaf Gaussian
perturbation, expressed as a fraction of leaf Frobenius norm, that
leaves the network's test loss within c × σ_run of the trained target
(σ_run = retraining-noise floor across K=10 seeds, c ∈ {1, 2}).

Procedure:

1. Train target ``WideKernelCNN-SiLU`` (target_seed=0) for 5 epochs at
   Adam(1e-3) — same recipe as phase 11B / 12.
2. K_target=10 retrains (same recipe, different seeds). σ_run = std of
   their final test losses; this is the empirical retraining-noise
   floor.
3. Whole-network perturbation sweep: δ ∈ logspace(-3, -1, 20). For each
   δ and each of K_noise=10 perturbation seeds, perturb every leaf
   simultaneously by ``δ × ||w||_leaf × N(0, I)``; evaluate on the test
   set.
4. Per-leaf perturbation sweep: for each leaf and each δ, perturb only
   that leaf (others frozen at target).
5. Derive δ_leaf(c) = largest δ where the median (whole-network) ratio
   ``Δloss / σ_run`` stays below c for c ∈ {1, 2}. Verdict: PROCEED to
   Brick 1 / KILL / RE-EXAMINE.

All artefacts land under ``runs/noise_probe/<iso-timestamp>/``; no hashes
in row identity or filenames. Numbers are reported verbatim; structural
invariants only are asserted. Eps for any floating-point checks comes
from ``jnp.finfo(dtype).eps × N`` with N noted.

Run::

    uv run python examples/noise_probe.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array


sys.path.insert(0, str(Path(__file__).parent))
from _common import data, mainnet, reporting  # noqa: E402


# --- Schedule ---------------------------------------------------------------
TARGET_EPOCHS = 5
TARGET_BATCH_SIZE = 128
TARGET_LR = 1e-3
K_TARGET = 10              # retraining-noise floor seeds
K_NOISE = 10               # perturbation-noise seeds per (δ, scope)
N_DELTAS = 20
# Initial sweep range [-3, -1] left the curve already above c=1 at the
# smallest δ tested (median ratio 1.105 at δ=1e-3). Widening to [-5, -1]
# brings the c=1 crossing well inside the sweep; the top stays at -1
# (median ratio ~ 2e5 at δ=1e-1 — well past the KILL regime).
DELTA_LOG10_LO = -5.0
DELTA_LOG10_HI = -1.0
TARGET_NAME = "cnn_cifar_v1"
TARGET_SEED = 0


# --- Training (same recipe as phase 11B / 12) -------------------------------
def train_cnn(
    train_x: np.ndarray, train_y: np.ndarray,
    test_x: np.ndarray, test_y: np.ndarray,
    *, seed: int, epochs: int = TARGET_EPOCHS,
    batch_size: int = TARGET_BATCH_SIZE, lr: float = TARGET_LR,
) -> tuple[dict[str, Array], list[tuple[int, float, float]]]:
    """Train ``WideKernelCNN-SiLU`` on CIFAR-10 via Adam(lr).

    Returns ``(final_params, [(epoch, test_loss, test_acc), ...])``.
    """
    key = jax.random.key(seed)
    params = mainnet.init_cnn_params(key)
    optimiser = optax.adam(lr)
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
    for epoch in range(epochs):
        idx = np.arange(n_train)
        rng.shuffle(idx)
        for s in range(steps_per_epoch):
            bi = idx[s * batch_size:(s + 1) * batch_size]
            batch = {"x": jnp.asarray(train_x[bi]), "y": jnp.asarray(train_y[bi])}
            params, opt_state, _ = step(params, opt_state, batch)
        tl, ta = eval_test_loss_acc(params, test_x, test_y)
        ckpts.append((epoch + 1, tl, ta))
        print(f"    seed {seed} epoch {epoch + 1}/{epochs}: "
              f"test_loss={tl:.4f} acc={ta:.4f}", flush=True)
    return params, ckpts


def eval_test_loss_acc(
    params: dict[str, Array], test_x: np.ndarray, test_y: np.ndarray,
    *, chunk: int = 1024,
) -> tuple[float, float]:
    """Mean test cross-entropy + accuracy on CIFAR-10."""
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


def eval_test_predictions(
    params: dict[str, Array], test_x: np.ndarray,
    *, chunk: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(predictions, max_softmax_confidences)`` for the full test set."""
    n = test_x.shape[0]
    preds = np.empty(n, dtype=np.int32)
    confs = np.empty(n, dtype=np.float32)
    for i in range(0, n, chunk):
        bx = jnp.asarray(test_x[i:i + chunk])
        logits = jax.vmap(lambda x: mainnet.cnn_forward(params, x))(bx)
        probs = jax.nn.softmax(logits, axis=-1)
        preds[i:i + bx.shape[0]] = np.asarray(jnp.argmax(logits, axis=-1))
        confs[i:i + bx.shape[0]] = np.asarray(jnp.max(probs, axis=-1))
    return preds, confs


# --- Perturbation -----------------------------------------------------------
def perturb_params(
    params: dict[str, Array], *, delta: float, key: Array,
    leaves: tuple[str, ...] | None = None,
) -> dict[str, Array]:
    """Return ``params`` with the chosen leaves perturbed by ``δ × ||w||_F × N(0, I)``.

    When ``leaves`` is ``None``, every leaf is perturbed simultaneously
    (whole-network sweep). When it lists specific leaves, only those are
    perturbed and the rest are returned unchanged (per-leaf sweep).
    """
    target_leaves = mainnet.LEAF_ORDER if leaves is None else leaves
    keys = jax.random.split(key, len(target_leaves))
    out = dict(params)
    for k, name in zip(keys, target_leaves, strict=True):
        w = params[name]
        frob = jnp.sqrt(jnp.sum(w ** 2))
        out[name] = w + delta * frob * jax.random.normal(k, w.shape, dtype=w.dtype)
    return out


# --- Row writer -------------------------------------------------------------
def append_row(path: Path, row: dict) -> None:
    """Append one JSON line to ``rows.jsonl``."""
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def make_row(
    *, measurement_name: str, value: float, units: str,
    leaf_name: str | None = None, delta: float | None = None,
    perturbation_seed: int | None = None, retrain_seed: int | None = None,
    extra: dict | None = None,
) -> dict:
    """Build a tidy row honouring the schema in the brief."""
    row: dict = {
        "target_name": TARGET_NAME,
        "target_seed": TARGET_SEED,
        "measurement_name": measurement_name,
        "leaf_name": leaf_name,
        "delta": delta,
        "perturbation_seed": perturbation_seed,
        "retrain_seed": retrain_seed,
        "value": value,
        "units": units,
        "dtype": "float32",
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
    }
    if extra:
        row.update(extra)
    return row


# --- Curated test-image picker ----------------------------------------------
def pick_curated_indices(
    preds: np.ndarray, confs: np.ndarray, test_y: np.ndarray,
) -> np.ndarray:
    """Pick 16 test indices: 8 high-conf correct, 4 high-conf wrong, 4 low-conf.

    Deterministic given the input arrays. The picker is intentionally
    biased toward visually informative cases — it is not a random sample.
    """
    correct = preds == test_y
    wrong = ~correct
    correct_idx = np.where(correct)[0]
    wrong_idx = np.where(wrong)[0]

    # 8 highest-conf correct.
    correct_by_conf = correct_idx[np.argsort(confs[correct_idx])[::-1]]
    pick_correct = correct_by_conf[:8]
    # 4 highest-conf wrong (the "confidently wrong" failures).
    wrong_by_conf = wrong_idx[np.argsort(confs[wrong_idx])[::-1]]
    pick_wrong_conf = wrong_by_conf[:4]
    # 4 lowest-conf overall (model is uncertain).
    all_by_conf = np.argsort(confs)
    pick_low = all_by_conf[:4]

    return np.concatenate([pick_correct, pick_wrong_conf, pick_low])


# --- Plots ------------------------------------------------------------------
def plot_sample_grid(
    raw_test_x: np.ndarray, test_y: np.ndarray,
    preds: np.ndarray, confs: np.ndarray, indices: np.ndarray,
    save_path: Path, *, title: str,
) -> None:
    """4×4 grid of test images with (true, pred, conf) labels colour-coded."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 4, figsize=(8.5, 9.0))
    for ax, idx in zip(axes.flat, indices, strict=True):
        img = raw_test_x[idx].transpose(1, 2, 0) / 255.0
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        truth = data.CIFAR_NAMES[int(test_y[idx])]
        pred = data.CIFAR_NAMES[int(preds[idx])]
        conf = float(confs[idx])
        correct = preds[idx] == test_y[idx]
        colour = reporting.WONG[3] if correct else reporting.WONG[6]
        ax.set_title(f"T:{truth}\nP:{pred} ({conf:.2f})", fontsize=7, color=colour)
        for sp in ax.spines.values():
            sp.set_edgecolor(colour)
            sp.set_linewidth(1.3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def plot_weight_magnitudes(
    target_W: dict[str, Array], retrain_Ws: list[dict[str, Array]],
    save_path: Path,
) -> None:
    """Per-leaf Frobenius norm: target (point) + K=10 retrained (boxplot)."""
    import matplotlib.pyplot as plt
    leaves = mainnet.LEAF_ORDER
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    box_data = []
    target_norms = []
    for name in leaves:
        target_norms.append(float(jnp.sqrt(jnp.sum(target_W[name] ** 2))))
        box_data.append([
            float(jnp.sqrt(jnp.sum(w[name] ** 2))) for w in retrain_Ws
        ])
    positions = np.arange(len(leaves))
    bp = ax.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True,
                    showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(reporting.WONG[1])
        patch.set_alpha(0.35)
        patch.set_edgecolor(reporting.WONG[1])
    for ml in bp["medians"]:
        ml.set_color(reporting.WONG[1])
        ml.set_linewidth(2.0)
    ax.scatter(positions, target_norms, color=reporting.WONG[6], s=90, zorder=4,
               edgecolor="white", linewidths=0.7, label="target")
    ax.set_xticks(positions)
    ax.set_xticklabels(leaves, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Frobenius norm")
    ax.set_yscale("log")
    ax.set_title("Per-leaf Frobenius norm — target vs K=10 retrained")
    ax.grid(True, axis="y", alpha=0.3, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_gradient_health(
    target_grads: dict[str, float], retrain_grads: list[dict[str, float]],
    save_path: Path,
) -> None:
    """Per-leaf gradient norm at convergence."""
    import matplotlib.pyplot as plt
    leaves = mainnet.LEAF_ORDER
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    box_data = [[g[name] for g in retrain_grads] for name in leaves]
    target_vals = [target_grads[name] for name in leaves]
    positions = np.arange(len(leaves))
    bp = ax.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True,
                    showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(reporting.WONG[3])
        patch.set_alpha(0.35)
        patch.set_edgecolor(reporting.WONG[3])
    for ml in bp["medians"]:
        ml.set_color(reporting.WONG[3])
        ml.set_linewidth(2.0)
    ax.scatter(positions, target_vals, color=reporting.WONG[6], s=90, zorder=4,
               edgecolor="white", linewidths=0.7, label="target")
    ax.set_xticks(positions)
    ax.set_xticklabels(leaves, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("‖∂L/∂w‖₂ on one held-out batch")
    ax.set_yscale("log")
    ax.set_title("Per-leaf gradient norm at convergence — target vs K=10 retrained")
    ax.grid(True, axis="y", alpha=0.3, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_retraining_noise(
    target_ckpts: list[tuple[int, float, float]],
    retrain_ckpts_list: list[list[tuple[int, float, float]]],
    sigma_run: float,
    save_path: Path,
) -> None:
    """K=10 training trajectories + final-loss boxplot + σ_run."""
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                   gridspec_kw={"width_ratios": [2.2, 1]})
    for ckpts in retrain_ckpts_list:
        epochs = [c[0] for c in ckpts]
        losses = [c[1] for c in ckpts]
        ax1.plot(epochs, losses, color=reporting.WONG[1], alpha=0.55, linewidth=1.0)
    t_epochs = [c[0] for c in target_ckpts]
    t_losses = [c[1] for c in target_ckpts]
    ax1.plot(t_epochs, t_losses, color=reporting.WONG[6], linewidth=2.3,
             label="target (seed 0)")
    ax1.set(xlabel="epoch", ylabel="test cross-entropy",
            title=f"K={K_TARGET} retraining trajectories")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)

    finals = [c[-1][1] for c in retrain_ckpts_list]
    bp = ax2.boxplot([finals], widths=0.55, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(reporting.WONG[1])
        patch.set_alpha(0.35)
    for ml in bp["medians"]:
        ml.set_color(reporting.WONG[1])
        ml.set_linewidth(2.0)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.10, 0.10, size=len(finals))
    ax2.scatter(np.full(len(finals), 1) + jitter, finals,
                color=reporting.WONG[1], s=40, zorder=3, edgecolor="white",
                linewidths=0.6)
    ax2.axhline(target_ckpts[-1][1], color=reporting.WONG[6], linestyle="--",
                linewidth=1.5, label=f"target final={target_ckpts[-1][1]:.4f}")
    ax2.set_xticks([1])
    ax2.set_xticklabels(["retrains"])
    ax2.set_ylabel("final test cross-entropy")
    ax2.set_title(f"σ_run = {sigma_run:.4e}")
    ax2.legend(loc="best", fontsize=7)
    ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _delta_leaf_from_curve(
    deltas: np.ndarray, ratios_median: np.ndarray, c: float,
) -> tuple[float | None, str]:
    """Largest δ where median ratio < c.

    Returns ``(δ_leaf, note)``. ``note`` is the empty string in the
    well-behaved case, otherwise documents the corner case
    (never-crosses → use largest δ tested; crosses-at-smallest → KILL).
    """
    below = ratios_median < c
    if not np.any(below):
        return None, f"curve never below c={c}; smallest δ tested = {deltas[0]:.3e}"
    if np.all(below):
        return float(deltas[-1]), (
            f"curve stays below c={c} for every δ; reporting largest tested = "
            f"{deltas[-1]:.3e} as a lower bound on δ_leaf(c={c})"
        )
    # Find the last index where ratio < c (the crossing transition).
    idx = int(np.where(below)[0].max())
    return float(deltas[idx]), ""


def plot_perturbation_curve(
    deltas: np.ndarray, ratios: np.ndarray, sigma_run: float,
    delta_c1: float | None, delta_c2: float | None,
    save_path: Path,
) -> None:
    """Median + IQR of Δloss/σ_run vs δ (whole-network sweep)."""
    import matplotlib.pyplot as plt
    median = np.median(ratios, axis=1)
    q25 = np.quantile(ratios, 0.25, axis=1)
    q75 = np.quantile(ratios, 0.75, axis=1)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.fill_between(deltas, q25, q75, color=reporting.WONG[1], alpha=0.25,
                    label="IQR")
    ax.plot(deltas, median, color=reporting.WONG[1], linewidth=2.0,
            marker="o", label="median")
    ax.axhline(1.0, color=reporting.WONG[3], linestyle="--", linewidth=1.2,
               label="c=1")
    ax.axhline(2.0, color=reporting.WONG[6], linestyle="--", linewidth=1.2,
               label="c=2")
    if delta_c1 is not None:
        ax.axvline(delta_c1, color=reporting.WONG[3], linestyle=":", linewidth=1.5,
                   label=f"δ_leaf(c=1)={delta_c1:.3e}")
    if delta_c2 is not None:
        ax.axvline(delta_c2, color=reporting.WONG[6], linestyle=":", linewidth=1.5,
                   label=f"δ_leaf(c=2)={delta_c2:.3e}")
    ax.set_xscale("log")
    ax.set(xlabel="δ (fraction of leaf Frobenius norm)",
           ylabel="Δloss / σ_run",
           title=f"Whole-network perturbation curve  (σ_run = {sigma_run:.4e})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_per_leaf_sensitivity(
    deltas: np.ndarray, ratios_by_leaf: dict[str, np.ndarray],
    worst_leaf: str, save_path: Path,
) -> None:
    """One curve per leaf on shared axes; worst leaf highlighted."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    palette = reporting.WONG
    for i, name in enumerate(mainnet.LEAF_ORDER):
        ratios = ratios_by_leaf[name]
        median = np.median(ratios, axis=1)
        is_worst = name == worst_leaf
        ax.plot(deltas, median,
                color=palette[i % len(palette)],
                linewidth=2.6 if is_worst else 1.4,
                marker="o" if is_worst else None,
                label=f"{name}{' (worst)' if is_worst else ''}")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.axhline(2.0, color="black", linestyle="--", linewidth=1.0, alpha=0.3)
    ax.set_xscale("log")
    ax.set(xlabel="δ (fraction of perturbed-leaf Frobenius norm)",
           ylabel="Δloss / σ_run (median over K_noise)",
           title="Per-leaf sensitivity (other leaves frozen at target)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# --- One-batch gradient health ---------------------------------------------
def per_leaf_grad_norms(
    params: dict[str, Array], batch: dict,
) -> dict[str, float]:
    """Compute ``‖∂L/∂w‖₂`` for each leaf on one batch — no training step."""
    grads = jax.grad(mainnet.cross_entropy_loss)(params, batch)
    return {n: float(jnp.sqrt(jnp.sum(grads[n] ** 2)))
            for n in mainnet.LEAF_ORDER}


# --- Main -------------------------------------------------------------------
def main() -> None:
    t_start = time.time()
    print("=" * 72)
    print("Brick 1.5 — Noise probe (behaviour-anchored δ_leaf derivation)")
    print("=" * 72)

    # Folder = ISO 8601 timestamp, no hashes.
    stamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = Path(__file__).parent.parent / "runs" / "noise_probe" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    rows_path = run_dir / "rows.jsonl"
    rows_path.write_text("")  # truncate
    print(f"Run directory: {run_dir}")

    print("Loading CIFAR-10 ...")
    train_x, train_y, test_x, test_y = data.load_cifar10()
    raw_test_x = data.load_raw_test()
    print(f"  train: {train_x.shape} | test: {test_x.shape}")

    # --- Step 1: train target ------------------------------------------------
    print(f"\n--- Step 1: train target (seed={TARGET_SEED}, "
          f"{TARGET_EPOCHS} epochs) ---")
    target_W, target_ckpts = train_cnn(
        train_x, train_y, test_x, test_y, seed=TARGET_SEED,
    )
    target_test_loss, target_test_acc = target_ckpts[-1][1], target_ckpts[-1][2]
    n_target_params = int(sum(np.asarray(v).size for v in target_W.values()))
    print(f"  target: test_loss={target_test_loss:.4f}  "
          f"acc={target_test_acc:.4f}  params={n_target_params}")
    append_row(rows_path, make_row(
        measurement_name="target_test_loss", value=target_test_loss,
        units="cross_entropy",
    ))
    append_row(rows_path, make_row(
        measurement_name="target_test_acc", value=target_test_acc,
        units="accuracy",
    ))

    # Per-leaf target diagnostics: Frobenius norm + one-batch gradient norm.
    rng = np.random.default_rng(TARGET_SEED)
    diag_idx = rng.choice(train_x.shape[0], size=TARGET_BATCH_SIZE, replace=False)
    diag_batch = {"x": jnp.asarray(train_x[diag_idx]),
                  "y": jnp.asarray(train_y[diag_idx])}
    target_grads = per_leaf_grad_norms(target_W, diag_batch)
    target_norms = {n: float(jnp.sqrt(jnp.sum(target_W[n] ** 2)))
                    for n in mainnet.LEAF_ORDER}
    for name in mainnet.LEAF_ORDER:
        append_row(rows_path, make_row(
            measurement_name="target_leaf_frob_norm",
            leaf_name=name, value=target_norms[name], units="frobenius",
        ))
        append_row(rows_path, make_row(
            measurement_name="target_leaf_grad_norm",
            leaf_name=name, value=target_grads[name], units="grad_l2",
        ))

    # Step 1 ping.

    # --- Step 2: K=10 retrains -> σ_run -------------------------------------
    print(f"\n--- Step 2: K={K_TARGET} retraining-noise floor ---")
    retrain_Ws: list[dict[str, Array]] = []
    retrain_ckpts_list: list[list[tuple[int, float, float]]] = []
    retrain_grads_list: list[dict[str, float]] = []
    for k in range(K_TARGET):
        retrain_seed = 1000 + k
        print(f"  retrain {k + 1}/{K_TARGET} (seed={retrain_seed})")
        W_k, ckpts_k = train_cnn(
            train_x, train_y, test_x, test_y, seed=retrain_seed,
        )
        retrain_Ws.append(W_k)
        retrain_ckpts_list.append(ckpts_k)
        final_loss_k = ckpts_k[-1][1]
        final_acc_k = ckpts_k[-1][2]
        append_row(rows_path, make_row(
            measurement_name="retrain_final_test_loss", value=final_loss_k,
            units="cross_entropy", retrain_seed=retrain_seed,
        ))
        append_row(rows_path, make_row(
            measurement_name="retrain_final_test_acc", value=final_acc_k,
            units="accuracy", retrain_seed=retrain_seed,
        ))
        grads_k = per_leaf_grad_norms(W_k, diag_batch)
        retrain_grads_list.append(grads_k)
        for name in mainnet.LEAF_ORDER:
            append_row(rows_path, make_row(
                measurement_name="retrain_leaf_frob_norm",
                leaf_name=name,
                value=float(jnp.sqrt(jnp.sum(W_k[name] ** 2))),
                units="frobenius", retrain_seed=retrain_seed,
            ))
            append_row(rows_path, make_row(
                measurement_name="retrain_leaf_grad_norm",
                leaf_name=name, value=grads_k[name], units="grad_l2",
                retrain_seed=retrain_seed,
            ))

    final_losses = np.array([c[-1][1] for c in retrain_ckpts_list])
    # ddof=1: empirical std with Bessel's correction (K=10 small sample).
    sigma_run = float(final_losses.std(ddof=1))
    print(f"  σ_run = {sigma_run:.4e}  (K={K_TARGET}, ddof=1)")
    print(f"  retrain final losses: {final_losses}")
    append_row(rows_path, make_row(
        measurement_name="sigma_run", value=sigma_run,
        units="cross_entropy", extra={"k_target": K_TARGET, "ddof": 1},
    ))

    # Step 2 ping.

    # --- Step 3: whole-network perturbation sweep ---------------------------
    print(f"\n--- Step 3: whole-network perturbation sweep "
          f"({N_DELTAS} deltas × K_noise={K_NOISE}) ---")
    deltas = jnp.logspace(DELTA_LOG10_LO, DELTA_LOG10_HI, N_DELTAS)
    deltas_np = np.asarray(deltas)
    whole_ratios = np.zeros((N_DELTAS, K_NOISE), dtype=np.float64)
    whole_losses = np.zeros_like(whole_ratios)
    base_key = jax.random.key(7919)
    for i, delta in enumerate(deltas_np):
        for j in range(K_NOISE):
            key = jax.random.fold_in(base_key, i * K_NOISE + j)
            W_pert = perturb_params(target_W, delta=float(delta), key=key)
            loss_p, acc_p = eval_test_loss_acc(W_pert, test_x, test_y)
            whole_losses[i, j] = loss_p
            ratio = (loss_p - target_test_loss) / sigma_run
            whole_ratios[i, j] = ratio
            append_row(rows_path, make_row(
                measurement_name="whole_perturb_loss", value=loss_p,
                units="cross_entropy", delta=float(delta),
                perturbation_seed=int(i * K_NOISE + j),
            ))
            append_row(rows_path, make_row(
                measurement_name="whole_perturb_ratio", value=ratio,
                units="dimensionless", delta=float(delta),
                perturbation_seed=int(i * K_NOISE + j),
            ))
        med = float(np.median(whole_ratios[i]))
        print(f"  δ={float(delta):.3e}  median Δloss/σ_run = {med:.3f}")

    # Step 3 ping.

    # --- Step 4: per-leaf perturbation sweep --------------------------------
    print(f"\n--- Step 4: per-leaf perturbation sweep "
          f"({mainnet.N_LEAVES} leaves × {N_DELTAS} deltas × K_noise={K_NOISE}) ---")
    per_leaf_ratios: dict[str, np.ndarray] = {}
    base_key_leaf = jax.random.key(15485863)
    for li, name in enumerate(mainnet.LEAF_ORDER):
        print(f"  leaf {li + 1}/{mainnet.N_LEAVES}: {name}")
        ratios = np.zeros((N_DELTAS, K_NOISE), dtype=np.float64)
        for i, delta in enumerate(deltas_np):
            for j in range(K_NOISE):
                key = jax.random.fold_in(
                    base_key_leaf, li * N_DELTAS * K_NOISE + i * K_NOISE + j,
                )
                W_pert = perturb_params(
                    target_W, delta=float(delta), key=key, leaves=(name,),
                )
                loss_p, _ = eval_test_loss_acc(W_pert, test_x, test_y)
                ratio = (loss_p - target_test_loss) / sigma_run
                ratios[i, j] = ratio
                append_row(rows_path, make_row(
                    measurement_name="per_leaf_perturb_loss",
                    leaf_name=name, value=loss_p, units="cross_entropy",
                    delta=float(delta), perturbation_seed=int(j),
                ))
                append_row(rows_path, make_row(
                    measurement_name="per_leaf_perturb_ratio",
                    leaf_name=name, value=ratio, units="dimensionless",
                    delta=float(delta), perturbation_seed=int(j),
                ))
        per_leaf_ratios[name] = ratios
        med_curve = np.median(ratios, axis=1)
        print(f"    median Δloss/σ_run curve: "
              f"min={med_curve.min():.3f}  max={med_curve.max():.3f}")

    # Step 4 ping.

    # --- Step 5: derive δ_leaf ----------------------------------------------
    print("\n--- Step 5: derive δ_leaf ---")
    whole_med = np.median(whole_ratios, axis=1)
    delta_c1, note_c1 = _delta_leaf_from_curve(deltas_np, whole_med, c=1.0)
    delta_c2, note_c2 = _delta_leaf_from_curve(deltas_np, whole_med, c=2.0)
    print(f"  δ_leaf(c=1) = {delta_c1}  {note_c1}")
    print(f"  δ_leaf(c=2) = {delta_c2}  {note_c2}")
    append_row(rows_path, make_row(
        measurement_name="delta_leaf_c1",
        value=float(delta_c1) if delta_c1 is not None else float("nan"),
        units="fraction_of_frob_norm",
        extra={"c": 1.0, "note": note_c1},
    ))
    append_row(rows_path, make_row(
        measurement_name="delta_leaf_c2",
        value=float(delta_c2) if delta_c2 is not None else float("nan"),
        units="fraction_of_frob_norm",
        extra={"c": 2.0, "note": note_c2},
    ))

    # Worst leaf — lowest δ where per-leaf median ratio first exceeds c=1.
    # If a leaf never crosses c=1, it's not the worst. Tie-break by ratio at
    # the largest δ.
    worst_leaf, worst_score = None, -np.inf
    for name in mainnet.LEAF_ORDER:
        med = np.median(per_leaf_ratios[name], axis=1)
        above = med >= 1.0
        if np.any(above):
            first_cross = int(np.argmax(above))
            # Lower first_cross index = more sensitive = worse.
            score = -first_cross + med[-1] * 1e-3
        else:
            score = -np.inf + med[-1] * 1e-3
        if score > worst_score:
            worst_score = score
            worst_leaf = name
    print(f"  worst leaf (by per-leaf sensitivity): {worst_leaf}")
    append_row(rows_path, make_row(
        measurement_name="worst_leaf_name", value=float("nan"),
        units="leaf_name", leaf_name=worst_leaf,
    ))

    # Step 5 ping.

    # --- Step 6: artefacts --------------------------------------------------
    print("\n--- Step 6: write artefacts ---")

    # 3. target_weights.npz: target + K=10 retrained, dict-keyed.
    weight_archive: dict[str, np.ndarray] = {}
    for name in mainnet.LEAF_ORDER:
        weight_archive[f"target__{name}"] = np.asarray(target_W[name])
        for k, W_k in enumerate(retrain_Ws):
            weight_archive[f"retrain_{k}__{name}"] = np.asarray(W_k[name])
    np.savez(run_dir / "target_weights.npz", **weight_archive)

    # 4. samples_target.png — curated 4×4 grid by the target.
    preds_target, confs_target = eval_test_predictions(target_W, test_x)
    curated_idx = pick_curated_indices(preds_target, confs_target, test_y)
    plot_sample_grid(
        raw_test_x, test_y, preds_target, confs_target, curated_idx,
        run_dir / "samples_target.png",
        title="Target predictions on curated test images",
    )

    # 5. samples_perturbed_at_transition.png — same indices, network perturbed
    #    at δ ≈ c=1 crossing.
    if delta_c1 is not None:
        # Use the δ on the grid closest to delta_c1 — c=1 transition.
        idx_close = int(np.argmin(np.abs(deltas_np - delta_c1)))
        delta_transition = float(deltas_np[idx_close])
    else:
        # Fall back to the largest tested δ if c=1 never crosses.
        delta_transition = float(deltas_np[-1])
    key_t = jax.random.key(31337)
    W_transition = perturb_params(target_W, delta=delta_transition, key=key_t)
    preds_t, confs_t = eval_test_predictions(W_transition, test_x)
    plot_sample_grid(
        raw_test_x, test_y, preds_t, confs_t, curated_idx,
        run_dir / "samples_perturbed_at_transition.png",
        title=(f"Predictions at δ ≈ c=1 crossing  "
               f"(δ={delta_transition:.3e})  — same images as samples_target"),
    )

    # 6 / 7. weight magnitudes + gradient health.
    plot_weight_magnitudes(target_W, retrain_Ws,
                           run_dir / "weight_magnitudes.png")
    plot_gradient_health(target_grads, retrain_grads_list,
                         run_dir / "gradient_health.png")

    # 8. retraining noise.
    plot_retraining_noise(target_ckpts, retrain_ckpts_list, sigma_run,
                          run_dir / "retraining_noise.png")

    # 9. perturbation curve.
    plot_perturbation_curve(
        deltas_np, whole_ratios, sigma_run, delta_c1, delta_c2,
        run_dir / "perturbation_curve.png",
    )

    # 10. per-leaf sensitivity.
    plot_per_leaf_sensitivity(
        deltas_np, per_leaf_ratios, worst_leaf or mainnet.LEAF_ORDER[0],
        run_dir / "per_leaf_sensitivity.png",
    )

    # 2. target_card.md — per-leaf table + summary.
    target_card_lines: list[str] = [
        "# Target card — cnn_cifar_v1\n",
        f"- Target seed: {TARGET_SEED}",
        f"- Total params: {n_target_params}",
        f"- Final test loss: **{target_test_loss:.4f}**",
        f"- Final test acc: **{target_test_acc:.4f}**",
        f"- Training: WideKernelCNN-SiLU, Adam(lr={TARGET_LR}), "
        f"{TARGET_EPOCHS} epochs, batch {TARGET_BATCH_SIZE}\n",
        "## Per-leaf statistics at convergence\n",
        "| leaf | params | Frobenius norm | ‖∂L/∂w‖₂ (one batch) |",
        "|---|---|---|---|",
    ]
    for name in mainnet.LEAF_ORDER:
        target_card_lines.append(
            f"| {name} | {mainnet.LEAF_SIZES[name]} | "
            f"{target_norms[name]:.4e} | {target_grads[name]:.4e} |"
        )
    (run_dir / "target_card.md").write_text("\n".join(target_card_lines) + "\n")

    # 11. summary.md — findings + recommendation.
    if delta_c1 is None or (note_c1 and "never below" in note_c1):
        recommendation = "KILL"
        rec_reason = (
            "The whole-network perturbation curve never drops below c=1 "
            f"even at the smallest tested δ={deltas_np[0]:.3e}. The target "
            "is exquisitely sensitive to any reconstruction error — no "
            "non-trivial tolerance for a weight-rendering machine exists "
            "against this target."
        )
    elif note_c2 and "stays below" in note_c2:
        recommendation = "RE-EXAMINE"
        rec_reason = (
            "The curve stays below c=2 for every tested δ; the upper end "
            f"of the sweep (δ_max={deltas_np[-1]:.3e}) is below the c=2 "
            "transition. The sweep range is too tight; widen the δ range "
            "and re-run before declaring δ_leaf(c=2)."
        )
    else:
        recommendation = "PROCEED"
        rec_reason = (
            f"A behaviour-anchored tolerance exists: δ_leaf(c=1)="
            f"{delta_c1:.3e}, δ_leaf(c=2)={delta_c2:.3e}. Brick 1 may "
            f"proceed with δ_leaf(c=1) as the conservative reconstruction "
            "tolerance and δ_leaf(c=2) as the loose tolerance for early "
            "diagnostic work."
        )

    median_whole_str = "\n".join(
        f"| {float(deltas_np[i]):.3e} | {whole_med[i]:.4f} | "
        f"{np.quantile(whole_ratios[i], 0.25):.4f} | "
        f"{np.quantile(whole_ratios[i], 0.75):.4f} |"
        for i in range(N_DELTAS)
    )

    per_leaf_med_str = "\n".join(
        f"| {name} | "
        + " | ".join(
            f"{np.median(per_leaf_ratios[name], axis=1)[i]:.3f}"
            for i in range(N_DELTAS)
        )
        + " |"
        for name in mainnet.LEAF_ORDER
    )
    per_leaf_hdr = (
        "| leaf | " + " | ".join(f"δ={float(d):.2e}" for d in deltas_np) + " |"
    )

    summary_lines: list[str] = [
        "# Brick 1.5 — Noise probe summary\n",
        f"Run: ``{stamp}``  ", f"Target: ``{TARGET_NAME}`` (seed={TARGET_SEED})\n",
        "## Headline findings\n",
        f"- **σ_run** = {sigma_run:.4e} (K={K_TARGET} retrains, ddof=1)",
        f"- **Target test loss** = {target_test_loss:.4f}",
        f"- **Target test acc**  = {target_test_acc:.4f}",
        "- **δ_leaf(c=1)** = "
        + (f"{delta_c1:.4e}" if delta_c1 is not None else "n/a")
        + (f"  ({note_c1})" if note_c1 else ""),
        "- **δ_leaf(c=2)** = "
        + (f"{delta_c2:.4e}" if delta_c2 is not None else "n/a")
        + (f"  ({note_c2})" if note_c2 else ""),
        f"- **Worst leaf** = ``{worst_leaf}``",
        f"- **Recommendation** = **{recommendation}**\n",
        "## Recommendation rationale\n",
        rec_reason + "\n",
        "## Interpretation\n",
        (
            "σ_run is the irreducible test-loss spread across independent retrains "
            "of the same recipe — it sets the floor at which a perturbed network "
            "is statistically indistinguishable from a retrained one. δ_leaf(c) "
            "answers: how much Gaussian noise (as a fraction of leaf Frobenius "
            "norm) can be injected into each leaf before the test loss departs "
            "from the target by more than c standard deviations of that "
            "retraining noise floor."
        ),
        (
            "δ_leaf(c=1) is the *strict* behaviour-preserving tolerance: a "
            "weight-rendering machine reconstructing every leaf within this "
            "noise scale is functionally within one retraining-σ of the target. "
            "δ_leaf(c=2) is the looser tolerance for early-stack diagnostic "
            "work — within two retraining-σ, distinguishable from but plausibly "
            "indistinct from a retrained network."
        ),
        (
            "The per-leaf curves identify which leaves drive the global "
            "tolerance: the ``worst leaf`` is the one whose isolated "
            "perturbation crosses c=1 at the smallest δ. A reconstruction "
            "machine that meets δ_leaf overall but blows budget on that leaf "
            "will functionally fail."
        ),
        "",
        "## Whole-network perturbation curve (median + IQR)\n",
        "| δ | median Δloss/σ_run | 25th | 75th |",
        "|---|---|---|---|",
        median_whole_str,
        "",
        "## Per-leaf median Δloss/σ_run\n",
        per_leaf_hdr,
        "|" + "---|" * (N_DELTAS + 1),
        per_leaf_med_str,
        "",
        "## Artefacts\n",
        "- ``rows.jsonl`` — every measurement as a JSON line",
        "- ``target_card.md`` — target params + per-leaf stats",
        "- ``target_weights.npz`` — target + K=10 retrained weights",
        "- ``samples_target.png`` — curated 4×4 test images (target preds)",
        "- ``samples_perturbed_at_transition.png`` — same images at δ ≈ c=1",
        "- ``weight_magnitudes.png`` — per-leaf Frobenius norm box",
        "- ``gradient_health.png`` — per-leaf gradient-norm box",
        "- ``retraining_noise.png`` — K=10 trajectories + σ_run",
        "- ``perturbation_curve.png`` — whole-network sweep",
        "- ``per_leaf_sensitivity.png`` — one curve per leaf",
        "",
        f"Wall time: {time.time() - t_start:.1f}s",
    ]
    summary_path = run_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"  wrote {summary_path}")

    # Step 6 ping.

    # --- Step 7: render summary.md → summary.pdf via quill ------------------
    print("\n--- Step 7: render summary.pdf via quill ---")
    pdf_path = reporting.render_pdf(summary_path)
    if pdf_path is None:
        print("  PDF rendering skipped or failed (markdown is the primary artefact)")
    else:
        print(f"  wrote {pdf_path}")

    print(f"\nDone. Run folder: {run_dir}")
    print(f"Total wall: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
