"""Wong-palette plots + research-log markdown writer + ``quill`` PDF render.

The plot family is parametric in the arm list — pass the list of arms that
were trained and the per-seed results, and the plot functions auto-key off
``arm.short`` for figure colours, labels, and dict access.

The research-log writer takes a small ``ReportContext`` dict and a list of
``(section_title, section_body_md)`` tuples and stitches them together
with the figures into a single markdown file; ``render_pdf`` then pipes
the markdown through ``quill`` to produce a PDF alongside it.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import mainnet
from .arms import (
    FWS_HYPER_COLOUR,
    FWS_PARALLEL_COLOUR,
    W_MATCHED_COLOUR,
    W_OVERPARAM_COLOUR,
    WONG,
    Arm,
)


# --- Helpers ----------------------------------------------------------------
def median_iqr_min_max(xs: np.ndarray) -> tuple[float, float, float, float]:
    """Return ``(median, IQR, min, max)`` of an array."""
    return (
        float(np.median(xs)),
        float(np.quantile(xs, 0.75) - np.quantile(xs, 0.25)),
        float(xs.min()),
        float(xs.max()),
    )


def _short_label(name: str) -> str:
    """Compress long arm labels for boxplot x-axis tick labels."""
    return name.replace("FWS-parallel-no-G_H", "FWS-par")


def _select_reps(per_seed: list[dict], acc_key: str) -> dict[str, dict]:
    """Pick best / median / worst seeds by ``acc_key`` for curated plots."""
    s = sorted(per_seed, key=lambda r: r[acc_key], reverse=True)
    return {"best": s[0], "median": s[len(s) // 2], "worst": s[-1]}


# --- Trajectory plots -------------------------------------------------------
def plot_loss_trajectories(
    per_seed: list[dict],
    arms_list: list[Arm],
    save_path: Path,
    *,
    title: str,
) -> None:
    """Per-arm training-loss trajectories: faint per-seed + bold median."""
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for a in arms_list:
        key = f"{a.short}_losses"
        for r in per_seed:
            ax.plot(r[key][:, 0], r[key][:, 1], color=a.color, alpha=0.3, linewidth=1.0)
        stack = np.stack([r[key][:, 1] for r in per_seed], axis=0)
        xs = per_seed[0][key][:, 0]
        ax.plot(xs, np.median(stack, axis=0), color=a.color, linewidth=2.0, label=a.name)
    ax.set_yscale("log")
    ax.set(xlabel="outer step", ylabel="training cross-entropy", title=title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_acc_trajectories(
    per_seed: list[dict],
    arms_list: list[Arm],
    save_path: Path,
    *,
    title: str,
) -> None:
    """Per-arm test-accuracy-by-epoch trajectories: faint per-seed + bold median."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for a in arms_list:
        key = f"{a.short}_ckpts"
        for r in per_seed:
            ax.plot(r[key][:, 0], r[key][:, 2], color=a.color, alpha=0.4, linewidth=1.0)
        stack = np.stack([r[key][:, 2] for r in per_seed], axis=0)
        ep = per_seed[0][key][:, 0]
        ax.plot(ep, np.median(stack, axis=0), color=a.color, linewidth=2.0,
                label=f"{a.name} (median)")
    ax.set(xlabel="epoch", ylabel="test accuracy", title=title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# --- α boxplots -------------------------------------------------------------
def _alpha_panel(ax, per_seed: list[dict], arms_list: list[Arm], leaf: str, title: str) -> None:
    # Per-arm α dicts live under "{arm.short}.alphas" — set by the arm's
    # diagnostics_at_convergence callback. Arms with no callback are
    # silently skipped so the rest of the panel still renders.
    arms_with_alphas = [
        a for a in arms_list
        if all(f"{a.short}.alphas" in r and leaf in r[f"{a.short}.alphas"] for r in per_seed)
    ]
    if not arms_with_alphas:
        ax.set_title(f"{title} — no α data", fontsize=10)
        return
    data = [[r[f"{a.short}.alphas"][leaf][0] for r in per_seed] for a in arms_with_alphas]
    labels = [_short_label(a.name) for a in arms_with_alphas]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False, tick_labels=labels)
    cols = [a.color for a in arms_with_alphas]
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
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        jitter = rng.uniform(-0.10, 0.10, size=v.size)
        ax.scatter(np.full_like(v, i) + jitter, v, color=c, s=30, zorder=3,
                   edgecolor="white", linewidths=0.6)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=7)


def plot_htsr_box(
    per_seed: list[dict],
    arms_list: list[Arm],
    save_path: Path,
    *,
    title: str,
) -> None:
    """HT-SR α boxplots on fc1 and fc2 weights."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    _alpha_panel(axes[0], per_seed, arms_list, "fc1", "HT-SR α (fc1)")
    _alpha_panel(axes[1], per_seed, arms_list, "fc2", "HT-SR α (fc2)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_fft_box(
    per_seed: list[dict],
    arms_list: list[Arm],
    save_path: Path,
    *,
    title: str,
) -> None:
    """Radial-FFT α boxplots on conv1 and conv2 kernels."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    _alpha_panel(axes[0], per_seed, arms_list, "conv1", "radial-FFT α (conv1)")
    _alpha_panel(axes[1], per_seed, arms_list, "conv2", "radial-FFT α (conv2)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# --- Final-accuracy boxplot ------------------------------------------------
def plot_final_acc_box(
    per_seed: list[dict],
    arms_list: list[Arm],
    save_path: Path,
    *,
    title: str,
) -> None:
    """Final test accuracy boxplot per arm."""
    data = [[r[f"final_{a.short}_acc"] for r in per_seed] for a in arms_list]
    cols = [a.color for a in arms_list]
    labels = [_short_label(a.name) for a in arms_list]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False, tick_labels=labels)
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
    ax.set_ylabel("final test accuracy")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# --- Jacobian σ-spectrum plot ----------------------------------------------
def plot_jacobian_spectrum(
    per_seed: list[dict],
    save_path: Path,
    *,
    title: str,
    sigma_keys: dict[str, str],
) -> None:
    """Plot Jacobian σ-spectra stored in ``per_seed[*][sigma_keys[label]]``.

    ``sigma_keys`` maps a legend label (e.g. ``"hyper: σ(∂render/∂z)"``)
    to the per-seed result-dict key holding a 1-D array of singular values.
    Solid lines are FWS-hyper; dashed lines are an alternate spectrum on
    the same arm; FWS-parallel is shown as its own colour.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    palette = {
        "hyper-z": (FWS_HYPER_COLOUR, "-"),
        "hyper-gh": (FWS_HYPER_COLOUR, "--"),
        "parallel": (FWS_PARALLEL_COLOUR, "-"),
    }
    for label, key in sigma_keys.items():
        if "hyper" in label and "G_H" in label:
            colour, style = palette["hyper-gh"]
        elif "parallel" in label:
            colour, style = palette["parallel"]
        else:
            colour, style = palette["hyper-z"]
        for r in per_seed:
            spec = r[key]
            ks = np.arange(1, spec.shape[0] + 1)
            ax.plot(ks, spec, color=colour, alpha=0.55, linewidth=1.0, linestyle=style)
        ax.plot([], [], color=colour, linewidth=2.0, linestyle=style, label=label)
    ax.set(xlabel="singular-value rank (1 = largest)", ylabel="σ", title=title)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# --- Stage-0 falsifier plot ------------------------------------------------
def plot_stage0(
    records: dict[str, list[tuple[int, float]]],
    save_path: Path,
    *,
    title: str,
    scale_bounds: dict[str, float] | None = None,
) -> None:
    """Plot σ_min vs step per scale_kind for the Stage-0 falsifier."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    colours = [WONG[5], WONG[3], WONG[6], WONG[1]]
    for (scale_kind, recs), colour in zip(records.items(), colours, strict=False):
        steps = [r[0] for r in recs]
        vals = [r[1] for r in recs]
        label = (f"{scale_kind} (bound={scale_bounds[scale_kind]:.3g})"
                 if scale_bounds is not None and scale_kind in scale_bounds
                 else scale_kind)
        ax.plot(steps, vals, marker="o", linewidth=2.0, color=colour, label=label)
    ax.set(xlabel="outer step", ylabel="sigma_min(d render / d z)", title=title)
    ax.set_yscale("log")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# --- Curated outputs: conv kernels + confusion + sample preds --------------
def plot_conv_kernels(
    per_seed: list[dict],
    arm: Arm,
    save_path: Path,
    *,
    title: str,
) -> None:
    """3-row (best/median/worst) curated panel of conv1 kernels for one arm."""
    rep = _select_reps(per_seed, f"final_{arm.short}_acc")
    row_labels = ["best", "median", "worst"]
    fig, axes = plt.subplots(3, mainnet.CONV1_OUT,
                             figsize=(mainnet.CONV1_OUT * 1.0, 3.4))
    for row, label in enumerate(row_labels):
        r = rep[label]
        c1 = r[f"{arm.short}_W"]["conv1_w"]
        for k in range(mainnet.CONV1_OUT):
            f = c1[k].transpose(1, 2, 0)
            mn, mx = float(f.min()), float(f.max())
            f_n = (f - mn) / (mx - mn + 1e-9)
            axes[row, k].imshow(f_n, interpolation="nearest")
            axes[row, k].set_xticks([])
            axes[row, k].set_yticks([])
            if k == 0:
                axes[row, k].set_ylabel(
                    f"{label}\nseed {r['seed']}\nacc {r[f'final_{arm.short}_acc']:.3f}",
                    fontsize=7,
                )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((mainnet.NUM_CLASSES, mainnet.NUM_CLASSES), dtype=np.int32)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[t, p] += 1
    return cm


def plot_arm_curation(
    per_seed: list[dict],
    arm: Arm,
    test_y: np.ndarray,
    raw_test_x: np.ndarray,
    save_path: Path,
    *,
    title: str,
    class_names: tuple[str, ...],
) -> None:
    """3-row (best/median/worst) confusion + 12 sample test-image predictions."""
    rep = _select_reps(per_seed, f"final_{arm.short}_acc")
    row_labels = ["best", "median", "worst"]
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(test_y.shape[0], size=12, replace=False)

    fig = plt.figure(figsize=(15.0, 11.0))
    gs = fig.add_gridspec(3, 15, wspace=0.4, hspace=0.5)

    for row, label in enumerate(row_labels):
        r = rep[label]
        preds = r[f"{arm.short}_test_preds"]
        ax_cm = fig.add_subplot(gs[row, 0:3])
        cm = _confusion_matrix(test_y, preds)
        ax_cm.imshow(cm, cmap="Blues", aspect="auto")
        ax_cm.set_title(
            f"{label} (seed {r['seed']}, acc {r[f'final_{arm.short}_acc']:.3f})\nconfusion",
            fontsize=8,
        )
        ax_cm.set_xticks(range(mainnet.NUM_CLASSES))
        ax_cm.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax_cm.set_yticks(range(mainnet.NUM_CLASSES))
        ax_cm.set_yticklabels(class_names, fontsize=6)
        for k, si in enumerate(sample_idx):
            col = 3 + k
            ax = fig.add_subplot(gs[row, col])
            img = raw_test_x[si].transpose(1, 2, 0)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            correct = bool(preds[si] == test_y[si])
            colour = WONG[3] if correct else WONG[6]
            ax.set_title(
                f"T:{class_names[test_y[si]]}\nP:{class_names[preds[si]]}",
                fontsize=5, color=colour,
            )
            for sp in ax.spines.values():
                sp.set_edgecolor(colour)
                sp.set_linewidth(1.2)
    fig.suptitle(title, fontsize=11)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --- G_leaf curation plots (FWS-hyper only) --------------------------------
def plot_fws_g_leaf_panel(
    per_seed: list[dict],
    save_path: Path,
    *,
    title: str,
    hidden_dim: int,
    arm_short: str = "fws_hyper",
) -> None:
    """Heatmaps of (W_in, W_h, W_out) for each leaf at the FWS-hyper median seed."""
    rep = _select_reps(per_seed, f"final_{arm_short}_acc")
    chosen = rep["median"]
    g_leaf = chosen[f"{arm_short}.g_leaf_params"]

    n_leaves = len(mainnet.LEAF_ORDER)
    fig, axes = plt.subplots(n_leaves, 3, figsize=(7.5, 1.0 * n_leaves + 0.5))
    for row, leaf_name in enumerate(mainnet.LEAF_ORDER):
        p = g_leaf[leaf_name]
        rank = mainnet.LEAF_RANKS[leaf_name]
        panels = [
            (f"W_in ({hidden_dim}×{rank})", p["W_in"]),
            (f"W_h ({hidden_dim}×{hidden_dim})", p["W_h"]),
            (f"W_out (1×{hidden_dim})", p["W_out"]),
        ]
        for col, (kn, arr) in enumerate(panels):
            M = arr if arr.ndim == 2 else arr.reshape(1, -1)
            vmax = float(np.percentile(np.abs(M), 95))
            vmax = vmax if vmax > 0 else (float(np.abs(M).max()) or 1.0)
            axes[row, col].imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            axes[row, col].set_title(f"{leaf_name} — {kn}", fontsize=7)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle(title.format(seed=chosen["seed"]), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_g_leaf_cosine(
    per_seed: list[dict],
    save_path: Path,
    *,
    title: str,
    arm_short: str = "fws_hyper",
) -> None:
    """Pairwise cosine between FWS-hyper's per-leaf flat G_leaf params."""
    rep = _select_reps(per_seed, f"final_{arm_short}_acc")
    chosen = rep["median"]
    M = chosen[f"{arm_short}.g_leaf_cosines"]
    n_leaves = len(mainnet.LEAF_ORDER)
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(M, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    ax.set_xticks(range(n_leaves))
    ax.set_xticklabels(mainnet.LEAF_ORDER, rotation=60, fontsize=7)
    ax.set_yticks(range(n_leaves))
    ax.set_yticklabels(mainnet.LEAF_ORDER, fontsize=7)
    ax.set_title(title.format(seed=chosen["seed"]), fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# --- Research-log writer ---------------------------------------------------
@dataclass
class ReportContext:
    """Inputs to :func:`render_research_log`.

    ``title`` is the H1 of the markdown doc. ``out_path`` is the markdown
    file to write; ``figures_dir`` is the directory each figure is saved
    under (the writer doesn't move them — phase code saves figures using
    paths it controls). ``sections`` is an ordered iterable of
    ``(section_heading, body_markdown)`` pairs, written as
    ``## {heading}\n\n{body}\n\n``.
    """

    title: str
    out_path: Path
    sections: list[tuple[str, str]]
    figures_dir: Path | None = None


def render_research_log(ctx: ReportContext) -> Path:
    """Write the markdown research log; return the written path."""
    ctx.out_path.parent.mkdir(parents=True, exist_ok=True)
    body = f"# {ctx.title}\n\n"
    for heading, content in ctx.sections:
        body += f"## {heading}\n\n{content.rstrip()}\n\n"
    ctx.out_path.write_text(body)
    print(f"\nWrote research log to {ctx.out_path}")
    return ctx.out_path


def render_pdf(md_path: Path, *, quill: str = "quill") -> Path | None:
    """Pipe ``md_path`` through ``quill --to=pdf`` to produce a sibling ``.pdf``.

    Returns the PDF path on success, or ``None`` if ``quill`` is not on
    PATH (PDF rendering is opportunistic; the markdown log is the
    primary artefact).
    """
    pdf_path = md_path.with_suffix(".pdf")
    try:
        with md_path.open("rb") as fin, pdf_path.open("wb") as fout:
            subprocess.run([quill, "--to=pdf"], stdin=fin, stdout=fout, check=True)
    except FileNotFoundError:
        print(f"  quill not on PATH; skipping PDF render of {md_path.name}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"  quill failed on {md_path.name}: {e}")
        return None
    print(f"  rendered {pdf_path}")
    return pdf_path


# --- Markdown table helpers -------------------------------------------------
def stage0_md_table(
    records: dict[str, list[tuple[int, float]]],
    checkpoints: tuple[int, ...],
    verdict_text: str,
    num_steps: int,
    *,
    figure_link: str,
) -> str:
    """Render the Stage-0 falsifier verdict block as markdown."""
    rows = []
    for kind, recs in records.items():
        row = f"| {kind} | " + " | ".join(f"{v:.4g}" for _, v in recs) + " |"
        rows.append(row)
    steps_hdr = " | ".join(f"step {s}" for s in checkpoints)
    return (
        f"For each ``G_H`` output-init scale we ran {num_steps} FWS-hyper "
        f"outer steps (seed 0) and probed ``σ_min(∂render/∂z)`` at the "
        f"checkpoints {checkpoints}.\n\n"
        f"| scale_kind | {steps_hdr} |\n"
        f"|---|" + "|".join(["---"] * len(checkpoints)) + "|\n"
        + "\n".join(rows)
        + "\n\n**Verdict (decision rule: ≥1 OoM spread at final step → proceed):**\n\n"
        + f"```\n{verdict_text}\n```\n"
        + f"\n![Stage 0 falsifier]({figure_link})\n"
    )


def acc_summary_md_table(
    per_seed: list[dict],
    arms_list: list[Arm],
    K_seed: int,
    epochs: int,
) -> str:
    """Render the per-arm median/IQR/min/max accuracy table + per-seed table."""
    summary_rows = []
    for a in arms_list:
        accs = np.array([r[f"final_{a.short}_acc"] for r in per_seed])
        med, iqr, lo, hi = median_iqr_min_max(accs)
        summary_rows.append(
            f"| {a.name} | {med:.4f} | {iqr:.4f} | {lo:.4f} | {hi:.4f} |"
        )

    per_seed_header = "| seed | " + " | ".join(f"{a.name} acc" for a in arms_list) + " |"
    per_seed_sep = "|---|" + "|".join(["---"] * len(arms_list)) + "|"
    per_seed_rows = []
    for r in per_seed:
        cells = " | ".join(f"{r[f'final_{a.short}_acc']:.4f}" for a in arms_list)
        per_seed_rows.append(f"| {r['seed']} | {cells} |")

    return (
        "### Final test accuracy (verbatim)\n\n"
        "| arm | median | IQR | min | max |\n"
        "|---|---|---|---|---|\n"
        + "\n".join(summary_rows)
        + f"\n\n### Per-seed table (K={K_seed}, {epochs} epochs)\n\n"
        + per_seed_header + "\n" + per_seed_sep + "\n"
        + "\n".join(per_seed_rows)
    )


# --- Re-exports (used by phase files for direct figure-key palette access) -
__all__ = [
    "FWS_HYPER_COLOUR",
    "FWS_PARALLEL_COLOUR",
    "W_MATCHED_COLOUR",
    "W_OVERPARAM_COLOUR",
    "WONG",
    "ReportContext",
    "acc_summary_md_table",
    "median_iqr_min_max",
    "plot_acc_trajectories",
    "plot_arm_curation",
    "plot_conv_kernels",
    "plot_final_acc_box",
    "plot_fft_box",
    "plot_fws_g_leaf_panel",
    "plot_g_leaf_cosine",
    "plot_htsr_box",
    "plot_jacobian_spectrum",
    "plot_loss_trajectories",
    "plot_stage0",
    "render_pdf",
    "render_research_log",
    "stage0_md_table",
]

