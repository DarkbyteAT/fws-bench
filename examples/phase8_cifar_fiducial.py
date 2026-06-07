"""Phase 8 — CIFAR-10 fiducial cell with per-leaf hyper-renderer.

Refactored to consume ``examples/_common`` for the shared CIFAR-10 +
WideKernelCNN scaffolding (mainnet, dataloader, W matched / W overparam /
FWS-parallel-no-G_H arms, paired training, diagnostics, plots, research
log). Phase-specific code below: the SiLU-MLP ``G_H`` for the FWS-hyper
arm, the Stage-0 σ-probe, and the experiment wiring.

Run::

    PHASE8_STAGE=falsifier uv run python examples/phase8_cifar_fiducial.py
    PHASE8_STAGE=k3        uv run python examples/phase8_cifar_fiducial.py
    PHASE8_STAGE=all       uv run python examples/phase8_cifar_fiducial.py   # default
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


sys.path.insert(0, str(Path(__file__).parent))
from _common import arms, data, diagnostics, mainnet, reporting, training  # noqa: E402
from _common.arms import (  # noqa: E402
    DIM_Z,
    G_LEAF_HIDDEN_DIM,
    G_LEAF_PARAM_SIZE,
    MAX_G_LEAF_PARAM_SIZE,
    OMEGA_0,
    slice_g_leaf_flat,
)


RESEARCH_DIR = Path("/Users/ammar/Documents/Research/fractal-weight-spaces/research")
FIGURES_DIR = RESEARCH_DIR / "figures"
RESEARCH_FILE = RESEARCH_DIR / "2026-06-07-phase8-cifar-fiducial.md"


# --- Phase-specific G_H: 2-layer SiLU MLP ----------------------------------
DIM_LEAF_EMB = 8
G_H_HIDDEN_DIM = 150


def g_h_w_out_scale(scale_kind: str) -> float:
    """Output-layer init scale for the SiLU-MLP ``G_H``.

    Three choices used by Stage 0:

    - ``"existing"``: ``0.1 / sqrt(G_H_HIDDEN_DIM)``, heuristic shrink.
    - ``"siren_derived"``: ``1 / (omega_0 * sqrt(G_H_HIDDEN_DIM))`` —
      the SIREN-paper bound for layers *following* a ``sin(omega_0 ·)``,
      propagated as a variance-preserving choice on ``G_leaf``'s first layer.
    - ``"siren_10x"``: ``10 × siren_derived``, the over-scaled control.
    """
    sqrt_H = float(jnp.sqrt(jnp.array(G_H_HIDDEN_DIM, dtype=jnp.float32)))
    if scale_kind == "existing":
        return 0.1 / sqrt_H
    if scale_kind == "siren_derived":
        return 1.0 / (OMEGA_0 * sqrt_H)
    if scale_kind == "siren_10x":
        return 10.0 / (OMEGA_0 * sqrt_H)
    raise ValueError(f"unknown scale_kind={scale_kind!r}")


SCALE_KINDS: tuple[str, ...] = ("existing", "siren_derived", "siren_10x")


class HyperRenderer(eqx.Module):
    """Phase-8 ``G_H``: 2-layer SiLU MLP over ``(z, leaf_emb[id])``.

    Maps to a flat vector of length ``MAX_G_LEAF_PARAM_SIZE``. Per leaf,
    take the first ``G_LEAF_PARAM_SIZE[rank]`` entries and unpack into the
    rank-appropriate ``G_leaf`` template.
    """

    leaf_embedding: Float[Array, "n_leaves dim_leaf_emb"]
    W_in: Float[Array, "hidden in_total"]
    b_in: Float[Array, " hidden"]
    W_out: Float[Array, "g_leaf_param_size hidden"]
    b_out: Float[Array, " g_leaf_param_size"]

    def __init__(self, *, key: Array, out_scale_kind: str = "existing") -> None:
        k_emb, k_w_in, k_b_in, k_w_out, k_b_out = jax.random.split(key, 5)
        in_total = DIM_Z + DIM_LEAF_EMB

        emb_bound = 1.0 / jnp.sqrt(jnp.array(DIM_LEAF_EMB, dtype=jnp.float32))
        self.leaf_embedding = jax.random.uniform(
            k_emb, (mainnet.N_LEAVES, DIM_LEAF_EMB), minval=-emb_bound, maxval=emb_bound,
        )

        bound_in = jnp.sqrt(jnp.array(6.0 / in_total, dtype=jnp.float32))
        self.W_in = jax.random.uniform(k_w_in, (G_H_HIDDEN_DIM, in_total),
                                       minval=-bound_in, maxval=bound_in)
        self.b_in = jax.random.uniform(k_b_in, (G_H_HIDDEN_DIM,),
                                       minval=-bound_in, maxval=bound_in)

        bound_out = g_h_w_out_scale(out_scale_kind)
        self.W_out = jax.random.uniform(k_w_out, (MAX_G_LEAF_PARAM_SIZE, G_H_HIDDEN_DIM),
                                        minval=-bound_out, maxval=bound_out)
        self.b_out = jax.random.uniform(k_b_out, (MAX_G_LEAF_PARAM_SIZE,),
                                        minval=-bound_out, maxval=bound_out)

    def produce_flat(self, z: Array, leaf_id: int) -> Array:
        emb = self.leaf_embedding[leaf_id]
        inp = jnp.concatenate([z, emb])
        h = jax.nn.silu(self.W_in @ inp + self.b_in)
        return self.W_out @ h + self.b_out

    def produce(self, z: Array, leaf_id: int, rank: int) -> dict[str, Array]:
        return slice_g_leaf_flat(self.produce_flat(z, leaf_id), rank)


# --- Phase-specific schedule ------------------------------------------------
EPOCHS = int(os.environ.get("PHASE8_EPOCHS", 5))
K_SEED = int(os.environ.get("PHASE8_K_SEED", 3))
FALSIFIER_STEPS = int(os.environ.get("PHASE8_FALSIFIER_STEPS", 5000))
FALSIFIER_CHECKPOINTS: tuple[int, ...] = (0, 100, 1000, FALSIFIER_STEPS)

SIGMA_TOP_K = 10
SIGMA_POWER_ITERS = 60
HESSIAN_POWER_ITERS = 40
EVAL_HESS_BATCH = 1024


# --- Build arms (phase 10's Kaiming remap is OFF in phase 8) ---------------
def make_fws_hyper_arm(out_scale_kind: str = "existing") -> arms.Arm:
    return arms.make_fws_hyper(
        g_h_init=lambda key: HyperRenderer(key=key, out_scale_kind=out_scale_kind),
        leaf_scale_fn=None,
    )


def make_arms_list() -> list[arms.Arm]:
    return [
        make_fws_hyper_arm("existing"),
        arms.make_fws_parallel(),
        arms.make_w_matched(),
        arms.make_w_overparam(),
    ]


# --- σ-probes (phase-specific because they reference HyperRenderer shape) ---
_HYPER_RENDER = arms.make_fws_hyper(g_h_init=lambda _k: None).render_W  # type: ignore[arg-type]
_PARALLEL_RENDER = arms.make_fws_parallel().render_W


def sigma_at_z_hyper(state: dict) -> Array:
    op = lambda z_var: _HYPER_RENDER({"G": state["G"], "z": z_var})  # noqa: E731
    return diagnostics.sigma_spectrum_op(op, state["z"],
                                        k=SIGMA_TOP_K, num_iterations=SIGMA_POWER_ITERS, seed=7)


def sigma_at_gh_hyper(state: dict) -> Array:
    op = lambda G_var: _HYPER_RENDER({"G": G_var, "z": state["z"]})  # noqa: E731
    return diagnostics.sigma_spectrum_op(op, state["G"],
                                        k=SIGMA_TOP_K, num_iterations=SIGMA_POWER_ITERS, seed=13)


def sigma_at_z_parallel(state: dict) -> Array:
    op = lambda z_var: _PARALLEL_RENDER({"P": state["P"], "z": z_var})  # noqa: E731
    return diagnostics.sigma_spectrum_op(op, state["z"],
                                        k=SIGMA_TOP_K, num_iterations=SIGMA_POWER_ITERS, seed=7)


# --- Per-seed phase-specific diagnostics callback ---------------------------
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
    train_x_l = _shared_train_x
    train_y_l = _shared_train_y
    hess_idx = rng.choice(train_x_l.shape[0], size=EVAL_HESS_BATCH, replace=False)
    hess_batch = {"x": jnp.asarray(train_x_l[hess_idx]), "y": jnp.asarray(train_y_l[hess_idx])}
    for short, W in final_Ws.items():
        hess[short] = float(np.asarray(diagnostics.hessian_top(
            W, hess_batch, k=SIGMA_TOP_K, num_iterations=HESSIAN_POWER_ITERS))[0])

    return {
        "sigma_z_hyper": sigma_z_hyper,
        "sigma_gh_hyper": sigma_gh_hyper,
        "sigma_z_par": sigma_z_par,
        "g_leaf_cosines": g_leaf_cosines,
        "g_leaf_params": g_leaf_params,
        **{f"hess_{k}": v for k, v in hess.items()},
    }


# --- Module-level train cache (set up in main(), used by per_seed_diagnostics)
_shared_train_x: np.ndarray
_shared_train_y: np.ndarray


# --- Main --------------------------------------------------------------------
def main() -> None:
    global _shared_train_x, _shared_train_y
    stage = os.environ.get("PHASE8_STAGE", "all")

    print("=" * 72)
    print(f"Phase 8 — CIFAR-10 fiducial cell  (stage={stage})")
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
    print(f"G_H hidden={G_H_HIDDEN_DIM}, z={DIM_Z}, leaf_emb=({mainnet.N_LEAVES},{DIM_LEAF_EMB})")
    for a in arms_list:
        print(f"  {a.name:32s}: {counts[a.short]}")
    print(f"K_seed={K_SEED}, epochs={EPOCHS}, batch={training.BATCH_SIZE}, "
          f"falsifier_steps={FALSIFIER_STEPS}")
    print()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Stage 0 -----------------------------------------------------------
    stage0_verdict: training.Stage0Verdict | None = None
    if stage in ("falsifier", "all"):
        t0 = time.time()
        stage0_verdict = training.stage0_falsifier(
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
            FIGURES_DIR / "2026-06-07-phase8-stage0-falsifier.png",
            title="Phase 8 — G_H W_out init-scale falsifier",
            scale_bounds={k: g_h_w_out_scale(k) for k in SCALE_KINDS},
        )

    # ---- Stage 1: K=3 ------------------------------------------------------
    per_seed: list[dict] = []
    nan_or_crash: list[str] = []
    k3_wall = 0.0
    if stage in ("k3", "all"):
        if stage0_verdict is not None and not stage0_verdict.proceed and stage == "all":
            print("\nSTOP — Stage 0 verdict says init recovery, not FWS prior. K=3 skipped.")
        else:
            per_seed, nan_or_crash, k3_wall = training.paired_train_4arm(
                arms_list=arms_list,
                train_x=train_x, train_y=train_y,
                test_x=test_x, test_y=test_y,
                num_epochs=EPOCHS, K_seed=K_SEED,
                per_seed_diagnostics=per_seed_diagnostics,
            )

    # ---- Plots + research log ----------------------------------------------
    summary_md = ""
    if per_seed:
        for a in arms_list:
            accs = np.array([r[f"final_{a.short}_acc"] for r in per_seed])
            m, iq, lo, hi = reporting.median_iqr_min_max(accs)
            print(f"  {a.name:32s}: median={m:.4f} IQR={iq:.4f} min={lo:.4f} max={hi:.4f}")
        summary_md = reporting.acc_summary_md_table(per_seed, arms_list, K_SEED, EPOCHS)

        title = f"Phase 8 — training-loss trajectories (K={K_SEED})"
        reporting.plot_loss_trajectories(per_seed, arms_list,
                                         FIGURES_DIR / "2026-06-07-phase8-loss-trajectories.png",
                                         title=title)
        reporting.plot_acc_trajectories(per_seed, arms_list,
                                        FIGURES_DIR / "2026-06-07-phase8-acc-trajectories.png",
                                        title=f"Phase 8 — test-accuracy trajectories (K={K_SEED})")
        reporting.plot_htsr_box(per_seed, arms_list,
                                FIGURES_DIR / "2026-06-07-phase8-htsr-alpha-boxplot.png",
                                title=f"Phase 8 — HT-SR α on fc leaves (K={K_SEED})")
        reporting.plot_fft_box(per_seed, arms_list,
                               FIGURES_DIR / "2026-06-07-phase8-radial-fft-alpha-boxplot.png",
                               title=f"Phase 8 — radial-FFT α on conv leaves (K={K_SEED}; k=5 under-recovery caveat)")
        reporting.plot_final_acc_box(per_seed, arms_list,
                                     FIGURES_DIR / "2026-06-07-phase8-final-acc-boxplot.png",
                                     title=f"Phase 8 — final test accuracy (K={K_SEED}, {EPOCHS} epochs)")
        reporting.plot_jacobian_spectrum(
            per_seed,
            FIGURES_DIR / "2026-06-07-phase8-jacobian-spectrum.png",
            title=f"Phase 8 — Jacobian σ-spectrum at final, top-{SIGMA_TOP_K} (K={K_SEED})",
            sigma_keys={
                "hyper: σ(∂render/∂z)": "sigma_z_hyper",
                "hyper: σ(∂render/∂G_H)": "sigma_gh_hyper",
                "parallel: σ(∂render/∂z)": "sigma_z_par",
            },
        )
        for a in arms_list:
            reporting.plot_conv_kernels(
                per_seed, a,
                FIGURES_DIR / f"2026-06-07-phase8-conv-kernels-{a.short}.png",
                title=f"Phase 8 — {a.name} conv1 kernels (8 × 5×5 RGB, per-filter normalised)",
            )
            reporting.plot_arm_curation(
                per_seed, a, test_y, raw_test_x,
                FIGURES_DIR / f"2026-06-07-phase8-curations-{a.short}.png",
                title=f"Phase 8 — {a.name} curation (best/median/worst by final test acc, K={K_SEED})",
                class_names=data.CIFAR_NAMES,
            )
        reporting.plot_fws_g_leaf_panel(
            per_seed,
            FIGURES_DIR / "2026-06-07-phase8-fws-hyper-leaves.png",
            title="Phase 8 — per-leaf G_leaf params (FWS-hyper median seed {seed})",
            hidden_dim=G_LEAF_HIDDEN_DIM,
        )
        reporting.plot_g_leaf_cosine(
            per_seed,
            FIGURES_DIR / "2026-06-07-phase8-g_leaf-cosine.png",
            title="FWS-hyper: pairwise G_leaf cosine (median seed {seed})",
        )

    # ---- Research log ------------------------------------------------------
    stage0_md = ""
    if stage0_verdict is not None:
        stage0_md = reporting.stage0_md_table(
            stage0_verdict.records, stage0_verdict.checkpoints, stage0_verdict.text,
            FALSIFIER_STEPS,
            figure_link="figures/2026-06-07-phase8-stage0-falsifier.png",
        )

    nan_md = "\n".join(f"- {line}" for line in nan_or_crash) if nan_or_crash else "- none."

    sections = [
        ("Background", _BACKGROUND_MD),
        ("Mainnet (WideKernelCNN-SiLU)", _MAINNET_MD.format(
            CONV1_OUT=mainnet.CONV1_OUT, CONV2_OUT=mainnet.CONV2_OUT,
            KERNEL_SIZE=mainnet.KERNEL_SIZE, FC1_IN_DIM=mainnet.FC1_IN_DIM,
            FC1_HIDDEN=mainnet.FC1_HIDDEN, NUM_CLASSES=mainnet.NUM_CLASSES,
            IN_CHANNELS=mainnet.IN_CHANNELS,
            w_n=counts["w_matched"],
        )),
    ]
    if stage0_md:
        sections.append(("Stage 0 — G_H W_out init-scale falsifier (BLOCKING)", stage0_md))
    if summary_md:
        figures_md = "\n\n".join([
            "![loss trajectories](figures/2026-06-07-phase8-loss-trajectories.png)",
            "![test accuracy trajectories](figures/2026-06-07-phase8-acc-trajectories.png)",
            "![HT-SR α (fc leaves)](figures/2026-06-07-phase8-htsr-alpha-boxplot.png)",
            "![radial-FFT α (conv leaves)](figures/2026-06-07-phase8-radial-fft-alpha-boxplot.png)",
            "![final test accuracy](figures/2026-06-07-phase8-final-acc-boxplot.png)",
            "![Jacobian σ-spectrum at convergence](figures/2026-06-07-phase8-jacobian-spectrum.png)",
            "### Curated outputs",
            "Per-arm best/median/worst by final test accuracy (pre-registered, not eyeballed).",
            *[f"![{a.name} conv1 kernels](figures/2026-06-07-phase8-conv-kernels-{a.short}.png)"
              for a in arms_list],
            *[f"![{a.name} curation](figures/2026-06-07-phase8-curations-{a.short}.png)"
              for a in arms_list],
            "![Per-leaf G_leaf params (FWS-hyper, median seed)](figures/2026-06-07-phase8-fws-hyper-leaves.png)",
            "![FWS-hyper pairwise G_leaf cosine](figures/2026-06-07-phase8-g_leaf-cosine.png)",
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
        title="Phase 8 — CIFAR-10 fiducial cell — 2026-06-07",
        out_path=RESEARCH_FILE,
        sections=sections,
        figures_dir=FIGURES_DIR,
    ))
    reporting.render_pdf(RESEARCH_FILE)


# --- Narrative blocks (kept as module-level templates for readability) ------
_BACKGROUND_MD = """
Phases 1–6 ran on a synthetic 2-layer SiLU teacher–student task. Phase 7
introduced the per-leaf hyper-renderer; phases 8–9 lifted it to MNIST and
then CIFAR-10. Per user direction this **renumbered Phase 8** is the
design-doc *fiducial cell*: CIFAR-10 + WideKernelCNN-SiLU with two
architecture corrections and three engineering-team-derived controls.
"""

_MAINNET_MD = """
- conv1: ``({CONV1_OUT}, {IN_CHANNELS}, {KERNEL_SIZE}, {KERNEL_SIZE})`` + bias ``({CONV1_OUT},)``
- SiLU + maxpool 2×2 → ``({CONV1_OUT}, 14, 14)``
- conv2: ``({CONV2_OUT}, {CONV1_OUT}, {KERNEL_SIZE}, {KERNEL_SIZE})`` + bias ``({CONV2_OUT},)``
- SiLU + maxpool 2×2 → ``({CONV2_OUT}, 5, 5)``  (flatten → {FC1_IN_DIM})
- fc1: ``({FC1_HIDDEN}, {FC1_IN_DIM})`` + bias ``({FC1_HIDDEN},)``  SiLU
- fc2: ``({NUM_CLASSES}, {FC1_HIDDEN})`` + bias ``({NUM_CLASSES},)``  (logits)

Total mainnet trainable parameters: **{w_n}**. Kernel size $k=5$ is large
enough to engage the radial-FFT α diagnostic, with the documented
under-recovery bias at small $k$ flagged in the caveats.
"""

_CAVEATS_MD = """
- **Fiducial cell, K={K_SEED} smoke** — not K=10 confirmation.
- **{EPOCHS} epochs only.** Absolute accuracies not benchmark-comparable;
  relative ordering across arms at fixed-budget early training is the signal.
- **Single architecture per arm.** No conv-topology sweep.
- **Radial-FFT α at k=5 has known under-recovery bias** ([Trello LYBZKFDi](https://trello.com/c/LYBZKFDi)).
- **FWS-parallel arm param count differs from FWS-hyper** — controlled
  architecture, not controlled param count.
- **σ probes are top-{SIGMA_TOP_K} power iteration**, not the global spectrum.
- **Hessian top eigenvalue** computed on a {EVAL_HESS_BATCH}-image subsample.
- **Wall-clock budget**: stage 1 K=3 ≈ {k3_wall:.1f}s.
"""

_NEXT_MD = """
Triaged depending on Stage 0 verdict:

- If Stage 0 said *init recovery*: report the failure verbatim. Either
  (a) try architectures that further separate from phase 6's failure
  mode (multiplicative coupling between $G_H$ and $G_{\\text{leaf}}$,
  or learned coord encodings), or (b) move to a different mainnet that
  exercises the FWS prior differently.
- If Stage 0 said *FWS prior doing geometric work* and Stage 1 ran:
  K=10 confirmation at the same budget, then a longer-training run
  (50 epochs) to see whether the early-training ordering survives convergence.
"""


if __name__ == "__main__":
    main()
