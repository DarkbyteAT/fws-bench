"""Phase 9 — Recursive SIREN (``G_H`` = SIREN, not SiLU MLP).

The only architectural change from phase 8: ``G_H`` is now a depth-3
sine-only SIREN over ``(z, leaf_emb[id])`` — every layer including the
output ends in a sine. Everything else (mainnet, ``G_leaf``, the four
arms, the Stage-0 falsifier structure, the CIFAR-10 task, the
measurements, the curation rules) carries over from phase 8 unchanged
via ``examples/_common``.

Hypothesis: phase 8's SiLU-MLP ``G_H`` Jacobian (``SiLU' · W`` — bounded,
no spectral structure) didn't survive the chain rule, so the SIREN
spectral character was lost at the ``G_H → G_leaf`` handoff. With
``G_H = SIREN`` the Jacobian is ``omega_0 · cos(omega_0 · x) · W`` at every
layer, so ``∂ render / ∂ z = (∂ G_leaf / ∂ params) · (∂ G_H / ∂ z)`` is
SIREN ∘ SIREN — recursion all the way down.

Run::

    PHASE9_STAGE=falsifier uv run python examples/phase9_recursive_siren.py
    PHASE9_STAGE=k3        uv run python examples/phase9_recursive_siren.py
    PHASE9_STAGE=all       uv run python examples/phase9_recursive_siren.py   # default
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
    slice_g_leaf_flat,
)


RESEARCH_DIR = Path("/Users/ammar/Documents/Research/fractal-weight-spaces/research")
FIGURES_DIR = RESEARCH_DIR / "figures"
RESEARCH_FILE = RESEARCH_DIR / "2026-06-07-phase9-recursive-siren.md"


# --- Phase-specific G_H: depth-3 SIREN -------------------------------------
DIM_LEAF_EMB = 8
G_H_HIDDEN_DIM = 100
G_H_DEPTH = 3                    # number of sine layers; last one outputs the param vector


def g_h_w_out_scale(scale_kind: str) -> float:
    """Output-layer (last sine layer's ``W``) init scale.

    With ``G_H`` a SIREN, ``siren_derived`` is the genuinely paper-faithful
    bound for a layer following ``sin(omega_0 ·)``; the other two are
    deliberately offset deviations from canonical SIREN.

    - ``"existing"``: ``0.1 / sqrt(G_H_HIDDEN_DIM)``, heuristic shrink.
    - ``"siren_derived"``: ``1 / (omega_0 * sqrt(G_H_HIDDEN_DIM))``,
      canonical SIREN-paper init for hidden / output layers.
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
    """Phase-9 ``G_H``: depth-3 sine-only SIREN over ``(z, leaf_emb[id])``.

    Every layer is a sine — including the output layer that produces the
    flat ``G_leaf`` parameter vector. This is what makes the recursion
    SIREN-all-the-way-through. The produced ``G_leaf`` parameters live in
    ``[-1, 1]``; ``G_leaf``'s own ``omega_0 = 30`` then handles the
    downstream linear maps as usual.
    """

    leaf_embedding: Float[Array, "n_leaves dim_leaf_emb"]
    W_in: Float[Array, "hidden in_total"]
    b_in: Float[Array, " hidden"]
    W_h: tuple[Float[Array, "hidden hidden"], ...]
    b_h: tuple[Float[Array, " hidden"], ...]
    W_out: Float[Array, "g_leaf_param_size hidden"]
    b_out: Float[Array, " g_leaf_param_size"]

    def __init__(self, *, key: Array, out_scale_kind: str = "existing") -> None:
        keys = jax.random.split(key, 5 + 2 * (G_H_DEPTH - 2))
        k_emb, k_w_in, k_b_in, k_w_out, k_b_out, *k_hidden = keys
        in_total = DIM_Z + DIM_LEAF_EMB

        emb_bound = 1.0 / jnp.sqrt(jnp.array(DIM_LEAF_EMB, dtype=jnp.float32))
        self.leaf_embedding = jax.random.uniform(
            k_emb, (mainnet.N_LEAVES, DIM_LEAF_EMB), minval=-emb_bound, maxval=emb_bound,
        )

        bound_first = 1.0 / in_total
        self.W_in = jax.random.uniform(k_w_in, (G_H_HIDDEN_DIM, in_total),
                                       minval=-bound_first, maxval=bound_first)
        self.b_in = jax.random.uniform(k_b_in, (G_H_HIDDEN_DIM,),
                                       minval=-bound_first, maxval=bound_first)

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
        return jnp.sin(OMEGA_0 * (self.W_out @ h + self.b_out))

    def produce(self, z: Array, leaf_id: int, rank: int) -> dict[str, Array]:
        return slice_g_leaf_flat(self.produce_flat(z, leaf_id), rank)


# --- Phase-specific schedule ------------------------------------------------
EPOCHS = int(os.environ.get("PHASE9_EPOCHS", 5))
K_SEED = int(os.environ.get("PHASE9_K_SEED", 3))
BATCH_SIZE = 128
FALSIFIER_STEPS = int(os.environ.get("PHASE9_FALSIFIER_STEPS", 5000))
FALSIFIER_CHECKPOINTS: tuple[int, ...] = (0, 100, 1000, FALSIFIER_STEPS)

# Phase-9 σ probes are far lighter than phase 8 because σ values for the
# recursive SIREN G_H are 5-6 OoM larger; block power iteration has poor
# conditioning at this scale. Task accuracy is the headline; spectral
# details are illustrative.
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


def make_fws_hyper_arm(out_scale_kind: str = "siren_derived") -> arms.Arm:
    return arms.make_fws_hyper(
        g_h_init=lambda key: HyperRenderer(key=key, out_scale_kind=out_scale_kind),
        leaf_scale_fn=None,
        name="FWS-hyper-rec",
        short="fws_hyper",
    )


def make_arms_list() -> list[arms.Arm]:
    return [
        # K=3 uses the paper-faithful siren_derived init scale: stage-0 in the
        # original phase 9 showed `existing` (0.01) drifts to a high-σ regime
        # (≈3e7 at step 5000) that destabilises classification training,
        # whereas siren_derived (0.0033 = 1/(omega_0·sqrt(H))) and siren_10x
        # stayed in a stable 1e5–1e6 band.
        make_fws_hyper_arm("siren_derived"),
        arms.make_fws_parallel(),
        arms.make_w_matched(),
        arms.make_w_overparam(),
    ]


# --- σ-probes --------------------------------------------------------------
_HYPER_RENDER = arms.make_fws_hyper(g_h_init=lambda _k: None).render_W  # type: ignore[arg-type]
_PARALLEL_RENDER = arms.make_fws_parallel().render_W


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
    stage = os.environ.get("PHASE9_STAGE", "all")

    print("=" * 72)
    print(f"Phase 9 — Recursive SIREN (G_H = SIREN)  (stage={stage})")
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
    print(f"G_H hidden={G_H_HIDDEN_DIM}, depth={G_H_DEPTH}, z={DIM_Z}, "
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
            FIGURES_DIR / "2026-06-07-phase9-stage0-falsifier.png",
            title="Phase 9 — G_H W_out init-scale falsifier (G_H = SIREN)",
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
                                         FIGURES_DIR / "2026-06-07-phase9-loss-trajectories.png",
                                         title=f"Phase 9 — training-loss trajectories (K={K_SEED})")
        reporting.plot_acc_trajectories(per_seed, arms_list,
                                        FIGURES_DIR / "2026-06-07-phase9-acc-trajectories.png",
                                        title=f"Phase 9 — test-accuracy trajectories (K={K_SEED})")
        reporting.plot_htsr_box(per_seed, arms_list,
                                FIGURES_DIR / "2026-06-07-phase9-htsr-alpha-boxplot.png",
                                title=f"Phase 9 — HT-SR α on fc leaves (K={K_SEED})")
        reporting.plot_fft_box(per_seed, arms_list,
                               FIGURES_DIR / "2026-06-07-phase9-radial-fft-alpha-boxplot.png",
                               title=f"Phase 9 — radial-FFT α on conv leaves (K={K_SEED}; k=5 under-recovery caveat)")
        reporting.plot_final_acc_box(per_seed, arms_list,
                                     FIGURES_DIR / "2026-06-07-phase9-final-acc-boxplot.png",
                                     title=f"Phase 9 — final test accuracy (K={K_SEED}, {EPOCHS} epochs)")
        reporting.plot_jacobian_spectrum(
            per_seed,
            FIGURES_DIR / "2026-06-07-phase9-jacobian-spectrum.png",
            title=f"Phase 9 — Jacobian σ-spectrum at final, top-{SIGMA_TOP_K} (K={K_SEED})",
            sigma_keys={
                "hyper: σ(∂render/∂z)": "sigma_z_hyper",
                "hyper: σ(∂render/∂G_H)": "sigma_gh_hyper",
                "parallel: σ(∂render/∂z)": "sigma_z_par",
            },
        )
        for a in arms_list:
            reporting.plot_conv_kernels(
                per_seed, a,
                FIGURES_DIR / f"2026-06-07-phase9-conv-kernels-{a.short}.png",
                title=f"Phase 9 — {a.name} conv1 kernels (8 × 5×5 RGB, per-filter normalised)",
            )
            reporting.plot_arm_curation(
                per_seed, a, test_y, raw_test_x,
                FIGURES_DIR / f"2026-06-07-phase9-curations-{a.short}.png",
                title=f"Phase 9 — {a.name} curation (best/median/worst by final test acc, K={K_SEED})",
                class_names=data.CIFAR_NAMES,
            )
        reporting.plot_fws_g_leaf_panel(
            per_seed,
            FIGURES_DIR / "2026-06-07-phase9-fws-hyper-leaves.png",
            title="Phase 9 — per-leaf G_leaf params (FWS-hyper median seed {seed})",
            hidden_dim=G_LEAF_HIDDEN_DIM,
        )
        reporting.plot_g_leaf_cosine(
            per_seed,
            FIGURES_DIR / "2026-06-07-phase9-g_leaf-cosine.png",
            title="FWS-hyper: pairwise G_leaf cosine (median seed {seed})",
        )

    stage0_md = ""
    if stage0_verdict is not None:
        stage0_md = reporting.stage0_md_table(
            stage0_verdict.records, stage0_verdict.checkpoints, stage0_verdict.text,
            FALSIFIER_STEPS,
            figure_link="figures/2026-06-07-phase9-stage0-falsifier.png",
        )

    nan_md = "\n".join(f"- {line}" for line in nan_or_crash) if nan_or_crash else "- none."

    sections: list[tuple[str, str]] = [
        ("Background", _BACKGROUND_MD),
    ]
    if stage0_md:
        sections.append(("Stage 0 — G_H W_out init-scale falsifier (BLOCKING)", stage0_md))
    if summary_md:
        figures_md = "\n\n".join([
            "![loss trajectories](figures/2026-06-07-phase9-loss-trajectories.png)",
            "![test accuracy trajectories](figures/2026-06-07-phase9-acc-trajectories.png)",
            "![HT-SR α (fc leaves)](figures/2026-06-07-phase9-htsr-alpha-boxplot.png)",
            "![radial-FFT α (conv leaves)](figures/2026-06-07-phase9-radial-fft-alpha-boxplot.png)",
            "![final test accuracy](figures/2026-06-07-phase9-final-acc-boxplot.png)",
            "![Jacobian σ-spectrum at convergence](figures/2026-06-07-phase9-jacobian-spectrum.png)",
            "### Curated outputs",
            *[f"![{a.name} conv1 kernels](figures/2026-06-07-phase9-conv-kernels-{a.short}.png)"
              for a in arms_list],
            *[f"![{a.name} curation](figures/2026-06-07-phase9-curations-{a.short}.png)"
              for a in arms_list],
            "![Per-leaf G_leaf params (FWS-hyper, median seed)](figures/2026-06-07-phase9-fws-hyper-leaves.png)",
            "![FWS-hyper pairwise G_leaf cosine](figures/2026-06-07-phase9-g_leaf-cosine.png)",
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
        title="Phase 9 — Recursive SIREN ($G_H$ = SIREN) — 2026-06-07",
        out_path=RESEARCH_FILE,
        sections=sections,
        figures_dir=FIGURES_DIR,
    ))
    reporting.render_pdf(RESEARCH_FILE)


_BACKGROUND_MD = """
Phase 8 set up the CIFAR-10 fiducial cell (WideKernelCNN-SiLU mainnet,
natural-rank coord scheme, sine-only ``G_leaf`` with no separate linear
readout, four arms — FWS-hyper / FWS-parallel-no-$G_H$ / W matched / W
overparam) and ran the K=1 ``G_H`` ``W_out`` output-scale falsifier as
its stage-0 gate. The verdict was *init recovery*: the three init scales
converged to within ~0.79 OoM of each other in $\\sigma_{\\min}(\\partial
\\text{render} / \\partial z)$ by step 5000, telling us the architecture
is mechanistically the same as phase 6's failure.

Phase 9 changes one thing and one thing only: **``G_H`` is now a SIREN,
not a SiLU MLP**. The hypothesis is that phase 8's SiLU MLP Jacobian
(``SiLU' · W``: bounded, smooth, no spectral structure) didn't survive
the chain rule, so the SIREN spectral character was lost at the hand-off
from ``G_H`` to ``G_leaf``.
"""

_CAVEATS_MD = """
- **Fiducial cell, K={K_SEED} smoke** — not K=10 confirmation.
- **{EPOCHS} epochs only.** Relative ordering across arms at fixed-budget
  early training is the signal, not the absolute number.
- **Radial-FFT α at k=5 has known under-recovery bias** ([Trello LYBZKFDi](https://trello.com/c/LYBZKFDi)).
- **σ probes are top-{SIGMA_TOP_K} power iteration with reduced iters** —
  recursive-SIREN ``G_H`` produces σ values 5-6 OoM larger than phase 8's.
- **Hessian top eigenvalue** computed on a {EVAL_HESS_BATCH}-image subsample.
- **Wall-clock budget**: stage 1 K=3 ≈ {k3_wall:.1f}s.
"""

_NEXT_MD = """
- If Stage 0 said *init recovery* (same as phase 8): the recursion
  doesn't survive optimisation either. Document the failure verbatim
  and consider pivoting to the post-May-18 supervised-distillation framing.
- If Stage 0 said *FWS prior doing geometric work* (different from
  phase 8): the SIREN $G_H$ change is load-bearing, and K=3 results
  follow below. Next phases: K=10 confirmation, 50-epoch training,
  ablation on $G_H$ depth.
"""


if __name__ == "__main__":
    main()
