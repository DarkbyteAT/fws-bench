# `examples/_common/` — Themed fixtures for CIFAR-10 FWS phases

Phases 8 / 9 / 10 / 11+ of the FWS programme share ~95% of their code: the
WideKernelCNN-SiLU mainnet, the CIFAR-10 loader, the W matched / W overparam /
FWS-parallel-no-G_H baselines, the Adam(1e-3) paired-training loop, the
HT-SR / radial-FFT / Jacobian-σ / Hessian diagnostics, the same eight
plots, and the research-log writer that pipes through `quill` for PDF
output. The actual phase-specific architectural change between two
consecutive phases is ~50-100 LoC of `G_H` plus a projection hook.

This package owns the shared 95%. Each phase file declares its own
`HyperRenderer` (the `G_H`), optional `projection_fn`, the Stage-0 σ-probe,
and a thin `main()` that wires arms + calls common training and reporting.

## Module layout

- **`mainnet.py`** — `LEAF_ORDER`, `LEAF_SHAPES`, `LEAF_RANKS`, `LEAF_SIZES`,
  `cnn_forward`, `cross_entropy_loss`, `init_cnn_params` (direct CNN init for
  the W matched arm), `OverparamCNN` (factored rank-P CNN for W overparam arm).
  The single source of truth for the mainnet topology.

- **`data.py`** — `load_cifar10()` and `load_raw_test()`. Cached to
  `~/.cache/cifar10`. Per-channel-normalised float32 tensors with train-set
  statistics.

- **`arms.py`** — `Arm` value type plus:
  - Shared `G_leaf` machinery (`G_LEAF_HIDDEN_DIM`, `OMEGA_0`, `slice_g_leaf_flat`,
    `g_leaf_forward`, `LEAF_COORDS`, `G_LEAF_PARAM_SIZE`, `MAX_G_LEAF_PARAM_SIZE`,
    `DIM_Z`).
  - `ParallelGLeaves`: three rank-keyed `G_leaf` instances + per-rank FiLM
    projection of the shared `z`.
  - Factory functions `make_w_matched`, `make_w_overparam`, `make_fws_parallel`,
    `make_fws_hyper` — each returns a configured `Arm`. `make_fws_*` take an
    optional `leaf_scale_fn` (multiplicative pre-factor per leaf) and an
    optional `projection_fn` (post-reshape per-leaf projection like polar
    decomposition).
  - `kaiming_leaf_scale`: per-leaf Kaiming-He fan-in std multiplier for the
    init-distribution remap.
  - Wong palette constants (`FWS_HYPER_COLOUR`, `FWS_PARALLEL_COLOUR`,
    `W_MATCHED_COLOUR`, `W_OVERPARAM_COLOUR`).

- **`diagnostics.py`** — `ht_sr_alpha(W)` (heavy-tailed self-regularisation α
  on fc weight matrices), `radial_fft_alpha(W)` (radial-FFT α on conv kernels),
  `per_leaf_alphas` (computes both, dict-keyed by leaf class), `sigma_spectrum_op`
  (wraps `landscape_archaeology.singular_spectrum`), `hessian_top` (top
  Hessian eigenvalues of the loss), `g_leaf_cosine_matrix` (pairwise cosine
  on FWS-hyper's per-leaf `G_leaf` params), `count_params`, `global_l2_norm`.

- **`training.py`** — `stage0_falsifier(arm_factory, scale_kinds, sigma_at_z, ...)`
  runs K=1 training for each scale_kind and decides the proceed/stop verdict
  by log10 spread of σ_min at the final checkpoint. `paired_train_4arm`
  runs all arms over the same batch sequence for K_seed seeds. `Stage0Verdict`
  is a frozen dataclass carrying the records + verdict text + checkpoints.

- **`reporting.py`** — Eight plot families (loss/acc trajectories,
  HT-SR/FFT α boxes, final-acc box, Jacobian-σ spectrum, conv1 kernel
  curation, sample-prediction curation, per-leaf `G_leaf` params, pairwise
  `G_leaf` cosine), plus `ReportContext` + `render_research_log` (writes
  markdown) and `render_pdf` (pipes through `quill`). `stage0_md_table`
  and `acc_summary_md_table` produce the markdown blocks for the research log.

## How a phase file should look

A typical phase example is now ~400-500 LoC of which only ~80-150 LoC is
genuinely phase-specific. The skeleton:

```python
"""Phase N — short description."""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import arms, data, diagnostics, mainnet, reporting, training
from _common.arms import (
    DIM_Z, G_LEAF_HIDDEN_DIM, MAX_G_LEAF_PARAM_SIZE,
    OMEGA_0, slice_g_leaf_flat,
)

# --- Phase-specific G_H ----------------------------------------------------
class HyperRenderer(eqx.Module):
    """Whatever G_H this phase is testing."""
    ...

# --- Optional phase-specific projection ------------------------------------
def project_thing(W, leaf_name): ...

# --- Phase-specific Stage-0 init scales ------------------------------------
SCALE_KINDS = (...)
def g_h_w_out_scale(scale_kind): ...

# --- Arm wiring (~30 LoC) --------------------------------------------------
def make_fws_hyper_arm(out_scale_kind):
    return arms.make_fws_hyper(
        g_h_init=lambda key: HyperRenderer(key=key, out_scale_kind=out_scale_kind),
        leaf_scale_fn=arms.kaiming_leaf_scale,  # if Kaiming remap
        projection_fn=project_thing,             # if polar / Newton-Schulz
    )

def make_arms_list():
    return [
        make_fws_hyper_arm(...),
        arms.make_fws_parallel(...),
        arms.make_w_matched(),
        arms.make_w_overparam(),
    ]

# --- σ-probes (phase-specific because of state shape) ----------------------
_HYPER_RENDER = arms.make_fws_hyper(g_h_init=lambda _k: None).render_W
def sigma_at_z_hyper(state):
    op = lambda z_var: _HYPER_RENDER({"G": state["G"], "z": z_var})
    return diagnostics.sigma_spectrum_op(op, state["z"], k=..., num_iterations=...)

# --- main() (~50 LoC) ------------------------------------------------------
def main():
    train_x, train_y, test_x, test_y = data.load_cifar10()
    raw_test_x = data.load_raw_test()

    stage0 = training.stage0_falsifier(
        arm_factory=make_fws_hyper_arm,
        scale_kinds=SCALE_KINDS,
        train_x=train_x, train_y=train_y,
        sigma_at_z=sigma_at_z_hyper,
        num_steps=5000,
        threshold_oom=1.0,
    )
    reporting.plot_stage0(stage0.records, ...)

    if stage0.proceed:
        per_seed, nan_log, wall = training.paired_train_4arm(
            arms_list=make_arms_list(),
            train_x=train_x, train_y=train_y,
            test_x=test_x, test_y=test_y,
            num_epochs=5, K_seed=3,
            per_seed_diagnostics=per_seed_diagnostics,
        )
        # ... call reporting.plot_* on per_seed

    reporting.render_research_log(reporting.ReportContext(
        title="Phase N — ...",
        out_path=...,
        sections=[("Background", ...), ...],
    ))
    reporting.render_pdf(out_path)
```

## When to put something in `_common/` vs the phase file

**Goes in `_common/`** — anything that's the same across phases:

- Mainnet shape, leaf layout, CIFAR-10 loader.
- Direct CNN init for W matched, factored CNN for W overparam.
- `G_leaf` machinery (forward pass, parameter layout, coord grids,
  ParallelGLeaves).
- Stage-0 falsifier orchestration (the loop, the decision rule).
- Paired 4-arm training (the JIT'd step, the per-seed runner, the
  test-acc evaluator).
- Diagnostics (α, σ, Hessian) — these are formula-driven, not phase-specific.
- Plot families (loss / acc / α / σ / kernels / curations / `G_leaf` heatmap).
- Markdown research-log writer + quill PDF renderer.

**Stays in the phase file** — anything that's *the experiment*:

- `HyperRenderer` (the `G_H`) — the architectural variable being tested.
- Stage-0 `SCALE_KINDS` and `g_h_w_out_scale` — phase-specific scale-rule choices.
- Any optional projection function (polar, Newton-Schulz, etc.).
- σ-probes — they reference the `G_H` shape, which is phase-specific.
- `per_seed_diagnostics` callback — phase-specific extras get layered on top.
- The narrative blocks (research-log body sections).

## Refactor history

This package was extracted on 2026-06-07 from phases 8 / 9 / 10, all of
which had ballooned to 1500-1700 LoC due to shared boilerplate. After
extraction, phases 8 and 9 dropped to ~410 LoC each (75% reduction); the
new phase 10 polar-projection example came in at ~480 LoC. The shared
`_common/` package totals ~1300 LoC across 6 themed modules.
